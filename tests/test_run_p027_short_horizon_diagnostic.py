from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import subprocess

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/run_p027_short_horizon_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("p027_launcher", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path: Path) -> tuple[Path, dict]:
    source = tmp_path / "source"
    scripts = source / "scripts"
    scripts.mkdir(parents=True)
    entry = scripts / module.ENTRY_BASENAME
    guard = scripts / module.GUARD_BASENAME
    entry.write_text("# diagnostic\n")
    guard.write_text("# guard\n")
    config = tmp_path / "config.yaml"
    baselines = tmp_path / "baselines.json"
    audit = tmp_path / "audit.json"
    config.write_text("x: 1\n")
    baselines.write_text("{}")
    audit.write_text("{}")
    predeclaration = tmp_path / "train16_predeclaration.json"
    predeclaration.write_text("{}")
    phase_mapping = tmp_path / "source_phase_mapping.json"
    phase_mapping.write_text("{}")
    data = {}
    for family in ("base", "train8", "train16"):
        root = tmp_path / family
        (root / "train").mkdir(parents=True)
        (root / "manifest.json").write_text("{}")
        (root / "normalization.json").write_text("{}")
        data[family] = {"root": str(root), "manifest_sha256": "a" * 64,
                        "normalization_sha256": "b" * 64, "train_files": {}}
    candidates = {}
    for arm in ("1", "4"):
        manifest = tmp_path / f"candidate{arm}" / "dual_manifest.json"
        manifest.parent.mkdir()
        manifest.write_text("{}")
        candidates[arm] = {"manifest": str(manifest), "manifest_sha256": digest(manifest)}
    payload = {
        "status": "P027_OFFLINE_DIAGNOSTIC_EXECUTION_APPROVED",
        "source_files": [{"path": str(entry), "sha256": digest(entry)},
                         {"path": str(guard), "sha256": digest(guard)}],
        "config": {"path": str(config), "sha256": digest(config)},
        "baselines_file": {"path": str(baselines), "sha256": digest(baselines)},
        "train_audit": {"path": str(audit)},
        "train16_predeclaration": {"path": str(predeclaration),
                                   "sha256": digest(predeclaration)},
        "source_phase_mapping": {"path": str(phase_mapping),
                                 "sha256": digest(phase_mapping)},
        "candidates": candidates, "data": data,
    }
    spec = tmp_path / "approval.json"
    spec.write_text(json.dumps(payload))
    return spec, payload


def test_exact_spec_hash_and_status(tmp_path):
    spec, payload = fixture(tmp_path)
    assert module.load_spec(spec, digest(spec)) == payload
    with pytest.raises(RuntimeError, match="SHA differs"):
        module.load_spec(spec, "0" * 64)
    payload["status"] = "DESIGN_ONLY"
    spec.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="not approved"):
        module.load_spec(spec, digest(spec))


def test_command_is_bounded_and_mounts_only_train_data(tmp_path):
    spec_path, payload = fixture(tmp_path)
    output = tmp_path / "exclusive-output"
    old_guard = module.GUARD_PATH
    module.GUARD_PATH = Path(payload["source_files"][1]["path"])
    try:
        command = module.create_command(payload, spec_path, digest(spec_path), output)
    finally:
        module.GUARD_PATH = old_guard
    joined = " ".join(command)
    assert command[:4] == ["docker", "create", "--name", module.CONTAINER]
    assert module.IMAGE in command
    assert "--cidfile" in command
    assert "--memory 12g --memory-swap 12g" in joined
    assert "--allocator-fraction .06 --margin-gib 4 --poll-seconds 2" in joined
    assert "timeout -k 20 900" in joined
    assert "--gpus device=0" in joined
    assert "/train,readonly" in joined
    assert "validation" not in joined and "frozen" not in joined
    assert joined.count("dst=" + str(output.resolve())) == 1


def test_created_container_requires_exact_owned_contract(tmp_path):
    _, payload = fixture(tmp_path)
    output = tmp_path / "output"
    cid = "1" * 64
    mounts = [(Path("/read"), Path("/read"), True)]
    expected_command = ["python", "diagnostic.py"]
    inspect = {
        "Id": cid, "Image": module.IMAGE, "Name": "/" + module.CONTAINER,
        "HostConfig": {"Memory": 12 * module.GIB, "MemorySwap": 12 * module.GIB,
                       "NetworkMode": "none", "ReadonlyRootfs": True,
                       "DeviceRequests": [{"DeviceIDs": ["0"], "Capabilities": [["gpu"]]}]},
        "Mounts": [{"Type": "bind", "Source": str(output.resolve()),
                    "Destination": str(output.resolve()), "RW": True},
                   {"Type": "bind", "Source": "/read", "Destination": "/read", "RW": False}],
        "Config": {"Cmd": expected_command},
    }
    module.validate_created(inspect, cid, output, mounts, expected_command)
    inspect["Image"] = "sha256:" + "0" * 64
    with pytest.raises(RuntimeError, match="identity"):
        module.validate_created(inspect, cid, output, mounts, expected_command)


@pytest.mark.parametrize("field,value", [("MemorySwap", 13 * module.GIB),
                                          ("NetworkMode", "bridge")])
def test_created_container_rejects_resource_or_isolation_drift(tmp_path, field, value):
    output = tmp_path / "output"
    cid = "2" * 64
    host = {"Memory": 12 * module.GIB, "MemorySwap": 12 * module.GIB,
            "NetworkMode": "none", "ReadonlyRootfs": True,
            "DeviceRequests": [{"DeviceIDs": ["0"], "Capabilities": [["gpu"]]}]}
    host[field] = value
    inspect = {"Id": cid, "Image": module.IMAGE, "Name": "/" + module.CONTAINER,
               "HostConfig": host,
               "Mounts": [{"Type": "bind", "Source": str(output.resolve()),
                           "Destination": str(output.resolve()), "RW": True}],
               "Config": {"Cmd": ["python"]}}
    with pytest.raises(RuntimeError):
        module.validate_created(inspect, cid, output, [], ["python"])


def test_only_output_may_be_writable(tmp_path):
    output = tmp_path / "output"
    cid = "3" * 64
    inspect = {"Id": cid, "Image": module.IMAGE, "Name": "/" + module.CONTAINER,
               "HostConfig": {"Memory": 12 * module.GIB, "MemorySwap": 12 * module.GIB,
                              "NetworkMode": "none", "ReadonlyRootfs": True,
                              "DeviceRequests": [{"DeviceIDs": ["0"],
                                                  "Capabilities": [["gpu"]]}]},
               "Mounts": [{"Type": "bind", "Source": str(output.resolve()),
                           "Destination": str(output.resolve()), "RW": True},
                          {"Type": "bind", "Source": "/candidate",
                           "Destination": "/candidate", "RW": True}],
               "Config": {"Cmd": ["python"]}}
    with pytest.raises(RuntimeError, match="mapping"):
        module.validate_created(inspect, cid, output, [], ["python"])


def test_missing_mount_source_is_rejected_before_container(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(RuntimeError, match="mount source absent"):
        module.validate_mount_sources([(missing, missing, True)])


def test_cleanup_attempts_remove_and_retains_failure_evidence(tmp_path, monkeypatch):
    output = tmp_path / "output"
    (output / "evidence").mkdir(parents=True)
    cid = "4" * 64

    class Attached:
        def poll(self):
            return None

        def wait(self, timeout):
            return 1

    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[1] == "stop":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(module.subprocess, "check_output",
                        lambda *_args, **_kwargs: cid + "\n")
    monkeypatch.setattr(module, "inspect_container",
                        lambda _cid: {"Id": _cid, "State": {"Running": True}})
    with pytest.raises(RuntimeError, match="cleanup failed"):
        module.cleanup_owned(cid, Attached(), output)
    assert any(command[1:3] == ["rm", "-f"] for command in calls)
    failure = json.loads((output / "cleanup_failure.json").read_text())
    assert failure["owned_container_id"] == cid
    assert any("stop/kill" in row for row in failure["errors"])
    assert any("still exists" in row for row in failure["errors"])


def test_create_timeout_cid_is_recovered_only_after_exact_identity(tmp_path, monkeypatch):
    output = tmp_path / "output"
    (output / "evidence").mkdir(parents=True)
    cid = "5" * 64
    (output / "evidence" / "container.cid").write_text(cid)
    command = ["python"]
    mounts = [(Path("/read"), Path("/read"), True)]
    inspect = {
        "Id": cid, "Image": module.IMAGE, "Name": "/" + module.CONTAINER,
        "HostConfig": {"Memory": 12 * module.GIB, "MemorySwap": 12 * module.GIB,
                       "NetworkMode": "none", "ReadonlyRootfs": True,
                       "DeviceRequests": [{"DeviceIDs": ["0"], "Capabilities": [["gpu"]]}]},
        "Mounts": [{"Type": "bind", "Source": "/read", "Destination": "/read", "RW": False},
                   {"Type": "bind", "Source": str(output.resolve()),
                    "Destination": str(output.resolve()), "RW": True}],
        "Config": {"Cmd": command}, "State": {"Running": False},
    }
    monkeypatch.setattr(module, "inspect_container", lambda _target: inspect)
    assert module.recover_created_cid(output, mounts, command) == cid
    inspect["Config"]["Cmd"] = ["wrong"]
    with pytest.raises(RuntimeError, match="command differs"):
        module.recover_created_cid(output, mounts, command)


def test_sigterm_request_enters_finally_and_retains_terminal_cleanup(tmp_path, monkeypatch):
    spec_path, payload = fixture(tmp_path)
    output = tmp_path / "output"
    cid = "6" * 64
    callbacks = {}
    state = {"Running": True, "Pid": 123, "ExitCode": 0, "OOMKilled": False}
    calls = []

    monkeypatch.setattr(module, "GUARD_PATH", Path(payload["source_files"][1]["path"]))
    monkeypatch.setattr(module, "GUARD_SHA256", digest(module.GUARD_PATH))
    monkeypatch.setattr(module, "host_memory",
                        lambda: {"MemFree": 60 * module.GIB,
                                 "MemAvailable": 60 * module.GIB})
    monkeypatch.setattr(module, "running_gpu_containers", lambda: [])
    monkeypatch.setattr(module, "validate_created", lambda *args: None)
    monkeypatch.setattr(module.signal, "signal",
                        lambda number, callback: callbacks.setdefault(number, callback))

    def check_output(command, **_kwargs):
        if command[:2] == ["nvidia-smi", "--query-compute-apps=pid"]:
            return ""
        if command[:3] == ["docker", "image", "inspect"]:
            return module.IMAGE + "\n"
        if command[:2] == ["docker", "create"]:
            (output / "evidence" / "container.cid").write_text(cid)
            return cid + "\n"
        if command[:4] == ["docker", "container", "ls", "-aq"]:
            return ""
        raise AssertionError(command)

    class Attached:
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            self.returncode = 143
            return self.returncode

    def popen(command, **_kwargs):
        assert command[:3] == ["docker", "start", "-a"]
        callbacks[signal.SIGTERM](signal.SIGTERM, None)
        return Attached()

    def run(command, **_kwargs):
        calls.append(command)
        if command[:2] == ["docker", "inspect"] and command[-1] == module.CONTAINER:
            return subprocess.CompletedProcess(command, 1)
        if command[:2] == ["docker", "stop"]:
            state["Running"] = False
            state["Pid"] = 0
            state["ExitCode"] = 143
            return subprocess.CompletedProcess(command, 0)
        if command[:3] == ["docker", "rm", "-f"]:
            return subprocess.CompletedProcess(command, 0)
        if command[:2] == ["docker", "inspect"]:
            return subprocess.CompletedProcess(command, 1)
        raise AssertionError(command)

    inspect = {"Id": cid, "Image": module.IMAGE, "Name": "/" + module.CONTAINER,
               "State": state}
    monkeypatch.setattr(module, "inspect_container", lambda _cid: inspect)
    monkeypatch.setattr(module.subprocess, "check_output", check_output)
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="termination requested"):
        module.execute(payload, spec_path, digest(spec_path), output)
    assert (output / "evidence" / "container_terminal.json").is_file()
    assert any(command[:3] == ["docker", "rm", "-f"] for command in calls)
