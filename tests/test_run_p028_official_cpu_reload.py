"""Mocked CPU-only lifecycle checks; no actual Docker/model/HDF access."""
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).with_name("run_p028_official_cpu_reload.py")
if not PATH.exists():
    PATH = Path(__file__).resolve().parents[1] / "scripts/run_p028_official_cpu_reload.py"
spec = importlib.util.spec_from_file_location("p028_cpu_launcher_test", PATH)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_command_cpu_only_readonly_inputs_deadline(tmp_path):
    value = dict(audit_path="/audit.json", audit_sha="a" * 64)
    command = launcher.create_command(value, tmp_path / "output")
    assert "--gpus" not in command
    assert command[command.index("--runtime") + 1] == "runc"
    assert command[command.index("--memory") + 1] == "8g"
    assert command[command.index("timeout") + 3] == "300"
    assert "CUDA_VISIBLE_DEVICES=" in command
    assert "NVIDIA_VISIBLE_DEVICES=void" in command
    assert all(ro for _, _, ro in launcher.readonly_mounts(value))
    assert len(launcher.readonly_mounts(value)) == 4


def test_actual_frozen_config_and_nine_source_layout_metadata_only():
    assert launcher.CONFIG.is_file()
    assert launcher.sha256(launcher.CONFIG) == launcher.CONFIG_SHA
    manifest = launcher.SOURCE / "source_manifest.json"
    assert manifest.is_file()
    assert launcher.sha256(manifest) == launcher.SOURCE_SHA
    mapping = json.loads(manifest.read_text())
    assert len(mapping) == 9
    for relative, digest in mapping.items():
        path = launcher.SOURCE / relative
        assert path.is_file() and not path.is_symlink()
        assert launcher.sha256(path) == digest


@pytest.mark.parametrize("wrong", [None, "InvocationID", "MainPID", "SubState", "ExecMainStatus"])
def test_exact_retained_training_terminal(monkeypatch, wrong):
    fields = dict(LoadState="loaded", ActiveState="active", SubState="exited", Result="success",
                  ExecMainCode="1", ExecMainStatus="0", MainPID="0", InvocationID=launcher.INVOCATION)
    if wrong:
        fields[wrong] = "wrong"
    monkeypatch.setattr(launcher.subprocess, "check_output", lambda *a, **kw: "\n".join(f"{k}={v}" for k, v in fields.items()))
    if wrong:
        with pytest.raises(RuntimeError):
            launcher.require_training_terminal()
    else:
        assert launcher.require_training_terminal() == fields


def test_existing_output_refused_before_system_calls(tmp_path):
    with pytest.raises(RuntimeError, match="exclusive"):
        launcher.execute({}, tmp_path)


def test_memory_failure_stops_owned_container_via_finally(tmp_path, monkeypatch):
    output = tmp_path / "cpu"
    value = dict(audit_path="/audit.json", audit_sha="a" * 64)
    cid = "c" * 64
    memory = iter([dict(MemFree=40 * launcher.GIB, MemAvailable=60 * launcher.GIB),
                   dict(MemFree=19 * launcher.GIB, MemAvailable=60 * launcher.GIB)])
    monkeypatch.setattr(launcher, "host_memory", lambda: next(memory))
    monkeypatch.setattr(launcher, "require_training_terminal", lambda: {"verified": True})
    monkeypatch.setattr(launcher, "validate_mount_sources", lambda *args: None)
    monkeypatch.setattr(launcher, "validate_created", lambda *args: None)
    monkeypatch.setattr(launcher, "inspect_container", lambda *args: {"State": {"Running": True, "Pid": 123}})
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1))
    monkeypatch.setattr(launcher.subprocess, "check_output", lambda args, **kw: launcher.IMAGE if args[:3] == ["docker", "image", "inspect"] else cid)
    process = SimpleNamespace(poll=lambda: None)
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **kw: process)
    cleaned = []
    monkeypatch.setattr(launcher, "cleanup_owned", lambda *args: cleaned.append(args[0]))
    with pytest.raises(RuntimeError, match="continuous dual20"):
        launcher.execute(value, output)
    assert cleaned == [cid]
    assert (output / "resource_watch.jsonl").exists()


def test_cleanup_and_create_recovery_ast_match_reviewed_base():
    base = Path("/workspace/fluid_control/scripts/run_p028_resource_probe.py")
    before = {n.name: n for n in ast.parse(base.read_text()).body if isinstance(n, ast.FunctionDef)}
    after = {n.name: n for n in ast.parse(PATH.read_text()).body if isinstance(n, ast.FunctionDef)}
    for name in ("cleanup_owned", "recover_created_cid", "inspect_container", "host_memory"):
        assert ast.dump(before[name], include_attributes=False) == ast.dump(after[name], include_attributes=False)


def test_actual_cpu_inspection_rejects_gpu(tmp_path):
    cid = "c" * 64
    inspect = dict(Id=cid, Image=launcher.IMAGE, Name="/" + launcher.CONTAINER,
                   HostConfig=dict(Memory=8 * launcher.GIB, MemorySwap=8 * launcher.GIB,
                                   NetworkMode="none", ReadonlyRootfs=True, Runtime="runc",
                                   DeviceRequests=[], Devices=[]),
                   Mounts=[dict(Source=str(tmp_path), Destination=str(tmp_path), RW=True)],
                   Config={"Cmd": ["fixture"]})
    launcher.validate_created(inspect, cid, tmp_path, [], ["fixture"])
    inspect["HostConfig"]["DeviceRequests"] = [{"gpu": True}]
    with pytest.raises(RuntimeError):
        launcher.validate_created(inspect, cid, tmp_path, [], ["fixture"])
