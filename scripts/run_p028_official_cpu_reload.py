#!/usr/bin/env python3
"""Bounded official CPU reload after exact P028 training terminal; no forward.

This file owns resources only.  The reviewed diagnostic owns the input schema
and all scientific calculations.  Without ``--execute`` no container is
created and no model or HDF file is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


IMAGE = "sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
CONTAINER = "fcp028-official-cpu-reload-20261006"
ENTRY_BASENAME = "train_p028_flow_rollout.py"
GUARD_BASENAME = "spark_ppo_gpu_guard.py"
GUARD_PATH = Path("/workspace/fluid_control/scripts/spark_ppo_gpu_guard.py")
GUARD_SHA256 = "3da61590dedb4ea95c02f83b15d79a935d20ad98a24453e47e1514c38868abec"
GIB = 1024**3


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


ROOT = Path("/workspace/fluid_control")
SOURCE = ROOT / "artifacts/fcp028_formal_source_20261006_immutable/cpu_reload_source"
SOURCE_SHA = "2899934d3f822749e1821088ae8dcb0e369e488fa8e170ffa03143219d26ad25"
CONFIG = ROOT / "artifacts/fcp028_formal_source_20261006_immutable/numerical_source/training_config.yaml"
CONFIG_SHA = "07e55fd11df8030313338cef0344490c3453e515aae9c6b6122e997bad5085d9"
CANDIDATE = ROOT / "artifacts/fcp028_flow_training_20261006/payload"
UNIT = "fluid-control-fcp028-flow-train-20261006.service"
INVOCATION = "c46c60f3c2634802b2646bb094f9d201"


def approval_spec(audit_path, audit_sha):
    require(re.fullmatch(r"[0-9a-f]{64}", audit_sha) is not None, "actual audit SHA required")
    require(audit_path.is_file() and not audit_path.is_symlink() and sha256(audit_path) == audit_sha,
            "actual audit bytes differ")
    audit = json.loads(audit_path.read_text())
    require(audit.get("status") == "FC_P028_CANDIDATE_INTEGRITY_VERIFIED_NOT_ADMISSION"
            and audit.get("training_unit") == UNIT and audit.get("training_invocation") == INVOCATION
            and audit.get("scientific_admission") is False, "actual terminal audit identity differs")
    manifest = SOURCE / "source_manifest.json"
    require(sha256(manifest) == SOURCE_SHA and sha256(CONFIG) == CONFIG_SHA, "runtime/config bytes differ")
    mapping = json.loads(manifest.read_text())
    require(len(mapping) == 9, "nine runtime sources required")
    for relative, digest in mapping.items():
        path = SOURCE / relative
        require(path.resolve().is_relative_to(SOURCE.resolve()) and not path.is_symlink()
                and path.is_file() and sha256(path) == digest, "runtime source differs")
    return {"audit_path": str(audit_path.resolve()), "audit_sha": audit_sha, "audit": audit}


def require_training_terminal():
    fields = ("LoadState", "ActiveState", "SubState", "Result", "ExecMainCode",
              "ExecMainStatus", "MainPID", "InvocationID")
    raw = subprocess.check_output(["systemctl", "--user", "show", UNIT,
                                   *["--property=" + key for key in fields]], text=True, timeout=10)
    actual = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
    require(actual == dict(LoadState="loaded", ActiveState="active", SubState="exited",
                          Result="success", ExecMainCode="1", ExecMainStatus="0", MainPID="0",
                          InvocationID=INVOCATION), "exact successful training terminal required")
    return actual


def readonly_mounts(spec, _unused=None):
    return [(SOURCE, SOURCE, True), (CONFIG, CONFIG, True), (CANDIDATE, CANDIDATE, True),
            (Path(spec["audit_path"]), Path(spec["audit_path"]), True)]


def validate_mount_sources(mounts: list[tuple[Path, Path, bool]]) -> None:
    for source, _, read_only in mounts:
        require(source.exists(), f"mount source absent: {source}")
        require(read_only, "non-output mount must be read-only")


def create_command(spec, output):
    command = ["docker", "create", "--name", CONTAINER,
               "--cidfile", str((output / "evidence/container.cid").resolve()),
               "--runtime", "runc", "--network", "none", "--read-only", "--cap-drop", "ALL",
               "--security-opt", "no-new-privileges", "--user", f"{os.getuid()}:{os.getgid()}",
               "--cpus", "2", "--memory", "8g", "--memory-swap", "8g", "--pids-limit", "512",
               "--tmpfs", "/tmp:rw,nosuid,nodev,size=1g",
               "-e", "CUDA_VISIBLE_DEVICES=", "-e", "NVIDIA_VISIBLE_DEVICES=void",
               "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "OMP_NUM_THREADS=2",
               "-e", "XDG_CACHE_HOME=/tmp/cache", "-e", "LOCAL_CACHE=/tmp/physicsnemo-cache",
               "-e", "WARP_CACHE_PATH=/tmp/warp",
               "-e", f"PYTHONPATH={SOURCE / 'scripts'}:{SOURCE / 'src'}"]
    for source, target, _ in readonly_mounts(spec):
        command += ["--mount", f"type=bind,src={source},dst={target},readonly"]
    command += ["--mount", f"type=bind,src={output.resolve()},dst={output.resolve()}",
                IMAGE, "timeout", "-k", "20", "300", "python", "-u",
                str(SOURCE / "scripts/verify_fcp028_dual_reload.py"),
                "--candidate", str(CANDIDATE), "--candidate-audit", spec["audit_path"],
                "--candidate-audit-sha256", spec["audit_sha"], "--config", str(CONFIG),
                "--source-root", str(SOURCE), "--runtime-source-manifest", str(SOURCE / "source_manifest.json"),
                "--runtime-source-manifest-sha256", SOURCE_SHA,
                "--output", str(output.resolve() / "dual_reload_receipt.json"), "--execute-cpu"]
    return command


def host_memory() -> dict[str, int]:
    rows = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(("MemFree:", "MemAvailable:")):
            key, value, _ = line.split()
            rows[key[:-1]] = int(value) * 1024
    require(set(rows) == {"MemFree", "MemAvailable"}, "host memory fields absent")
    return rows


def validate_created(
    inspect: dict,
    cid: str,
    output: Path,
    expected_mounts: list[tuple[Path, Path, bool]],
    expected_command: list[str],
) -> None:
    require(inspect["Id"] == cid and inspect["Image"] == IMAGE, "created identity differs")
    require(inspect["Name"] == "/" + CONTAINER, "created name differs")
    host = inspect["HostConfig"]
    require(host["Memory"] == 8 * GIB and host["MemorySwap"] == 8 * GIB,
            "8GiB/no-extra-swap contract differs")
    require(host["NetworkMode"] == "none" and host["ReadonlyRootfs"] is True,
            "isolation contract differs")
    require(not host.get("DeviceRequests") and not host.get("Devices") and host.get("Runtime") == "runc",
            "CPU-only runc/no-GPU contract differs")
    expected = {(str(source.resolve()), str(target), not read_only)
                for source, target, read_only in expected_mounts}
    expected.add((str(output.resolve()), str(output.resolve()), True))
    actual = {(row["Source"], row["Destination"], bool(row["RW"]))
              for row in inspect["Mounts"] if row.get("Type", "bind") == "bind"}
    require(actual == expected, "exact bind mapping differs")
    require(inspect["Config"]["Cmd"] == expected_command,
            "created diagnostic/GPU guard command differs")


def inspect_container(cid: str) -> dict:
    raw = subprocess.check_output(["docker", "inspect", cid], text=True, timeout=10)
    rows = json.loads(raw)
    require(len(rows) == 1, "container inspect cardinality differs")
    return rows[0]


def cleanup_owned(cid: str, attached: subprocess.Popen | None, output: Path) -> None:
    """Attempt every cleanup step; retain exact CID and errors for recovery."""
    errors = []
    if attached is not None and attached.poll() is None:
        try:
            stopped = subprocess.run(["docker", "stop", "--timeout", "20", cid],
                                     timeout=30, check=False)
            if stopped.returncode:
                subprocess.run(["docker", "kill", cid], timeout=10, check=True)
        except (OSError, subprocess.SubprocessError) as error:
            errors.append(f"stop/kill: {error!r}")
        try:
            attached.wait(timeout=30)
        except (OSError, subprocess.SubprocessError) as error:
            errors.append(f"attach wait: {error!r}")
    terminal_path = output / "evidence" / "container_terminal.json"
    if not terminal_path.exists():
        try:
            snapshot = inspect_container(cid)
            if snapshot["State"]["Running"] is False:
                terminal_path.write_text(json.dumps(snapshot, indent=2))
            else:
                (output / "evidence" / "container_cleanup_snapshot_nonterminal.json").write_text(
                    json.dumps(snapshot, indent=2))
                errors.append("container remained nonterminal before removal")
        except (OSError, subprocess.SubprocessError, RuntimeError) as error:
            errors.append(f"terminal inspect: {error!r}")
    try:
        subprocess.run(["docker", "rm", "-f", cid], timeout=30, check=True,
                       stdout=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError) as error:
        errors.append(f"remove: {error!r}")
    try:
        remaining_ids = subprocess.check_output(
            ["docker", "container", "ls", "-aq", "--no-trunc"],
            text=True, timeout=10).split()
        if cid in remaining_ids:
            errors.append("CID still exists after owned cleanup")
    except (OSError, subprocess.SubprocessError) as error:
        errors.append(f"post-remove inspect: {error!r}")
    if errors:
        (output / "cleanup_failure.json").write_text(
            json.dumps({"owned_container_id": cid, "errors": errors}, indent=2))
        raise RuntimeError("owned-container cleanup failed; recovery evidence retained")


def recover_created_cid(
    output: Path,
    expected_mounts: list[tuple[Path, Path, bool]],
    expected_command: list[str],
) -> str | None:
    """Recover a create-timeout container only after exact identity validation."""
    cidfile = output / "evidence" / "container.cid"
    deadline = time.monotonic() + 5
    candidates = []
    while not candidates and time.monotonic() < deadline:
        if cidfile.is_file():
            candidates.append(cidfile.read_text().strip())
        try:
            named = inspect_container(CONTAINER)
            candidates.append(named["Id"])
        except (OSError, subprocess.SubprocessError, RuntimeError):
            pass
        candidates = list(dict.fromkeys(candidates))
        if not candidates:
            time.sleep(0.1)
    if not candidates:
        return None
    require(len(candidates) == 1 and re.fullmatch(r"[0-9a-f]{64}", candidates[0]) is not None,
            "unable to recover unique create-timeout CID")
    cid = candidates[0]
    inspect = inspect_container(cid)
    validate_created(inspect, cid, output, expected_mounts, expected_command)
    return cid


def execute(spec: dict, output: Path) -> None:
    require(not output.exists(), "exclusive output already exists")
    memory = host_memory()
    require(memory["MemFree"] >= 30 * GIB and memory["MemAvailable"] >= 50 * GIB,
            "startup 30/50GiB host guard")
    training_terminal = require_training_terminal()
    require(subprocess.run(["docker", "inspect", CONTAINER], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL).returncode != 0,
            "owned container name already exists")
    image = subprocess.check_output(
        ["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"],
        text=True, timeout=10).strip()
    require(image == IMAGE, "official image ID differs")
    validate_mount_sources(readonly_mounts(spec))
    output.mkdir(parents=False)
    (output / "evidence").mkdir()
    (output / "evidence/training_terminal.json").write_text(json.dumps(training_terminal, indent=2))
    cid = ""
    attached = None
    log = None
    stop_requested = False
    deadline = time.monotonic() + 300

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_signals = {
        number: signal.signal(number, request_stop) for number in (signal.SIGINT, signal.SIGTERM)
    }
    command = create_command(spec, output)
    image_index = command.index(IMAGE)
    expected_mounts = readonly_mounts(spec)
    expected_command = command[image_index + 1:]
    try:
        created_cid = subprocess.check_output(command, text=True, timeout=60).strip()
        require(re.fullmatch(r"[0-9a-f]{64}", created_cid) is not None,
                "invalid created CID")
        cid = created_cid
        created = inspect_container(cid)
        validate_created(created, cid, output, expected_mounts, expected_command)
        (output / "evidence" / "container_created.json").write_text(
            json.dumps(created, indent=2))
        log = (output / "run.log").open("x")
        attached = subprocess.Popen(["docker", "start", "-a", cid], stdout=log,
                                    stderr=subprocess.STDOUT)
        start_wait = time.monotonic() + 10
        while attached.poll() is None and not inspect_container(cid)["State"]["Running"]:
            require(time.monotonic() < start_wait, "owned container did not enter running state")
            time.sleep(0.1)
        with (output / "resource_watch.jsonl").open("x") as watch:
            while attached.poll() is None:
                row = host_memory()
                state = inspect_container(cid)["State"]
                row.update(time_unix=time.time(), container_running=state["Running"],
                           container_pid=state["Pid"])
                watch.write(json.dumps(row) + "\n")
                watch.flush()
                if not state["Running"]:
                    # Docker may publish terminal state just before start -a is reaped.
                    attached.wait(timeout=5)
                    break
                require(not stop_requested, "launcher termination requested")
                require(min(row["MemFree"], row["MemAvailable"]) >= 20 * GIB,
                        "continuous dual20GiB host guard")
                require(time.monotonic() < deadline, "whole-job300s deadline")
                time.sleep(2)
        require(attached.returncode == 0, "docker attach reported failure")
        terminal = inspect_container(cid)
        (output / "evidence" / "container_terminal.json").write_text(
            json.dumps(terminal, indent=2))
        state = terminal["State"]
        require(state["Running"] is False and state["ExitCode"] == 0
                and state["OOMKilled"] is False, "container terminal state differs")
        result_path = output / "dual_reload_receipt.json"
        require(result_path.is_file(), "official CPU reload receipt absent")
        result = json.loads(result_path.read_text())
        require(result.get("status") == "FC_P028_OFFICIAL_CPU_DUAL_RELOAD_VERIFIED_NOT_ADMISSION"
                and result.get("candidate_audit_sha256") == spec["audit_sha"]
                and result.get("candidate_sha256") == spec["audit"]["candidate_sha256"]
                and result.get("tensor_sha256") == spec["audit"]["tensor_sha256"]
                and result.get("runtime_source_manifest_sha256") == SOURCE_SHA
                and result.get("official_dual_reload_verified") is True
                and result.get("scientific_admission") is False
                and result.get("gpu_used") is False
                and result.get("optimizer_created") is False
                and result.get("model_saved") is False
                and result.get("forward_performed") is False,
                "actual CPU reload receipt differs")

    finally:
        cleanup_error = None
        if not cid:
            try:
                cid = recover_created_cid(output, expected_mounts, expected_command) or ""
            except RuntimeError as error:
                (output / "create_recovery_failure.json").write_text(
                    json.dumps({"container_name": CONTAINER, "error": repr(error)}, indent=2))
                cid = ""
        if cid:
            try:
                cleanup_owned(cid, attached, output)
            except RuntimeError as error:
                cleanup_error = error
        if log is not None:
            log.close()
        for number, handler in previous_signals.items():
            signal.signal(number, handler)
        if cleanup_error is not None and sys.exc_info()[0] is None:
            raise cleanup_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    spec = approval_spec(args.audit, args.audit_sha256)
    if not args.execute:
        print(json.dumps({"status": "P028_CPU_RELOAD_COMMAND_PREPARED_NO_EXECUTION",
                          "command": create_command(spec, args.output)}, indent=2))
        return
    execute(spec, args.output.resolve())


if __name__ == "__main__":
    main()
