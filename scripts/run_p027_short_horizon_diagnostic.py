#!/usr/bin/env python3
"""Bounded launcher for the approved P027 read-only diagnostic.

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
CONTAINER = "fcp027-short-horizon-diagnostic-20261006"
ENTRY_BASENAME = "diagnose_p027_short_horizon_errors.py"
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


def load_spec(path: Path, expected_sha256: str) -> dict:
    require(re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is not None,
            "invalid exact spec SHA")
    require(path.is_file() and not path.is_symlink(), "spec must be a regular file")
    require(sha256(path) == expected_sha256, "exact spec SHA differs")
    spec = json.loads(path.read_text())
    require(spec.get("status") == "P027_OFFLINE_DIAGNOSTIC_EXECUTION_APPROVED",
            "P027 execution is not approved")
    require(set(spec["candidates"]) == {"1", "4"}, "exact K1/K4 candidates required")
    require(set(spec["data"]) == {"base", "train8", "train16"},
            "exact train families required")
    require(len(spec["source_files"]) > 0, "source closure is empty")
    return spec


def named_source(spec: dict, basename: str) -> Path:
    found = [Path(item["path"]) for item in spec["source_files"]
             if Path(item["path"]).name == basename]
    require(len(found) == 1, f"exactly one {basename} is required")
    return found[0]


def readonly_mounts(spec: dict, spec_path: Path) -> list[tuple[Path, Path, bool]]:
    """Return narrow bind mounts; the only writable mount is added separately."""
    mounts: list[tuple[Path, Path, bool]] = [(spec_path, spec_path, True)]
    mounts += [(Path(item["path"]), Path(item["path"]), True)
               for item in spec["source_files"]]
    for key in ("config", "baselines_file", "train_audit"):
        path = Path(spec[key]["path"])
        mounts.append((path, path, True))
    predeclaration = Path(spec["train16_predeclaration"]["path"])
    mounts.append((predeclaration, predeclaration, True))
    phase_mapping = Path(spec["source_phase_mapping"]["path"])
    mounts.append((phase_mapping, phase_mapping, True))
    mounts.append((GUARD_PATH, GUARD_PATH, True))
    for arm in ("1", "4"):
        manifest = Path(spec["candidates"][arm]["manifest"])
        mounts.append((manifest.parent, manifest.parent, True))
    for family in ("base", "train8", "train16"):
        root = Path(spec["data"][family]["root"])
        # Never mount the whole data root: validation/frozen siblings stay absent.
        for child in (root / "manifest.json", root / "normalization.json", root / "train"):
            mounts.append((child, child, True))
    unique: dict[tuple[str, str], tuple[Path, Path, bool]] = {}
    for source, target, read_only in mounts:
        unique[(str(source.resolve()), str(target))] = (source.resolve(), target, read_only)
    return list(unique.values())


def validate_mount_sources(mounts: list[tuple[Path, Path, bool]]) -> None:
    for source, _, read_only in mounts:
        require(source.exists(), f"mount source absent: {source}")
        require(read_only, "non-output mount must be read-only")


def pythonpath(spec: dict) -> str:
    paths: set[str] = set()
    for item in spec["source_files"]:
        parent = Path(item["path"]).parent
        paths.update((str(parent), str(parent.parent)))
    return ":".join(sorted(paths))


def create_command(spec: dict, spec_path: Path, spec_sha: str, output: Path) -> list[str]:
    entry = named_source(spec, ENTRY_BASENAME)
    guard = GUARD_PATH
    command = [
        "docker", "create", "--name", CONTAINER,
        "--cidfile", str((output / "evidence" / "container.cid").resolve()),
        "--gpus", "device=0",
        "--network", "none", "--read-only", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges", "--user", f"{os.getuid()}:{os.getgid()}",
        "--cpus", "8", "--memory", "12g", "--memory-swap", "12g",
        "--pids-limit", "1024", "--shm-size", "2g",
        "--tmpfs", "/tmp:rw,nosuid,nodev,size=2g",
        "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "XDG_CACHE_HOME=/tmp/cache",
        "-e", "LOCAL_CACHE=/tmp/physicsnemo-cache", "-e", "WARP_CACHE_PATH=/tmp/warp",
        "-e", f"PYTHONPATH={pythonpath(spec)}",
    ]
    for source, target, _ in readonly_mounts(spec, spec_path):
        command += ["--mount", f"type=bind,src={source},dst={target},readonly"]
    command += ["--mount", f"type=bind,src={output.resolve()},dst={output.resolve()}"]
    command += [IMAGE, "python", "-u", str(guard), "--min-free-gib", "20",
                "--allocator-fraction", ".06", "--margin-gib", "4", "--poll-seconds", "2", "--",
                "timeout", "-k", "20", "900", "python", "-u", str(entry),
                "--spec", str(spec_path.resolve()), "--spec-sha256", spec_sha,
                "--output", str((output / "result.json").resolve()), "--execute"]
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
    require(host["Memory"] == 12 * GIB and host["MemorySwap"] == 12 * GIB,
            "12GiB/no-extra-swap contract differs")
    require(host["NetworkMode"] == "none" and host["ReadonlyRootfs"] is True,
            "isolation contract differs")
    requests = host.get("DeviceRequests") or []
    require(any("gpu" in row.get("Capabilities", [[]])[0] and row.get("DeviceIDs") == ["0"]
                for row in requests), "GPU0 request differs")
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


def running_gpu_containers() -> list[str]:
    ids = subprocess.check_output(["docker", "ps", "-q"], text=True, timeout=10).split()
    active = []
    for cid in ids:
        inspect = inspect_container(cid)
        requests = inspect.get("HostConfig", {}).get("DeviceRequests") or []
        if any("gpu" in capability
               for row in requests for group in row.get("Capabilities", [])
               for capability in group):
            active.append(cid)
    return active


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


def execute(spec: dict, spec_path: Path, spec_sha: str, output: Path) -> None:
    require(not output.exists(), "exclusive output already exists")
    memory = host_memory()
    require(memory["MemFree"] >= 30 * GIB and memory["MemAvailable"] >= 50 * GIB,
            "startup 30/50GiB host guard")
    gpu_pids = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True, timeout=10).strip()
    require(not gpu_pids, "another GPU compute task is active")
    require(not running_gpu_containers(), "another running GPU container is active")
    require(subprocess.run(["docker", "inspect", CONTAINER], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL).returncode != 0,
            "owned container name already exists")
    image = subprocess.check_output(
        ["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"],
        text=True, timeout=10).strip()
    require(image == IMAGE, "official image ID differs")
    require(GUARD_PATH.is_file() and sha256(GUARD_PATH) == GUARD_SHA256,
            "reviewed CUDA guard differs")
    validate_mount_sources(readonly_mounts(spec, spec_path))
    output.mkdir(parents=False)
    (output / "evidence").mkdir()
    cid = ""
    attached = None
    log = None
    stop_requested = False
    deadline = time.monotonic() + 900

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_signals = {
        number: signal.signal(number, request_stop) for number in (signal.SIGINT, signal.SIGTERM)
    }
    command = create_command(spec, spec_path, spec_sha, output)
    image_index = command.index(IMAGE)
    expected_mounts = readonly_mounts(spec, spec_path)
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
                require(time.monotonic() < deadline, "whole-job900s deadline")
                time.sleep(2)
        require(attached.returncode == 0, "docker attach reported failure")
        terminal = inspect_container(cid)
        (output / "evidence" / "container_terminal.json").write_text(
            json.dumps(terminal, indent=2))
        state = terminal["State"]
        require(state["Running"] is False and state["ExitCode"] == 0
                and state["OOMKilled"] is False, "container terminal state differs")
        require((output / "result.json").is_file(), "diagnostic result absent")
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
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    spec = load_spec(args.spec, args.spec_sha256)
    if not args.execute:
        print(json.dumps({"status": "P027_LAUNCH_PREPARED_NO_DOCKER_NO_GPU",
                          "image": IMAGE,
                          "command": create_command(spec, args.spec.resolve(),
                                                    args.spec_sha256, args.output.resolve())}))
        return
    execute(spec, args.spec.resolve(), args.spec_sha256, args.output.resolve())


if __name__ == "__main__":
    main()
