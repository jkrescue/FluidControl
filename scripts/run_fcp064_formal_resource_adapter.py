#!/usr/bin/env python3
"""Explicit P064-only MemAvailable resource adapter for unchanged formal numerics."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


FORMAL_SHA = "aac728bcf7f568073b723fb640a84f2fcad1e671b55b68ba8a331b77856089c9"
FORMAL = Path(__file__).with_name("run_fcp064_posteval.py")
GIB = 1024**3
STARTUP_AVAILABLE = 50 * GIB
RUNTIME_AVAILABLE = 22 * GIB
RESOURCE_CONTRACT = {
    "startup_mem_available_gib": 50,
    "runtime_mem_available_gib": 22,
    "reserved_mem_available_gib": 20,
    "container_memory_gib": 72,
    "container_memory_swap_gib": 72,
    "container_pids_limit": 1024,
    "allocator_fraction": 0.06,
    "deadline_seconds": 10800,
}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def available_memory() -> dict[str, int]:
    rows = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return {
        key: int(rows[key].split()[0]) * 1024
        for key in ("MemFree", "MemAvailable")
    }


def load_formal():
    require(FORMAL.is_file() and not FORMAL.is_symlink(), "formal wrapper must be regular")
    require(hashlib.sha256(FORMAL.read_bytes()).hexdigest() == FORMAL_SHA, "formal wrapper bytes differ")
    spec = importlib.util.spec_from_file_location("_p064_resource_formal", FORMAL)
    require(spec is not None and spec.loader is not None, "formal wrapper import unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def adapted_run_container(base, args, approval, name, gpu, command, deadline):
    """Copy the reviewed container contract; change only explicit resource handling."""
    output = args.output
    mounts = [(args.source / part, f"/workspace/{part}") for part in ("src", "scripts", "conf", "cfd")]
    mounts += [
        (args.source / "training_config.yaml", "/workspace/training_config.yaml"),
        (args.candidate, "/workspace/dual"),
        (args.repo / base.PREDECL, "/workspace/predecl.json"),
        (args.repo / base.QC, "/workspace/physical_qc.json"),
    ]
    mounts += [(args.repo / base.DATA[part], f"/workspace/{part}") for part in ("devdata", "dynamic")]
    if name == "endpoint_gate":
        mounts = [(path, target) for path, target in mounts if target != "/workspace/devdata"]
        mounts += [
            (args.repo / base.DATA["full40"] / item, f"/workspace/devdata/{item}")
            for item in ("validation", "manifest.json", "normalization.json")
        ]
    create = [
        "docker", "create", "--network", "none", "--cpus", "8",
        "--memory", "72g", "--memory-swap", "72g", "--shm-size", "2g",
        "--pids-limit", "1024", "--cap-drop", "ALL", "--security-opt",
        "no-new-privileges", "--read-only", "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=4g", "--user", f"{os.getuid()}:{os.getgid()}",
        "-w", "/workspace", "-e", "XDG_CACHE_HOME=/tmp/cache", "-e",
        "LOCAL_CACHE=/tmp/physicsnemo-cache", "-e", "WARP_CACHE_PATH=/tmp/warp",
        "-e", "PYTHONDONTWRITEBYTECODE=1", "-e",
        "PYTHONPATH=/workspace/src:/workspace/scripts",
    ]
    if gpu:
        create += ["--gpus", "device=0"]
        command = [
            "python", "-u", "/workspace/scripts/spark_gpu_guard.py",
            "--min-free-gib", "20", "--allocator-fraction", ".06",
            "--margin-gib", "4", "--poll-seconds", "2", "--", *command,
        ]
    else:
        create += ["--runtime", "runc", "-e", "NVIDIA_VISIBLE_DEVICES=void", "-e", "CUDA_VISIBLE_DEVICES="]
    for path, target in mounts:
        create += ["-v", f"{path.resolve()}:{target}:ro"]
    create += ["-v", f"{output.resolve()}:/workspace/output:rw", base.IMAGE, *command]
    cid = subprocess.check_output(create, text=True, timeout=60).strip()
    base.require(re.fullmatch(r"[0-9a-f]{64}", cid) is not None, "invalid created container ID")
    process = None
    try:
        inspect = json.loads(subprocess.check_output(["docker", "inspect", cid], text=True, timeout=10))[0]
        base.require(
            inspect["Image"] == base.IMAGE
            and any(m["Source"] == str(output.resolve()) and m["Destination"] == "/workspace/output" for m in inspect["Mounts"])
            and inspect["HostConfig"]["Memory"] == 72 * GIB
            and inspect["HostConfig"]["MemorySwap"] == 72 * GIB
            and inspect["HostConfig"]["PidsLimit"] == 1024,
            "created container identity/resources differ",
        )
        (output / "evidence" / f"{name}_container.json").write_text(json.dumps(inspect, indent=2))
        with (output / f"{name}.log").open("x") as log, (output / "memory.jsonl").open("a") as memory_log:
            process = subprocess.Popen(["docker", "start", "-a", cid], stdout=log, stderr=subprocess.STDOUT)
            while True:
                row = available_memory()
                row.update(time_unix=time.time(), step=name, resource_profile="p064_memavailable_50_22")
                memory_log.write(json.dumps(row) + "\n")
                memory_log.flush()
                base.require(row["MemAvailable"] >= RUNTIME_AVAILABLE, "P064 MemAvailable below 22 GiB")
                base.require(time.monotonic() < deadline, "formal 3-hour deadline")
                if process.poll() is not None:
                    break
                time.sleep(2)
            base.require(process.returncode == 0, f"{name} failed")
        terminal = json.loads(subprocess.check_output(["docker", "inspect", cid], text=True, timeout=10))[0]
        (output / "evidence" / f"{name}_container_terminal.json").write_text(json.dumps(terminal, indent=2))
        base.require(
            terminal["State"]["ExitCode"] == 0 and not terminal["State"]["OOMKilled"]
            and not terminal["State"]["Running"], "container terminal failure",
        )
    finally:
        subprocess.run(["docker", "rm", "-f", cid], check=True, timeout=30, stdout=subprocess.DEVNULL)
        if process is not None:
            process.wait(timeout=10)


def main():
    require("--approval" in sys.argv, "formal approval argument absent")
    approval_path = Path(sys.argv[sys.argv.index("--approval") + 1])
    approval = json.loads(approval_path.read_text())
    require(
        approval.get("resource_adapter_sha256")
        == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "resource adapter bytes differ",
    )
    require(approval.get("resource_contract") == RESOURCE_CONTRACT, "resource contract differs")
    formal = load_formal()
    if "--execute" in sys.argv:
        require(available_memory()["MemAvailable"] >= STARTUP_AVAILABLE, "P064 startup MemAvailable below 50 GiB")
    inherited = formal.p028._base_runner

    def adapted_base(path):
        base = inherited(path)
        # The inherited main performs one startup check through this hook.
        def startup_memory():
            available = available_memory()["MemAvailable"]
            require(available >= STARTUP_AVAILABLE, "P064 second startup MemAvailable below 50 GiB")
            return {"MemAvailable": available}
        base.memory = startup_memory
        base.run_container = lambda args, approval, name, gpu, command, deadline: adapted_run_container(
            base, args, approval, name, gpu, command, deadline
        )
        return base

    formal.p028._base_runner = adapted_base
    formal.p028.main(formal.EXPERIMENT, entry_file=FORMAL)


if __name__ == "__main__":
    main()
