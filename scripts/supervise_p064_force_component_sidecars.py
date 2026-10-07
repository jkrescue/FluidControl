"""Thin resource-bound launcher for the approved P064 label-sidecar conversion."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/workspace/fluid_control")
PYTHON = REPO / ".venv-curator-py312/bin/python"
SOURCE = REPO / "src"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def mem_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    raise ValueError("MemAvailable missing")


def unit_properties(unit: str) -> dict[str, str]:
    keys = ("InvocationID", "ActiveState", "MemoryMax", "MemorySwapMax",
            "CPUQuotaPerSecUSec", "RuntimeMaxUSec")
    result = subprocess.run(
        ["systemctl", "--user", "show", unit, "--property=" + ",".join(keys)],
        check=True, capture_output=True, text=True,
    )
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    require(sha(args.spec) == args.spec_sha256, "spec SHA")
    spec = json.loads(args.spec.read_text())
    require(sha(Path(spec["supervisor"]["path"])) == spec["supervisor"]["sha256"], "supervisor SHA")
    require(mem_available_gib() >= 50, "startup memory")
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(SOURCE),
                "P064_COMPONENT_SIDECAR_TOKEN": "APPROVED"})
    command = [str(PYTHON), spec["worker"]["path"], "--spec", str(args.spec),
               "--spec-sha256", args.spec_sha256, "--output", spec["planned_output"]]
    if not args.execute:
        subprocess.run(command, check=True, env=env)
        print("P064_FORCE_COMPONENT_SIDECAR_SUPERVISOR_PREPARATION_PASS")
        return
    require(spec["execution_authorized"] is True, "execution not authorized")
    require(not Path(spec["planned_output"]).exists(), "output exists")
    props = unit_properties(spec["planned_unit"])
    require(props["InvocationID"] == os.environ.get("INVOCATION_ID"), "invocation")
    require(props["ActiveState"] == "active", "unit state")
    require(props["MemoryMax"] == str(8 * 2**30), "MemoryMax")
    require(props["MemorySwapMax"] == "0", "MemorySwapMax")
    require(props["CPUQuotaPerSecUSec"] == "1s", "CPU quota")
    require(props["RuntimeMaxUSec"] == "2min", "runtime max")
    command.append("--execute")
    child = subprocess.Popen(command, env=env, start_new_session=True)
    deadline = time.monotonic() + 100
    while child.poll() is None:
        if time.monotonic() >= deadline or mem_available_gib() < 22:
            child.terminate()
            try:
                child.wait(10)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()
            raise RuntimeError("child timeout or memory guard")
        time.sleep(0.5)
    require(child.returncode == 0, f"child exit {child.returncode}")
    print(json.dumps({"status": "P064_FORCE_COMPONENT_SIDECAR_SUPERVISOR_COMPLETE",
                      "unit": spec["planned_unit"], "invocation": props["InvocationID"],
                      "limits": props}, sort_keys=True))


if __name__ == "__main__":
    main()
