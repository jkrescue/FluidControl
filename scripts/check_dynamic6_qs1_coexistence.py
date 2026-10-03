"""Fail-closed admission check for one dynamic6 CFD beside reviewed qs1 training."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE_ID = "sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
DATA = ROOT / "data/curated/tandem_cylinders_matched_start_full40_dev30_v1"
ALLOWED = {
    "fluid-control-dev30-quickscreen-h20-qs1.service": {
        "trainer": "train_tandem_fno_rollout.py",
        "config": "tandem_fno_full40_quickscreen_h20",
        "output": ROOT / "artifacts/tandem_fno_full40_dev30_quickscreen_h20_qs1",
    },
}
TRACKED = (
    "train_tandem_fno.py",
    "train_tandem_fno_rollout.py",
    "evaluate_tandem_fno.py",
)


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def active_units() -> list[str]:
    return [
        unit
        for unit in ALLOWED
        if run(["systemctl", "--user", "is-active", "--quiet", unit]).returncode == 0
    ]


def host_science_pids() -> set[int]:
    result = run(["ps", "-eo", "pid=,comm=,args="])
    if result.returncode:
        raise RuntimeError("cannot inspect host processes")
    pids: set[int] = set()
    for line in result.stdout.splitlines():
        fields = line.split(None, 2)
        if len(fields) != 3:
            continue
        pid, command, args = fields
        if command in {"python", "python3", "physicsnemo_env"} and any(
            marker in args for marker in TRACKED
        ):
            pids.add(int(pid))
    return pids


def container_pids(container: str) -> set[int]:
    result = run(["docker", "top", container, "-eo", "pid,args"])
    if result.returncode:
        raise RuntimeError("cannot inspect reviewed training container PIDs")
    return {
        int(fields[0])
        for line in result.stdout.splitlines()
        if (fields := line.split(None, 1)) and fields[0].isdigit()
    }


def valid_container(payload: dict, contract: dict) -> bool:
    host = payload.get("HostConfig", {})
    state = payload.get("State", {})
    command = " ".join(payload.get("Config", {}).get("Cmd") or [])
    mounts = {
        row.get("Destination"): (row.get("Source"), row.get("RW"))
        for row in payload.get("Mounts", [])
    }
    device_requests = host.get("DeviceRequests") or []
    gpu0_only = any(row.get("DeviceIDs") == ["0"] for row in device_requests)
    return (
        state.get("Running") is True
        and payload.get("Image") == IMAGE_ID
        and host.get("NetworkMode") == "none"
        and host.get("Memory") == 64 * 1024**3
        and host.get("NanoCpus") == 8_000_000_000
        and host.get("ReadonlyRootfs") is True
        and host.get("AutoRemove") is True
        and gpu0_only
        and contract["trainer"] in command
        and f'--config-name {contract["config"]}' in command
        and mounts.get("/workspace/devdata") == (str(DATA), False)
        and mounts.get("/workspace/output") == (str(contract["output"]), True)
        and "/workspace/frozen" not in mounts
    )


def mem_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    raise RuntimeError("MemAvailable is unavailable")


def main() -> int:
    units = active_units()
    tracked = host_science_pids()
    if not units:
        if tracked:
            raise RuntimeError("unreviewed training/evaluation process is active")
        print("DYNAMIC6_NO_CONCURRENT_TRAINING")
        return 0
    if len(units) != 1:
        raise RuntimeError("more than one reviewed qs1 H20 service is active")
    if mem_available_gib() < 60:
        raise RuntimeError("qs1 coexistence requires at least 60 GiB MemAvailable")

    listed = run(["docker", "ps", "--format", "{{.ID}}"])
    if listed.returncode:
        raise RuntimeError("cannot enumerate containers")
    matches: list[str] = []
    for container in listed.stdout.split():
        inspected = run(["docker", "inspect", container])
        if inspected.returncode:
            raise RuntimeError("cannot inspect running container")
        payload = json.loads(inspected.stdout)
        if len(payload) != 1:
            raise RuntimeError("unexpected docker inspect result")
        if valid_container(payload[0], ALLOWED[units[0]]):
            matches.append(container)
    if len(matches) != 1:
        raise RuntimeError("reviewed qs1 service lacks exactly one bound training container")
    if not tracked or not tracked <= container_pids(matches[0]):
        raise RuntimeError("training/evaluation PID exists outside reviewed qs1 container")
    print(f"DYNAMIC6_REVIEWED_QS1_COEXISTENCE_OK {units[0]} {matches[0]}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"dynamic6 coexistence rejected: {error}", flush=True)
        raise SystemExit(1) from error
