#!/usr/bin/env python3
"""Persist operational state for train16 training and post-evaluation.

The watchdog is read-only with respect to research work: it records state and
raises explicit alerts, but never starts, restarts, or repairs a job.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path


TRAINING_UNIT = "fluid-control-train16-h100-full-v1-20261004.service"
POSTEVAL_UNIT = "fluid-control-train16-posteval-queue-v1-20261004.service"
MAIN_RUN = Path("artifacts/tandem_fno_control_train16_h100_20261004")
BALANCED_RUN = Path(
    "artifacts/tandem_fno_control_train16_h100_lift_balanced_worker_20261004"
)
IDLE_ALERT_SECONDS = 300


def utc_now() -> datetime:
    return datetime.now(UTC)


def read_json(path: Path, fallback):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return fallback


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def _command(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        args, text=True, capture_output=True, check=False, timeout=15
    )


def most_specific_error(lines: list[str]) -> str | None:
    """Prefer the runtime exception over systemd's later generic failure line."""
    specific = re.compile(
        r"ValueError:|RuntimeError:|Traceback \(most recent call last\)|error: argument",
        re.I,
    )
    generic = re.compile(r"Error:|FAILED|failed", re.I)
    for pattern in (specific, generic):
        for line in reversed(lines):
            if pattern.search(line):
                return line[-500:]
    return None


def unit_state(unit: str) -> dict:
    properties = (
        "ActiveState,SubState,Result,ExecMainStatus,ExecMainStartTimestamp,"
        "ExecMainExitTimestamp"
    )
    result = _command(
        ["systemctl", "--user", "show", unit, f"--property={properties}"]
    )
    values = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    journal = _command(
        ["journalctl", "--user", "-u", unit, "--no-pager", "-n", "40"]
    )
    error = most_specific_error(journal.stdout.splitlines())
    return {
        "unit": unit,
        "active_state": values.get("ActiveState", "unknown"),
        "sub_state": values.get("SubState", "unknown"),
        "result": values.get("Result", "unknown"),
        "exec_main_status": values.get("ExecMainStatus", "unknown"),
        "started_at": values.get("ExecMainStartTimestamp") or None,
        "exited_at": values.get("ExecMainExitTimestamp") or None,
        "last_error_line": error,
    }


def discover_related_units() -> list[str]:
    result = _command(
        [
            "systemctl",
            "--user",
            "list-units",
            "--type=service",
            "--all",
            "--plain",
            "--no-legend",
        ]
    )
    names = []
    for line in result.stdout.splitlines():
        columns = line.lstrip("● ").split()
        if columns and columns[0].startswith("fluid-control-train16-"):
            names.append(columns[0])
    return sorted(set((TRAINING_UNIT, POSTEVAL_UNIT, *names)))


def _cpu_counters() -> list[int]:
    line = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0]
    return [int(value) for value in line.split()[1:]]


def _cpu_usage(current: list[int], previous: list[int] | None):
    if not previous or len(previous) != len(current):
        return None
    total = sum(current) - sum(previous)
    idle = (current[3] + current[4]) - (previous[3] + previous[4])
    if total <= 0 or idle < 0:
        return None
    return 100.0 * (total - idle) / total


def resource_state(previous: dict | None) -> dict:
    counters = _cpu_counters()
    memory = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith(("MemAvailable:", "MemTotal:")):
            key, value, _unit = line.split()
            memory[key.rstrip(":")] = int(value)
    gpu = _command(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
    )
    values = [part.strip() for part in gpu.stdout.splitlines()[0].split(",")] if gpu.stdout.strip() else []
    return {
        "cpu_utilization_pct": _cpu_usage(
            counters, previous.get("cpu_counters") if previous else None
        ),
        "cpu_counters": counters,
        "gpu_utilization_pct": float(values[0]) if len(values) == 3 else None,
        "gpu_temperature_c": float(values[1]) if len(values) == 3 else None,
        "gpu_power_w": float(values[2]) if len(values) == 3 else None,
        "mem_available_gib": memory.get("MemAvailable", 0) / 1024**2,
        "mem_total_gib": memory.get("MemTotal", 0) / 1024**2,
    }


def latest_relevant_file(repo: Path) -> dict | None:
    candidates = []
    for root in (repo / MAIN_RUN, repo / BALANCED_RUN):
        if not root.is_dir():
            continue
        for pattern in (
            "train.log",
            "physicsnemo.log",
            "training_history.json",
            "worker_transfer_complete.json",
            "posteval_complete/**/*.log",
            "posteval_complete/**/*.json",
        ):
            candidates.extend(path for path in root.glob(pattern) if path.is_file())
    if not candidates:
        return None
    path = max(candidates, key=lambda item: item.stat().st_mtime)
    modified = datetime.fromtimestamp(path.stat().st_mtime, UTC)
    return {
        "path": str(path.relative_to(repo)),
        "modified_at_utc": modified.isoformat(timespec="seconds"),
    }


def workflow_progress(repo: Path) -> dict:
    main_history = read_json(repo / MAIN_RUN / "training_history.json", [])
    balanced_history = read_json(repo / BALANCED_RUN / "training_history.json", [])
    main_receipt = read_json(repo / MAIN_RUN / "posteval_complete/receipt.json", None)
    balanced_receipt = read_json(
        repo / BALANCED_RUN / "posteval_complete/receipt.json", None
    )
    return {
        "main_training_epochs": len(main_history) if isinstance(main_history, list) else 0,
        "balanced_training_epochs": (
            len(balanced_history) if isinstance(balanced_history, list) else 0
        ),
        "main_training_complete": (
            isinstance(main_history, list)
            and [row.get("epoch") for row in main_history] == [1, 2]
        ),
        "balanced_training_complete": (
            isinstance(balanced_history, list)
            and [row.get("epoch") for row in balanced_history] == [1, 2]
        ),
        "main_posteval_complete": (
            isinstance(main_receipt, dict)
            and main_receipt.get("status") == "CONTROL_TRAIN16_POSTEVAL_COMPLETE"
        ),
        "balanced_posteval_complete": (
            isinstance(balanced_receipt, dict)
            and balanced_receipt.get("status") == "CONTROL_TRAIN16_POSTEVAL_COMPLETE"
        ),
        "scientific_gate_bypassed": False,
    }


def build_sample(
    repo: Path,
    previous: dict | None,
    units: dict[str, dict],
    resources: dict,
    now: datetime,
) -> dict:
    progress = workflow_progress(repo)
    pending = not (
        progress["main_posteval_complete"]
        and progress["balanced_posteval_complete"]
    )
    active_units = sorted(
        name for name, state in units.items() if state["active_state"] == "active"
    )
    if not pending or active_units:
        idle_since = None
        idle_seconds = 0
    else:
        idle_since = previous.get("no_running_since_utc") if previous else None
        if not idle_since:
            idle_since = now.isoformat(timespec="seconds")
        idle_seconds = max(
            0, int((now - datetime.fromisoformat(idle_since)).total_seconds())
        )
    failed_units = sorted(
        name
        for name, state in units.items()
        if state["active_state"] == "failed" or state["result"] == "exit-code"
    )
    alerts = []
    blocker_reasons = []
    if pending and not active_units and failed_units:
        alerts.append("TRAIN16_POSTEVAL_FAILED_WITH_PENDING_WORK")
        for name in failed_units:
            state = units[name]
            blocker_reasons.append(
                f"{name}: result={state['result']} status={state['exec_main_status']}"
                + (
                    f"; {state['last_error_line']}"
                    if state.get("last_error_line")
                    else ""
                )
            )
    if pending and not active_units and idle_seconds >= IDLE_ALERT_SECONDS:
        alerts.append("TRAIN16_PENDING_WITH_NO_RUNNING_UNIT_FOR_300_SECONDS")
    if resources["mem_available_gib"] < 20.0:
        alerts.append("SPARK_MEMORY_BELOW_20_GIB")
    return {
        "status": "ALERT" if alerts else "MONITORING",
        "timestamp_utc": now.isoformat(timespec="seconds"),
        "policy": {
            "sample_interval_seconds": 60,
            "no_running_alert_seconds": IDLE_ALERT_SECONDS,
            "automatic_restart_or_repair": False,
            "scientific_failure_is_never_bypassed": True,
        },
        "workflow_pending": pending,
        "no_running_since_utc": idle_since,
        "no_running_duration_seconds": idle_seconds,
        "active_units": active_units,
        "failed_units": failed_units,
        "alerts": alerts,
        "blocker_reasons": blocker_reasons,
        "progress": progress,
        "units": units,
        "resources": resources,
        "latest_relevant_file": latest_relevant_file(repo),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/monitor/training_evaluation_watchdog"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output_dir
    if not output.is_absolute():
        output = repo / output
    previous = read_json(output / "latest.json", None)
    units = {name: unit_state(name) for name in discover_related_units()}
    resources = resource_state(previous.get("resources") if previous else None)
    sample = build_sample(repo, previous, units, resources, utc_now())
    output.mkdir(parents=True, exist_ok=True)
    with (output / "samples.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(sample, sort_keys=True, allow_nan=False) + "\n")
    old_alerts = set(previous.get("alerts", [])) if previous else set()
    new_alerts = set(sample["alerts"]) - old_alerts
    if new_alerts:
        with (output / "alerts.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "timestamp_utc": sample["timestamp_utc"],
                        "new_alerts": sorted(new_alerts),
                        "blocker_reasons": sample["blocker_reasons"],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    atomic_json(output / "latest.json", sample)
    print(json.dumps({key: sample[key] for key in ("status", "alerts", "blocker_reasons")}))


if __name__ == "__main__":
    main()
