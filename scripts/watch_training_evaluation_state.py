#!/usr/bin/env python3
"""Persist authoritative train16 training and post-evaluation state.

This sampler classifies current authority separately from superseded failures.
It never starts work; the separate reconciler may execute only reviewed,
idempotent actions from its fixed allow-list.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path


TRAINING_UNIT = "fluid-control-train16-h100-full-v1-20261004.service"
MAIN_AUTHORITY_UNIT = "fluid-control-train16-posteval-main-v3-20261004.service"
WORKER_AUTHORITY_UNIT = (
    "fluid-control-balanced-posteval-worker-resume-v1-20261004.service"
)
POSTEVAL_UNIT = MAIN_AUTHORITY_UNIT  # Backward-compatible test/import alias.
SUPERSEDED_MAIN_UNITS = (
    "fluid-control-train16-posteval-queue-v1-20261004.service",
    "fluid-control-train16-posteval-main-v2-20261004.service",
)
WORKER_HOST = "USER@WORKER_HOST"
MAIN_RUN = Path("artifacts/tandem_fno_control_train16_h100_20261004")
BALANCED_RUN = Path(
    "artifacts/tandem_fno_control_train16_h100_lift_balanced_worker_20261004"
)
IDLE_ALERT_SECONDS = 300
MAIN_RECEIPT = MAIN_RUN / "posteval_complete_v2/receipt.json"
WORKER_RECEIPT = BALANCED_RUN / "posteval_worker_v1/receipt.json"
REVIEWED_MAIN_RESUME_ACTION = "main-posteval-resume-979fbd6"
PRODUCTION_AUTO_RECOVERY_ENABLED = False


def classify_authority_task(state: dict, complete: bool, *, allow_resume: bool) -> tuple[str, str | None]:
    if complete:
        return "STAGE_COMPLETE", None
    if state.get("active_state") == "active":
        return "RUNNING", None
    error = (state.get("last_error_line") or "").lower()
    must_stop = (
        "lineage",
        "sha mismatch",
        "hash mismatch",
        "schema",
        "nonfinite",
        "out of memory",
        "oom",
        "20 gib",
    )
    if any(marker in error for marker in must_stop):
        return "NEEDS_AGENT_ANALYSIS", None
    known_transient = (
        "invalid choice: 'control_train16_development'",
        "container startup",
        "gpu_guard",
    )
    if PRODUCTION_AUTO_RECOVERY_ENABLED and allow_resume and (
        state.get("result") == "success"
        or any(marker in error for marker in known_transient)
    ):
        return "RETRY_ELIGIBLE", REVIEWED_MAIN_RESUME_ACTION
    return "NEEDS_AGENT_ANALYSIS", None


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_receipt(path: Path, expected_status: str) -> tuple[bool, list[str]]:
    receipt = read_json(path, None)
    if not isinstance(receipt, dict):
        return False, ["receipt missing or invalid JSON"]
    issues = []
    if receipt.get("status") != expected_status:
        issues.append("receipt status differs")
    hashes = receipt.get("sha256")
    if not isinstance(hashes, dict) or not hashes:
        issues.append("receipt sha256 table missing")
        return False, issues
    root = path.parent.resolve()
    for relative, expected in hashes.items():
        target = (root / relative).resolve()
        if root not in target.parents or not target.is_file():
            issues.append(f"missing/unsafe payload: {relative}")
        elif file_sha256(target) != expected:
            issues.append(f"payload SHA differs: {relative}")
    return not issues, issues


def select_main_authority(units: dict[str, dict]) -> str:
    candidates = [
        name
        for name in units
        if name.startswith("fluid-control-train16-posteval-main-")
    ]
    active = [name for name in candidates if units[name].get("active_state") == "active"]
    if active:
        return max(active, key=lambda name: units[name].get("started_at") or "")
    versioned = []
    for name in candidates:
        match = re.search(r"-v(\d+)-", name)
        versioned.append((int(match.group(1)) if match else 0, name))
    return max(versioned, default=(0, MAIN_AUTHORITY_UNIT))[1]


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


def worker_unit_state(unit: str) -> dict:
    """Read Worker systemd state through the existing key-only SSH path."""
    command = (
        "systemctl --user show "
        f"{unit} --property=ActiveState,SubState,Result,ExecMainStatus,"
        "ExecMainStartTimestamp,ExecMainExitTimestamp --no-pager"
    )
    result = _command(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            WORKER_HOST,
            command,
        ]
    )
    if result.returncode != 0:
        return {
            "unit": unit,
            "node": "worker78",
            "active_state": "unknown",
            "sub_state": "unknown",
            "result": "unknown",
            "exec_main_status": "unknown",
            "started_at": None,
            "exited_at": None,
            "last_error_line": result.stderr.strip()[-500:] or "Worker SSH unavailable",
        }
    values = dict(
        line.split("=", 1)
        for line in result.stdout.splitlines()
        if "=" in line
    )
    return {
        "unit": unit,
        "node": "worker78",
        "active_state": values.get("ActiveState", "unknown"),
        "sub_state": values.get("SubState", "unknown"),
        "result": values.get("Result", "unknown"),
        "exec_main_status": values.get("ExecMainStatus", "unknown"),
        "started_at": values.get("ExecMainStartTimestamp") or None,
        "exited_at": values.get("ExecMainExitTimestamp") or None,
        "last_error_line": None,
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
    return sorted(set((TRAINING_UNIT, MAIN_AUTHORITY_UNIT, *SUPERSEDED_MAIN_UNITS, *names)))


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
            "posteval_complete_v2/**/*.log",
            "posteval_complete_v2/**/*.json",
            "posteval_worker_v1/**/*.log",
            "posteval_worker_v1/**/*.json",
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
    main_complete, main_receipt_issues = verify_receipt(
        repo / MAIN_RECEIPT, "CONTROL_TRAIN16_POSTEVAL_COMPLETE"
    )
    balanced_complete, balanced_receipt_issues = verify_receipt(
        repo / WORKER_RECEIPT,
        "CONTROL_TRAIN16_H100_LIFT_BALANCED_WORKER_POSTEVAL_COMPLETE",
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
        "main_posteval_complete": main_complete,
        "balanced_posteval_complete": balanced_complete,
        "main_receipt_issues": main_receipt_issues,
        "balanced_receipt_issues": balanced_receipt_issues,
        "main_receipt_path": str(MAIN_RECEIPT),
        "balanced_receipt_path": str(WORKER_RECEIPT),
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
    stage_complete = (
        progress["main_posteval_complete"]
        and progress["balanced_posteval_complete"]
    )
    pending = not stage_complete
    main_authority = select_main_authority(units)
    authority_names = (main_authority, WORKER_AUTHORITY_UNIT)
    active_units = sorted(
        name
        for name in authority_names
        if units.get(name, {}).get("active_state") == "active"
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
    authority_failures = sorted(
        name
        for name in authority_names
        if (state := units.get(name, {}))
        if state["active_state"] == "failed" or state["result"] == "exit-code"
    )
    failed_units = authority_failures if pending else []
    post_completion_incidents = authority_failures if stage_complete else []
    superseded_failures = sorted(
        name
        for name, state in units.items()
        if name != main_authority
        and (
            name in SUPERSEDED_MAIN_UNITS
            or name.startswith("fluid-control-train16-posteval-main-")
        )
        and (state.get("active_state") == "failed" or state.get("result") == "exit-code")
    )
    tasks = {
        "main_posteval": {
            "authority_unit": main_authority,
            "node": "spark",
            "receipt_path": str(MAIN_RECEIPT),
            "stage_complete": progress["main_posteval_complete"],
        },
        "balanced_posteval": {
            "authority_unit": WORKER_AUTHORITY_UNIT,
            "node": "worker78",
            "receipt_path": str(WORKER_RECEIPT),
            "stage_complete": progress["balanced_posteval_complete"],
        },
    }
    for task_name, task in tasks.items():
        state = units.get(task["authority_unit"], {})
        task["state"], action_id = classify_authority_task(
            state,
            task["stage_complete"],
            allow_resume=task_name == "main_posteval",
        )
        task["automatic_recovery_eligible"] = action_id is not None
        if action_id:
            task["approved_action_id"] = action_id
    alerts = []
    blocker_reasons = []
    needs_analysis = [name for name, task in tasks.items() if task["state"] == "NEEDS_AGENT_ANALYSIS"]
    retryable = [name for name, task in tasks.items() if task["state"] == "RETRY_ELIGIBLE"]
    if pending and needs_analysis:
        alerts.append("TRAIN16_POSTEVAL_NEEDS_AGENT_ANALYSIS")
        for task_name in needs_analysis:
            name = tasks[task_name]["authority_unit"]
            state = units.get(
                name,
                {
                    "result": "unknown",
                    "exec_main_status": "unknown",
                    "last_error_line": "authority state missing",
                },
            )
            blocker_reasons.append(
                f"{task_name}/{name}: missing authoritative completion receipt; "
                f"result={state['result']} status={state['exec_main_status']}"
                + (
                    f"; {state['last_error_line']}"
                    if state.get("last_error_line")
                    else ""
                )
            )
    if pending and retryable and not active_units:
        alerts.append("TRAIN16_REVIEWED_RECOVERY_PENDING")
        blocker_reasons.append(
            "Reviewed idempotent Main --resume is eligible for one guarded attempt: "
            + ", ".join(retryable)
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
            "auto_recovery_enabled": PRODUCTION_AUTO_RECOVERY_ENABLED,
            "auto_recovery_reason": (
                "Disabled pending content-hash verification fixes to reviewed Main resume; "
                "Worker remains external takeover"
            ),
            "scientific_failure_is_never_bypassed": True,
        },
        "stage_complete": stage_complete,
        "project_goal_complete": False,
        "project_status": (
            "NEEDS_MODEL_IMPROVEMENT" if stage_complete else "POSTEVAL_INCOMPLETE"
        ),
        "workflow_pending": pending,
        "no_running_since_utc": idle_since,
        "no_running_duration_seconds": idle_seconds,
        "active_units": active_units,
        "failed_units": failed_units,
        "post_completion_incidents": post_completion_incidents,
        "superseded_failures": superseded_failures,
        "authority_tasks": tasks,
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
    units[WORKER_AUTHORITY_UNIT] = worker_unit_state(WORKER_AUTHORITY_UNIT)
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
