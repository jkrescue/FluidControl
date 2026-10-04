#!/usr/bin/env python3
"""Fail-closed reconciler for reviewed train16 evaluation recovery actions.

The production allow-list is deliberately empty until an idempotent resume
entry point and its exact hash are reviewed. Unknown failures are diagnosed,
never retried. The pure planning functions accept injected actions for tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path


MIN_MEM_AVAILABLE_GIB = 20.0
MAX_RETRIES_PER_ACTION = 1
REPO = Path("/workspace/fluid_control")
RUNNER = REPO / "scripts/run_control_train16_posteval_queue_spark.sh"
VALIDATOR = REPO / "scripts/validate_control_train16_posteval_step.py"
APPROVED_ACTIONS: dict[str, dict] = {
    "main-posteval-resume-78d827f": {
        "required_files": {
            str(RUNNER): "a80aabb74b783e791cbbe33ea02146643b6395574e3817e8a8be842afcddecea",
            str(VALIDATOR): "9e3ef79d43fef37ff6a2f41b6d1729f69f82f2bddd9e50e5ecac86e54be389e9",
        },
        "max_attempts": 1,
        "conflict_pattern": "evaluate_tandem_fno.py.*tandem_fno_control_train16_h100_20261004",
        "command": [
            "systemd-run",
            "--user",
            "--unit=fluid-control-train16-posteval-main-auto-resume1-20261004",
            "--collect",
            "--property=Restart=no",
            "--property=KillMode=mixed",
            f"--working-directory={REPO}",
            "--setenv=CONTROL_TRAIN16_POSTEVAL_TOKEN=EXECUTE_REVIEWED_CONTROL_TRAIN16_POSTEVAL",
            str(RUNNER),
            "--resume",
        ],
    }
}


def read_json(path: Path, fallback):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return fallback


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def state_digest(sample: dict) -> str:
    relevant = {
        "timestamp_utc": sample.get("timestamp_utc"),
        "alerts": sample.get("alerts", []),
        "blocker_reasons": sample.get("blocker_reasons", []),
        "authority_tasks": sample.get("authority_tasks", {}),
    }
    encoded = json.dumps(relevant, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plan_recovery(sample: dict, ledger: dict, approved_actions: dict) -> dict:
    """Return a deterministic decision without executing external work."""
    if sample.get("stage_complete"):
        return {"decision": "NO_ACTION_STAGE_COMPLETE"}
    tasks = sample.get("authority_tasks", {})
    needs_analysis = [
        name
        for name, task in tasks.items()
        if task.get("state") == "NEEDS_AGENT_ANALYSIS"
    ]
    if needs_analysis:
        return {
            "decision": "NEEDS_AGENT_ANALYSIS",
            "reason": "Current authority requires diagnosis: " + ", ".join(needs_analysis),
        }
    if sample.get("active_units"):
        return {"decision": "NO_ACTION_AUTHORITY_RUNNING"}
    candidates = [
        (name, task)
        for name, task in tasks.items()
        if task.get("state") == "RETRY_ELIGIBLE"
    ]
    if not candidates:
        return {
            "decision": "NEEDS_AGENT_ANALYSIS",
            "reason": "No authority task has a reviewed retry-eligible classification",
        }
    task_name, task = candidates[0]
    action_id = task.get("approved_action_id")
    action = approved_actions.get(action_id)
    if not action:
        return {
            "decision": "NEEDS_AGENT_ANALYSIS",
            "reason": f"No fixed allow-list action for {action_id!r}",
        }
    if sample.get("resources", {}).get("mem_available_gib", 0) < MIN_MEM_AVAILABLE_GIB:
        return {"decision": "RESOURCE_BLOCKED", "reason": "MemAvailable below 20 GiB"}
    attempts = int(ledger.get("attempts", {}).get(action_id, 0))
    maximum = int(action.get("max_attempts", MAX_RETRIES_PER_ACTION))
    if attempts >= maximum:
        return {
            "decision": "NEEDS_AGENT_ANALYSIS",
            "reason": f"Reviewed retry budget exhausted ({attempts}/{maximum})",
        }
    return {
        "decision": "EXECUTE_REVIEWED_RESUME",
        "task": task_name,
        "action_id": action_id,
        "command": list(action["command"]),
        "attempt_number": attempts + 1,
    }


def execute_reviewed_resume(
    decision: dict, ledger: dict, approved_actions: dict, output: Path
) -> dict:
    """Execute one exact command under an O_EXCL lock; never loop internally."""
    if decision.get("decision") != "EXECUTE_REVIEWED_RESUME":
        return decision
    action_id = decision["action_id"]
    action = approved_actions[action_id]
    for required_file, expected_sha in action.get("required_files", {}).items():
        path = Path(required_file)
        if not path.is_file() or file_sha256(path) != expected_sha:
            return {
                "decision": "NEEDS_AGENT_ANALYSIS",
                "action_id": action_id,
                "reason": f"Reviewed recovery dependency SHA differs: {path}",
            }
    conflict_pattern = action.get("conflict_pattern")
    if conflict_pattern:
        conflict = subprocess.run(
            ["pgrep", "-af", conflict_pattern],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
        if conflict.returncode == 0 and conflict.stdout.strip():
            return {
                "decision": "RESOURCE_BLOCKED",
                "action_id": action_id,
                "reason": "Candidate evaluation process is already active",
            }
    lock = output / f"{action_id}.lock"
    output.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return {"decision": "NO_ACTION_DUPLICATE_LOCK", "action_id": action_id}
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(datetime.now(UTC).isoformat(timespec="seconds") + "\n")
    result = subprocess.run(
        action["command"], text=True, capture_output=True, check=False, timeout=30
    )
    ledger.setdefault("attempts", {})[action_id] = decision["attempt_number"]
    ledger.setdefault("executions", []).append(
        {
            "action_id": action_id,
            "attempt_number": decision["attempt_number"],
            "returncode": result.returncode,
            "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "stdout_tail": result.stdout[-1000:],
            "stderr_tail": result.stderr[-1000:],
        }
    )
    atomic_json(output / "recovery_ledger.json", ledger)
    return {
        "decision": "REVIEWED_RESUME_DISPATCHED"
        if result.returncode == 0
        else "REVIEWED_RESUME_DISPATCH_FAILED",
        "action_id": action_id,
        "returncode": result.returncode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument(
        "--monitor-dir",
        type=Path,
        default=Path("artifacts/monitor/training_evaluation_watchdog"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.monitor_dir
    if not output.is_absolute():
        output = repo / output
    sample = read_json(output / "latest.json", {})
    ledger = read_json(output / "recovery_ledger.json", {"attempts": {}, "executions": []})
    decision = plan_recovery(sample, ledger, APPROVED_ACTIONS)
    if decision["decision"] == "EXECUTE_REVIEWED_RESUME":
        decision = execute_reviewed_resume(decision, ledger, APPROVED_ACTIONS, output)
    record = {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "auto_recovery_enabled": bool(APPROVED_ACTIONS),
        "decision": decision,
        "monitor_state_sha256": state_digest(sample),
        "does_not_modify_code": True,
        "unknown_failures_require_agent_analysis": True,
    }
    atomic_json(output / "reconciliation_latest.json", record)
    if decision["decision"] == "NEEDS_AGENT_ANALYSIS":
        diagnostic = output / "diagnostics" / f"{record['monitor_state_sha256']}.json"
        if not diagnostic.exists():
            atomic_json(diagnostic, {"sample": sample, "reconciliation": record})
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
