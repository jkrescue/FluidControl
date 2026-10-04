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
PAIRED_DATAPIPE_ROOT = Path("artifacts/train20_paired_stat_datapipe_v1")
PAIRED_DATAPIPE_WORKER_UNIT = "fluid-control-train20-paired-datapipe-probe-20261004.service"
PAIRED_LAMBDA0_UNIT = "fluid-control-paired-stats-lambda0-full-v1-20261004.service"
PAIRED_LAMBDA0_ROOT = Path("artifacts/tandem_fno_paired_stats_lambda0_20261004")
PAIRED_LAMBDA10_ROOT = Path("artifacts/tandem_fno_paired_stats_lambda10_20261004")
PAIRED_TRAINING_STATUS = "PAIRED_STATS_CONTROLLED_TRAINING_COMPLETE"
PAIRED_LAMBDA10_TRANSFER_STATUS = (
    "PAIRED_STATS_LAMBDA10_WORKER_TO_SPARK_TRANSFER_VERIFIED"
)
PAIRED_POSTEVAL_APPROVAL = Path("docs/FC-P001_APPROVAL.md")
FC_P003_APPROVAL = Path("docs/FC-P003_APPROVAL.md")
FC_P003_UNIT = "fluid-control-fcp003-interleaved-lambda10-20261005.service"
FC_P003_PREFIX = "fluid-control-fcp003-interleaved-lambda10-"
FC_P003_PROBE_UNIT = "fluid-control-fcp003-interleaved-probe-20261005.service"
FC_P003_PROBE_PREFIX = "fluid-control-fcp003-interleaved-probe-"
FC_P003_ROOT = Path("artifacts/tandem_fno_paired_stats_interleaved_lambda10_20261005")
FC_P003_PROBE_ROOT = Path(
    "artifacts/tandem_fno_paired_stats_interleaved_lambda10_probe_20261005"
)
FC_P003_LAUNCH_RECEIPT = FC_P003_ROOT / "launch_receipt.json"
FC_P003_COMPLETION_RECEIPT = FC_P003_ROOT / "completion_receipt.json"
FC_P003_DEVELOPMENT_GATE = FC_P003_ROOT / "posteval_fc_p003/development_gate.json"
FC_P003_LAUNCH_STATUS = "FC_P003_INTERLEAVED_LAUNCH_STAGED"
FC_P003_COMPLETION_STATUS = "FC_P003_INTERLEAVED_TRAINING_COMPLETE"
FC_P003_PROBE_STATUS = "FC_P003_INTERLEAVED_PROBE_PASS"
PAIRED_POSTEVAL_STATUS = "PAIRED_STATS_FC_P001_POSTEVAL_COMPLETE"
PAIRED_LAMBDA0_POSTEVAL_UNIT = (
    "fluid-control-paired-lambda0-posteval-fcp001-v2-20261004.service"
)
PAIRED_LAMBDA0_POSTEVAL_PREFIX = "fluid-control-paired-lambda0-posteval-fcp001-"
PAIRED_LAMBDA0_POSTEVAL_RECEIPT = (
    PAIRED_LAMBDA0_ROOT / "posteval_fc_p001/receipt.json"
)
PAIRED_LAMBDA10_POSTEVAL_UNIT = (
    "fluid-control-paired-stats-lambda10-posteval-worker-20261004.service"
)
PAIRED_LAMBDA10_POSTEVAL_RECEIPT = (
    PAIRED_LAMBDA10_ROOT / "posteval_fc_p001/receipt.json"
)
PAIRED_LAMBDA0_DEVELOPMENT_GATE = (
    PAIRED_LAMBDA0_ROOT / "posteval_fc_p001/development_gate.json"
)
PAIRED_LAMBDA10_DEVELOPMENT_GATE = (
    PAIRED_LAMBDA10_ROOT / "posteval_fc_p001/development_gate.json"
)
PAIRED_DEVELOPMENT_FAIL_STATUS = "DYNAMIC_FNO_DEVELOPMENT_ADMISSION_FAIL"
PAIRED_PROTOCOL_SHA256 = {
    "evaluator": "eb93ec1e95ee5ac491bcf929e9e00bc9da83d2c97240c0d9c87d517faffe9d5f",
    "force_window": "4ef0a878ac6f3ab3a8e0a957b7b16882731b2aff8093d0efa46cb4e29a954df8",
    "dynamic6_audit": "4e78d8473d1d0f93b25031a3bf9dcc0582f43604f7b1f67c65e6754f032af100",
    "endpoint_audit": "eedc114549eacd787951ef8d8531db6a1ef40c088ce8bff6fcf10d351fa09269",
    "development_gate": "ca6da0afdce5859be1c060eb48ba2cdd1ccc5ee3aeb2570d9c9b53067d5bc412",
    "step_validator": "c2588d1ad4e7fbea0840fe4d97ccf780865cf912cf25b0ef86d2f4e2689af3c0",
}
REVIEWED_MAIN_RESUME_ACTION = "main-posteval-resume-78d827f"
PRODUCTION_AUTO_RECOVERY_ENABLED = True


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


def classify_approved_evaluation(state: dict, complete: bool) -> str:
    """Classify FC-P001 without treating approved preflight as a failure."""
    if complete:
        return "STAGE_COMPLETE"
    if state.get("active_state") == "active":
        return "RUNNING"
    if state.get("active_state") == "failed" or state.get("result") == "exit-code":
        return "NEEDS_AGENT_ANALYSIS"
    return "LEAD_APPROVED_PREFLIGHT"


def utc_now() -> datetime:
    return datetime.now(UTC)


def read_json(path: Path, fallback):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return fallback


def paired_scientific_verdict(repo: Path, paired_complete: bool) -> dict:
    """Read receipt-bound development gates without turning a science fail into a retry."""
    if not paired_complete:
        return {"status": "PENDING", "lambda0": None, "lambda10": None}
    lambda0 = read_json(repo / PAIRED_LAMBDA0_DEVELOPMENT_GATE, {}).get("status")
    lambda10 = read_json(repo / PAIRED_LAMBDA10_DEVELOPMENT_GATE, {}).get("status")
    status = (
        "FC_P001_SCIENTIFIC_REJECTED"
        if lambda0 == lambda10 == PAIRED_DEVELOPMENT_FAIL_STATUS
        else "FC_P001_COMPLETE_REQUIRES_AGENT_REVIEW"
    )
    return {"status": status, "lambda0": lambda0, "lambda10": lambda10}


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


def select_versioned_authority(
    units: dict[str, dict], prefix: str, default: str
) -> str:
    candidates = [name for name in units if name.startswith(prefix)]
    active = [name for name in candidates if units[name].get("active_state") == "active"]
    if active:
        return max(active, key=lambda name: units[name].get("started_at") or "")
    versioned = []
    for name in candidates:
        match = re.search(r"-[vr](\d+)-", name)
        versioned.append((int(match.group(1)) if match else 1, name))
    return max(versioned, default=(0, default))[1]


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
        "ExecMainExitTimestamp,MainPID"
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
    main_pid = int(values.get("MainPID") or 0)
    training_process_pids = descendant_processes_matching(
        main_pid, "train_tandem_fno_paired_stats.py"
    )
    return {
        "unit": unit,
        "active_state": values.get("ActiveState", "unknown"),
        "sub_state": values.get("SubState", "unknown"),
        "result": values.get("Result", "unknown"),
        "exec_main_status": values.get("ExecMainStatus", "unknown"),
        "main_pid": main_pid,
        "training_process_pids": training_process_pids,
        "started_at": values.get("ExecMainStartTimestamp") or None,
        "exited_at": values.get("ExecMainExitTimestamp") or None,
        "last_error_line": error,
    }


def descendant_processes_matching(root_pid: int, needle: str) -> list[int]:
    """Return descendants whose command line proves the actual training payload."""
    if root_pid <= 0:
        return []
    children: dict[int, list[int]] = {}
    commands: dict[int, str] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
            close = stat.rfind(")")
            parent = int(stat[close + 2 :].split()[1])
            pid = int(entry.name)
            children.setdefault(parent, []).append(pid)
            commands[pid] = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                errors="replace"
            )
        except (OSError, ValueError, IndexError):
            continue
    descendants = []
    queue = [root_pid]
    while queue:
        parent = queue.pop()
        for pid in children.get(parent, []):
            queue.append(pid)
            if needle in commands.get(pid, ""):
                descendants.append(pid)
    return sorted(descendants)


def worker_unit_state(unit: str, *, user_scope: bool = True) -> dict:
    """Read Worker systemd state through the existing key-only SSH path."""
    command = (
        "systemctl " + ("--user " if user_scope else "") + "show "
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
        if columns and (
            columns[0].startswith("fluid-control-train16-")
            or columns[0].startswith(PAIRED_LAMBDA0_POSTEVAL_PREFIX)
            or columns[0].startswith(FC_P003_PREFIX)
            or columns[0].startswith(FC_P003_PROBE_PREFIX)
        ):
            names.append(columns[0])
    return sorted(
        set(
            (
                TRAINING_UNIT,
                MAIN_AUTHORITY_UNIT,
                PAIRED_LAMBDA0_UNIT,
                PAIRED_LAMBDA0_POSTEVAL_UNIT,
                FC_P003_UNIT,
                FC_P003_PROBE_UNIT,
                *SUPERSEDED_MAIN_UNITS,
                *names,
            )
        )
    )


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
    for root in (
        repo / MAIN_RUN,
        repo / BALANCED_RUN,
        repo / PAIRED_LAMBDA0_ROOT,
        repo / PAIRED_LAMBDA10_ROOT,
    ):
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
            "posteval_fc_p001/**/*.log",
            "posteval_fc_p001/**/*.json",
            "completion_receipt.json",
            "worker_transfer_complete.json",
            "training_history.json",
            "train.log",
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


def paired_training_progress(repo: Path) -> dict:
    lambda0_complete, lambda0_issues = verify_receipt(
        repo / PAIRED_LAMBDA0_ROOT / "completion_receipt.json",
        PAIRED_TRAINING_STATUS,
    )
    lambda10_complete, lambda10_issues = verify_receipt(
        repo / PAIRED_LAMBDA10_ROOT / "completion_receipt.json",
        PAIRED_TRAINING_STATUS,
    )
    lambda10_transfer_complete, lambda10_transfer_issues = verify_receipt(
        repo / PAIRED_LAMBDA10_ROOT / "worker_transfer_complete.json",
        PAIRED_LAMBDA10_TRANSFER_STATUS,
    )
    lambda0_posteval_complete, lambda0_posteval_issues = verify_receipt(
        repo / PAIRED_LAMBDA0_POSTEVAL_RECEIPT,
        PAIRED_POSTEVAL_STATUS,
    )
    lambda10_posteval_complete, lambda10_posteval_issues = verify_receipt(
        repo / PAIRED_LAMBDA10_POSTEVAL_RECEIPT,
        PAIRED_POSTEVAL_STATUS,
    )
    return {
        "lambda0_training_complete": lambda0_complete,
        "lambda0_receipt_issues": lambda0_issues,
        "lambda10_training_complete": lambda10_complete,
        "lambda10_receipt_issues": lambda10_issues,
        "lambda10_transfer_complete": lambda10_transfer_complete,
        "lambda10_transfer_issues": lambda10_transfer_issues,
        "paired_training_complete": (
            lambda0_complete and lambda10_complete and lambda10_transfer_complete
        ),
        "lambda0_posteval_complete": lambda0_posteval_complete,
        "lambda0_posteval_issues": lambda0_posteval_issues,
        "lambda0_posteval_receipt": str(PAIRED_LAMBDA0_POSTEVAL_RECEIPT),
        "lambda10_posteval_complete": lambda10_posteval_complete,
        "lambda10_posteval_issues": lambda10_posteval_issues,
        "lambda10_posteval_receipt": str(PAIRED_LAMBDA10_POSTEVAL_RECEIPT),
        "paired_posteval_complete": (
            lambda0_posteval_complete and lambda10_posteval_complete
        ),
        "paired_posteval_status": "PAIRED_POSTEVAL_APPROVED_PREFLIGHT",
        "approval_state": "LEAD_APPROVED",
        "approval_reference": str(PAIRED_POSTEVAL_APPROVAL),
        "next_owner": "Surrogate + Physics/Data",
        "approval_required": None,
    }


def build_sample(
    repo: Path,
    previous: dict | None,
    units: dict[str, dict],
    resources: dict,
    now: datetime,
) -> dict:
    progress = workflow_progress(repo)
    prior_posteval_stage_complete = (
        progress["main_posteval_complete"]
        and progress["balanced_posteval_complete"]
    )
    paired = paired_training_progress(repo)
    paired_verdict = paired_scientific_verdict(
        repo, paired["paired_posteval_complete"]
    )
    fc_p003_approved = (repo / FC_P003_APPROVAL).is_file()
    fc_p003_authority = select_versioned_authority(
        units, FC_P003_PREFIX, FC_P003_UNIT
    )
    fc_p003_unit = units.get(fc_p003_authority, {})
    fc_p003_probe_authority = select_versioned_authority(
        units, FC_P003_PROBE_PREFIX, FC_P003_PROBE_UNIT
    )
    fc_p003_probe_unit = units.get(fc_p003_probe_authority, {})
    fc_p003_running = (
        fc_p003_unit.get("active_state") == "active"
        and bool(fc_p003_unit.get("training_process_pids"))
    )
    fc_p003_probe_running = (
        fc_p003_probe_unit.get("active_state") == "active"
        and bool(fc_p003_probe_unit.get("training_process_pids"))
    )
    fc_p003_launch_present = (repo / FC_P003_LAUNCH_RECEIPT).is_file()
    fc_p003_completion_present = (repo / FC_P003_COMPLETION_RECEIPT).is_file()
    fc_p003_launch_verified, fc_p003_launch_issues = verify_receipt(
        repo / FC_P003_LAUNCH_RECEIPT, FC_P003_LAUNCH_STATUS
    )
    fc_p003_completion_verified, fc_p003_completion_issues = verify_receipt(
        repo / FC_P003_COMPLETION_RECEIPT, FC_P003_COMPLETION_STATUS
    )
    fc_p003_probe_verified, fc_p003_probe_issues = verify_receipt(
        repo / FC_P003_PROBE_ROOT / "completion_receipt.json", FC_P003_PROBE_STATUS
    )
    fc_p003_probe_launch_verified, fc_p003_probe_launch_issues = verify_receipt(
        repo / FC_P003_PROBE_ROOT / "launch_receipt.json", FC_P003_LAUNCH_STATUS
    )
    fc_p003_gate_status = read_json(
        repo / FC_P003_DEVELOPMENT_GATE, {}
    ).get("status")
    if fc_p003_gate_status == PAIRED_DEVELOPMENT_FAIL_STATUS:
        fc_p003_state = "SCIENTIFIC_FAIL_NEEDS_LEAD_NEXT_HYPOTHESIS"
    elif fc_p003_gate_status:
        fc_p003_state = "SCIENTIFIC_RESULT_REQUIRES_AGENT_REVIEW"
    elif fc_p003_running and fc_p003_launch_verified:
        fc_p003_state = "RUNNING"
    elif fc_p003_unit.get("active_state") == "failed":
        fc_p003_state = "OPERATIONAL_FAILURE_NEEDS_AGENT_ANALYSIS"
    elif fc_p003_completion_verified:
        fc_p003_state = "TRAINING_COMPLETE_POSTEVAL_PENDING"
    elif fc_p003_completion_present or (
        fc_p003_launch_present and not fc_p003_launch_verified
    ):
        fc_p003_state = "OPERATIONAL_FAILURE_NEEDS_AGENT_ANALYSIS"
    elif fc_p003_launch_verified:
        fc_p003_state = "PREFLIGHT_COMPLETE_WAITING_RUN"
    elif fc_p003_probe_running and fc_p003_probe_launch_verified:
        fc_p003_state = "RESOURCE_PROBE_RUNNING"
    elif fc_p003_probe_verified:
        fc_p003_state = "RESOURCE_PROBE_PASS_FULL_RUN_PENDING"
    elif fc_p003_probe_unit.get("active_state") == "failed":
        fc_p003_state = "OPERATIONAL_FAILURE_NEEDS_AGENT_ANALYSIS"
    elif fc_p003_approved:
        fc_p003_state = "PREFLIGHT_IMPLEMENTATION"
    else:
        fc_p003_state = "PLANNED_NOT_APPROVED"
    progress.update(paired)
    fc_p001_stage_complete = (
        prior_posteval_stage_complete
        and paired["paired_training_complete"]
        and paired["paired_posteval_complete"]
    )
    stage_complete = (
        fc_p003_gate_status is not None if fc_p003_approved else fc_p001_stage_complete
    )
    pending = True  # The accepted surrogate/controller/real-CFD project goal is unmet.
    main_authority = select_main_authority(units)
    lambda0_posteval_authority = select_versioned_authority(
        units, PAIRED_LAMBDA0_POSTEVAL_PREFIX, PAIRED_LAMBDA0_POSTEVAL_UNIT
    )
    authority_names = (
        main_authority,
        WORKER_AUTHORITY_UNIT,
        lambda0_posteval_authority,
        PAIRED_LAMBDA10_POSTEVAL_UNIT,
        fc_p003_authority,
        fc_p003_probe_authority,
    )
    active_units = sorted(
        name
        for name in authority_names
        if units.get(name, {}).get("active_state") == "active"
    )
    paired_evaluation_active = any(
        name in active_units
        for name in (lambda0_posteval_authority, PAIRED_LAMBDA10_POSTEVAL_UNIT)
    )
    if paired["paired_posteval_complete"]:
        paired["paired_posteval_status"] = (
            "PAIRED_POSTEVAL_COMPLETE_SCIENTIFIC_REJECTED"
            if paired_verdict["status"] == "FC_P001_SCIENTIFIC_REJECTED"
            else "PAIRED_POSTEVAL_COMPLETE_REQUIRES_AGENT_REVIEW"
        )
    elif paired_evaluation_active:
        paired["paired_posteval_status"] = "PAIRED_POSTEVAL_RUNNING"
    progress["paired_posteval_status"] = paired["paired_posteval_status"]
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
    failed_units = authority_failures if not prior_posteval_stage_complete else []
    post_completion_incidents = (
        authority_failures if prior_posteval_stage_complete else []
    )
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
        "paired_lambda0_posteval": {
            "authority_unit": lambda0_posteval_authority,
            "node": "spark",
            "receipt_path": str(PAIRED_LAMBDA0_POSTEVAL_RECEIPT),
            "stage_complete": paired["lambda0_posteval_complete"],
        },
        "paired_lambda10_posteval": {
            "authority_unit": PAIRED_LAMBDA10_POSTEVAL_UNIT,
            "node": "worker78",
            "receipt_path": str(PAIRED_LAMBDA10_POSTEVAL_RECEIPT),
            "stage_complete": paired["lambda10_posteval_complete"],
        },
    }
    for task_name, task in tasks.items():
        state = units.get(task["authority_unit"], {})
        if task_name in ("paired_lambda0_posteval", "paired_lambda10_posteval"):
            task["state"] = classify_approved_evaluation(
                state, task["stage_complete"]
            )
            task["automatic_recovery_eligible"] = False
            continue
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
    if fc_p003_state == "OPERATIONAL_FAILURE_NEEDS_AGENT_ANALYSIS":
        alerts.append("FC_P003_OPERATIONAL_FAILURE_NEEDS_AGENT_ANALYSIS")
        blocker_reasons.append(
            f"{fc_p003_authority}: approved FC-P003 has no scientific gate; "
            f"result={fc_p003_unit.get('result')} "
            f"status={fc_p003_unit.get('exec_main_status')}; blind restart forbidden"
        )
    needs_analysis = [name for name, task in tasks.items() if task["state"] == "NEEDS_AGENT_ANALYSIS"]
    retryable = [name for name, task in tasks.items() if task["state"] == "RETRY_ELIGIBLE"]
    if pending and needs_analysis:
        alerts.append(
            "PAIRED_POSTEVAL_NEEDS_AGENT_ANALYSIS"
            if any(name.startswith("paired_") for name in needs_analysis)
            else "TRAIN16_POSTEVAL_NEEDS_AGENT_ANALYSIS"
        )
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
        paired_wait = (
            prior_posteval_stage_complete
            and paired["paired_training_complete"]
            and not paired["paired_posteval_complete"]
        )
        alerts.append(
            "PAIRED_POSTEVAL_APPROVED_WITH_NO_RUNNING_UNIT_FOR_300_SECONDS"
            if paired_wait
            else "TRAIN16_PENDING_WITH_NO_RUNNING_UNIT_FOR_300_SECONDS"
        )
        if paired_wait:
            blocker_reasons.append(
                "FC-P001 paired λ0/λ10 post-evaluation is Lead-approved; "
                "implementation/lineage preflight has no active evaluation unit"
            )
    if resources["mem_available_gib"] < 20.0:
        alerts.append("SPARK_MEMORY_BELOW_20_GIB")
    lambda0_receipt_path = PAIRED_LAMBDA0_ROOT / "completion_receipt.json"
    lambda0_complete = paired["lambda0_training_complete"]
    lambda0_unit = units.get(PAIRED_LAMBDA0_UNIT, {})
    if paired["lambda0_posteval_complete"]:
        lambda0_state = "TRAINING_AND_POSTEVAL_STAGE_COMPLETE"
    elif lambda0_complete:
        lambda0_state = "TRAINING_STAGE_COMPLETE_POSTEVAL_PENDING"
    elif lambda0_unit.get("active_state") == "active":
        lambda0_state = "RUNNING"
    elif lambda0_unit.get("active_state") == "failed":
        lambda0_state = "NEEDS_AGENT_ANALYSIS"
        alerts.append("PAIRED_LAMBDA0_TRAINING_FAILED_NEEDS_AGENT_ANALYSIS")
        blocker_reasons.append(
            f"{PAIRED_LAMBDA0_UNIT}: result={lambda0_unit.get('result')} "
            f"status={lambda0_unit.get('exec_main_status')}; automatic restart forbidden"
        )
    else:
        lambda0_state = "PENDING"
    scientific_status = (
        f"FC_P003_{fc_p003_state}"
        if paired_verdict["status"] == "FC_P001_SCIENTIFIC_REJECTED"
        and fc_p003_approved
        else paired["paired_posteval_status"]
        if paired["paired_posteval_complete"]
        else (
            "PAIRED_POSTEVAL_RUNNING"
            if paired_evaluation_active
            else (
                "PAIRED_STATS_CONTROLLED_TRAINING_RUNNING"
                if lambda0_state == "RUNNING"
                else (
                    "PAIRED_POSTEVAL_APPROVED_PREFLIGHT"
                    if paired["paired_training_complete"]
                    else "PAIRED_STATS_CONTROLLED_TRAINING_AND_POSTEVAL_PENDING"
                )
            )
        )
    )
    return {
        "status": "ALERT" if alerts else "MONITORING",
        "timestamp_utc": now.isoformat(timespec="seconds"),
        "policy": {
            "sample_interval_seconds": 60,
            "no_running_alert_seconds": IDLE_ALERT_SECONDS,
            "automatic_restart_or_repair": False,
            "auto_recovery_enabled": PRODUCTION_AUTO_RECOVERY_ENABLED,
            "auto_recovery_reason": (
                "Main strict content-verified --resume at commit 78d827f, at most once; "
                "Worker remains external takeover"
            ),
            "scientific_failure_is_never_bypassed": True,
        },
        "stage_complete": stage_complete,
        "fc_p001_stage_complete": fc_p001_stage_complete,
        "prior_posteval_stage_complete": prior_posteval_stage_complete,
        "project_goal_complete": False,
        "project_status": (
            "NEEDS_MODEL_IMPROVEMENT"
            if prior_posteval_stage_complete
            else "POSTEVAL_INCOMPLETE"
        ),
        "scientific_next_stage": {
            "status": scientific_status,
            "active_work": (
                (
                    "fc_p003_interleaved_paired_supervision_training"
                    if fc_p003_state == "RUNNING"
                    else "fc_p003_bounded_resource_probe"
                    if fc_p003_state == "RESOURCE_PROBE_RUNNING"
                    else "fc_p003_unchanged_formal_posteval"
                    if fc_p003_state == "TRAINING_COMPLETE_POSTEVAL_PENDING"
                    else "fc_p003_scientific_fail_awaiting_lead_next_hypothesis"
                    if fc_p003_state == "SCIENTIFIC_FAIL_NEEDS_LEAD_NEXT_HYPOTHESIS"
                    else "fc_p003_interleaved_paired_supervision_preflight"
                )
                if paired_verdict["status"] == "FC_P001_SCIENTIFIC_REJECTED"
                and fc_p003_approved
                else "fc_p002_failure_map"
                if paired["paired_posteval_complete"]
                else (
                    "paired_posteval_running"
                    if paired_evaluation_active
                    else "paired_posteval_approved_preflight"
                    if paired["paired_training_complete"]
                    else "paired_stats_controlled_training"
                )
            ),
            "purpose": (
                "Lead-approved FC-P003 changes only paired-update timing from frontloaded "
                "to uniformly interleaved; training completion is not scientific admission, "
                "and a failed gate returns control to Lead for the next hypothesis"
            ),
            "fc_p001_verdict": paired_verdict,
            "fc_p003": {
                "approval_state": "LEAD_APPROVED" if fc_p003_approved else "PLANNED",
                "approval_reference": str(FC_P003_APPROVAL),
                "state": (
                    fc_p003_state
                ),
                "authority_unit": (
                    fc_p003_authority
                    if fc_p003_running
                    else fc_p003_probe_authority
                    if fc_p003_probe_running
                    else fc_p003_authority
                ),
                "main_pid": (
                    fc_p003_unit.get("main_pid", 0)
                    or fc_p003_probe_unit.get("main_pid", 0)
                ),
                "output_root": str(FC_P003_ROOT),
                "launch_receipt": str(FC_P003_LAUNCH_RECEIPT),
                "launch_receipt_present": fc_p003_launch_present,
                "launch_receipt_verified": fc_p003_launch_verified,
                "launch_receipt_issues": fc_p003_launch_issues,
                "completion_receipt": str(FC_P003_COMPLETION_RECEIPT),
                "completion_receipt_present": fc_p003_completion_present,
                "completion_receipt_verified": fc_p003_completion_verified,
                "completion_receipt_issues": fc_p003_completion_issues,
                "probe_unit": fc_p003_probe_authority,
                "probe_receipt": str(FC_P003_PROBE_ROOT / "completion_receipt.json"),
                "probe_receipt_verified": fc_p003_probe_verified,
                "probe_receipt_issues": fc_p003_probe_issues,
                "probe_launch_receipt_verified": fc_p003_probe_launch_verified,
                "probe_launch_receipt_issues": fc_p003_probe_launch_issues,
                "development_gate": str(FC_P003_DEVELOPMENT_GATE),
                "development_gate_status": fc_p003_gate_status,
                "single_factor": "paired_update_schedule_frontloaded_to_interleaved",
                "automatic_recovery_eligible": False,
            },
            "parallel_cpu_work": {
                "owner": "Physics/Data",
                "status": read_json(
                    repo / "artifacts/fc_p003_dynamic8_pair_candidate_20261005/manifest.json",
                    {},
                ).get("status", "DYNAMIC8_EXISTING_RESTART_PAIR_QC"),
                "scope": "same-restart train-only dynamic8 action/zero pairing audit",
                "training_authorized": False,
                "reference": "docs/FC-P003_DYNAMIC8_PAIR_CANDIDATE.md",
                "artifact": (
                    "artifacts/fc_p003_dynamic8_pair_candidate_20261005/manifest.json"
                ),
            },
            "planned_spark_root": str(PAIRED_DATAPIPE_ROOT),
            "planned_worker_unit": PAIRED_DATAPIPE_WORKER_UNIT,
            "lambda0": {
                "unit": PAIRED_LAMBDA0_UNIT,
                "output_root": str(PAIRED_LAMBDA0_ROOT),
                "source_commit": "b6aada926942161da4430d52664c92db1a2269c7",
                "state": lambda0_state,
                "completion_receipt": str(lambda0_receipt_path),
                "posteval_unit": lambda0_posteval_authority,
                "posteval_state": tasks["paired_lambda0_posteval"]["state"],
                "posteval_receipt": str(PAIRED_LAMBDA0_POSTEVAL_RECEIPT),
            },
            "lambda10": {
                "state": (
                    "TRAINING_AND_POSTEVAL_STAGE_COMPLETE"
                    if paired["lambda10_posteval_complete"]
                    else (
                        "TRAINING_STAGE_COMPLETE_TRANSFER_VERIFIED"
                        if paired["lambda10_transfer_complete"]
                        else "WORKER_TRANSFER_PENDING"
                    )
                ),
                "output_root": str(PAIRED_LAMBDA10_ROOT),
                "posteval_unit": PAIRED_LAMBDA10_POSTEVAL_UNIT,
                "posteval_state": tasks["paired_lambda10_posteval"]["state"],
                "posteval_receipt": str(PAIRED_LAMBDA10_POSTEVAL_RECEIPT),
            },
            "paired_training_complete": paired["paired_training_complete"],
            "paired_posteval_complete": paired["paired_posteval_complete"],
            "paired_posteval_status": paired["paired_posteval_status"],
            "next_owner": paired["next_owner"],
            "approval_required": paired["approval_required"],
            "approval_state": paired["approval_state"],
            "approval_reference": paired["approval_reference"],
            "protocol_review": {
                "status": "LAMBDA0_LAMBDA10_PROTOCOL_MATCH_VERIFIED",
                "physicsnemo_image_id": (
                    "sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
                ),
                "normalization_sha256": (
                    "f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1"
                ),
                "implementation_sha256": PAIRED_PROTOCOL_SHA256,
                "validation10_expected_segments": {
                    "H1": 320,
                    "H10": 320,
                    "H50": 310,
                    "H100": 290,
                },
                "dynamic6_expected_segments": {
                    "H1": 1200,
                    "H10": 1146,
                    "H50": 906,
                    "H100": 606,
                    "all": 3858,
                },
                "training_epoch_metrics_used_for_verdict": False,
                "frozen_test_accessed": False,
            },
            "automatic_restart_allowed": False,
            "formal_training_units_assigned": True,
        },
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
    units[PAIRED_LAMBDA10_POSTEVAL_UNIT] = worker_unit_state(
        PAIRED_LAMBDA10_POSTEVAL_UNIT, user_scope=False
    )
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
