#!/usr/bin/env python3
"""Bounded, read-only research watchdog for the two DGX Spark nodes.

Runs under a user-level transient systemd service; it never restarts training
or relaxes a scientific gate. All samples remain on the primary Spark.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/monitor/research_window_20261002"
WORKER = "USER@WORKER_HOST"
HOST_PROBE = (
    "awk '/^MemAvailable:/ {print $2}' /proc/meminfo; "
    "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1; "
    "ps -eo args | grep -E 'train_tandem_fno_rollout.py|evaluate_tandem_fno.py|train_tandem_hydrogym_ppo_pilot.py' "
    "| grep -v grep | wc -l; "
    "jq -r 'length' /tmp/fluid_control_gateb_20261002/artifacts/"
    "tandem_fno_gate_b_aug_v3_rollout_seed20261005_10epoch/"
    "training_history.json 2>/dev/null || echo 0; "
    "jq -r 'length' /tmp/fluid_control_gateb_20261002/artifacts/"
    "tandem_fno_gate_b_aug_v3_rollout_h20_seed20261005_10epoch/"
    "training_history.json 2>/dev/null || echo 0; "
    "jq -r 'length' /tmp/fluid_control_gateb_20261002/artifacts/"
    "tandem_fno_gate_b_aug_v3_rollout_h20_seed20261002_dense_10epoch/"
    "training_history.json 2>/dev/null || echo 0; "
    "jq -r 'length' /tmp/fluid_control_gateb_20261002/artifacts/"
    "tandem_fno_gate_b_aug_v3_h20_rear_drag_seed20261002_10epoch/"
    "training_history.json 2>/dev/null || echo 0"
)


def probe(worker: bool) -> dict:
    command = (
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", WORKER, HOST_PROBE]
        if worker else ["sh", "-c", HOST_PROBE]
    )
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=12)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or f"exit {result.returncode}")
        lines = result.stdout.strip().splitlines()
        if len(lines) != 7:
            raise ValueError(f"unexpected host probe: {lines!r}")
        return {
            "reachable": True,
            "mem_available_gib": round(int(lines[0]) / 1024**2, 2),
            "gpu_utilization_pct": int(lines[1].strip().split()[0]),
            "training_or_evaluation_processes": int(lines[2]),
            "worker_multistep_epoch": int(lines[3]),
            "worker_h20_epoch": int(lines[4]),
            "worker_primary_seed_h20_epoch": int(lines[5]),
            "worker_h20_rear_drag_epoch": int(lines[6]),
        }
    except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as exc:
        return {"reachable": False, "error": str(exc)[:300]}


def read_audit(relative: str) -> dict:
    path = ROOT / relative
    if not path.is_file():
        return {"state": "pending"}
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        return {
            "state": report.get("status", "invalid"),
            "failed_checks": report.get("failed_checks", []),
            "heldout_nrmse": report.get("checks", {})
                .get("heldout_full_period_total_drag_nrmse", {})
                .get("total_drag_nrmse"),
        }
    except (OSError, ValueError) as exc:
        return {"state": "invalid", "error": str(exc)[:300]}


def training_epoch(relative: str) -> int | None:
    path = ROOT / relative
    if not path.is_file():
        return None
    try:
        history = json.loads(path.read_text(encoding="utf-8"))
        return int(history[-1]["epoch"]) if history else 0
    except (OSError, ValueError, IndexError, KeyError, TypeError):
        return None


def sample(deadline: datetime) -> dict:
    now = datetime.now(timezone.utc)
    primary = probe(False)
    worker = probe(True)
    audits = {
        "primary_one_step": read_audit("artifacts/tandem_fno_gate_b_aug_v3_30epoch/gate_b_audit.json"),
        "worker_one_step": read_audit("artifacts/distributed_runs/gateb_aug_v3_seed20261005_20261002/formal/tandem_fno_gate_b_aug_v3_seed20261005_30epoch/gate_b_audit.json"),
        "primary_multistep": read_audit("artifacts/tandem_fno_gate_b_aug_v3_rollout_seed20261002_10epoch/gate_b_audit.json"),
        "worker_multistep": read_audit("artifacts/distributed_runs/gateb_aug_v3_rollout_seed20261005_20261002/formal/tandem_fno_gate_b_aug_v3_rollout_seed20261005_10epoch/gate_b_audit.json"),
        "rear_drag_weighted": read_audit("artifacts/tandem_fno_gate_b_aug_v3_rear_drag_seed20261007_10epoch/gate_b_audit.json"),
        "primary_seed_h20": read_audit("artifacts/distributed_runs/gateb_aug_v3_rollout_h20_seed20261002_dense_20261002/formal/tandem_fno_gate_b_aug_v3_rollout_h20_seed20261002_dense_10epoch/gate_b_audit.json"),
        "combined_h20_rear_drag": read_audit("artifacts/distributed_runs/gateb_aug_v3_h20_rear_drag_seed20261002_20261003/formal/tandem_fno_gate_b_aug_v3_h20_rear_drag_seed20261002_10epoch/gate_b_audit.json"),
    }
    alerts = []
    for name, host in (("primary", primary), ("worker", worker)):
        if not host["reachable"]:
            alerts.append(f"{name}_unreachable")
        elif host["training_or_evaluation_processes"] and host["mem_available_gib"] < 20:
            alerts.append(f"{name}_memory_below_20_gib")
    for name, audit in audits.items():
        if audit["state"] == "invalid":
            alerts.append(f"{name}_audit_invalid")
    primary_epoch = training_epoch(
        "artifacts/tandem_fno_gate_b_aug_v3_rollout_seed20261002_10epoch/training_history.json"
    )
    rear_weighted_epoch = training_epoch(
        "artifacts/tandem_fno_gate_b_aug_v3_rear_drag_seed20261007_10epoch/training_history.json"
    )
    worker_epoch = worker.get("worker_multistep_epoch")
    worker_h20_epoch = worker.get("worker_h20_epoch")
    worker_primary_seed_h20_epoch = worker.get("worker_primary_seed_h20_epoch")
    worker_h20_rear_drag_epoch = worker.get("worker_h20_rear_drag_epoch")
    if primary["reachable"] and primary_epoch is not None and primary_epoch < 10:
        if primary["training_or_evaluation_processes"] == 0:
            alerts.append("primary_multistep_stopped_before_epoch_10")
    if primary["reachable"] and (rear_weighted_epoch is None or rear_weighted_epoch < 10):
        if (ROOT / "artifacts/tandem_cylinders/gate_b_aug_v3_rear_drag_seed20261007.log").exists() \
            and primary["training_or_evaluation_processes"] == 0:
            alerts.append("rear_drag_weighted_stopped_before_epoch_10")
    if worker["reachable"] and worker_epoch is not None and worker_epoch < 10:
        if worker["training_or_evaluation_processes"] == 0:
            alerts.append("worker_multistep_stopped_before_epoch_10")
    if worker["reachable"] and worker_h20_epoch is not None and worker_h20_epoch < 10:
        if (ROOT / "artifacts/distributed_runs/gateb_aug_v3_rollout_h20_preflight_20261002/runner.log").exists() \
            and worker["training_or_evaluation_processes"] == 0:
            alerts.append("worker_h20_stopped_before_epoch_10")
    if worker["reachable"] and worker_primary_seed_h20_epoch is not None and worker_primary_seed_h20_epoch < 10:
        if (ROOT / "artifacts/tandem_cylinders/gate_b_aug_v3_rollout_h20_seed20261002_dense_finalize.log").exists() \
            and worker["training_or_evaluation_processes"] == 0:
            alerts.append("worker_primary_seed_h20_stopped_before_epoch_10")
    if worker["reachable"] and worker_h20_rear_drag_epoch is not None and worker_h20_rear_drag_epoch < 10:
        if (ROOT / "docs/RESEARCH_VALUE_AUDIT_20261003.md").exists() \
            and worker["training_or_evaluation_processes"] == 0:
            alerts.append("worker_h20_rear_drag_stopped_before_epoch_10")
    for marker, name in (
        ("artifacts/tandem_cylinders/GATE_B_AUG_V3_ROLLOUT_SEED20261002_FAILED", "primary_rollout_pipeline_failed"),
        ("artifacts/distributed_runs/gateb_aug_v3_rollout_seed20261005_20261002/formal/MULTISTEP_GATE_B_FAILED", "worker_rollout_finalizer_failed"),
        ("artifacts/distributed_runs/gateb_aug_v3_rollout_h20_seed20261005_20261002/formal/MULTISTEP_GATE_B_FAILED", "worker_h20_finalizer_failed"),
        ("artifacts/tandem_cylinders/GATE_B_AUG_V3_REAR_DRAG_FINALIZE_FAILED", "rear_drag_finalizer_failed"),
        ("artifacts/distributed_runs/gateb_aug_v3_rollout_h20_seed20261002_dense_20261002/formal/MULTISTEP_GATE_B_FAILED", "primary_seed_h20_finalizer_failed"),
        ("artifacts/distributed_runs/gateb_aug_v3_h20_rear_drag_seed20261002_20261003/formal/MULTISTEP_GATE_B_FAILED", "combined_h20_rear_drag_finalizer_failed"),
    ):
        if (ROOT / marker).exists():
            alerts.append(name)
    return {
        "timestamp_utc": now.isoformat(timespec="seconds"),
        "deadline_utc": deadline.isoformat(timespec="seconds"),
        "seconds_remaining": max(0, int((deadline - now).total_seconds())),
        "primary": primary,
        "worker": worker,
        "audits": audits,
        "primary_multistep_epoch": primary_epoch,
        "rear_drag_weighted_epoch": rear_weighted_epoch,
        "worker_multistep_epoch": worker_epoch,
        "worker_h20_epoch": worker_h20_epoch,
        "worker_primary_seed_h20_epoch": worker_primary_seed_h20_epoch,
        "worker_h20_rear_drag_epoch": worker_h20_rear_drag_epoch,
        "alerts": alerts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline-utc", required=True)
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not 15 <= args.interval_seconds <= 600:
        parser.error("interval must be 15..600 seconds")
    deadline = datetime.fromisoformat(args.deadline_utc.replace("Z", "+00:00"))
    if deadline.tzinfo is None:
        parser.error("deadline must be timezone-aware")
    OUT.mkdir(parents=True, exist_ok=True)
    while True:
        row = sample(deadline)
        with (OUT / "samples.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary = OUT / "latest.json.tmp"
        temporary.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
        temporary.replace(OUT / "latest.json")
        print(json.dumps({key: row[key] for key in (
            "timestamp_utc", "seconds_remaining", "primary_multistep_epoch",
            "worker_multistep_epoch", "alerts"
        )}), flush=True)
        if args.once or row["seconds_remaining"] <= 0:
            break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
