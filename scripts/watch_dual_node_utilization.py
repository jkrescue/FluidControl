#!/usr/bin/env python3
"""Read-only dual-node utilization and pipeline-progress watchdog.

The watchdog records alerts and reviewed recovery candidates.  It never starts,
restarts, or changes a solver, Curator, scheduler, training, or research task.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO / "artifacts/monitor/dual_node_watchdog_20261003"
WORKER = "USER@WORKER_HOST"
IDLE_ALERT_SECONDS = 300
SPARK_MIN_MEMORY_GIB = 20.0
WORKER_MIN_MEMORY_GIB = 40.0
USEFUL_PROCESS_CPU_PCT = 5.0
COMPUTE_RE = re.compile(
    r"(?:pimpleFoam|foamToVTK|curate_low_action_phase94_validation\.py|"
    r"curate_matched_start_full40_remainder\.py|train_tandem|evaluate_tandem)"
)
SERVICES = {
    "full40_scheduler": "fluid-control-full40-extension-20261003.service",
    "matched9_curator": "fluid-control-curator-matched9-20261003.service",
}
RAW_AGGREGATE_STATUS = "MATCHED_START_FULL40_EXTENSION_31_CASE_RAW_QC_PASS"

REMOTE_PROBE = r"""
import json, subprocess
def read_cpu():
    row = open('/proc/stat').readline().split()[1:]
    return [int(v) for v in row]
def mem():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])
def run(args):
    p=subprocess.run(args,text=True,capture_output=True,timeout=8)
    return p.stdout if p.returncode == 0 else ''
gpu=run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.free','--format=csv,noheader,nounits'])
ps=run(['ps','-eo','pid=,etimes=,pcpu=,args='])
print(json.dumps({'cpu':read_cpu(),'mem_available_kib':mem(),'gpu':gpu,'ps':ps}))
"""


def utc_now() -> datetime:
    return datetime.now(UTC)


def local_probe_payload() -> dict:
    cpu = [int(value) for value in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
    mem_available = None
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            mem_available = int(line.split()[1])
            break
    if mem_available is None:
        raise ValueError("MemAvailable missing")
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        timeout=8,
        check=False,
    ).stdout
    processes = subprocess.run(
        ["ps", "-eo", "pid=,etimes=,pcpu=,args="],
        text=True,
        capture_output=True,
        timeout=8,
        check=True,
    ).stdout
    return {"cpu": cpu, "mem_available_kib": mem_available, "gpu": gpu, "ps": processes}


def remote_probe_payload() -> dict:
    result = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            WORKER,
            f"python3 -c {shlex.quote(REMOTE_PROBE)}",
        ],
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"ssh exit {result.returncode}")
    return json.loads(result.stdout)


def cpu_utilization(previous: list[int] | None, current: list[int]) -> float | None:
    if previous is None or len(previous) < 5 or len(current) != len(previous):
        return None
    total_delta = sum(current) - sum(previous)
    idle_delta = (current[3] + current[4]) - (previous[3] + previous[4])
    if total_delta <= 0 or idle_delta < 0:
        return None
    return 100.0 * (total_delta - idle_delta) / total_delta


def parse_gpu(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        try:
            utilization, used, free = [float(value.strip()) for value in line.split(",")]
        except (TypeError, ValueError):
            continue
        rows.append(
            {
                "utilization_pct": utilization,
                "memory_used_mib": used,
                "memory_free_mib": free,
            }
        )
    return rows


def parse_compute_processes(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        parts = line.strip().split(maxsplit=3)
        if len(parts) != 4 or not COMPUTE_RE.search(parts[3]):
            continue
        try:
            rows.append(
                {
                    "pid": int(parts[0]),
                    "elapsed_seconds": int(parts[1]),
                    "cpu_pct": float(parts[2]),
                    "command": parts[3][:500],
                }
            )
        except ValueError:
            continue
    return rows


def summarize_host(
    name: str, payload: dict, previous_cpu: list[int] | None, memory_floor: float
) -> dict:
    gpu = parse_gpu(str(payload.get("gpu", "")))
    processes = parse_compute_processes(str(payload.get("ps", "")))
    process_cpu = sum(row["cpu_pct"] for row in processes)
    memory_gib = float(payload["mem_available_kib"]) / 1024**2
    return {
        "node": name,
        "reachable": True,
        "cpu_counters": payload["cpu"],
        "node_cpu_utilization_pct": cpu_utilization(previous_cpu, payload["cpu"]),
        "mem_available_gib": round(memory_gib, 3),
        "memory_floor_gib": memory_floor,
        "memory_guard_pass": memory_gib >= memory_floor,
        "gpu": gpu,
        "project_compute_process_count": len(processes),
        "project_compute_cpu_pct_sum": process_cpu,
        "project_compute_processes": processes[:20],
        # GPU telemetry is informational.  An unrelated GPU job must not mask
        # the absence of an approved fluid-control compute process.
        "useful_compute_active": bool(process_cpu >= USEFUL_PROCESS_CPU_PCT),
    }


def read_json_status(path: Path, expected: str) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == expected
    except (OSError, ValueError):
        return False


def project_progress(repo: Path) -> dict:
    receipts = repo / "artifacts/matched_start_full40_extension/transfer_verified"
    matched9_staging = repo / "data/curated/.staging/matched_start_commissioning_train9_v1"
    full40_staging = repo / "data/curated/.staging/matched_start_full40_v1"
    matched9_final = (
        repo / "data/curated/tandem_cylinders_matched_start_commissioning_train9_v1"
    )
    full40_final = repo / "data/curated/tandem_cylinders_matched_start_full40_v1"
    raw_complete = read_json_status(
        repo / "artifacts/matched_start_full40_extension/aggregate_qc/result.json",
        RAW_AGGREGATE_STATUS,
    )
    matched9_complete = False
    matched9_manifest = matched9_final / "manifest.json"
    if matched9_manifest.is_file():
        try:
            manifest = json.loads(matched9_manifest.read_text(encoding="utf-8"))
            matched9_complete = (
                manifest.get("profile") == "matched_start_commissioning_train9_v1"
                and manifest.get("trajectory_counts")
                == {"train": 9, "validation": 0, "test": 0}
            )
        except (OSError, ValueError):
            matched9_complete = False
    final_manifest = full40_final / "manifest.json"
    final_complete = False
    if final_manifest.is_file():
        try:
            manifest = json.loads(final_manifest.read_text(encoding="utf-8"))
            final_complete = (
                manifest.get("profile") == "matched_start_full40_v1"
                and manifest.get("trajectory_counts")
                == {"train": 20, "validation": 10, "frozen_test": 10}
            )
        except (OSError, ValueError):
            final_complete = False
    return {
        "strict_full40_receipt_count": len(list(receipts.glob("*.json"))),
        "strict_full40_receipt_target": 31,
        "raw_31_case_aggregate_complete": raw_complete,
        "matched9_staging_hdf5_count": len(list(matched9_staging.glob("*/*/*.h5"))),
        "matched9_staging_tmp_count": len(list(matched9_staging.glob("*/*/*.h5.tmp"))),
        "matched9_final_complete": matched9_complete,
        "full40_remainder_staging_hdf5_count": len(list(full40_staging.glob("*/*/*.h5"))),
        "full40_remainder_staging_tmp_count": len(list(full40_staging.glob("*/*/*.h5.tmp"))),
        "full40_final_manifest_complete": final_complete,
        "full40_final_hdf5_count": len(list(full40_final.glob("*/*.h5"))),
        "project_complete": final_complete,
    }


def service_states() -> dict:
    states = {}
    for label, unit in SERVICES.items():
        result = subprocess.run(
            ["systemctl", "--user", "show", unit, "--property=ActiveState,SubState", "--value"],
            text=True,
            capture_output=True,
            timeout=8,
            check=False,
        )
        values = result.stdout.strip().splitlines()
        states[label] = {
            "unit": unit,
            "active_state": values[0] if values else "unknown",
            "sub_state": values[1] if len(values) > 1 else "unknown",
        }
    return states


def update_idle_state(
    node: dict, incomplete: bool, previous: dict | None, now: datetime
) -> dict:
    if not node.get("reachable") or not incomplete or node.get("useful_compute_active"):
        idle_since = None
        duration = 0
    else:
        idle_since = previous.get("idle_since_utc") if previous else None
        if not idle_since:
            idle_since = now.isoformat(timespec="seconds")
        duration = max(
            0,
            int((now - datetime.fromisoformat(idle_since)).total_seconds()),
        )
    node["assigned_task_incomplete"] = incomplete
    node["idle_since_utc"] = idle_since
    node["idle_duration_seconds"] = duration
    node["idle_alert_threshold_seconds"] = IDLE_ALERT_SECONDS
    return node


def recovery_candidates(services: dict, progress: dict) -> list[dict]:
    candidates = []
    if not progress["raw_31_case_aggregate_complete"]:
        candidates.append({
            "task": "full40_raw_acquisition_scheduler",
            "unit": services["full40_scheduler"]["unit"],
            "current_state": services["full40_scheduler"]["active_state"],
            "manual_review_command": (
                "systemctl --user restart " + services["full40_scheduler"]["unit"]
            ),
            "automatic_execution_permitted": False,
        })
    if not progress["matched9_final_complete"]:
        candidates.append({
            "task": "matched_start_curator",
            "unit": services["matched9_curator"]["unit"],
            "current_state": services["matched9_curator"]["active_state"],
            "manual_review_command": (
                "systemctl --user restart " + services["matched9_curator"]["unit"]
            ),
            "automatic_execution_permitted": False,
        })
    if (
        progress["raw_31_case_aggregate_complete"]
        and progress["matched9_final_complete"]
        and not progress["full40_final_manifest_complete"]
    ):
        candidates.append(
            {
                "task": "full40_remainder_curation",
                "unit": None,
                "current_state": "no reviewed systemd restart unit is installed",
                "manual_review_command": None,
                "automatic_execution_permitted": False,
            }
        )
    return candidates


def build_sample(
    repo: Path, previous: dict | None, local_payload: dict, worker_payload: dict
) -> dict:
    now = utc_now()
    progress = project_progress(repo)
    previous_nodes = previous.get("nodes", {}) if previous else {}
    spark = summarize_host(
        "spark", local_payload, previous_nodes.get("spark", {}).get("cpu_counters"), SPARK_MIN_MEMORY_GIB
    )
    worker = summarize_host(
        "worker78", worker_payload, previous_nodes.get("worker78", {}).get("cpu_counters"), WORKER_MIN_MEMORY_GIB
    )
    spark = update_idle_state(
        spark,
        not progress["full40_final_manifest_complete"],
        previous_nodes.get("spark"),
        now,
    )
    worker = update_idle_state(
        worker,
        not progress["raw_31_case_aggregate_complete"],
        previous_nodes.get("worker78"),
        now,
    )
    services = service_states()
    alerts = []
    for name, node in (("spark", spark), ("worker78", worker)):
        if not node["memory_guard_pass"]:
            alerts.append(f"{name}_MEMORY_GUARD_BREACH")
        if (
            node["assigned_task_incomplete"]
            and node["idle_duration_seconds"] >= IDLE_ALERT_SECONDS
        ):
            alerts.append(f"{name}_NO_USEFUL_COMPUTE_FOR_300_SECONDS")
    if (
        not progress["raw_31_case_aggregate_complete"]
        and services["full40_scheduler"]["active_state"] != "active"
    ):
        alerts.append("FULL40_SCHEDULER_INACTIVE_BEFORE_RAW_AGGREGATE")
    if (
        not progress["matched9_final_complete"]
        and services["matched9_curator"]["active_state"] != "active"
    ):
        alerts.append("MATCHED9_CURATOR_INACTIVE_BEFORE_FINALIZATION")
    return {
        "status": "ALERT" if alerts else "MONITORING",
        "timestamp_utc": now.isoformat(timespec="seconds"),
        "policy": {
            "sample_interval_seconds": None,
            "idle_alert_seconds": IDLE_ALERT_SECONDS,
            "spark_min_mem_available_gib": SPARK_MIN_MEMORY_GIB,
            "worker_min_mem_available_gib": WORKER_MIN_MEMORY_GIB,
            "automatic_restart_or_research_action": False,
        },
        "nodes": {"spark": spark, "worker78": worker},
        "progress": progress,
        "services": services,
        "alerts": alerts,
        "reviewed_recovery_candidates": (
            recovery_candidates(services, progress) if alerts else []
        ),
    }


def atomic_replace_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(payload, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_previous(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=int, default=15)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not 10 <= args.interval <= 30:
        parser.error("interval must be between 10 and 30 seconds")
    latest = args.output_dir / "latest.json"
    while True:
        previous = read_previous(latest)
        try:
            local_payload = local_probe_payload()
            worker_payload = remote_probe_payload()
            sample = build_sample(REPO, previous, local_payload, worker_payload)
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
            sample = {
                "status": "ALERT",
                "timestamp_utc": utc_now().isoformat(timespec="seconds"),
                "alerts": ["WATCHDOG_PROBE_FAILURE"],
                "error": f"{type(error).__name__}: {error}"[:500],
                "automatic_restart_or_research_action": False,
            }
        sample.setdefault("policy", {})["sample_interval_seconds"] = args.interval
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / "samples.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(sample, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        old_alerts = set(previous.get("alerts", [])) if previous else set()
        new_alerts = set(sample.get("alerts", [])) - old_alerts
        if new_alerts:
            with (args.output_dir / "alerts.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "timestamp_utc": sample["timestamp_utc"],
                            "alerts": sorted(new_alerts),
                            "recovery_candidates": sample.get(
                                "reviewed_recovery_candidates", []
                            ),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                stream.flush()
                os.fsync(stream.fileno())
        atomic_replace_json(latest, sample)
        print(
            json.dumps(
                {
                    "timestamp_utc": sample["timestamp_utc"],
                    "status": sample["status"],
                    "alerts": sample.get("alerts", []),
                }
            ),
            flush=True,
        )
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
