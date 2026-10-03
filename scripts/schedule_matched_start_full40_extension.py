#!/usr/bin/env python3
"""Dry-run-first scheduler for the independently authorized full40 remainder."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CASES = REPO / "cfd/tandem_cylinders/cases"
PREDECLARATION = (
    REPO / "artifacts/tandem_cylinders/matched_start_full40_predeclared_20261003.json"
)
PREDECLARATION_SHA256 = "d7ff174ef10194a8739357376335ca13ff9b45c8079970846bb71f15a715d24b"
AUTHORIZATION = (
    REPO / "artifacts/tandem_cylinders/matched_start_full40_extension_authorized_20261003.json"
)
AUTHORIZATION_SHA256 = "REVIEW_REQUIRED_AFTER_NINE_CASE_AGGREGATE"
PHASE_MANIFEST = (
    "artifacts/tandem_cylinders/"
    "matched_start_phase_restart_predeclared_v3_20261003.json"
)
WORKER = "USER@WORKER_HOST"
WORKER_REPO = "/home/USER/workspace/fluid_control_v4_compute_415a50f/project"
ROOT = REPO / "artifacts/matched_start_full40_extension"
MAX_PARALLEL = 4
START_MEM_KIB = 64 * 1024 * 1024
RUNNING_MEM_KIB = 40 * 1024 * 1024
WORKER_DISK_BYTES = 100 * 1024**3
SPARK_DISK_BYTES = 250 * 1024**3
EXECUTION_TOKEN = "EXECUTE_COMMITTED_FULL40_EXTENSION"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_queue() -> tuple[str, ...]:
    if sha256(PREDECLARATION) != PREDECLARATION_SHA256:
        raise ValueError("full40 predeclaration SHA-256 differs")
    data = json.loads(PREDECLARATION.read_text(encoding="utf-8"))
    queue = sorted(
        name
        for name, row in data.get("cases", {}).items()
        if row.get("disposition") == "planned_new_remainder_case"
    )
    if len(queue) != 31:
        raise ValueError("full40 remainder queue is not exactly 31 cases")
    return tuple(queue)


QUEUE = load_queue()


def validate_execution_authorization() -> dict:
    if len(AUTHORIZATION_SHA256) != 64:
        raise ValueError("extension authorization is not committed and SHA-bound")
    if sha256(AUTHORIZATION) != AUTHORIZATION_SHA256:
        raise ValueError("extension authorization SHA-256 differs")
    data = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    resources = {
        "worker_start_mem_available_gib_at_least": 64,
        "worker_running_mem_available_gib_at_least": 40,
        "worker_free_disk_gib_at_least": 100,
        "spark_free_disk_gib_at_least": 250,
    }
    if (
        data.get("status") != "MATCHED_START_FULL40_EXTENSION_AUTHORIZED"
        or data.get("full40_predeclaration_sha256") != PREDECLARATION_SHA256
        or data.get("authorized_cases") != list(QUEUE)
        or data.get("maximum_parallel_cases") != MAX_PARALLEL
        or data.get("resource_guards") != resources
        or len(data.get("nine_case_aggregate_qc_sha256", "")) != 64
        or len(data.get("nine_raw_transfer_receipt_sha256", {})) != 9
    ):
        raise ValueError("extension authorization content differs")
    return data


def is_case_process(comm: str, command: str, name: str) -> bool:
    if name not in command:
        return False
    if comm == "pimpleFoam":
        return True
    return comm in {"bash", "sh"} and "run_matched_start_full40_case.sh" in command


def plan(snapshot: dict, receipts: set[str]) -> dict:
    states = snapshot["cases"]
    failures = [
        name
        for name, row in states.items()
        if row["status"] in {"FAILED", "ORPHANED", "DIRTY_PENDING"}
    ]
    if failures:
        return {"stop": True, "failures": failures, "transfer": [], "start": []}
    transfer = [
        name for name in QUEUE if states[name]["status"] == "COMPLETED" and name not in receipts
    ]
    active = sum(row["status"] == "ACQUIRED" for row in states.values())
    slots = max(0, MAX_PARALLEL - active)
    start = []
    if not transfer:
        start = [name for name in QUEUE if states[name]["status"] == "PENDING"][:slots]
    return {
        "stop": False,
        "failures": [],
        "transfer": transfer,
        "start": start,
        "active": active,
    }


def resource_gate(snapshot: dict, spark_free_bytes: int, starting: bool) -> list[str]:
    failures = []
    memory_floor = START_MEM_KIB if starting else RUNNING_MEM_KIB
    if snapshot["mem_available_kib"] < memory_floor:
        failures.append("worker_mem_available")
    if snapshot["disk_free_bytes"] < WORKER_DISK_BYTES:
        failures.append("worker_free_disk")
    if spark_free_bytes < SPARK_DISK_BYTES:
        failures.append("spark_free_disk")
    return failures


def worker_snapshot() -> dict:
    code = r'''
import json, os, sys
from pathlib import Path
root=Path(sys.argv[1]); names=sys.argv[2:]; cases={}; processes=[]
for cmdline in Path('/proc').glob('[0-9]*/cmdline'):
    try:
        command=cmdline.read_bytes().replace(b'\0',b' ').decode(errors='replace')
        comm=(cmdline.parent/'comm').read_text().strip()
    except (FileNotFoundError,PermissionError): continue
    is_runner = comm in {'bash','sh'} and 'run_matched_start_full40_case.sh' in command
    if comm=='pimpleFoam' or is_runner:
        processes.append((comm,command))
for name in names:
    case=root/'cfd/tandem_cylinders/cases'/name
    lock=case/'.matched_start_full40_solver_lock/state.json'
    if not lock.exists():
        dirty=(
            (case/'log.pimpleFoam.matched_start_full40').exists()
            or (case/'solver_complete.full40.json').exists()
        )
        cases[name]={'status':'DIRTY_PENDING' if dirty else 'PENDING'}; continue
    state=json.loads(lock.read_text()); status=state.get('status')
    alive=status=='ACQUIRED' and any(name in command for _,command in processes)
    if status=='ACQUIRED' and not alive: status='ORPHANED'
    cases[name]={'status':status,'pid':state.get('pid'),'alive':alive}
stat=os.statvfs(root)
meminfo=Path('/proc/meminfo').read_text().splitlines()
mem=int(next(line.split()[1] for line in meminfo if line.startswith('MemAvailable:')))
print(json.dumps({
    'cases':cases, 'mem_available_kib':mem,
    'disk_free_bytes':stat.f_bavail*stat.f_frsize,
}))
'''
    command = shlex.join(["python3", "-c", code, WORKER_REPO, *QUEUE])
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", WORKER, command],
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(result.stdout)


def valid_receipts() -> set[str]:
    receipt_dir = ROOT / "transfer_verified"
    valid = set()
    for name in QUEUE:
        path = receipt_dir / f"{name}.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") != "FULL40_RAW_TRANSFER_VERIFIED" or data.get("case") != name:
            raise ValueError(f"invalid full40 raw receipt: {name}")
        valid.add(name)
    return valid


def dry_run_report(snapshot: dict | None = None) -> dict:
    if snapshot is None:
        snapshot = {
            "cases": {name: {"status": "NOT_GENERATED"} for name in QUEUE},
            "mem_available_kib": 0,
            "disk_free_bytes": 0,
        }
    return {
        "mode": "dry-run",
        "execution_enabled": len(AUTHORIZATION_SHA256) == 64,
        "authorization": "BLOCKED_PENDING_NINE_CASE_AGGREGATE_AND_COMMITTED_SHA",
        "queue_count": len(QUEUE),
        "maximum_parallel_cases": MAX_PARALLEL,
        "resource_guards": {
            "start_worker_mem_gib": 64,
            "running_worker_mem_gib": 40,
            "worker_disk_gib": 100,
            "spark_disk_gib": 250,
        },
        "snapshot": snapshot,
    }


def write_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(payload, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise ValueError(f"refusing to overwrite {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run-output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approval-token")
    args = parser.parse_args()
    if not args.execute:
        report = dry_run_report()
        if args.dry_run_output:
            write_exclusive(args.dry_run_output, report)
        print(json.dumps(report, indent=2))
        return
    if args.approval_token != EXECUTION_TOKEN:
        parser.error("execution requires the committed-extension approval token")
    validate_execution_authorization()
    lock_path = ROOT / "scheduler.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit("another full40 extension scheduler holds the lock") from error
        spark_free = shutil.disk_usage(REPO).free
        snapshot = worker_snapshot()
        actions = plan(snapshot, valid_receipts())
        gates = resource_gate(snapshot, spark_free, starting=bool(actions["start"]))
        # Execution implementation remains deliberately unavailable until the
        # authorization SHA is bound and this branch receives a second review.
        raise SystemExit(
            json.dumps(
                {
                    "status": "EXECUTION_IMPLEMENTATION_REVIEW_REQUIRED",
                    "plan": actions,
                    "resource_failures": gates,
                }
            )
        )


if __name__ == "__main__":
    main()
