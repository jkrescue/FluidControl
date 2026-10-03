#!/usr/bin/env python3
"""Safely schedule and repatriate the nine reviewed matched-start CFD cases."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CASES = REPO / "cfd/tandem_cylinders/cases"
WORKER = "USER@WORKER_HOST"
WORKER_REPO = "/home/USER/workspace/fluid_control_v4_compute_415a50f/project"
MANIFEST = "artifacts/tandem_cylinders/matched_start_phase_restart_predeclared_v3_20261003.json"
APPROVAL_TOKEN = "EXECUTE_REVIEWED_9_CASE_COMMISSIONING"
ROOT = REPO / "artifacts/matched_start_acquisition"
RECEIPTS = ROOT / "transfer_verified"
WORKER_MANIFESTS = ROOT / "worker_transfer_manifests"
CASE_AUDITS = ROOT / "case_audits"
QUEUE = (
    "matched_start_acquisition_train_b00_zero",
    "matched_start_acquisition_train_b00_p075",
    "matched_start_acquisition_train_b00_m075",
    "matched_start_acquisition_train_b02_zero",
    "matched_start_acquisition_train_b02_p075",
    "matched_start_acquisition_train_b02_m075",
    "matched_start_acquisition_train_b04_zero",
    "matched_start_acquisition_train_b04_p075",
    "matched_start_acquisition_train_b04_m075",
)


def is_case_process(comm: str, command: str, name: str) -> bool:
    if name not in command:
        return False
    if comm == "pimpleFoam":
        return True
    return comm in {"bash", "sh"} and "run_matched_start_acquisition_case.sh" in command


def run(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(args, check=True, text=True, **kwargs)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(payload, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def worker_snapshot() -> dict:
    code = r'''
import json, os, shutil, sys
from pathlib import Path
root=Path(sys.argv[1]); names=sys.argv[2:]; cases={}
process_commands=[]
for command_path in Path('/proc').glob('[0-9]*/cmdline'):
    try:
        command=command_path.read_bytes().replace(b'\0',b' ').decode(errors='replace')
        comm=(command_path.parent/'comm').read_text().strip()
    except (FileNotFoundError,PermissionError): continue
    if comm=='pimpleFoam' or (comm in {'bash','sh'} and 'run_matched_start_acquisition_case.sh' in command):
        process_commands.append((comm,command))
for name in names:
    case=root/'cfd/tandem_cylinders/cases'/name
    lock=case/'.matched_start_solver_lock/state.json'
    if not lock.exists():
        dirty=(case/'log.pimpleFoam.matched_start_acquisition').exists() or (case/'solver_complete.json').exists()
        cases[name]={"status":"DIRTY_PENDING" if dirty else "PENDING"}
        continue
    state=json.loads(lock.read_text()); status=state.get('status'); alive=False
    pid=state.get('pid')
    if status=='ACQUIRED': alive=any(name in command for _,command in process_commands)
    if status=='ACQUIRED' and not alive: status='ORPHANED'
    cases[name]={"status":status,"pid":pid,"alive":alive}
stat=os.statvfs(root)
mem=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))
print(json.dumps({"cases":cases,"mem_available_kib":mem,"disk_free_bytes":stat.f_bavail*stat.f_frsize}))
'''
    remote = shlex.join(["python3", "-c", code, WORKER_REPO, *QUEUE])
    result = run(
        ["ssh", "-o", "BatchMode=yes", WORKER, remote],
        capture_output=True,
    )
    return json.loads(result.stdout)


def valid_receipts() -> set[str]:
    valid = set()
    for name in QUEUE:
        path = RECEIPTS / f"{name}.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") != "RAW_TRANSFER_VERIFIED" or data.get("case") != name:
            raise ValueError(f"invalid raw-transfer receipt: {path}")
        valid.add(name)
    return valid


def plan(snapshot: dict, receipts: set[str]) -> dict:
    states = snapshot["cases"]
    failures = [name for name, row in states.items() if row["status"] in {"FAILED", "ORPHANED", "DIRTY_PENDING"}]
    if failures:
        return {"stop": True, "failures": failures, "transfer": [], "start": []}
    transfer = [name for name in QUEUE if states[name]["status"] == "COMPLETED" and name not in receipts]
    active = sum(row["status"] == "ACQUIRED" for row in states.values())
    slots = max(0, 2 - active)
    start = [] if transfer else [name for name in QUEUE if states[name]["status"] == "PENDING"][:slots]
    return {"stop": False, "failures": [], "transfer": transfer, "start": start, "active": active}


def copy_manifest(name: str) -> Path:
    WORKER_MANIFESTS.mkdir(parents=True, exist_ok=True)
    target = WORKER_MANIFESTS / f"{name}.sha256"
    with tempfile.NamedTemporaryFile(dir=WORKER_MANIFESTS, prefix=f".{name}.", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        run(
            [
                "scp", "-q", f"{WORKER}:{WORKER_REPO}/artifacts/matched_start_acquisition/worker_transfer_manifests/{name}.sha256", str(temporary)
            ]
        )
        if target.exists():
            if target.read_bytes() != temporary.read_bytes():
                raise ValueError(f"worker manifest changed for {name}")
        else:
            os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def transfer_and_audit(name: str) -> None:
    target = CASES / name
    if not target.is_dir():
        raise ValueError(f"missing generated Spark case: {name}")
    run(
        [
            "rsync", "-a", "--exclude=.matched_start_solver_lock",
            f"{WORKER}:{WORKER_REPO}/cfd/tandem_cylinders/cases/{name}/", f"{target}/",
        ]
    )
    copy_manifest(name)
    output = CASE_AUDITS / name / "result.json"
    if output.exists() or (RECEIPTS / f"{name}.json").exists():
        raise ValueError(f"refusing to overwrite audit/receipt for {name}")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO / "cfd/tandem_cylinders")
    manifest_dir_arg = WORKER_MANIFESTS.relative_to(REPO)
    receipt_dir_arg = RECEIPTS.relative_to(REPO)
    output_arg = output.relative_to(REPO)
    run(
        [
            sys.executable,
            str(REPO / "scripts/audit_matched_start_acquisition_commissioning.py"),
            "--worker-raw-manifest-dir", str(manifest_dir_arg),
            "--receipt-dir", str(receipt_dir_arg), "--output", str(output_arg), "--case", name,
        ],
        env=environment,
        cwd=REPO,
    )


def finalize_aggregate() -> None:
    output = ROOT / "aggregate_qc/result.json"
    if output.is_file():
        report = json.loads(output.read_text(encoding="utf-8"))
        if report.get("status") != "MATCHED_START_9_CASE_COMMISSIONING_QC_PASS":
            raise ValueError("existing aggregate QC is not a pass")
        return
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO / "cfd/tandem_cylinders")
    manifest_dir_arg = WORKER_MANIFESTS.relative_to(REPO)
    receipt_dir_arg = RECEIPTS.relative_to(REPO)
    output_arg = output.relative_to(REPO)
    run(
        [
            sys.executable,
            str(REPO / "scripts/audit_matched_start_acquisition_commissioning.py"),
            "--worker-raw-manifest-dir", str(manifest_dir_arg),
            "--receipt-dir", str(receipt_dir_arg), "--output", str(output_arg),
        ],
        env=environment,
        cwd=REPO,
    )


def start_case(name: str) -> None:
    launch_dir = f"{WORKER_REPO}/artifacts/matched_start_acquisition/runner_logs"
    remote = (
        f"set -eu; cd {WORKER_REPO}; mkdir -p {launch_dir}; "
        f"test ! -e {launch_dir}/{name}.log; "
        f"nohup bash cfd/tandem_cylinders/run_matched_start_acquisition_case.sh "
        f"{name} {MANIFEST} > {launch_dir}/{name}.log 2>&1 < /dev/null & echo $!"
    )
    result = run(["ssh", "-o", "BatchMode=yes", WORKER, remote], capture_output=True)
    if not result.stdout.strip().isdigit():
        raise ValueError(f"worker did not return runner PID for {name}: {result.stdout!r}")


def write_state(snapshot: dict, actions: dict, mode: str) -> None:
    phases = {}
    receipts = valid_receipts()
    for phase in ("b00", "b02", "b04"):
        members = [name for name in QUEUE if phase in name]
        phases[phase] = {"raw_transfer_complete": all(name in receipts for name in members), "members": members}
    atomic_json(
        ROOT / "scheduler_state.json",
        {
            "updated_utc": datetime.now(UTC).isoformat(),
            "mode": mode,
            "snapshot": snapshot,
            "actions": actions,
            "phases": phases,
            "guard": "any FAILED/ORPHANED/DIRTY_PENDING case stops all new launches",
        },
    )


def iteration(execute: bool) -> bool:
    snapshot = worker_snapshot()
    actions = plan(snapshot, valid_receipts())
    write_state(snapshot, actions, "execute" if execute else "dry-run")
    print(json.dumps(actions, indent=2))
    if actions["stop"]:
        raise RuntimeError(f"fail-stop: {actions['failures']}")
    if not execute:
        return False
    for name in actions["transfer"]:
        transfer_and_audit(name)
    if actions["transfer"]:
        return False
    if snapshot["mem_available_kib"] < 40 * 1024 * 1024:
        raise RuntimeError("worker MemAvailable is below 40 GiB")
    if snapshot["disk_free_bytes"] < 50 * 1024**3:
        raise RuntimeError("worker free disk is below 50 GiB")
    for name in actions["start"]:
        start_case(name)
    if len(valid_receipts()) == len(QUEUE):
        finalize_aggregate()
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approval-token")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=15.0)
    args = parser.parse_args()
    if args.execute and args.approval_token != APPROVAL_TOKEN:
        parser.error("execution requires the reviewed approval token")
    ROOT.mkdir(parents=True, exist_ok=True)
    lock_stream = (ROOT / "scheduler.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise SystemExit("another scheduler instance holds the lock") from error
    while True:
        done = iteration(args.execute)
        if done or not args.watch or not args.execute:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
