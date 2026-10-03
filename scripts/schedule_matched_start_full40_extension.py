#!/usr/bin/env python3
"""Dry-run-first scheduler for the independently authorized full40 remainder."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
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
AUTHORIZATION_SHA256 = "da8bccaf18a86666ac78804e775c1608d8fc390bfca1484cbd1eaabe93c17151"
PHASE_MANIFEST = (
    "artifacts/tandem_cylinders/"
    "matched_start_phase_restart_predeclared_v3_20261003.json"
)
PHASE_MANIFEST_SHA256 = "6279492bd3a79eff868a4333e1f642e3be4a4d58a78e46dc47dde040e4e39603"
WORKER = "USER@WORKER_HOST"
WORKER_REPO = "/home/USER/workspace/fluid_control_v4_compute_415a50f/project"
ROOT = REPO / "artifacts/matched_start_full40_extension"
RECEIPTS = ROOT / "transfer_verified"
WORKER_MANIFESTS = ROOT / "worker_transfer_manifests"
CASE_AUDITS = ROOT / "case_audits"
TRANSFER_STAGING = ROOT / "return_staging"
GENERATOR = REPO / "cfd/tandem_cylinders/make_matched_start_full40.py"
RUNNER = "cfd/tandem_cylinders/run_matched_start_full40_case.sh"
AUDITOR = REPO / "scripts/audit_matched_start_full40_extension.py"
OPENFOAM_IMAGE = "opencfd/openfoam-default@sha256:33fb575aa9980d2bc42fd58c75ae698c489293ba30c991380fe3f899c622f319"
WORKER_DEPENDENCIES = {
    "cfd/tandem_cylinders/run_openfoam.sh": "ee15b3fb9efe6079b5c88bbce698281907184ad760403622a4e7708dcb3d9976",
    "cfd/tandem_cylinders/matched_start_case_lock.py": "2dabae9af1631e35f72d991861152b52fb9b55fd4d4df480a54cd92925159e80",
    "cfd/tandem_cylinders/check_matched_start_solver_log.py": "e4ab47ae9a30ccf6c87f1d0d43ca79ad7e99489665c6582140b5e6263a20db40",
}
MAX_PARALLEL = 4
START_MEM_KIB = 64 * 1024 * 1024
RUNNING_MEM_KIB = 40 * 1024 * 1024
WORKER_DISK_BYTES = 100 * 1024**3
SPARK_DISK_BYTES = 250 * 1024**3
EXECUTION_TOKEN = "EXECUTE_COMMITTED_FULL40_EXTENSION"
IMPLEMENTATION_REVIEWED = True
GENERATION_TOKEN = "GENERATE_REVIEWED_FULL40_REMAINDER"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_queue() -> tuple[str, ...]:
    if sha256(PREDECLARATION) != PREDECLARATION_SHA256:
        raise ValueError("full40 predeclaration SHA-256 differs")
    data = json.loads(PREDECLARATION.read_text(encoding="utf-8"))
    rows = [
        (name, row) for name, row in data.get("cases", {}).items()
        if row.get("disposition") == "planned_new_remainder_case"
    ]
    split_order = {"train": 0, "validation": 1, "frozen_test": 2}
    action_order = {-0.75: 0, -0.375: 1, 0.0: 2, 0.375: 3, 0.75: 4}
    queue = [name for name, _ in sorted(
        rows,
        key=lambda item: (
            split_order[item[1]["split"]], item[1]["phase_bin"],
            action_order[float(item[1]["action_target"])],
        ),
    )]
    if len(queue) != 31:
        raise ValueError("full40 remainder queue is not exactly 31 cases")
    return tuple(queue)


QUEUE = load_queue()
PREDECLARED = json.loads(PREDECLARATION.read_text(encoding="utf-8"))["cases"]


def authorized_case_set_matches(values: object) -> bool:
    return (
        isinstance(values, list)
        and len(values) == len(QUEUE)
        and len(set(values)) == len(QUEUE)
        and sorted(values) == sorted(QUEUE)
    )


def count_external_matched_cases(
    process_commands: list[list[str]], queue: tuple[str, ...] = QUEUE
) -> int:
    pattern = re.compile(
        r"matched_start_acquisition_(?:train|validation|frozen_test)_b0[0-7]_"
        r"(?:m075|m0375|zero|p0375|p075)"
    )
    external = {
        case
        for _, command in process_commands
        for case in pattern.findall(command)
        if case not in queue
    }
    return len(external)


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
    baseline = data.get("baseline_force_source_sha256")
    if not isinstance(baseline, dict) or set(baseline) != {"forceFront", "forceRear"}:
        raise ValueError("extension authorization lacks baseline force provenance")
    source = CASES / "tandem_backward_dt005"
    for force_name, expected_sha in baseline.items():
        if sha256(source / f"postProcessing/{force_name}/0/coefficient.dat") != expected_sha:
            raise ValueError(f"authorized baseline force changed: {force_name}")
    if (
        data.get("status") != "MATCHED_START_FULL40_EXTENSION_AUTHORIZED"
        or data.get("phase_manifest_sha256") != PHASE_MANIFEST_SHA256
        or data.get("full40_predeclaration_sha256") != PREDECLARATION_SHA256
        or not authorized_case_set_matches(data.get("authorized_cases"))
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
        return {
            "stop": True, "failures": failures, "transfer": [], "start": [],
            "prepare": [], "current_group": [], "eligible_groups": [],
        }
    transfer = [
        name for name in QUEUE if states[name]["status"] == "COMPLETED" and name not in receipts
    ]
    active = sum(row["status"] == "ACQUIRED" for row in states.values())
    active += int(snapshot.get("external_matched_start_active", 0))
    slots = max(0, MAX_PARALLEL - active)
    start = []
    prepare = []
    current_group = []
    eligible_groups = []
    for split in ("train", "validation", "frozen_test"):
        split_names = [name for name in QUEUE if PREDECLARED[name]["split"] == split]
        if all(name in receipts for name in split_names):
            continue
        for phase_bin in sorted({PREDECLARED[name]["phase_bin"] for name in split_names}):
            group = [name for name in split_names if PREDECLARED[name]["phase_bin"] == phase_bin]
            if not all(name in receipts for name in group):
                eligible_groups.append(group)
        break
    if eligible_groups:
        current_group = eligible_groups[0]
    eligible = [name for group in eligible_groups for name in group]
    if not transfer and eligible:
        start = [
            name for name in eligible
            if name not in receipts and states[name]["status"] in {"PENDING", "WORKER_STAGED"}
        ][:slots]
        remaining_slots = max(0, slots - len(start))
        prepare = [
            name for name in eligible
            if name not in receipts
            and states[name]["status"] in {"NOT_GENERATED", "SPARK_GENERATED"}
        ][:remaining_slots]
    return {
        "stop": False,
        "failures": [],
        "transfer": transfer,
        "start": start,
        "prepare": prepare,
        "active": active,
        "current_group": current_group,
        "eligible_groups": eligible_groups,
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
    if cmdline.parent.name == str(os.getpid()): continue
    try:
        command=cmdline.read_bytes().replace(b'\0',b' ').decode(errors='replace')
        comm=(cmdline.parent/'comm').read_text().strip()
    except (FileNotFoundError,PermissionError): continue
    is_runner = (
        comm in {'bash','sh'} and 'run_matched_start_' in command
        and 'python3 -c' not in command
    )
    if comm=='pimpleFoam' or is_runner:
        processes.append((comm,command))
for name in names:
    case=root/'cfd/tandem_cylinders/cases'/name
    if not (case/'case_config.json').is_file():
        cases[name]={'status':'NOT_STAGED'}; continue
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
    'matched_process_commands':processes,
}))
'''
    command = shlex.join(["python3", "-c", code, WORKER_REPO, *QUEUE])
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", WORKER, command],
        check=True,
        text=True,
        capture_output=True,
    )
    snapshot = json.loads(result.stdout)
    snapshot["external_matched_start_active"] = count_external_matched_cases(
        snapshot.pop("matched_process_commands", [])
    )
    return snapshot


def enrich_snapshot(snapshot: dict) -> dict:
    """Combine worker state with immutable Spark generation state."""
    for name, row in snapshot["cases"].items():
        if row["status"] != "NOT_STAGED":
            if row["status"] == "PENDING":
                row["status"] = "WORKER_STAGED"
            continue
        spark_config = CASES / name / "case_config.json"
        row["status"] = "SPARK_GENERATED" if spark_config.is_file() else "NOT_GENERATED"
    return snapshot


def valid_receipts() -> set[str]:
    valid = set()
    for name in QUEUE:
        path = RECEIPTS / f"{name}.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        expected = PREDECLARED[name]
        required = {
            "status": "FULL40_RAW_TRANSFER_VERIFIED", "case": name,
            "split": expected["split"], "phase_bin": expected["phase_bin"],
            "action_target": expected["action_target"],
            "full40_predeclaration_sha256": PREDECLARATION_SHA256,
            "full40_extension_authorization_sha256": AUTHORIZATION_SHA256,
        }
        if any(data.get(key) != value for key, value in required.items()):
            raise ValueError(f"invalid full40 raw receipt: {name}")
        manifest = REPO / data.get("worker_raw_manifest", "")
        if (
            not manifest.is_file()
            or sha256(manifest) != data.get("worker_raw_manifest_sha256")
            or data.get("raw_file_count", 0) <= 0
        ):
            raise ValueError(f"invalid full40 worker manifest receipt: {name}")
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
        "execution_enabled": len(AUTHORIZATION_SHA256) == 64 and IMPLEMENTATION_REVIEWED,
        "authorization": (
            "BOUND_BUT_BLOCKED_PENDING_IMPLEMENTATION_REVIEW"
            if len(AUTHORIZATION_SHA256) == 64 and not IMPLEMENTATION_REVIEWED
            else "ENABLED" if IMPLEMENTATION_REVIEWED
            else "BLOCKED_PENDING_COMMITTED_SHA"
        ),
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


def run(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(args, check=True, text=True, **kwargs)


def tree_hashes(root: Path, prefix: str) -> dict[str, str]:
    return {
        f"{prefix}/{path.relative_to(root)}": sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not any(part.startswith(".matched_start_") for part in path.parts)
    }


def parse_worker_manifest(path: Path, name: str) -> dict[str, str]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        candidate = Path(relative)
        if (
            separator != "  " or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or candidate.is_absolute() or ".." in candidate.parts
            or not candidate.parts or candidate.parts[0] != name or relative in rows
        ):
            raise ValueError(f"invalid worker manifest row for {name}")
        rows[relative] = digest
    if not rows:
        raise ValueError(f"empty worker manifest for {name}")
    return rows


def sync_worker_artifact(path: Path, expected_sha: str) -> None:
    relative = path.relative_to(REPO)
    remote = f"{WORKER_REPO}/{relative}"
    command = (
        f"set -eu; mkdir -p {shlex.quote(str(Path(remote).parent))}; "
        f"if test -e {shlex.quote(remote)}; then "
        f"test \"$(sha256sum {shlex.quote(remote)} | awk '{{print $1}}')\" = {shlex.quote(expected_sha)}; "
        "else "
        f"tmp={shlex.quote(remote)}.stage.$$; cat >\"$tmp\"; "
        f"test \"$(sha256sum \"$tmp\" | awk '{{print $1}}')\" = {shlex.quote(expected_sha)}; "
        f"ln \"$tmp\" {shlex.quote(remote)}; rm \"$tmp\"; "
        f"test \"$(sha256sum {shlex.quote(remote)} | awk '{{print $1}}')\" = {shlex.quote(expected_sha)}; fi"
    )
    run(["ssh", "-o", "BatchMode=yes", WORKER, command], input=path.read_text(encoding="utf-8"))


def sync_worker_baseline_forces(authorization: dict | None = None) -> None:
    if authorization is None:
        if sha256(AUTHORIZATION) != AUTHORIZATION_SHA256:
            raise ValueError("extension authorization SHA-256 differs")
        authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    expected = authorization.get("baseline_force_source_sha256")
    if not isinstance(expected, dict) or set(expected) != {"forceFront", "forceRear"}:
        raise ValueError("authorization lacks exact baseline force hashes")
    source = CASES / "tandem_backward_dt005"
    for force_name in ("forceFront", "forceRear"):
        path = source / f"postProcessing/{force_name}/0/coefficient.dat"
        if sha256(path) != expected[force_name]:
            raise ValueError(f"Spark baseline force SHA differs: {force_name}")
        sync_worker_artifact(path, expected[force_name])


def sync_worker_runner() -> None:
    local = REPO / RUNNER
    expected_sha = sha256(local)
    target = f"{WORKER_REPO}/{RUNNER}"
    update_root = f"{WORKER_REPO}/artifacts/matched_start_full40_extension/worker_script_updates"
    code = r'''
import hashlib, json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path
target=Path(sys.argv[1]); expected=sys.argv[2]; update_root=Path(sys.argv[3])
payload=sys.stdin.buffer.read()
actual=hashlib.sha256(payload).hexdigest()
if actual != expected: raise SystemExit('incoming runner SHA differs')
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
if target.is_file() and digest(target) == expected:
    target.chmod(0o755); raise SystemExit(0)
active=[]
for cmdline in Path('/proc').glob('[0-9]*/cmdline'):
    if cmdline.parent.name == str(os.getpid()): continue
    try: command=cmdline.read_bytes().replace(b'\0',b' ').decode(errors='replace')
    except (FileNotFoundError,PermissionError): continue
    if ('run_matched_start_full40_case.sh' in command and 'python3 -c' not in command) or (
        'pimpleFoam' in command and 'matched_start_acquisition_' in command
    ): active.append(command)
if active: raise SystemExit('refusing runner update while matched-start execution is active')
target.parent.mkdir(parents=True,exist_ok=True); update_root.mkdir(parents=True,exist_ok=True)
old_sha=digest(target) if target.is_file() else None
if old_sha:
    backup=update_root/f'{target.name}.{old_sha}.backup'
    if backup.exists() and digest(backup) != old_sha: raise SystemExit('runner backup differs')
    if not backup.exists(): os.link(target,backup)
fd,tmp_name=tempfile.mkstemp(prefix=f'.{target.name}.',dir=target.parent)
with os.fdopen(fd,'wb') as stream:
    stream.write(payload); stream.flush(); os.fsync(stream.fileno())
tmp=Path(tmp_name); tmp.chmod(0o755)
os.replace(tmp,target)
record=update_root/f'{expected}.json'
entry={'status':'WORKER_RUNNER_SHA_UPDATED','old_sha256':old_sha,'new_sha256':expected,
       'target':str(target),'updated_utc':datetime.now(timezone.utc).isoformat()}
text=json.dumps(entry,indent=2)+'\n'
if record.exists() and record.read_text() != text:
    # Timestamp differs on a repeated update; existing same-target record is sufficient.
    prior=json.loads(record.read_text())
    if prior.get('new_sha256') != expected or prior.get('old_sha256') != old_sha:
        raise SystemExit('runner update record differs')
elif not record.exists():
    fd,record_tmp=tempfile.mkstemp(prefix=f'.{record.name}.',dir=record.parent)
    with os.fdopen(fd,'w') as stream:
        stream.write(text); stream.flush(); os.fsync(stream.fileno())
    os.link(record_tmp,record); os.unlink(record_tmp)
'''
    command = shlex.join(["python3", "-c", code, target, expected_sha, update_root])
    run(["ssh", "-o", "BatchMode=yes", WORKER, command], input=local.read_text(encoding="utf-8"))


def verify_worker_solver_dependencies() -> None:
    checks = " && ".join(
        f"test \"$(sha256sum {shlex.quote(WORKER_REPO + '/' + relative)} | awk '{{print $1}}')\" = {sha}"
        for relative, sha in WORKER_DEPENDENCIES.items()
    )
    command = f"set -eu; {checks}; docker image inspect {shlex.quote(OPENFOAM_IMAGE)} >/dev/null"
    run(["ssh", "-o", "BatchMode=yes", WORKER, command])


def generate_case(name: str) -> None:
    if (CASES / name).exists():
        raise FileExistsError(f"refusing existing Spark case: {name}")
    run([
        "python3", str(GENERATOR), "--case", name,
        "--approval-token", GENERATION_TOKEN,
    ], cwd=REPO)


def stage_case(name: str) -> None:
    case = CASES / name
    if not (case / "case_config.json").is_file():
        raise ValueError(f"Spark case is not generated: {name}")
    sync_worker_runner()
    verify_worker_solver_dependencies()
    for path, digest in (
        (REPO / PHASE_MANIFEST, PHASE_MANIFEST_SHA256),
        (PREDECLARATION, PREDECLARATION_SHA256),
        (AUTHORIZATION, AUTHORIZATION_SHA256),
    ):
        sync_worker_artifact(path, digest)
    sync_worker_baseline_forces()
    remote_cases = f"{WORKER_REPO}/cfd/tandem_cylinders/cases"
    remote_stage_command = (
        f"set -eu; mkdir -p {shlex.quote(remote_cases)}; "
        f"test ! -e {shlex.quote(remote_cases + '/' + name)}; "
        f"mktemp -d {shlex.quote(remote_cases + '/.' + name + '.stage.XXXXXX')}"
    )
    result = run(
        ["ssh", "-o", "BatchMode=yes", WORKER, remote_stage_command],
        capture_output=True,
    )
    staging = result.stdout.strip()
    run(["rsync", "-a", f"{case}/", f"{WORKER}:{staging}/"])
    hashes = tree_hashes(case, name)
    verification = r'''
import hashlib, json, os, sys
from pathlib import Path
stage=Path(sys.argv[1]); target=Path(sys.argv[2]); expected=json.load(sys.stdin)
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1<<20), b''): h.update(block)
    return h.hexdigest()
actual={f"%s/{p.relative_to(stage)}" % target.name:digest(p) for p in sorted(stage.rglob('*')) if p.is_file()}
if actual != expected: raise SystemExit('worker staging hashes differ')
if target.exists(): raise SystemExit('worker target appeared during staging')
os.rename(stage,target)
'''
    target = f"{remote_cases}/{name}"
    command = shlex.join(["python3", "-c", verification, staging, target])
    run(["ssh", "-o", "BatchMode=yes", WORKER, command], input=json.dumps(hashes))
    runner = f"{WORKER_REPO}/{RUNNER}"
    args = [
        runner, name, f"{WORKER_REPO}/{PHASE_MANIFEST}",
        f"{WORKER_REPO}/{PREDECLARATION.relative_to(REPO)}",
        f"{WORKER_REPO}/{AUTHORIZATION.relative_to(REPO)}", "--preflight-only",
    ]
    run(["ssh", "-o", "BatchMode=yes", WORKER, shlex.join(args)])


def start_case(name: str) -> None:
    launch_root = f"{WORKER_REPO}/artifacts/matched_start_full40_extension/runner_logs"
    args = [
        f"{WORKER_REPO}/{RUNNER}", name, f"{WORKER_REPO}/{PHASE_MANIFEST}",
        f"{WORKER_REPO}/{PREDECLARATION.relative_to(REPO)}",
        f"{WORKER_REPO}/{AUTHORIZATION.relative_to(REPO)}",
    ]
    command = (
        f"set -eu; mkdir -p {shlex.quote(launch_root)}; "
        f"test ! -e {shlex.quote(launch_root + '/' + name + '.log')}; "
        f"nohup {shlex.join(args)} >{shlex.quote(launch_root + '/' + name + '.log')} "
        "2>&1 < /dev/null & echo $!"
    )
    result = run(["ssh", "-o", "BatchMode=yes", WORKER, command], capture_output=True)
    if not result.stdout.strip().isdigit():
        raise ValueError(f"worker did not return runner PID: {name}")


def load_auditor():
    spec = importlib.util.spec_from_file_location("full40_raw_auditor", AUDITOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def transfer_case(name: str) -> None:
    WORKER_MANIFESTS.mkdir(parents=True, exist_ok=True)
    manifest = WORKER_MANIFESTS / f"{name}.sha256"
    with tempfile.NamedTemporaryFile(dir=WORKER_MANIFESTS, prefix=f".{name}.", delete=False) as stream:
        temporary_manifest = Path(stream.name)
    run(["scp", "-q", f"{WORKER}:{WORKER_REPO}/artifacts/matched_start_full40_extension/worker_transfer_manifests/{name}.sha256", str(temporary_manifest)])
    if manifest.exists():
        if manifest.read_bytes() != temporary_manifest.read_bytes():
            raise ValueError(f"existing worker manifest differs on resume: {name}")
        temporary_manifest.unlink()
    else:
        os.link(temporary_manifest, manifest)
        temporary_manifest.unlink()
    worker_hashes = parse_worker_manifest(manifest, name)
    skeleton = tree_hashes(CASES / name, name)
    mismatched = [path for path, digest in skeleton.items() if worker_hashes.get(path) != digest]
    if mismatched:
        raise ValueError(f"worker would overwrite Spark skeleton: {mismatched[:5]}")
    staging = TRANSFER_STAGING / name
    if not staging.exists():
        staging.mkdir(parents=True)
        run(["rsync", "-a", f"{WORKER}:{WORKER_REPO}/cfd/tandem_cylinders/cases/{name}/", f"{staging}/"])
    if tree_hashes(staging, name) != worker_hashes:
        raise ValueError(f"returned case staging is absent, partial, or differs: {name}")
    target = CASES / name
    for path in sorted(staging.rglob("*")):
        if not path.is_file() or any(
            part.startswith(".matched_start_") for part in path.parts
        ):
            continue
        destination = target / path.relative_to(staging)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if sha256(destination) != sha256(path):
                raise ValueError(f"refusing Spark overwrite: {destination}")
        else:
            os.link(path, destination)
    auditor = load_auditor()
    result = auditor.audit_case(name, manifest, AUTHORIZATION_SHA256)
    audit_path = CASE_AUDITS / name / "result.json"
    if audit_path.exists():
        if json.loads(audit_path.read_text(encoding="utf-8")) != result:
            raise ValueError(f"existing case audit differs on resume: {name}")
    else:
        auditor.write_exclusive(audit_path, result)
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    receipt = RECEIPTS / f"{name}.json"
    if receipt.exists():
        raise FileExistsError(f"refusing existing receipt: {name}")
    os.link(audit_path, receipt)


def finalize_aggregate() -> None:
    auditor = load_auditor()
    output = ROOT / "aggregate_qc/result.json"
    auditor.write_exclusive(output, auditor.aggregate(RECEIPTS))


def aggregate_complete() -> bool:
    output = ROOT / "aggregate_qc/result.json"
    if not output.exists():
        return False
    data = json.loads(output.read_text(encoding="utf-8"))
    required = {
        "status": "MATCHED_START_FULL40_EXTENSION_31_CASE_RAW_QC_PASS",
        "case_count": 31,
        "split_counts": {"train": 11, "validation": 10, "frozen_test": 10},
        "full40_predeclaration_sha256": PREDECLARATION_SHA256,
        "full40_extension_authorization_sha256": AUTHORIZATION_SHA256,
    }
    if any(data.get(key) != value for key, value in required.items()):
        raise ValueError("existing full40 aggregate QC differs")
    if len(valid_receipts()) != len(QUEUE):
        raise ValueError("aggregate QC exists without 31 strict receipts")
    return True


def execute_actions(actions: dict, receipts: set[str]) -> None:
    if actions["stop"]:
        raise RuntimeError(f"full40 fail-stop: {actions['failures']}")
    for name in actions["transfer"]:
        transfer_case(name)
    if actions["transfer"]:
        return
    for name in actions["prepare"]:
        if not (CASES / name / "case_config.json").is_file():
            generate_case(name)
        stage_case(name)
    for name in actions["start"]:
        start_case(name)
    if len(receipts) == len(QUEUE):
        finalize_aggregate()


def write_scheduler_state(payload: dict) -> None:
    path = ROOT / "scheduler_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(payload, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def iteration() -> bool:
    validate_execution_authorization()
    if aggregate_complete():
        return True
    spark_free = shutil.disk_usage(REPO).free
    snapshot = enrich_snapshot(worker_snapshot())
    receipts = valid_receipts()
    actions = plan(snapshot, receipts)
    gates = resource_gate(
        snapshot, spark_free, starting=bool(actions["start"] or actions["prepare"])
    )
    write_scheduler_state({
        "status": "FULL40_EXTENSION_WATCH_ACTIVE",
        "receipt_count": len(receipts),
        "snapshot": snapshot,
        "plan": actions,
        "resource_failures": gates,
    })
    if gates and (actions["start"] or actions["prepare"]):
        raise RuntimeError(f"resource guards block new work: {gates}")
    if actions["transfer"] and "spark_free_disk" in gates:
        raise RuntimeError("Spark disk guard blocks worker return transfer")
    execute_actions(actions, receipts)
    return aggregate_complete()


def run_iterations(watch: bool, interval: float) -> bool:
    while True:
        if iteration():
            return True
        if not watch:
            return False
        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run-output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approval-token")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=15.0)
    args = parser.parse_args()
    if not args.execute:
        report = dry_run_report()
        if args.dry_run_output:
            write_exclusive(args.dry_run_output, report)
        print(json.dumps(report, indent=2))
        return
    if args.approval_token != EXECUTION_TOKEN:
        parser.error("execution requires the committed-extension approval token")
    if args.interval <= 0:
        parser.error("--interval must be positive")
    validate_execution_authorization()
    if not IMPLEMENTATION_REVIEWED:
        raise SystemExit("execution implementation is complete but blocked pending second review")
    lock_path = ROOT / "scheduler.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit("another full40 extension scheduler holds the lock") from error
        done = run_iterations(args.watch, args.interval)
        status = "FULL40_EXTENSION_COMPLETE" if done else "ONE_SAFE_STEP_COMPLETE"
        print(json.dumps({"status": status}, indent=2))


if __name__ == "__main__":
    main()
