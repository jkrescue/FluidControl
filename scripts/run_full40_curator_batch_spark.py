#!/usr/bin/env python3
"""Reviewed, fail-closed Spark scheduler for full40 per-case curation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "artifacts/matched_start_full40_extension/spark_curator_batch"
RECEIPTS = REPO / "artifacts/matched_start_full40_extension/transfer_verified"
STAGING = REPO / "data/curated/.staging/matched_start_full40_v1"
NINE_FINAL = REPO / "data/curated/tandem_cylinders_matched_start_commissioning_train9_v1"
NINE_STAGING = REPO / "data/curated/.staging/matched_start_commissioning_train9_v1"
PILOT = "matched_start_acquisition_train_b00_m0375"
PILOT_SHA256 = "85e5b5a60c01c8f3521d580afc68f81f501e4bc94e092a8e873a582e11fd0d82"
EXECUTION_REVIEWED = True
EXECUTION_TOKEN = "EXECUTE_REVIEWED_FULL40_SPARK_TRAIN_BATCH"
MAX_PARALLEL = 4
MIN_MEMORY_GIB = 40
MIN_DISK_GIB = 250


def load(relative: str, module_name: str):
    path = REPO / relative
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(payload, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def available_memory_gib() -> float:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / 1024**2
    raise RuntimeError("MemAvailable is absent")


def available_disk_gib(path: Path) -> float:
    return shutil.disk_usage(path).free / 1024**3


def selected_train_cases(matrix: dict[str, dict]) -> list[str]:
    return sorted(
        name
        for name, row in matrix.items()
        if row.get("disposition") == "planned_new_remainder_case"
        and row.get("split") == "train"
    )


def hdf_path(name: str) -> Path:
    return STAGING / name / "train" / f"{name}.h5"


def preflight() -> tuple[object, object, dict[str, dict], list[str]]:
    curator = load("scripts/curate_matched_start_full40_remainder.py", "batch_curator")
    finalizer = load("scripts/finalize_matched_start_full40.py", "batch_finalizer")
    matrix = curator.load_matrix(REPO)
    cases = selected_train_cases(matrix)
    if len(cases) != 11:
        raise ValueError("full40 train remainder must contain exactly 11 cases")
    pilot = hdf_path(PILOT)
    if not pilot.is_file() or finalizer.sha256(pilot) != PILOT_SHA256:
        raise ValueError("reviewed Spark pilot HDF5 is absent or differs")
    nine_final_ready = (NINE_FINAL / "commissioning_qc.json").is_file()
    nine_staged = list(NINE_STAGING.glob("*/train/*.h5"))
    nine_temporary = list(NINE_STAGING.glob("*/*/*.h5.tmp"))
    if not nine_final_ready and (len(nine_staged) != 9 or nine_temporary):
        raise FileNotFoundError("nine-case Curator outputs are not atomically complete")
    return curator, finalizer, matrix, cases


def case_command(name: str) -> list[str]:
    prepare = REPO / "scripts/prepare_matched_start_full40_vtk.py"
    curate = REPO / "scripts/curate_matched_start_full40_remainder.py"
    marker = (
        REPO / "artifacts/matched_start_full40_extension/vtk_ready" / f"{name}.json"
    )
    commands = []
    if not marker.is_file():
        commands.append(
            f"/home/USER/env_isaaclab/bin/python -u {prepare} --case {name} "
            "--execute --approval-token EXECUTE_REVIEWED_FULL40_VTK_EXPORT"
        )
    commands.append(
        f"{REPO}/.venv-curator-py312/bin/python -u {curate} --case {name} "
        "--execute --approval-token EXECUTE_REVIEWED_FULL40_CURATOR_REMAINDER"
    )
    return ["/bin/bash", "-lc", " && ".join(commands)]


def audit_case(curator, finalizer, matrix: dict[str, dict], name: str) -> dict:
    raw_sha = curator.validate_raw_receipt_files(REPO, name, matrix[name])
    vtk = curator.validate_vtk_receipt(REPO, name, raw_sha, "train")
    force_validator = finalizer.load_existing_finalizer(REPO)
    result = finalizer.validate_remainder_hdf(
        REPO, name, matrix[name], hdf_path(name), force_validator
    )
    result["vtk_manifest_sha256"] = vtk["vtk_manifest_sha256"]
    return result


def run() -> None:
    curator, finalizer, matrix, cases = preflight()
    lock = ROOT / ".scheduler.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise RuntimeError("Spark full40 curator scheduler lock exists") from error
    atomic_json(
        lock / "OWNER.json",
        {
            "status": "SPARK_FULL40_CURATOR_SCHEDULER_ACQUIRED",
            "pid": os.getpid(),
            "max_parallel": MAX_PARALLEL,
        },
    )
    pending = [name for name in cases if name != PILOT and not hdf_path(name).is_file()]
    completed = {PILOT: audit_case(curator, finalizer, matrix, PILOT)}
    running: dict[str, tuple[subprocess.Popen, object]] = {}
    failure = None
    while pending or running:
        while pending and len(running) < MAX_PARALLEL and failure is None:
            if available_memory_gib() < MIN_MEMORY_GIB:
                break
            if available_disk_gib(REPO) < MIN_DISK_GIB:
                raise RuntimeError("Spark free-disk guard failed")
            name = pending[0]
            receipt = RECEIPTS / f"{name}.json"
            if not receipt.is_file():
                break
            curator.validate_raw_receipt_files(REPO, name, matrix[name])
            log_path = ROOT / "logs" / f"{name}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("x", encoding="utf-8")
            process = subprocess.Popen(
                case_command(name), cwd=REPO, stdout=log, stderr=subprocess.STDOUT
            )
            running[name] = (process, log)
            pending.pop(0)
        for name, (process, log) in list(running.items()):
            status = process.poll()
            if status is None:
                continue
            log.close()
            del running[name]
            if status != 0:
                failure = {"case": name, "returncode": status}
                continue
            completed[name] = audit_case(curator, finalizer, matrix, name)
        atomic_json(
            ROOT / "state.json",
            {
                "status": "FAILED" if failure else "RUNNING",
                "completed": sorted(completed),
                "running": {name: process.pid for name, (process, _) in running.items()},
                "pending": pending,
                "failure": failure,
                "memory_available_gib": available_memory_gib(),
                "disk_available_gib": available_disk_gib(REPO),
            },
        )
        if failure and not running:
            raise RuntimeError(f"Spark curation batch failed: {failure}")
        if pending or running:
            time.sleep(10)
    atomic_json(
        ROOT / "result.json",
        {
            "status": "FULL40_SPARK_TRAIN11_CURATOR_QC_PASS",
            "cases": completed,
            "case_count": len(completed),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approval-token")
    args = parser.parse_args()
    curator, _, matrix, cases = preflight()
    report = {
        "status": "FULL40_SPARK_TRAIN_BATCH_READY",
        "cases": cases,
        "strict_receipts_ready": sum(
            (RECEIPTS / f"{name}.json").is_file() for name in cases
        ),
        "execution_reviewed": EXECUTION_REVIEWED,
        "max_parallel": MAX_PARALLEL,
        "memory_guard_gib": MIN_MEMORY_GIB,
        "disk_guard_gib": MIN_DISK_GIB,
    }
    if not args.execute:
        print(json.dumps(report, indent=2))
        return
    if not EXECUTION_REVIEWED or args.approval_token != EXECUTION_TOKEN:
        parser.error("Spark full40 curator batch execution is not reviewed")
    del curator, matrix
    run()


if __name__ == "__main__":
    main()
