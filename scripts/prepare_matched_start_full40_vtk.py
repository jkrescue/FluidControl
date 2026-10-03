#!/usr/bin/env python3
"""Dry-run-first pinned foamToVTK stage for the full40 remainder."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CASES = REPO / "cfd/tandem_cylinders/cases"
VTK_RECEIPTS = REPO / "artifacts/matched_start_full40_extension/vtk_ready"
VTK_MANIFESTS = REPO / "artifacts/matched_start_full40_extension/vtk_manifests"
EXPORT = REPO / "scripts/export_tandem_vtk.sh"
CURATOR_PREFLIGHT = REPO / "scripts/curate_matched_start_full40_remainder.py"
OPENFOAM_IMAGE = (
    "opencfd/openfoam-default@sha256:"
    "33fb575aa9980d2bc42fd58c75ae698c489293ba30c991380fe3f899c622f319"
)
EXECUTION_REVIEWED = True
EXECUTION_TOKEN = "EXECUTE_REVIEWED_FULL40_VTK_EXPORT"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_preflight(repo: Path):
    path = repo / CURATOR_PREFLIGHT.relative_to(REPO)
    spec = importlib.util.spec_from_file_location("full40_vtk_preflight", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plan(repo: Path, name: str) -> dict:
    preflight = load_preflight(repo)
    matrix = preflight.load_matrix(repo)
    row = matrix.get(name)
    if row is None or row.get("disposition") != "planned_new_remainder_case":
        raise ValueError("case is not in the 31-case remainder")
    case = repo / CASES.relative_to(REPO) / name
    config = json.loads((case / "case_config.json").read_text(encoding="utf-8"))
    preflight.validate_case_config(name, config, row)
    preflight.validate_source_force(repo, name, config)
    raw_receipt_sha256 = preflight.validate_raw_receipt_files(repo, name, row)
    outputs = [
        case / "VTK_curator",
        case / "log.foamToVTK_curator",
        case / ".full40_vtk_export.lock",
        repo / VTK_RECEIPTS.relative_to(REPO) / f"{name}.json",
        repo / VTK_MANIFESTS.relative_to(REPO) / f"{name}.sha256",
    ]
    if any(path.exists() for path in outputs):
        raise FileExistsError(f"refusing existing VTK output/receipt: {name}")
    return {
        "status": "FULL40_REMAINDER_CASE_READY_FOR_VTK_EXPORT",
        "case": name,
        "split": row["split"],
        "raw_aggregate_gate": "DEFERRED_TO_FINALIZER_AFTER_ALL_31_CASES",
        "raw_transfer_receipt_sha256": raw_receipt_sha256,
        "command": ["bash", str(repo / EXPORT.relative_to(REPO)), name],
        "expected_frames": 801,
        "openfoam_image": OPENFOAM_IMAGE,
        "lock": str((case / ".full40_vtk_export.lock").relative_to(repo)),
    }


def write_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise ValueError(f"refusing to overwrite {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def verify_and_record(repo: Path, item: dict) -> dict:
    name = item["case"]
    vtk = repo / CASES.relative_to(REPO) / name / "VTK_curator"
    files = sorted(vtk.glob("*/internal.vtu"))
    if len(files) != 801 or any(not path.is_file() or path.stat().st_size == 0 for path in files):
        raise ValueError(f"{name}: expected exactly 801 nonempty VTK frames")
    manifest = repo / VTK_MANIFESTS.relative_to(REPO) / f"{name}.sha256"
    marker = repo / VTK_RECEIPTS.relative_to(REPO) / f"{name}.json"
    rows = "".join(
        f"{sha256(path)}  {path.relative_to(repo / CASES.relative_to(REPO))}\n"
        for path in files
    )
    write_exclusive(manifest, rows)
    result = {
        "status": "VTK_READY",
        "case": name,
        "split": item["split"],
        "frames": 801,
        "vtk_manifest": str(manifest.relative_to(repo)),
        "vtk_manifest_sha256": sha256(manifest),
        "raw_transfer_receipt_sha256": item["raw_transfer_receipt_sha256"],
        "export_command": item["command"],
        "openfoam_image": OPENFOAM_IMAGE,
        "overwrite_policy": "REFUSE",
    }
    write_exclusive(marker, json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--case", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approval-token")
    args = parser.parse_args()
    repo = args.repo.resolve()
    item = plan(repo, args.case)
    if not args.execute:
        print(json.dumps(item, indent=2))
        return
    if not EXECUTION_REVIEWED or args.approval_token != EXECUTION_TOKEN:
        parser.error("full40 VTK execution is not reviewed/enabled")
    lock = repo / item["lock"]
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise RuntimeError(f"VTK export lock already exists: {lock}") from error
    write_exclusive(
        lock / "ACQUIRED.json",
        json.dumps({"status": "VTK_EXPORT_ACQUIRED", "case": item["case"]}) + "\n",
    )
    try:
        subprocess.run(item["command"], cwd=repo, check=True)
        result = verify_and_record(repo, item)
    except Exception as error:
        write_exclusive(
            lock / "FAILED.json",
            json.dumps(
                {"status": "VTK_EXPORT_FAILED_REVIEW_REQUIRED", "error": str(error)}
            )
            + "\n",
        )
        raise
    write_exclusive(
        lock / "COMPLETED.json",
        json.dumps({"status": "VTK_EXPORT_COMPLETED", "case": item["case"]}) + "\n",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
