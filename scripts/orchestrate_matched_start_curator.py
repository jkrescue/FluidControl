#!/usr/bin/env python3
"""Preflight and optionally run isolated Curator jobs for nine commissioning cases."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import subprocess
from itertools import pairwise
from pathlib import Path

PROFILE = "matched_start_commissioning_train9_v1"
CASE_RE = re.compile(
    r"^matched_start_acquisition_train_b(00|02|04)_(m075|zero|p075)$"
)
STATE_KEYS = {"U", "U_0", "p", "phi", "phi_0"}
PHASE_MANIFEST = Path(
    "artifacts/tandem_cylinders/matched_start_phase_restart_predeclared_v3_20261003.json"
)
STAGING_ROOT = Path("data/curated/.staging") / PROFILE
CURATOR = Path("scripts/curate_low_action_phase94_validation.py")
VTK_EXPORT = Path("scripts/export_tandem_vtk.sh")
VTK_VERIFIED_ROOT = Path("artifacts/matched_start_acquisition/vtk_ready")
VTK_MANIFEST_ROOT = Path("artifacts/matched_start_acquisition/vtk_manifests")
OPENFOAM_IMAGE = "opencfd/openfoam-default@sha256:33fb575aa9980d2bc42fd58c75ae698c489293ba30c991380fe3f899c622f319"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_cases() -> list[str]:
    return [
        f"matched_start_acquisition_train_b{phase}_{branch}"
        for phase in ("00", "02", "04")
        for branch in ("m075", "zero", "p075")
    ]


def parse_sha256_manifest(path: Path, case: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise ValueError(f"{path}:{line_number}: invalid sha256 manifest row")
        digest, relative = match.groups()
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"{path}:{line_number}: unsafe relative path")
        if not candidate.parts or candidate.parts[0] != case:
            raise ValueError(f"{path}:{line_number}: path is outside case prefix")
        if relative in rows:
            raise ValueError(f"{path}:{line_number}: duplicate path")
        rows[relative] = digest
    if not rows:
        raise ValueError(f"empty worker raw manifest: {path}")
    return rows


def parse_rear_omega_table(path: Path) -> list[list[float]]:
    text = path.read_text(encoding="utf-8")
    patch = re.search(r"rearCylinder\s*\{(?P<body>.*?)\n\s*\}", text, re.DOTALL)
    if not patch:
        raise ValueError(f"rearCylinder patch missing: {path}")
    table = re.search(
        r"omega\s+table\s*\((?P<rows>.*?)\)\s*;", patch.group("body"), re.DOTALL
    )
    if not table:
        raise ValueError(f"rearCylinder omega table missing: {path}")
    rows = [
        [float(left), float(right)]
        for left, right in re.findall(
            r"\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)", table.group("rows")
        )
    ]
    if not rows:
        raise ValueError(f"empty rearCylinder omega table: {path}")
    return rows


def validate_action_contract(config: dict, velocity_path: Path, case: str) -> None:
    points = [[float(value) for value in row] for row in config["action_points"]]
    if any(len(row) != 2 or not all(math.isfinite(value) for value in row) for row in points):
        raise ValueError(f"{case}: invalid action_points")
    if any(right[0] <= left[0] for left, right in pairwise(points)):
        raise ValueError(f"{case}: action times are not strictly increasing")
    table = parse_rear_omega_table(velocity_path)
    if len(table) != len(points) or any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-10)
        for actual_row, expected_row in zip(table, points, strict=True)
        for actual, expected in zip(actual_row, expected_row, strict=True)
    ):
        raise ValueError(f"{case}: OpenFOAM U omega table differs from action_points")


def validate_generator_scalar_contract(
    config: dict, case: str, expected_phase: str
) -> None:
    """Validate scalar fields exactly as emitted by the acquisition generator."""
    try:
        phase_bin = int(config["phase_bin"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{case}: phase_bin must be an integer") from error
    if phase_bin != int(expected_phase):
        raise ValueError(f"{case}: phase_bin mismatch")
    if int(config.get("expected_field_frames", -1)) != 801:
        raise ValueError(f"{case}: expected_field_frames mismatch")
    if int(config.get("expected_aligned_force_samples_with_source_t0", -1)) != 16001:
        raise ValueError(
            f"{case}: expected_aligned_force_samples_with_source_t0 mismatch"
        )


def validate_case(repo: Path, entry: dict, phase_sha: str) -> dict:
    case = entry["case"]
    match = CASE_RE.fullmatch(case)
    if not match:
        raise ValueError(f"unexpected commissioning case: {case}")
    expected_receipt = (
        Path("artifacts/matched_start_acquisition/transfer_verified")
        / f"{case}.json"
    )
    if entry.get("receipt", str(expected_receipt)) != str(expected_receipt):
        raise ValueError(f"{case}: contract receipt path mismatch")
    receipt_path = repo / expected_receipt
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    required_receipt = {
        "status": "RAW_TRANSFER_VERIFIED",
        "case": case,
        "phase_manifest_sha256": phase_sha,
        "coverage": "raw OpenFOAM solver files only; VTK is not included",
        "next_required_stage": "SPARK_PINNED_FOAMTOVTK_801_FRAMES",
        "curator_guard": "do not curate until a separate VTK_READY receipt exists",
    }
    for key, expected in required_receipt.items():
        if receipt.get(key) != expected:
            raise ValueError(f"{case}: receipt {key} mismatch")
    worker_manifest_rel = Path(receipt["worker_raw_manifest"])
    expected_manifest = (
        Path("artifacts/matched_start_acquisition/worker_transfer_manifests")
        / f"{case}.sha256"
    )
    if worker_manifest_rel != expected_manifest:
        raise ValueError(f"{case}: worker_raw_manifest path mismatch")
    worker_manifest = repo / worker_manifest_rel
    if sha256(worker_manifest) != receipt["worker_raw_manifest_sha256"]:
        raise ValueError(f"{case}: worker raw manifest SHA mismatch")
    raw_rows = parse_sha256_manifest(worker_manifest, case)
    if int(receipt["raw_file_count"]) != len(raw_rows):
        raise ValueError(f"{case}: raw_file_count mismatch")
    cases_root = repo / "cfd/tandem_cylinders/cases"
    for relative, expected in raw_rows.items():
        actual_path = cases_root / relative
        if not actual_path.is_file() or sha256(actual_path) != expected:
            raise ValueError(f"{case}: raw SHA mismatch: {relative}")
    case_root = cases_root / case
    config = json.loads((case_root / "case_config.json").read_text(encoding="utf-8"))
    phase, branch = match.groups()
    target_by_branch = {"m075": -0.75, "zero": 0.0, "p075": 0.75}
    if config.get("split") != "train" or config.get("case") != case:
        raise ValueError(f"{case}: case/split contract mismatch")
    validate_generator_scalar_contract(config, case, phase)
    if not math.isclose(float(config.get("action_target")), target_by_branch[branch]):
        raise ValueError(f"{case}: action_target mismatch")
    if config.get("phase_manifest_sha256") != phase_sha:
        raise ValueError(f"{case}: phase manifest SHA mismatch")
    source_state = config.get("source_state_sha256")
    if not isinstance(source_state, dict) or set(source_state) != STATE_KEYS:
        raise ValueError(f"{case}: source_state_sha256 must contain exact five-field keys")
    if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in source_state.values()):
        raise ValueError(f"{case}: invalid source-state SHA")
    if config.get("source_restart_case") != "tandem_backward_dt005":
        raise ValueError(f"{case}: source_restart_case mismatch")
    if config.get("source_state_provenance_dir") != "source_restart_provenance":
        raise ValueError(f"{case}: source_state_provenance_dir mismatch")
    provenance = case_root / "source_restart_provenance"
    for field, expected in source_state.items():
        path = provenance / field
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"{case}: source-state provenance mismatch: {field}")
        relative = f"{case}/source_restart_provenance/{field}"
        if raw_rows.get(relative) != expected:
            raise ValueError(f"{case}: source-state provenance absent from raw manifest")
    start = float(config["start_time"])
    end = float(config["end_time"])
    if not end > start or config.get("analysis_window") is None:
        raise ValueError(f"{case}: invalid time/analysis contract")
    validate_action_contract(config, case_root / f"{start:g}" / "U", case)
    return {
        "case": case,
        "phase_bin": f"b{phase}",
        "branch": branch,
        "receipt": str(expected_receipt),
        "receipt_sha256": sha256(receipt_path),
        "worker_raw_manifest": str(worker_manifest_rel),
        "worker_raw_manifest_sha256": sha256(worker_manifest),
        "raw_file_count": len(raw_rows),
        "source_state_sha256": source_state,
    }


def preflight(repo: Path, contract: dict) -> dict:
    if contract.get("schema_version") != 1 or contract.get("profile") != PROFILE:
        raise ValueError("wrong contract schema/profile")
    if contract.get("status") != "BOUND_APPROVED_FOR_CURATION":
        raise ValueError("contract is not bound and approved for curation")
    if Path(contract.get("phase_manifest", "")) != PHASE_MANIFEST:
        raise ValueError("wrong authoritative phase manifest path")
    phase_path = repo / PHASE_MANIFEST
    phase_sha = sha256(phase_path)
    if contract.get("phase_manifest_sha256") != phase_sha:
        raise ValueError("contract phase manifest SHA mismatch")
    entries = contract.get("cases")
    if not isinstance(entries, list) or sorted(
        row.get("case", "") for row in entries
    ) != sorted(expected_cases()):
        raise ValueError("contract must bind exactly the canonical nine cases")
    cases = [validate_case(repo, entry, phase_sha) for entry in entries]
    for phase in ("b00", "b02", "b04"):
        group = [row for row in cases if row["phase_bin"] == phase]
        if {row["branch"] for row in group} != {"m075", "zero", "p075"}:
            raise ValueError(f"{phase}: incomplete three-branch panel")
        if len({json.dumps(row["source_state_sha256"], sort_keys=True) for row in group}) != 1:
            raise ValueError(f"{phase}: five-field restart hashes differ across branches")
    return {
        "status": "CURATOR_PREFLIGHT_OK",
        "profile": PROFILE,
        "phase_manifest": str(PHASE_MANIFEST),
        "phase_manifest_sha256": phase_sha,
        "cases": cases,
        "isolation": "one process and one staging output directory per case",
        "training_use": "FORBIDDEN_COMMISSIONING_ONLY",
        "validation_or_frozen_access": "FORBIDDEN",
    }


def write_vtk_verification(repo: Path, case: str, receipt_sha256: str) -> dict:
    vtk_root = repo / "cfd/tandem_cylinders/cases" / case / "VTK_curator"
    files = sorted(vtk_root.glob("*/internal.vtu"))
    if len(files) != 801 or any(not path.is_file() or path.stat().st_size == 0 for path in files):
        raise ValueError(f"{case}: expected 801 nonempty internal.vtu files")
    manifest = repo / VTK_MANIFEST_ROOT / f"{case}.sha256"
    marker = repo / VTK_VERIFIED_ROOT / f"{case}.json"
    if manifest.exists() or marker.exists() or marker.with_suffix(".json.tmp").exists():
        raise FileExistsError(f"{case}: refusing existing VTK verification output")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    marker.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(repo / 'cfd/tandem_cylinders/cases')}\n"
            for path in files
        ),
        encoding="utf-8",
    )
    result = {
        "status": "VTK_READY",
        "case": case,
        "frames": 801,
        "vtk_manifest": str(manifest.relative_to(repo)),
        "vtk_manifest_sha256": sha256(manifest),
        "raw_transfer_receipt_sha256": receipt_sha256,
        "export_command": f"bash {VTK_EXPORT} {case}",
        "openfoam_image": OPENFOAM_IMAGE,
        "overwrite_policy": "REFUSE",
        "curator_output_contract": f"{case}.h5.tmp -> {case}.h5; refuse overwrite; require VTK_READY",
    }
    temporary = marker.with_suffix(".json.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, marker)
    temporary.unlink()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--plan-output", type=Path)
    parser.add_argument("--max-parallel", type=int, choices=(2, 3), default=2)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    plan = preflight(repo, contract)
    commands = {}
    curator_python = repo / ".venv-curator-py312/bin/python"
    for row in plan["cases"]:
        case = row["case"]
        output = repo / STAGING_ROOT / case
        commands[case] = {
            "vtk_export": ["bash", str(repo / VTK_EXPORT), case],
            "curator": [
                str(curator_python), "-u", str(repo / CURATOR),
                "--profile", PROFILE, "--output", str(output),
                "--nx", "256", "--ny", "128", "--cases", case,
                "--defer-finalize", "--atomic-hdf5",
            ],
        }
    plan["commands"] = commands
    plan["max_parallel"] = args.max_parallel
    if args.plan_output:
        if args.plan_output.exists():
            raise FileExistsError(args.plan_output)
        args.plan_output.parent.mkdir(parents=True, exist_ok=True)
        args.plan_output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    if not curator_python.is_file():
        raise FileNotFoundError(curator_python)
    log_root = repo / "artifacts/matched_start_acquisition/curator_logs"
    vtk_log_root = repo / "artifacts/matched_start_acquisition/vtk_export_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    vtk_log_root.mkdir(parents=True, exist_ok=True)
    for case in commands:
        output = repo / STAGING_ROOT / case
        log = log_root / f"{case}.log"
        vtk_log = vtk_log_root / f"{case}.log"
        vtk_marker = repo / VTK_VERIFIED_ROOT / f"{case}.json"
        vtk_manifest = repo / VTK_MANIFEST_ROOT / f"{case}.sha256"
        raw_case = repo / "cfd/tandem_cylinders/cases" / case
        if (
            output.exists() or log.exists() or vtk_log.exists()
            or vtk_marker.exists() or vtk_marker.with_suffix(".json.tmp").exists()
            or vtk_manifest.exists() or (raw_case / "VTK_curator").exists()
            or (raw_case / "log.foamToVTK_curator").exists()
        ):
            raise FileExistsError(f"refusing existing staging/log for {case}")

    def run(case: str) -> dict:
        row = next(item for item in plan["cases"] if item["case"] == case)
        vtk_log = vtk_log_root / f"{case}.log"
        with vtk_log.open("xb") as stream:
            exported = subprocess.run(
                commands[case]["vtk_export"], cwd=repo, stdout=stream,
                stderr=subprocess.STDOUT, check=False,
            )
        if exported.returncode:
            raise RuntimeError(f"{case}: pinned foamToVTK export failed")
        vtk = write_vtk_verification(repo, case, row["receipt_sha256"])
        log = log_root / f"{case}.log"
        with log.open("xb") as stream:
            completed = subprocess.run(
                commands[case]["curator"], cwd=repo, stdout=stream, stderr=subprocess.STDOUT,
                check=False,
            )
        target = repo / STAGING_ROOT / case / "train" / f"{case}.h5"
        temporary = target.with_suffix(".h5.tmp")
        if completed.returncode or not target.is_file() or temporary.exists():
            raise RuntimeError(f"{case}: Curator failed or atomic output incomplete")
        return {
            "case": case, "exit_code": 0,
            "vtk_verification": vtk,
            "hdf5": str(target.relative_to(repo)),
            "hdf5_sha256": sha256(target),
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        results = list(pool.map(run, commands))
    print(json.dumps({"status": "NINE_CASE_STAGING_COMPLETE", "results": results}, indent=2))


if __name__ == "__main__":
    main()
