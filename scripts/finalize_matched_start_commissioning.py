#!/usr/bin/env python3
"""Validate and atomically assemble the nine-case train-only commissioning profile."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import h5py
import numpy as np

PROFILE = "matched_start_commissioning_train9_v1"
STAGING_ROOT = Path("data/curated/.staging") / PROFILE
FINAL_ROOT = Path("data/curated/tandem_cylinders_matched_start_commissioning_train9_v1")
VTK_VERIFIED_ROOT = Path("artifacts/matched_start_acquisition/vtk_ready")


def validate_staging_layout(repo: Path, expected_cases: list[str]) -> None:
    root = repo / STAGING_ROOT
    expected = {
        root / case / "train" / f"{case}.h5" for case in expected_cases
    }
    actual = set(root.glob("*/train/*.h5"))
    temporary = list(root.glob("*/*/*.h5.tmp"))
    forbidden = list(root.glob("*/validation/*.h5")) + list(root.glob("*/test/*.h5"))
    if actual != expected:
        raise ValueError(
            f"incomplete or extra staging HDF5 set: actual={len(actual)}, expected=9"
        )
    if temporary or forbidden:
        raise ValueError("staging contains temporary, validation, or test HDF5 files")


def load_orchestrator(repo: Path):
    path = repo / "scripts/orchestrate_matched_start_curator.py"
    spec = importlib.util.spec_from_file_location("matched_start_orchestrator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_generator_contract(
    config: dict, case: str, preflight_row: dict, orchestrator
) -> None:
    """Use the orchestrator's exact generator schema during finalization too."""
    orchestrator.validate_generator_scalar_contract(
        config, case, preflight_row["phase_bin"].removeprefix("b")
    )


def load_force_files(paths: list[Path]) -> np.ndarray:
    samples: dict[float, np.ndarray] = {}
    if not paths:
        raise FileNotFoundError("no force coefficient files")
    for path in paths:
        header = None
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# Time"):
                header = line[1:].split()
                continue
            if not line or line.startswith("#"):
                continue
            if header is None or any(name not in header for name in ("Time", "Cd", "Cl")):
                raise ValueError(f"invalid coefficient header: {path}")
            fields = line.split()
            row = np.asarray(
                [float(fields[header.index(name)]) for name in ("Time", "Cd", "Cl")],
                dtype=np.float64,
            )
            if not np.isfinite(row).all():
                raise ValueError(f"nonfinite force row: {path}")
            key = round(float(row[0]), 8)
            if key in samples and not np.allclose(samples[key], row, rtol=0.0, atol=1e-10):
                raise ValueError(f"conflicting duplicate force time {key}: {path}")
            samples[key] = row
    return np.asarray([samples[key] for key in sorted(samples)], dtype=np.float64)


def assembled_force(
    cases_root: Path, case: Path, config: dict, object_name: str, start: float, end: float
) -> tuple[np.ndarray, dict]:
    raw_paths = sorted(case.glob(f"postProcessing/{object_name}/*/coefficient.dat"))
    raw = load_force_files(raw_paths)
    if len(raw) != 16000:
        raise ValueError(f"{case.name}:{object_name}: expected 16000 new raw force rows")
    source_name = config.get("source_restart_case")
    if not source_name:
        raise ValueError(f"{case.name}: source_restart_case required for exact t0 force")
    source_paths = sorted(
        (cases_root / source_name).glob(
            f"postProcessing/{object_name}/*/coefficient.dat"
        )
    )
    source = load_force_files(source_paths)
    matches = np.flatnonzero(np.isclose(source[:, 0], start, rtol=0.0, atol=1e-8))
    if len(matches) != 1:
        raise ValueError(f"{case.name}:{object_name}: no unique exact source force at t0")
    assembled = np.concatenate((source[matches], raw), axis=0)
    if len(assembled) != 16001:
        raise ValueError(f"{case.name}:{object_name}: assembled force must have 16001 rows")
    expected = np.linspace(start, end, 16001)
    if not np.allclose(assembled[:, 0], expected, rtol=0.0, atol=2e-6):
        raise ValueError(f"{case.name}:{object_name}: assembled force time grid mismatch")
    return assembled, {
        "raw_rows": 16000,
        "assembled_rows": 16001,
        "source_restart_case": source_name,
        "source_t0_coefficient_paths": [str(path) for path in source_paths],
        "source_t0_row": assembled[0].tolist(),
    }


def validate_vtk(repo: Path, case: str, orchestrator) -> dict:
    marker_path = repo / VTK_VERIFIED_ROOT / f"{case}.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("status") != "VTK_READY" or marker.get("case") != case:
        raise ValueError(f"{case}: invalid VTK marker")
    if marker.get("curator_output_contract") != (
        f"{case}.h5.tmp -> {case}.h5; refuse overwrite; require VTK_READY"
    ):
        raise ValueError(f"{case}: VTK receipt Curator contract mismatch")
    if marker.get("openfoam_image") != orchestrator.OPENFOAM_IMAGE:
        raise ValueError(f"{case}: VTK receipt OpenFOAM image mismatch")
    manifest_path = repo / marker["vtk_manifest"]
    if orchestrator.sha256(manifest_path) != marker["vtk_manifest_sha256"]:
        raise ValueError(f"{case}: VTK manifest SHA mismatch")
    rows = orchestrator.parse_sha256_manifest(manifest_path, case)
    if len(rows) != 801:
        raise ValueError(f"{case}: VTK manifest does not contain 801 files")
    cases_root = repo / "cfd/tandem_cylinders/cases"
    for relative, expected in rows.items():
        path = cases_root / relative
        if not path.is_file() or orchestrator.sha256(path) != expected:
            raise ValueError(f"{case}: VTK file SHA mismatch: {relative}")
    return marker


def validate_case(repo: Path, case: str, preflight_row: dict, orchestrator) -> dict:
    cases_root = repo / "cfd/tandem_cylinders/cases"
    raw_case = cases_root / case
    config = json.loads((raw_case / "case_config.json").read_text(encoding="utf-8"))
    validate_generator_contract(config, case, preflight_row, orchestrator)
    vtk = validate_vtk(repo, case, orchestrator)
    target = repo / STAGING_ROOT / case / "train" / f"{case}.h5"
    if target.with_suffix(".h5.tmp").exists() or not target.is_file():
        raise ValueError(f"{case}: atomic HDF5 output incomplete")
    start = float(config["start_time"])
    end = float(config["end_time"])
    expected_times = np.linspace(start, end, 801)
    action = np.asarray(config["action_points"], dtype=np.float64)
    front, front_audit = assembled_force(cases_root, raw_case, config, "forceFront", start, end)
    rear, rear_audit = assembled_force(cases_root, raw_case, config, "forceRear", start, end)
    max_pressure_mean = 0.0
    min_coverage = 1.0
    max_coverage = 0.0
    with h5py.File(target, "r") as handle:
        required = {"state", "mask", "omega", "force", "time", "x", "y"}
        if not required.issubset(handle.keys()):
            raise ValueError(f"{case}: missing HDF5 datasets")
        expected_shapes = {
            "state": (801, 3, 128, 256), "mask": (801, 1, 128, 256),
            "omega": (801, 1), "force": (801, 4), "time": (801, 1),
        }
        for key, shape in expected_shapes.items():
            if handle[key].shape != shape:
                raise ValueError(f"{case}: wrong {key} shape {handle[key].shape}")
        if handle.attrs.get("case") != case or handle.attrs.get("split") != "train":
            raise ValueError(f"{case}: HDF5 attrs are not train-only")
        embedded = json.loads(handle.attrs["config_json"])
        if embedded != config:
            raise ValueError(f"{case}: embedded config differs from raw case config")
        times = handle["time"][:, 0].astype(np.float64)
        omega = handle["omega"][:, 0].astype(np.float64)
        force = handle["force"][:].astype(np.float64)
        if not np.allclose(times, expected_times, rtol=0.0, atol=2e-5):
            raise ValueError(f"{case}: HDF5 time grid mismatch")
        if not np.allclose(
            omega, np.interp(times, action[:, 0], action[:, 1]),
            rtol=0.0, atol=1e-6,
        ):
            raise ValueError(f"{case}: HDF5 action alignment mismatch")
        expected_force = np.stack(
            [
                np.interp(times, front[:, 0], front[:, 1]),
                np.interp(times, front[:, 0], front[:, 2]),
                np.interp(times, rear[:, 0], rear[:, 1]),
                np.interp(times, rear[:, 0], rear[:, 2]),
            ],
            axis=1,
        )
        if not np.allclose(force, expected_force, rtol=2e-6, atol=2e-6):
            raise ValueError(f"{case}: HDF5 force alignment mismatch")
        if not np.isfinite(force).all():
            raise ValueError(f"{case}: nonfinite force")
        for index in range(801):
            state = handle["state"][index]
            mask = handle["mask"][index, 0].astype(bool)
            if not np.isfinite(state).all() or not mask.any():
                raise ValueError(f"{case}:{index}: nonfinite or empty field")
            coverage = float(mask.mean())
            min_coverage = min(min_coverage, coverage)
            max_coverage = max(max_coverage, coverage)
            pressure_mean = abs(float(state[2][mask].mean(dtype=np.float64)))
            max_pressure_mean = max(max_pressure_mean, pressure_mean)
    if max_pressure_mean > 1e-6 or not (0.8 < min_coverage <= max_coverage < 1.0):
        raise ValueError(f"{case}: pressure gauge or coverage QC failed")
    return {
        "case": case, "split": "train", "frames": 801,
        "phase_bin": preflight_row["phase_bin"], "branch": preflight_row["branch"],
        "source_state_sha256": preflight_row["source_state_sha256"],
        "hdf5": str(target.relative_to(repo)),
        "hdf5_sha256": orchestrator.sha256(target),
        "max_abs_pressure_mean": max_pressure_mean,
        "coverage_range": [min_coverage, max_coverage],
        "force_front": front_audit, "force_rear": rear_audit,
        "vtk_marker_sha256": orchestrator.sha256(
            repo / VTK_VERIFIED_ROOT / f"{case}.json"
        ),
        "vtk_manifest_sha256": vtk["vtk_manifest_sha256"],
        "raw_transfer_receipt_sha256": preflight_row["receipt_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--qc-output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    orchestrator = load_orchestrator(repo)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    preflight = orchestrator.preflight(repo, contract)
    if args.qc_output.exists() or args.qc_output.with_suffix(
        args.qc_output.suffix + ".tmp"
    ).exists():
        raise FileExistsError(args.qc_output)
    validate_staging_layout(
        repo, [row["case"] for row in preflight["cases"]]
    )
    rows = [
        validate_case(repo, row["case"], row, orchestrator)
        for row in preflight["cases"]
    ]
    for phase in ("b00", "b02", "b04"):
        group = [row for row in rows if row["phase_bin"] == phase]
        if len({json.dumps(row["source_state_sha256"], sort_keys=True) for row in group}) != 1:
            raise ValueError(f"{phase}: aggregate five-field restart mismatch")
    qc = {
        "status": "MATCHED_START_COMMISSIONING_NINE_CASE_QC_OK",
        "profile": PROFILE,
        "trajectory_counts": {"train": 9, "validation": 0, "test": 0},
        "normalization": "NOT_COMPUTED",
        "training_use": "FORBIDDEN_COMMISSIONING_ONLY",
        "validation_or_frozen_access": "NOT_ACCESSED",
        "cases": rows,
    }
    if not args.validate_only:
        final = repo / FINAL_ROOT
        temporary = final.with_name(f".{final.name}.tmp")
        if final.exists() or temporary.exists():
            raise FileExistsError("refusing existing final or temporary profile")
        (temporary / "train").mkdir(parents=True)
        for row in rows:
            source = repo / row["hdf5"]
            os.link(source, temporary / "train" / source.name)
        manifest = {
            "schema_version": 1, "profile": PROFILE,
            "purpose": "train-split commissioning evidence only; model training forbidden until a separately reviewed promotion",
            "trajectory_counts": {"train": 9, "validation": 0, "test": 0},
            "frames_per_trajectory": 801, "pairs_per_trajectory": 800,
            "normalization": "NOT_COMPUTED_COMMISSIONING_ONLY",
            "training_use": "FORBIDDEN",
            "phase_manifest": preflight["phase_manifest"],
            "phase_manifest_sha256": preflight["phase_manifest_sha256"],
            "hdf5_sha256": {row["case"]: row["hdf5_sha256"] for row in rows},
            "curator_pipeline": [
                "PhysicsNeMo Curator Source[VTKSource + Mesh/BVH]",
                "NumericalQualityFilter", "atomic TrajectoryHDF5Sink",
            ],
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (temporary / "commissioning_qc.json").write_text(json.dumps(qc, indent=2) + "\n")
        os.replace(temporary, final)
    args.qc_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_qc = args.qc_output.with_suffix(args.qc_output.suffix + ".tmp")
    with temporary_qc.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(qc, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary_qc, args.qc_output)
    temporary_qc.unlink()
    print(json.dumps({"status": qc["status"], "cases": len(rows)}))


if __name__ == "__main__":
    main()
