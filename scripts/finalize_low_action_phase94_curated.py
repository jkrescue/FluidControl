#!/usr/bin/env python3
"""Finalize and validate the validation-only phase-94 Curator profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


NAMES = (
    "validation_signed_low_pulse_p_phase94_v1_20261003",
    "validation_signed_low_pulse_m_phase94_v1_20261003",
)
TIME_ATOL = 2.0e-5


def validate_time_grid(times: np.ndarray) -> None:
    expected = np.linspace(94.0, 174.0, 801)
    # Curator stores float32 time.  At this absolute time range the expected
    # roundoff is up to 6.11e-6, and adjacent 0.1 steps vary by 9.16e-6.
    if not np.allclose(times, expected, rtol=0.0, atol=TIME_ATOL):
        raise ValueError("time mismatch")
    if not np.allclose(np.diff(times), 0.1, rtol=0.0, atol=TIME_ATOL):
        raise ValueError("nonuniform time grid")


def load_force(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        rows.append((float(fields[0]), float(fields[1]), float(fields[4])))
    return np.asarray(rows, dtype=np.float64)


def validate_case(data_root: Path, cases_root: Path, name: str) -> dict:
    case = cases_root / name
    config = json.loads((case / "case_config.json").read_text(encoding="utf-8"))
    target = data_root / "validation" / f"{name}.h5"
    with h5py.File(target, "r") as handle:
        required = {"state", "mask", "omega", "force", "time", "x", "y"}
        if not required.issubset(handle.keys()):
            raise ValueError(f"{name}: missing HDF5 datasets")
        if handle["state"].shape != (801, 3, 128, 256):
            raise ValueError(f"{name}: wrong state shape {handle['state'].shape}")
        if handle["mask"].shape != (801, 1, 128, 256):
            raise ValueError(f"{name}: wrong mask shape")
        if handle["force"].shape != (801, 4):
            raise ValueError(f"{name}: wrong force shape")
        times = handle["time"][:, 0].astype(np.float64)
        omega = handle["omega"][:, 0].astype(np.float64)
        force = handle["force"][:].astype(np.float64)
        mask = handle["mask"][:].astype(bool)
        state = handle["state"][:]
        if handle.attrs["case"] != name or handle.attrs["split"] != "validation":
            raise ValueError(f"{name}: wrong HDF5 attributes")
    try:
        validate_time_grid(times)
    except ValueError as error:
        raise ValueError(f"{name}: {error}") from error
    table = np.asarray(config["action_points"], dtype=np.float64)
    expected_omega = np.interp(times, table[:, 0], table[:, 1])
    if not np.allclose(omega, expected_omega, rtol=0.0, atol=1.0e-6):
        raise ValueError(f"{name}: action alignment mismatch")
    if np.max(np.abs(omega)) > 0.750001:
        raise ValueError(f"{name}: action support exceeded")
    aligned = []
    for object_name in ("forceFront", "forceRear"):
        raw = load_force(case / "postProcessing" / object_name / "94" / "coefficient.dat")
        if times[0] < raw[0, 0] - 1.0e-8:
            source_name = config.get("source_restart_case")
            if not source_name:
                raise ValueError(f"{name}: missing exact force restart provenance")
            source = load_force(
                cases_root
                / source_name
                / "postProcessing"
                / object_name
                / "0"
                / "coefficient.dat"
            )
            matches = np.flatnonzero(
                np.isclose(source[:, 0], times[0], rtol=0.0, atol=1.0e-8)
            )
            if len(matches) != 1:
                raise ValueError(f"{name}: no unique source force at restart")
            raw = np.concatenate((source[matches], raw), axis=0)
        aligned.extend(np.interp(times, raw[:, 0], raw[:, column]) for column in (1, 2))
    expected_force = np.stack(aligned, axis=1)
    if not np.allclose(force, expected_force, rtol=2.0e-6, atol=2.0e-6):
        raise ValueError(f"{name}: force alignment mismatch")
    if not np.isfinite(state).all() or not np.isfinite(force).all():
        raise ValueError(f"{name}: nonfinite curated values")
    pressure_means = []
    for index in range(801):
        valid = mask[index, 0]
        pressure_means.append(float(state[index, 2][valid].mean(dtype=np.float64)))
    max_pressure_mean = max(abs(value) for value in pressure_means)
    if max_pressure_mean > 1.0e-6:
        raise ValueError(f"{name}: pressure gauge mismatch {max_pressure_mean}")
    return {
        "case": name,
        "split": "validation",
        "frames": 801,
        "coverage": float(mask.mean()),
        "omega_min": float(omega.min()),
        "omega_max": float(omega.max()),
        "max_abs_pressure_mean": max_pressure_mean,
        "source_restart_u_sha256": config["source_restart_u_sha256"],
        "source_restart_p_sha256": config["source_restart_p_sha256"],
        "force_alignment": "exact raw CFD interpolation at 801 field times",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cases-root", type=Path, required=True)
    parser.add_argument("--qc-output", type=Path, required=True)
    args = parser.parse_args()
    files = sorted((args.data / "validation").glob("*.h5"))
    if [path.stem for path in files] != sorted(NAMES):
        raise ValueError("validation-only profile does not contain exactly the frozen pair")
    if any((args.data / split).exists() and any((args.data / split).iterdir()) for split in ("train", "test")):
        raise ValueError("validation-only profile contains train/test data")
    cases = [validate_case(args.data, args.cases_root, name) for name in NAMES]
    manifest = {
        "schema_version": 1,
        "profile": "low_action_phase94_validation_v1",
        "purpose": "OOD validation-only low-action diagnosis; never training or model selection",
        "trajectory_counts": {"train": 0, "validation": 2, "test": 0},
        "frames_per_trajectory": 801,
        "pairs_per_trajectory": 800,
        "grid": {"nx": 256, "ny": 128, "x_range": [8, 25], "y_range": [4, 11]},
        "fields": ["u", "v", "gauge_pressure", "valid_mask", "rear_omega", "front_cd_cl", "rear_cd_cl"],
        "pressure_preprocessing": "subtract valid-domain spatial mean independently at every frame",
        "normalization": "external immutable training normalization required; use control_gap_v4 for v4 checkpoint",
        "max_abs_omega": 0.75,
        "source_restart_time": 94.0,
        "leakage_guard": "do not merge with control_gap_v4 train/validation/test",
        "curator_pipeline": [
            "TandemTrajectorySource[VTKSource + Mesh.sample_data_at_points]",
            "NumericalQualityFilter",
            "TrajectoryHDF5Sink",
        ],
    }
    (args.data / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    qc = {
        "status": "LOW_ACTION_PHASE94_CURATED_VALIDATION_OK",
        "profile": manifest["profile"],
        "trajectory_counts": manifest["trajectory_counts"],
        "cases": cases,
        "scientific_scope": "validation-only OOD diagnostic; not used to train or select the fixed model",
    }
    args.qc_output.parent.mkdir(parents=True, exist_ok=True)
    args.qc_output.write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": qc["status"], "cases": [row["case"] for row in cases]}))


if __name__ == "__main__":
    main()
