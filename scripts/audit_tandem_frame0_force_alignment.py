#!/usr/bin/env python3
"""Audit the original expanded-v2 frame-0 force timestamp against raw OpenFOAM."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from fluid_control.openfoam_observation import observation_at  # noqa: E402


def first_rear_force(path: Path) -> tuple[float, float, float]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            fields = line.split()
            return float(fields[0]), float(fields[1]), float(fields[4])
    raise ValueError(f"missing OpenFOAM coefficient rows: {path}")


def audit(data: Path, cases: Path) -> dict:
    source = cases / "tandem_backward_dt005"
    source_obs, _ = observation_at(source, 80.0, 0.0)
    true_rear = np.asarray(source_obs[64:66], dtype=np.float64)
    rows = []
    for split, expected in (("train", 24), ("validation", 4), ("test", 4)):
        files = sorted((data / split).glob("expanded_*.h5"))
        if len(files) != expected:
            raise ValueError(f"{split}: expected {expected} HDF5 trajectories, got {len(files)}")
        for path in files:
            with h5py.File(path, "r") as handle:
                t0 = float(handle["time"][0, 0])
                curated = np.asarray(handle["force"][0, 2:4], dtype=np.float64)
            if abs(t0 - 80.0) > 1e-8 or not np.isfinite(curated).all():
                raise ValueError(f"invalid frame-0 time/force in {path}")
            raw_path = cases / path.stem / "postProcessing/forceRear/80/coefficient.dat"
            first_time, first_cd, first_cl = first_rear_force(raw_path)
            first_force = np.asarray([first_cd, first_cl])
            source_error = np.abs(curated - true_rear)
            first_error = np.abs(curated - first_force)
            rows.append({
                "split": split,
                "case": path.stem,
                "curated_frame_time": t0,
                "first_case_force_time": first_time,
                "curated_rear_cd_cl": curated.tolist(),
                "true_restart_rear_cd_cl": true_rear.tolist(),
                "curated_minus_true_restart": (curated - true_rear).tolist(),
                "max_abs_error_vs_true_restart": float(source_error.max()),
                "max_abs_error_vs_first_case_force": float(first_error.max()),
            })
    first_sample_match = all(row["max_abs_error_vs_first_case_force"] < 1e-5 for row in rows)
    first_time_later = all(row["first_case_force_time"] > 80.0 for row in rows)
    mismatch = max(row["max_abs_error_vs_true_restart"] for row in rows)
    return {
        "status": (
            "FRAME0_CURATED_FORCE_TIME_MISALIGNMENT_CONFIRMED"
            if first_sample_match and first_time_later and mismatch > 1e-4
            else "FRAME0_FORCE_PROVENANCE_INCONCLUSIVE"
        ),
        "scientific_scope": "original_curated_v2_frame0_rear_force_only_not_model_accuracy",
        "trajectory_count": len(rows),
        "source_restart_case": source.name,
        "source_restart_time": 80.0,
        "true_restart_rear_cd_cl": true_rear.tolist(),
        "max_abs_error_vs_true_restart_across_cases": mismatch,
        "max_abs_error_vs_first_case_force_across_cases": max(
            row["max_abs_error_vs_first_case_force"] for row in rows
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = audit(args.data, args.cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    if report["status"] != "FRAME0_CURATED_FORCE_TIME_MISALIGNMENT_CONFIRMED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
