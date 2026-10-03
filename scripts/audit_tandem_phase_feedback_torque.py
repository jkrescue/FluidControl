#!/usr/bin/env python3
"""Audit rear-cylinder CmPitch and rotation-work proxies for the paired pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
CASES = PROJECT / "cfd" / "tandem_cylinders" / "cases"


def read_moment_series(case: Path) -> np.ndarray:
    samples: dict[float, float] = {}
    paths = sorted(case.glob("postProcessing/forceRear/*/coefficient.dat"))
    if not paths:
        raise FileNotFoundError(f"missing rear force coefficients: {case}")
    for path in paths:
        header: list[str] | None = None
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# Time"):
                header = line[1:].split()
                if "CmPitch" not in header:
                    raise ValueError(f"CmPitch absent from {path}")
                continue
            if not line or line.startswith("#"):
                continue
            if header is None:
                raise ValueError(f"force header absent from {path}")
            fields = line.split()
            time = float(fields[header.index("Time")])
            moment = float(fields[header.index("CmPitch")])
            if not all(math.isfinite(value) for value in (time, moment)):
                raise ValueError(f"non-finite moment row in {path}")
            key = round(time, 8)
            if key in samples and not math.isclose(
                samples[key], moment, rel_tol=0.0, abs_tol=1.0e-10
            ):
                raise ValueError(f"conflicting restart CmPitch at t={time}")
            samples[key] = moment
    return np.asarray([[time, samples[time]] for time in sorted(samples)], dtype=float)


def power_metrics(
    time: np.ndarray,
    cm_pitch: np.ndarray,
    action_points: np.ndarray,
    baseline_total_cd: float,
) -> tuple[np.ndarray, dict]:
    if baseline_total_cd <= 0 or not math.isfinite(baseline_total_cd):
        raise ValueError("baseline total drag must be positive and finite")
    omega = np.interp(time, action_points[:, 0], action_points[:, 1])
    fluid_work = omega * cm_pitch / baseline_total_cd
    actuator_signed = -fluid_work
    actuator_positive_only = np.maximum(actuator_signed, 0.0)
    rows = np.column_stack(
        (time, omega, cm_pitch, fluid_work, actuator_signed, actuator_positive_only)
    )
    metrics = {
        "samples": len(rows),
        "cm_pitch_mean": float(np.mean(cm_pitch)),
        "cm_pitch_rms": float(np.sqrt(np.mean(np.square(cm_pitch)))),
        "mean_omega_times_cm_pitch": float(np.mean(omega * cm_pitch)),
        "mean_fluid_to_cylinder_power_over_zero_drag_power": float(
            np.mean(fluid_work)
        ),
        "mean_signed_actuator_power_over_zero_drag_power": float(
            np.mean(actuator_signed)
        ),
        "mean_positive_only_actuator_power_over_zero_drag_power": float(
            np.mean(actuator_positive_only)
        ),
        "mean_abs_rotation_work_over_zero_drag_power": float(
            np.mean(np.abs(actuator_signed))
        ),
    }
    return rows, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    if not result_dir.is_relative_to(PROJECT / "artifacts" / "tandem_cylinders"):
        parser.error("result_dir must be under artifacts/tandem_cylinders")
    result_path = result_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "REAL_OPENFOAM_PHASE_FEEDBACK_PAIR_COMPLETED":
        raise ValueError("paired pilot is incomplete")
    progress = json.loads((result_dir / "progress.json").read_text(encoding="utf-8"))
    action_points = np.asarray(
        [[result["simulation_window"][0], 0.0]]
        + [[row["end_time"], row["applied_omega"]] for row in progress["rows"]],
        dtype=float,
    )
    begin, end = map(float, result["predeclared_evaluation_window"])
    series = read_moment_series(CASES / result["pair"]["feedback"])
    selected = series[(series[:, 0] >= begin - 1.0e-8) & (series[:, 0] <= end + 1.0e-8)]
    if len(selected) < 100:
        raise ValueError("insufficient CmPitch samples in evaluation window")
    baseline_drag = float(result["metrics"]["zero"]["total_cd_mean"])
    rows, metrics = power_metrics(
        selected[:, 0], selected[:, 1], action_points, baseline_drag
    )
    csv_path = result_dir / "rear_cylinder_torque_timeseries.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "time",
                "omega",
                "cm_pitch_fluid_on_cylinder",
                "fluid_to_cylinder_power_over_zero_drag_power",
                "signed_actuator_power_over_zero_drag_power",
                "positive_only_actuator_power_over_zero_drag_power",
            ]
        )
        writer.writerows(rows)
    audit = {
        "status": "REAR_CYLINDER_CMPITCH_AUDIT_OK",
        "case": result["pair"]["feedback"],
        "window": [begin, end],
        "force_coefficient_definition": (
            "OpenFOAM forceCoeffs CmPitch=M_fluid/(0.5*rho*U_inf^2*Aref*lRef)"
        ),
        "normalization": (
            "D=U_inf=lRef=1, so signed actuator power divided by the matched "
            "zero-case drag power is -omega*CmPitch/Cd_total_zero"
        ),
        "metrics": metrics,
        "timeseries": str(csv_path),
        "timeseries_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "interpretation_guard": (
            "kinematic fluid-torque work proxy, not electrical motor input power; "
            "positive-only assumes no regenerative energy recovery"
        ),
    }
    (result_dir / "torque_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
