#!/usr/bin/env python3
"""Audit controlled CFD force histories against their common restart baseline.

This is a data sanity check, not evidence of improved flow control.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--baseline", type=Path,
        default=Path(
            "cfd/tandem_cylinders/cases/tandem_backward_dt005/"
            "postProcessing/forceRear/0/coefficient.dat"
        ),
    )
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.expected_count < 1:
        parser.error("--expected-count must be positive")

    baseline = np.loadtxt(args.baseline, usecols=(0, 1, 4))
    if (baseline.ndim != 2 or baseline.shape[1] != 3
            or not np.isfinite(baseline).all()
            or not np.all(np.diff(baseline[:, 0]) > 0)):
        raise ValueError("invalid source OpenFOAM rear-force baseline")

    paths = sorted(args.data.glob("*/*.h5"))
    if len(paths) != args.expected_count:
        raise ValueError(f"expected {args.expected_count} HDF5 trajectories, found {len(paths)}")
    rows = []
    for path in paths:
        with h5py.File(path, "r") as handle:
            time = np.asarray(handle["time"][:, 0], dtype=np.float64)
            omega = np.asarray(handle["omega"][:, 0], dtype=np.float64)
            rear_force = np.asarray(handle["force"][:, 2:4], dtype=np.float64)
        if (len(time) != 801 or time[0] < baseline[0, 0]
                or time[-1] > baseline[-1, 0] + 1.0e-6
                or not np.all(np.diff(time) > 0)
                or not np.isfinite(omega).all()
                or not np.isfinite(rear_force).all()):
            raise ValueError(f"{path.name}: invalid time, action, or force series")

        zero_force = np.stack(
            [np.interp(time, baseline[:, 0], baseline[:, column]) for column in (1, 2)],
            axis=1,
        )
        absolute_difference = np.abs(rear_force - zero_force)
        initial_cd, initial_cl = absolute_difference[0]
        late_cd, late_cl = absolute_difference[time >= 120.0].mean(axis=0)
        omega_rms = float(np.sqrt(np.mean(omega**2)))
        row = {
            "case": path.stem,
            "split": path.parent.name,
            "omega_rms": omega_rms,
            "initial_abs_cd_difference": float(initial_cd),
            "initial_abs_cl_difference": float(initial_cl),
            "late_mean_abs_cd_difference": float(late_cd),
            "late_mean_abs_cl_difference": float(late_cl),
        }
        # Gross same-restart and post-transient checks, not optimization targets.
        if (initial_cd > 0.01 or initial_cl > 0.02
                or omega_rms < 0.1 or late_cd < 0.05 or late_cl < 0.1):
            raise ValueError(f"{path.name}: no credible controlled response: {row}")
        rows.append(row)
        print(json.dumps(row), flush=True)

    report = {
        "status": "CONTROLLED_CFD_RESPONSE_AUDIT_OK",
        "baseline": str(args.baseline),
        "late_window": "t >= 120 D/U_inf",
        "interpretation": (
            "Common-start and nonzero physical response sanity check only; "
            "absolute deviation is not a control benefit or causal effect estimate."
        ),
        "trajectory_count": len(rows),
        "minimum_late_mean_abs_cd_difference": min(
            row["late_mean_abs_cd_difference"] for row in rows
        ),
        "minimum_late_mean_abs_cl_difference": min(
            row["late_mean_abs_cl_difference"] for row in rows
        ),
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("CONTROLLED_CFD_RESPONSE_AUDIT_OK", flush=True)


if __name__ == "__main__":
    main()
