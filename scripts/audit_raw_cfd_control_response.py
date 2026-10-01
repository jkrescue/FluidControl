#!/usr/bin/env python3
"""Check common restart and nonzero force response in raw OpenFOAM cases.

This is a CFD-data sanity check, not evidence of drag reduction or closed-loop
control. Input columns are time, Cd, Cl from OpenFOAM forceCoeffs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_BASELINE = Path(
    "cfd/tandem_cylinders/cases/tandem_backward_dt005/"
    "postProcessing/forceRear/0/coefficient.dat"
)
DEFAULT_CASES_ROOT = Path("cfd/tandem_cylinders/cases")


def read_forces(path: Path) -> np.ndarray:
    values = np.loadtxt(path, usecols=(0, 1, 4), ndmin=2)
    if (
        values.shape[1] != 3
        or len(values) < 2
        or not np.isfinite(values).all()
        or not np.all(np.diff(values[:, 0]) > 0)
    ):
        raise ValueError(f"Invalid OpenFOAM force coefficients: {path}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="+", help="OpenFOAM case directory names")
    parser.add_argument("--cases-root", type=Path, default=DEFAULT_CASES_ROOT)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--start-time", default="80")
    parser.add_argument("--late-time", type=float, default=120.0)
    parser.add_argument("--expected-samples", type=int, default=16000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.expected_samples < 2:
        parser.error("--expected-samples must be at least two")
    if len(set(args.cases)) != len(args.cases):
        parser.error("case names must be unique")

    baseline = read_forces(args.baseline)
    rows = []
    for name in args.cases:
        if Path(name).name != name:
            parser.error(f"expected a case directory name, got {name!r}")
        path = (
            args.cases_root / name / "postProcessing" / "forceRear"
            / args.start_time / "coefficient.dat"
        )
        controlled = read_forces(path)
        time = controlled[:, 0]
        if (
            len(controlled) != args.expected_samples
            or time[0] < baseline[0, 0]
            or time[-1] > baseline[-1, 0] + 1.0e-6
        ):
            raise ValueError(f"{name}: wrong sample count or baseline time range")
        late = time >= args.late_time
        if not late.any():
            raise ValueError(f"{name}: no samples in late window")
        reference = np.column_stack(
            [np.interp(time, baseline[:, 0], baseline[:, column])
             for column in (1, 2)]
        )
        delta = np.abs(controlled[:, 1:3] - reference)
        initial_cd, initial_cl = delta[0]
        late_cd, late_cl = delta[late].mean(axis=0)
        row = {
            "case": name,
            "sample_count": len(controlled),
            "initial_abs_cd_difference": float(initial_cd),
            "initial_abs_cl_difference": float(initial_cl),
            "late_mean_abs_cd_difference": float(late_cd),
            "late_mean_abs_cl_difference": float(late_cl),
        }
        if initial_cd > 0.01 or initial_cl > 0.02 or late_cd < 0.05 or late_cl < 0.1:
            raise ValueError(f"{name}: no credible common-start response: {row}")
        rows.append(row)
        print(json.dumps(row), flush=True)

    report = {
        "status": "RAW_CFD_CONTROL_RESPONSE_AUDIT_OK",
        "baseline": str(args.baseline),
        "late_window": f"t >= {args.late_time} D/U_inf",
        "interpretation": (
            "Common-start and nonzero response sanity check only; absolute "
            "deviation is not a control benefit or causal effect estimate."
        ),
        "case_count": len(rows),
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report["status"], flush=True)


if __name__ == "__main__":
    main()
