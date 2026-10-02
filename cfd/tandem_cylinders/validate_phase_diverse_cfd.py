#!/usr/bin/env python3
"""Fail-closed validation for the independent-phase OpenFOAM panel."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from make_phase_diverse_control_dataset import PHASE_CASES


def coefficient_rows(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            columns = line.split()
            rows.append([float(columns[index]) for index in (0, 1, 4)])
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError(f"invalid force coefficients: {path}")
    return values


def validate_case(case: Path) -> dict:
    config = json.loads((case / "case_config.json").read_text(encoding="utf-8"))
    start = float(config["start_time"])
    end = float(config["end_time"])
    frames = int(config["expected_frames"])
    delta_t = float(config["delta_t"])
    log_path = case / "log.pimpleFoam"
    log = log_path.read_text(encoding="utf-8")
    nonempty = [line.strip() for line in log.splitlines() if line.strip()]
    if not nonempty or nonempty[-1] != "End":
        raise ValueError(f"{case.name}: solver did not end normally")

    numeric = sorted(
        float(path.name)
        for path in case.iterdir()
        if path.is_dir() and re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", path.name)
    )
    expected_times = np.linspace(start, end, frames)
    if len(numeric) != frames or not np.allclose(numeric, expected_times, atol=2e-8):
        raise ValueError(f"{case.name}: incomplete field times")
    for time in numeric:
        directory = case / f"{time:g}"
        if not (directory / "U").is_file() or not (directory / "p").is_file():
            raise ValueError(f"{case.name}: missing U/p at {time:g}")

    expected_force_samples = int(round((end - start) / delta_t))
    force_summary = {}
    for object_name in ("forceFront", "forceRear"):
        path = case / "postProcessing" / object_name / f"{start:g}" / "coefficient.dat"
        values = coefficient_rows(path)
        if len(values) != expected_force_samples:
            raise ValueError(
                f"{case.name}:{object_name}: expected {expected_force_samples} rows, "
                f"found {len(values)}"
            )
        if not math.isclose(values[-1, 0], end, abs_tol=1e-8):
            raise ValueError(f"{case.name}:{object_name}: force series ends early")
        force_summary[object_name] = {
            "samples": len(values),
            "final_time": float(values[-1, 0]),
            "mean_cd": float(values[:, 1].mean()),
            "cl_rms": float(np.sqrt(np.mean(np.square(values[:, 2])))),
        }

    courant = [
        float(value)
        for value in re.findall(r"Courant Number mean: \S+ max: (\S+)", log)
    ]
    if not courant or max(courant) >= 1.0:
        raise ValueError(f"{case.name}: invalid or unstable Courant history")
    return {
        "case": case.name,
        "split": config["split"],
        "source_restart_time": config["source_restart_time"],
        "frames": frames,
        "force": force_summary,
        "max_courant": max(courant),
        "status": "complete",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases-root", type=Path, default=Path("cfd/tandem_cylinders/cases")
    )
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    known = {spec.name for spec in PHASE_CASES}
    selected = args.cases or sorted(known)
    unknown = set(selected) - known
    if unknown:
        parser.error(f"unknown case(s): {', '.join(sorted(unknown))}")
    report = {
        "dataset": "tandem_phase_diverse_v1",
        "cases": [validate_case(args.cases_root / name) for name in selected],
        "status": "PHASE_DIVERSE_CFD_OK",
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
