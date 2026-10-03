#!/usr/bin/env python3
"""Recompute total and fluctuating lift RMS from retained raw OpenFOAM forces."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from analyze_baseline import load_coefficients
from make_control_landscape_panel import CASES, file_sha256

OLD_REPORTED_TOTAL_RMS = 1.516647


def force_metrics(rows: list[tuple[float, float, float]], begin: float, end: float) -> dict:
    selected = [cl for time, _, cl in rows if begin <= time <= end]
    if len(selected) < 1000:
        raise ValueError("insufficient raw lift samples")
    mean = statistics.mean(selected)
    fluctuation = math.sqrt(statistics.mean((value - mean) ** 2 for value in selected))
    total = math.sqrt(statistics.mean(value * value for value in selected))
    if not math.isclose(total**2, fluctuation**2 + mean**2, rel_tol=1e-12):
        raise ValueError("RMS identity failed")
    return {
        "sample_count": len(selected),
        "cl_mean": mean,
        "cl_fluctuation_rms": fluctuation,
        "cl_total_rms": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    paths = {
        "original_from_t0": CASES / "control_small_p100/postProcessing/forceRear/0/coefficient.dat",
        "matched_step_from_t80": CASES / "landscape_step_val_p100_20261003/postProcessing/forceRear/80/coefficient.dat",
    }
    rows = {key: load_coefficients(path) for key, path in paths.items()}
    old_full = force_metrics(rows["original_from_t0"], 80.0, 160.0)
    old_late = force_metrics(rows["original_from_t0"], 120.0, 160.0)
    new_late = force_metrics(rows["matched_step_from_t80"], 120.0, 160.0)
    if abs(old_full["cl_total_rms"] / OLD_REPORTED_TOTAL_RMS - 1) > 0.005:
        raise ValueError("historical ~1.517 does not match total RMS")
    if abs(old_late["cl_fluctuation_rms"] / new_late["cl_fluctuation_rms"] - 1) > 0.02:
        raise ValueError("matched-window fluctuating RMS changed by >2%")
    report = {
        "status": "LIFT_RMS_METRIC_DEFINITION_RESOLVED",
        "historical_table_label": "Cl RMS (ambiguous)",
        "historical_table_value_p100": OLD_REPORTED_TOTAL_RMS,
        "historical_value_matches": "total RMS including nonzero mean, not fluctuating RMS",
        "definitions": {
            "fluctuation_rms": "sqrt(mean((Cl-mean(Cl))^2))",
            "total_rms": "sqrt(mean(Cl^2)) = hypot(mean(Cl), fluctuation_rms)",
        },
        "raw_force_sha256": {key: file_sha256(path) for key, path in paths.items()},
        "old_t80_160": old_full,
        "old_t120_160": old_late,
        "new_step_t120_160": new_late,
        "old_vs_new_late_fluctuation_rms_ratio": old_late["cl_fluctuation_rms"] / new_late["cl_fluctuation_rms"],
        "interpretation_guard": "metric-label correction; raw solver data and canonical physical thresholds unchanged",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "old": old_full, "new_late": new_late}))


if __name__ == "__main__":
    main()
