#!/usr/bin/env python3
"""Exploratory locked-objective screen of pre-existing train/validation CFD.

Never reads the five frozen test trajectories. A passing open-loop candidate
here would be a *baseline to beat*, not a learned closed-loop result.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from analyze_baseline import load_coefficients, log_health, window_stats
from make_control_landscape_long_panel import ANALYSIS_START, END_TIME
from make_control_landscape_panel import CASES, SOURCE_CASE, file_sha256

DATA = Path(__file__).resolve().parents[2] / "data/curated/tandem_cylinders_gate_b_aug_v3"
MIN_DRAG_REDUCTION = 0.02
MAX_FLUCTUATION_RATIO = 1.05
MAX_MEAN_LIFT_RATIO = 0.10


def accept(drag_reduction: float, fluctuation_ratio: float, mean_lift_ratio: float) -> dict:
    checks = {
        "total_cd_reduction_at_least_2pct": drag_reduction >= MIN_DRAG_REDUCTION,
        "rear_cl_fluctuation_increase_at_most_5pct": fluctuation_ratio <= MAX_FLUCTUATION_RATIO,
        "abs_mean_rear_cl_at_most_10pct_zero_fluct_rms": mean_lift_ratio <= MAX_MEAN_LIFT_RATIO,
    }
    return {"checks": checks, "joint_pass": all(checks.values())}


def force_metrics(case: Path) -> dict:
    front = load_coefficients(case / "postProcessing/forceFront/80/coefficient.dat")
    rear = load_coefficients(case / "postProcessing/forceRear/80/coefficient.dat")
    if len(front) != len(rear) or len(front) != 16000:
        raise ValueError(f"incomplete matched force histories: {case.name}")
    if any(abs(a[0] - b[0]) > 1e-8 for a, b in zip(front, rear)):
        raise ValueError(f"front/rear force timestamps differ: {case.name}")
    window = [(a[1] + b[1], b[2]) for a, b in zip(front, rear)
              if ANALYSIS_START <= a[0] <= END_TIME]
    if len(window) != 8001 or abs(front[-1][0] - END_TIME) > 1e-8:
        raise ValueError(f"incomplete fixed analysis window: {case.name}")
    return {
        "total_cd_mean": sum(cd for cd, _ in window) / len(window),
        "front": window_stats(front, ANALYSIS_START, END_TIME),
        "rear": window_stats(rear, ANALYSIS_START, END_TIME),
    }


def check_fields(case: Path) -> dict:
    times = []
    for path in case.iterdir():
        try:
            value = float(path.name)
        except ValueError:
            continue
        if path.is_dir() and value >= 80.0:
            if not (path / "U").is_file() or not (path / "p").is_file():
                raise ValueError(f"missing U/p field: {case.name}/{path.name}")
            times.append(value)
    times.sort()
    if (len(times) != 801 or not math.isclose(times[0], 80.0, abs_tol=1e-8)
            or not math.isclose(times[-1], 160.0, abs_tol=1e-8)
            or any(not math.isclose(t, 80.0 + index * 0.1, abs_tol=2e-8)
                   for index, t in enumerate(times))):
        raise ValueError(f"incomplete 801-frame legacy CFD sequence: {case.name}")
    return {"frames": len(times), "start": times[0], "end": times[-1]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    names = [(split, path.stem) for split in ("train", "validation")
             for path in sorted((DATA / split).glob("*.h5"))]
    if len(names) != 30 or [sum(split == s for split, _ in names) for s in ("train", "validation")] != [26, 4]:
        raise ValueError("v3 exploratory cohort is not the expected 26/4 train/validation split")
    baseline_case = CASES / "landscape_long_val_zero_20261003"
    baseline = force_metrics(baseline_case)
    zero_drag, zero_fluct = baseline["total_cd_mean"], baseline["rear"]["cl_rms"]
    if min(zero_drag, zero_fluct) <= 0:
        raise ValueError("invalid physical zero-action reference")
    source_p = file_sha256(SOURCE_CASE / "80/p")
    rows = []
    for split, name in names:
        case = CASES / name
        cfg = json.loads((case / "case_config.json").read_text(encoding="utf-8"))
        if cfg["split"] != split or cfg["source_restart_time"] != 80.0 or cfg["source_restart_case"] != SOURCE_CASE.name:
            raise ValueError(f"wrong split or source restart: {name}")
        if file_sha256(case / "80/p") != source_p:
            raise ValueError(f"source pressure restart differs: {name}")
        raw = check_fields(case)
        health = log_health(case / "log.pimpleFoam")
        if not health["solver_ended_cleanly"] or health["max_courant"] >= 1 or health["steps"] != 16000:
            raise ValueError(f"numerical health failed: {name}")
        metrics = force_metrics(case)
        drag_reduction = 1 - metrics["total_cd_mean"] / zero_drag
        fluctuation_ratio = metrics["rear"]["cl_rms"] / zero_fluct
        mean_lift_ratio = abs(metrics["rear"]["cl_mean"]) / zero_fluct
        decision = accept(drag_reduction, fluctuation_ratio, mean_lift_ratio)
        rows.append({
            "case": name,
            "split": split,
            "schedule_kind": cfg.get("schedule_kind"),
            "total_cd_mean": metrics["total_cd_mean"],
            "front_cd_mean": metrics["front"]["cd_mean"],
            "rear_cd_mean": metrics["rear"]["cd_mean"],
            "rear_cl_mean": metrics["rear"]["cl_mean"],
            "rear_cl_fluctuation_rms": metrics["rear"]["cl_rms"],
            "drag_reduction": drag_reduction,
            "rear_cl_fluctuation_ratio": fluctuation_ratio,
            "abs_mean_rear_cl_over_zero_fluct_rms": mean_lift_ratio,
            **decision,
            "numerical_health": health,
            "raw_case_audit": raw,
        })
    rows.sort(key=lambda row: row["drag_reduction"], reverse=True)
    result = {
        "status": "EXPLORATORY_EXISTING_OPEN_LOOP_COHORT_AUDITED",
        "scientific_scope": "pre-existing train/validation CFD only; no frozen test, controller selection or closed-loop claim",
        "baseline_case": baseline_case.name,
        "analysis_window": [ANALYSIS_START, END_TIME],
        "baseline_total_cd": zero_drag,
        "baseline_rear_cl_fluctuation_rms": zero_fluct,
        "locked_thresholds": {
            "min_total_cd_reduction": MIN_DRAG_REDUCTION,
            "max_rear_cl_fluctuation_ratio": MAX_FLUCTUATION_RATIO,
            "max_abs_mean_rear_cl_over_zero_fluct_rms": MAX_MEAN_LIFT_RATIO,
        },
        "trajectory_count": len(rows),
        "joint_pass_count": sum(row["joint_pass"] for row in rows),
        "drag_only_pass_count": sum(row["checks"]["total_cd_reduction_at_least_2pct"] for row in rows),
        "cases": rows,
        "interpretation_guard": "retrospective one-phase coarse-grid screen; passing candidate is a baseline to beat, not a validated final policy",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "joint_pass_count": result["joint_pass_count"],
                      "drag_only_pass_count": result["drag_only_pass_count"],
                      "top_five": [{k: row[k] for k in ("case", "drag_reduction", "rear_cl_fluctuation_ratio", "abs_mean_rear_cl_over_zero_fluct_rms", "joint_pass")}
                                   for row in rows[:5]]}))


if __name__ == "__main__":
    main()
