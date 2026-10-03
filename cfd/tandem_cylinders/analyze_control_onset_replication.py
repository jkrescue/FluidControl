#!/usr/bin/env python3
"""Fail-closed, same-window OpenFOAM comparison of instant and ramped onset."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from analyze_baseline import load_coefficients, log_health, window_stats
from analyze_control_landscape_long_panel import analyze_one, block_means, full_effort
from analyze_control_landscape_panel import total_drag_by_time
from make_control_landscape_long_panel import ANALYSIS_START, END_TIME
from make_control_landscape_panel import CASES, SOURCE_CASE, file_sha256
from make_control_onset_replication import PANEL
from validate_phase_diverse_cfd import validate_case


def analyze_step(name: str) -> dict:
    case = CASES / name
    cfg = json.loads((case / "case_config.json").read_text(encoding="utf-8"))
    if cfg["panel"] != "tandem_rotation_onset_replication_validation_20261003" or cfg["split"] != "validation":
        raise ValueError(f"wrong onset panel or split: {name}")
    if cfg["omega_final"] != PANEL[name] or cfg["action_points"][0] != [80.0, PANEL[name]]:
        raise ValueError(f"onset schedule mismatch: {name}")
    for field in ("U", "p"):
        if cfg[f"source_restart_{field.lower()}_sha256"] != file_sha256(SOURCE_CASE / "80" / field):
            raise ValueError(f"source {field} hash mismatch: {name}")
    raw = validate_case(case)
    health = log_health(case / "log.pimpleFoam")
    if (not health["solver_ended_cleanly"] or health["steps"] != 16000
            or health["max_courant"] >= 1
            or health["max_abs_global_continuity_per_step"] >= 1e-5):
        raise ValueError(f"solver health failed: {name}: {health}")
    front = load_coefficients(case / "postProcessing/forceFront/80/coefficient.dat")
    rear = load_coefficients(case / "postProcessing/forceRear/80/coefficient.dat")
    drag = total_drag_by_time(front, rear, ANALYSIS_START, END_TIME)
    return {
        "case": name,
        "onset": "instant",
        "omega_final": PANEL[name],
        "source_restart_u_sha256": cfg["source_restart_u_sha256"],
        "source_restart_p_sha256": cfg["source_restart_p_sha256"],
        "analysis_window": [ANALYSIS_START, END_TIME],
        "total_cd_mean": statistics.mean(value for _, value in drag),
        "total_cd_block_means": block_means(drag),
        "front": window_stats(front, ANALYSIS_START, END_TIME),
        "rear": window_stats(rear, ANALYSIS_START, END_TIME),
        "effort": full_effort(cfg["action_points"]),
        "numerical_health": health,
        "raw_case_audit": raw,
    }


def compare(rows: list[dict]) -> dict:
    if len(rows) != 5:
        raise ValueError("expected zero, two ramps, two instant-onset cases")
    indexed = {(row["onset"], row["omega_final"]): row for row in rows}
    zero = indexed[("ramp", 0.0)]
    if len({(row["source_restart_u_sha256"], row["source_restart_p_sha256"]) for row in rows}) != 1:
        raise ValueError("matched comparison does not share the same restart")
    comparisons = []
    for omega in (-1.0, 1.0):
        ramp = indexed[("ramp", omega)]
        step = indexed[("instant", omega)]
        reference = zero["rear"]["cl_rms"]
        if min(reference, ramp["rear"]["cl_rms"], ramp["total_cd_mean"]) <= 0:
            raise ValueError("invalid zero/ramp denominator")
        comparisons.append({
            "omega": omega,
            "instant_vs_ramp_rear_cl_fluctuation_rms_ratio": step["rear"]["cl_rms"] / ramp["rear"]["cl_rms"],
            "ramp_vs_zero_rear_cl_fluctuation_rms_ratio": ramp["rear"]["cl_rms"] / reference,
            "instant_vs_zero_rear_cl_fluctuation_rms_ratio": step["rear"]["cl_rms"] / reference,
            "instant_vs_ramp_total_cd_change_percent": 100 * (step["total_cd_mean"] / ramp["total_cd_mean"] - 1),
            "instant_vs_zero_total_cd_change_percent": 100 * (step["total_cd_mean"] / zero["total_cd_mean"] - 1),
            "instant_vs_ramp_rear_total_cl_rms_ratio": math.hypot(step["rear"]["cl_mean"], step["rear"]["cl_rms"]) / math.hypot(ramp["rear"]["cl_mean"], ramp["rear"]["cl_rms"]),
            "instant_vs_ramp_block_cd_difference": [a - b for a, b in zip(step["total_cd_block_means"], ramp["total_cd_block_means"])],
        })
    return {
        "status": "MATCHED_CFD_ROTATION_ONSET_REPLICATION_OK",
        "scientific_scope": "one-phase coarse-grid open-loop onset comparison, not a controlled-policy or general bistability claim",
        "analysis_window": [ANALYSIS_START, END_TIME],
        "cases": rows,
        "comparisons": comparisons,
        "interpretation_guard": "same restart and window; check cycle drift and repeat at independent phases before mechanism claims",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    refs = [analyze_one(name) for name in (
        "landscape_long_val_zero_20261003", "landscape_long_val_p100_20261003", "landscape_long_val_m100_20261003"
    )]
    for row in refs:
        row["onset"] = "ramp"
    report = compare(refs + [analyze_step(name) for name in PANEL])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "comparisons": report["comparisons"]}))


if __name__ == "__main__":
    main()
