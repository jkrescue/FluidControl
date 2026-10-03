#!/usr/bin/env python3
"""Check predeclared periodic OpenFOAM controls against locked physical gates."""

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
from make_periodic_rotation_benchmark import PANEL, audit, action_points
from validate_phase_diverse_cfd import validate_case


def analyze_periodic(name: str) -> dict:
    case = CASES / name
    cfg = json.loads((case / "case_config.json").read_text(encoding="utf-8"))
    if cfg["panel"] != "tandem_periodic_open_loop_benchmark_validation_20261003" or cfg["split"] != "validation":
        raise ValueError(f"wrong periodic panel or split: {name}")
    if cfg["period"] != PANEL[name] or cfg["action_points"] != [list(x) for x in action_points(PANEL[name])]:
        raise ValueError(f"periodic action table mismatch: {name}")
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
        "period": PANEL[name],
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


def evaluate(rows: list[dict]) -> dict:
    if len(rows) != 3:
        raise ValueError("expected one zero and two periodic cases")
    zero = rows[0]
    if zero["case"] != "landscape_long_val_zero_20261003":
        raise ValueError("wrong matched zero control")
    if len({(row["source_restart_u_sha256"], row["source_restart_p_sha256"]) for row in rows}) != 1:
        raise ValueError("different source restart")
    bounds = audit()["acceptance_from_RESEARCH_OBJECTIVE_md"]
    if zero["total_cd_mean"] <= 0 or zero["rear"]["cl_rms"] <= 0:
        raise ValueError("invalid zero denominator")
    decisions = []
    for row in rows[1:]:
        drag_reduction = 1 - row["total_cd_mean"] / zero["total_cd_mean"]
        rear_cl_fluct_ratio = row["rear"]["cl_rms"] / zero["rear"]["cl_rms"]
        mean_rear_ratio = abs(row["rear"]["cl_mean"]) / zero["rear"]["cl_rms"]
        checks = {
            "total_drag_reduction_at_least_2pct": drag_reduction >= bounds["total_cd_reduction_min"],
            "rear_cl_fluctuation_increase_at_most_5pct": rear_cl_fluct_ratio <= 1 + bounds["rear_cl_fluctuation_rms_increase_max"],
            "abs_mean_rear_cl_at_most_10pct_zero_fluct_rms": mean_rear_ratio <= bounds["abs_mean_rear_cl_over_zero_fluctuation_rms_max"],
        }
        decisions.append({
            "case": row["case"],
            "total_cd_reduction": drag_reduction,
            "rear_cl_fluctuation_rms_ratio": rear_cl_fluct_ratio,
            "abs_mean_rear_cl_over_zero_fluctuation_rms": mean_rear_ratio,
            "rear_cl_total_rms_ratio": math.hypot(row["rear"]["cl_mean"], row["rear"]["cl_rms"]) / math.hypot(zero["rear"]["cl_mean"], zero["rear"]["cl_rms"]),
            "checks": checks,
            "one_phase_coarse_grid_screen_pass": all(checks.values()),
        })
    return {
        "status": "PERIODIC_OPEN_LOOP_CFD_BENCHMARK_OK",
        "scientific_scope": "one-phase coarse-grid open-loop screen only; not full multi-phase Gate-D pass",
        "analysis_window": [ANALYSIS_START, END_TIME],
        "acceptance": bounds,
        "cases": rows,
        "decisions": decisions,
        "interpretation_guard": "phase/mesh validation and actuator torque are still required; block means are correlated time segments",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    zero = analyze_one("landscape_long_val_zero_20261003")
    report = evaluate([zero] + [analyze_periodic(name) for name in PANEL])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "decisions": report["decisions"]}))


if __name__ == "__main__":
    main()
