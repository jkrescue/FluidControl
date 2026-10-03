#!/usr/bin/env python3
"""Audit and compare the frozen matched-start OpenFOAM action panel."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from analyze_baseline import load_coefficients, log_health, window_stats
from make_control_landscape_panel import (
    CASES,
    END_TIME,
    PANEL,
    SOURCE_CASE,
    START_TIME,
    file_sha256,
)
from validate_phase_diverse_cfd import validate_case

ANALYSIS_START = 92.0
BLOCK_EDGES = (92.0, 98.0, 104.0, 110.0, 116.0)


def total_drag_by_time(
    front: list[tuple[float, float, float]],
    rear: list[tuple[float, float, float]],
    begin: float,
    end: float,
) -> list[tuple[float, float]]:
    if len(front) != len(rear):
        raise ValueError("front and rear force series have different lengths")
    rows = []
    for (time_front, cd_front, _), (time_rear, cd_rear, _) in zip(front, rear):
        if not math.isclose(time_front, time_rear, abs_tol=1e-8):
            raise ValueError("front and rear force timestamps are not paired")
        if begin <= time_front <= end:
            rows.append((time_front, cd_front + cd_rear))
    if len(rows) < 3 or not math.isclose(rows[-1][0], end, abs_tol=1e-8):
        raise ValueError("paired force window is incomplete")
    return rows


def block_drag_means(rows: list[tuple[float, float]]) -> list[float]:
    means = []
    for index, (begin, end) in enumerate(zip(BLOCK_EDGES, BLOCK_EDGES[1:])):
        values = [
            drag for time, drag in rows
            if begin <= time < end or (index == len(BLOCK_EDGES) - 2 and time == end)
        ]
        if len(values) < 100:
            raise ValueError("insufficient force samples in shedding-period block")
        means.append(statistics.mean(values))
    return means


def action_effort(points: list[list[float]]) -> dict[str, float]:
    full = [(float(time), float(omega)) for time, omega in points]
    window = [(time, omega) for time, omega in full if ANALYSIS_START <= time <= END_TIME]
    if len(window) < 3:
        raise ValueError("action window is incomplete")
    omega_rms = math.sqrt(statistics.mean(value * value for _, value in window))
    rates = [(b[1] - a[1]) / (b[0] - a[0]) for a, b in zip(window, window[1:])]
    full_rates = [(b[1] - a[1]) / (b[0] - a[0]) for a, b in zip(full, full[1:])]
    return {
        "analysis_window_omega_rms": omega_rms,
        "analysis_window_domega_dt_rms": math.sqrt(statistics.mean(value * value for value in rates)),
        "full_horizon_omega_rms": math.sqrt(statistics.mean(value * value for _, value in full)),
        "full_horizon_domega_dt_rms": math.sqrt(statistics.mean(value * value for value in full_rates)),
        "note": "kinematic effort proxies, not actuator torque or physical work",
    }


def analyze_one(name: str) -> dict:
    case = CASES / name
    cfg = json.loads((case / "case_config.json").read_text())
    if cfg["panel"] != "tandem_control_landscape_validation_20261003" or cfg["split"] != "validation":
        raise ValueError(f"wrong panel or split: {name}")
    if cfg["source_restart_u_sha256"] != file_sha256(SOURCE_CASE / "80" / "U"):
        raise ValueError(f"source U hash mismatch: {name}")
    if cfg["source_restart_p_sha256"] != file_sha256(SOURCE_CASE / "80" / "p"):
        raise ValueError(f"source p hash mismatch: {name}")
    raw = validate_case(case)
    health = log_health(case / "log.pimpleFoam")
    if (not health["solver_ended_cleanly"] or health["steps"] != 7200
            or health["max_courant"] >= 1
            or health["max_abs_global_continuity_per_step"] >= 1e-5):
        raise ValueError(f"solver-health gate failed: {name}: {health}")
    front = load_coefficients(case / "postProcessing" / "forceFront" / "80" / "coefficient.dat")
    rear = load_coefficients(case / "postProcessing" / "forceRear" / "80" / "coefficient.dat")
    drag = total_drag_by_time(front, rear, ANALYSIS_START, END_TIME)
    blocks = block_drag_means(drag)
    return {
        "case": name,
        "action_kind": cfg["action_kind"],
        "source_restart_u_sha256": cfg["source_restart_u_sha256"],
        "source_restart_p_sha256": cfg["source_restart_p_sha256"],
        "analysis_window": [ANALYSIS_START, END_TIME],
        "total_cd_mean": statistics.mean(value for _, value in drag),
        "total_cd_block_means": blocks,
        "total_cd_block_std": statistics.stdev(blocks),
        "front": window_stats(front, ANALYSIS_START, END_TIME),
        "rear": window_stats(rear, ANALYSIS_START, END_TIME),
        "effort": action_effort(cfg["action_points"]),
        "numerical_health": health,
        "raw_case_audit": raw,
    }


def compare(rows: list[dict]) -> dict:
    baseline = next(row for row in rows if row["action_kind"] == "zero")
    denominator = baseline["total_cd_mean"]
    lift_denominator = baseline["rear"]["cl_rms"]
    rear_root_mean_square_baseline = math.hypot(
        baseline["rear"]["cl_rms"], baseline["rear"]["cl_mean"]
    )
    if denominator <= 0 or lift_denominator <= 0:
        raise ValueError("invalid zero-action baseline")
    for row in rows:
        row["relative_to_zero"] = {
            "total_cd_change_percent": 100 * (row["total_cd_mean"] / denominator - 1),
            "rear_cl_rms_ratio": row["rear"]["cl_rms"] / lift_denominator,
            "rear_cl_abs_mean_ratio": row["rear"]["cl_abs_mean"] / baseline["rear"]["cl_abs_mean"],
            "rear_cl_root_mean_square_ratio": math.hypot(
                row["rear"]["cl_rms"], row["rear"]["cl_mean"]
            ) / rear_root_mean_square_baseline,
            "front_cl_rms_ratio": row["front"]["cl_rms"] / baseline["front"]["cl_rms"],
        }
    return {
        "status": "EXPLORATORY_MATCHED_CFD_PANEL_OK",
        "scientific_scope": "paired open-loop physical response, not learned closed-loop drag reduction",
        "solver": "OpenFOAM v2512 pimpleFoam, coarse 19290-cell mesh, dt=0.005",
        "source_restart": "uncontrolled t=80; identical U/p hashes in each case",
        "analysis_window": [ANALYSIS_START, END_TIME],
        "cases": rows,
        "interpretation_guard": "report drag, lift and actuation proxies together; no torque/work measurement",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    rows = [analyze_one(spec.name) for spec in PANEL]
    report = compare(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "cases": [
        {"case": row["case"], "total_cd_mean": row["total_cd_mean"],
         **row["relative_to_zero"]} for row in rows
    ]}))


if __name__ == "__main__":
    main()
