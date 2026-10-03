#!/usr/bin/env python3
"""Fail-closed force audit for long paired OpenFOAM rotation cases."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from analyze_baseline import load_coefficients, log_health, window_stats
from analyze_control_landscape_panel import total_drag_by_time
from make_control_landscape_long_panel import ANALYSIS_START, END_TIME, LONG_PANEL
from make_control_landscape_panel import CASES, SOURCE_CASE, file_sha256
from validate_phase_diverse_cfd import validate_case


def block_means(rows: list[tuple[float, float]]) -> list[float]:
    edges = [ANALYSIS_START + index * (END_TIME - ANALYSIS_START) / 6 for index in range(7)]
    result = []
    for index, (begin, end) in enumerate(zip(edges, edges[1:])):
        values = [value for time, value in rows
                  if begin <= time < end or (index == 5 and time == end)]
        if len(values) < 500:
            raise ValueError("incomplete long-panel drag block")
        result.append(statistics.mean(values))
    return result


def full_effort(points: list[list[float]]) -> dict[str, float | str]:
    samples = [(float(time), float(omega)) for time, omega in points]
    rates = [(b[1] - a[1]) / (b[0] - a[0]) for a, b in zip(samples, samples[1:])]
    return {
        "omega_rms": math.sqrt(statistics.mean(omega * omega for _, omega in samples)),
        "domega_dt_rms": math.sqrt(statistics.mean(rate * rate for rate in rates)),
        "note": "kinematic proxies, not measured actuator torque or power",
    }


def analyze_one(name: str) -> dict:
    case = CASES / name
    cfg = json.loads((case / "case_config.json").read_text())
    if cfg["panel"] != "tandem_control_landscape_long_validation_20261003" or cfg["split"] != "validation":
        raise ValueError(f"wrong long panel or split: {name}")
    for field in ("U", "p"):
        if cfg[f"source_restart_{field.lower()}_sha256"] != file_sha256(SOURCE_CASE / "80" / field):
            raise ValueError(f"source {field} hash mismatch: {name}")
    raw = validate_case(case)
    health = log_health(case / "log.pimpleFoam")
    if (not health["solver_ended_cleanly"] or health["steps"] != 16000
            or health["max_courant"] >= 1
            or health["max_abs_global_continuity_per_step"] >= 1e-5):
        raise ValueError(f"solver-health gate failed: {name}: {health}")
    front = load_coefficients(case / "postProcessing" / "forceFront" / "80" / "coefficient.dat")
    rear = load_coefficients(case / "postProcessing" / "forceRear" / "80" / "coefficient.dat")
    drag = total_drag_by_time(front, rear, ANALYSIS_START, END_TIME)
    blocks = block_means(drag)
    return {
        "case": name,
        "omega_final": cfg["omega_final"],
        "source_restart_u_sha256": cfg["source_restart_u_sha256"],
        "source_restart_p_sha256": cfg["source_restart_p_sha256"],
        "analysis_window": [ANALYSIS_START, END_TIME],
        "total_cd_mean": statistics.mean(value for _, value in drag),
        "total_cd_block_means": blocks,
        "total_cd_block_std": statistics.stdev(blocks),
        "front": window_stats(front, ANALYSIS_START, END_TIME),
        "rear": window_stats(rear, ANALYSIS_START, END_TIME),
        "effort": full_effort(cfg["action_points"]),
        "numerical_health": health,
        "raw_case_audit": raw,
    }


def compare(rows: list[dict]) -> dict:
    baseline = next(row for row in rows if row["omega_final"] == 0.0)
    if baseline["total_cd_mean"] <= 0 or baseline["rear"]["cl_rms"] <= 0:
        raise ValueError("invalid long-panel zero baseline")
    rear_total_baseline = math.hypot(baseline["rear"]["cl_mean"], baseline["rear"]["cl_rms"])
    for row in rows:
        row["relative_to_zero"] = {
            "total_cd_change_percent": 100 * (row["total_cd_mean"] / baseline["total_cd_mean"] - 1),
            "rear_cl_fluctuation_rms_ratio": row["rear"]["cl_rms"] / baseline["rear"]["cl_rms"],
            "rear_cl_total_rms_ratio": math.hypot(row["rear"]["cl_mean"], row["rear"]["cl_rms"]) / rear_total_baseline,
            "rear_cl_abs_mean_ratio": row["rear"]["cl_abs_mean"] / baseline["rear"]["cl_abs_mean"],
        }
    return {
        "status": "EXPLORATORY_LONG_MATCHED_CFD_PANEL_OK",
        "scientific_scope": "long paired open-loop response; not learned closed-loop benefit",
        "solver": "OpenFOAM v2512 pimpleFoam, coarse 19290-cell mesh, dt=0.005",
        "source_restart": "uncontrolled t=80, identical source U/p hashes",
        "analysis_window": [ANALYSIS_START, END_TIME],
        "cases": rows,
        "interpretation_guard": "separate drag, fluctuating lift, total lift and kinematic effort; no wall torque measured",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    report = compare([analyze_one(spec.name) for spec in LONG_PANEL])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "cases": [
        {"case": row["case"], "total_cd_mean": row["total_cd_mean"],
         **row["relative_to_zero"]} for row in report["cases"]
    ]}))


if __name__ == "__main__":
    main()
