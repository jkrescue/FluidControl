#!/usr/bin/env python3
"""Fail-closed numerical/provenance QC for the train-only signed CFD pair."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from analyze_baseline import load_coefficients, log_health, window_stats
from make_control_landscape_panel import CASES, SOURCE_CASE, file_sha256
from make_signed_pulse_training_pair import END_TIME, NAMES, START_TIME, action_points
from validate_phase_diverse_cfd import validate_case


def validate_one(name: str) -> dict:
    case = CASES / name
    cfg = json.loads((case / "case_config.json").read_text(encoding="utf-8"))
    if cfg["dataset"] != "tandem_control_gap_targeted_train_v4" or cfg["split"] != "train":
        raise ValueError(f"wrong data profile or split: {name}")
    if cfg["sign"] != NAMES[name] or cfg["action_points"] != [list(x) for x in action_points(NAMES[name])]:
        raise ValueError(f"action table mismatch: {name}")
    for field in ("U", "p"):
        if cfg[f"source_restart_{field.lower()}_sha256"] != file_sha256(SOURCE_CASE / "82" / field):
            raise ValueError(f"source {field} mismatch: {name}")
    raw = validate_case(case)
    health = log_health(case / "log.pimpleFoam")
    if (not health["solver_ended_cleanly"] or health["steps"] != 16000
            or health["max_courant"] >= 1
            or health["max_abs_global_continuity_per_step"] >= 1e-5):
        raise ValueError(f"solver health failed: {name}: {health}")
    front = load_coefficients(case / "postProcessing/forceFront/82/coefficient.dat")
    rear = load_coefficients(case / "postProcessing/forceRear/82/coefficient.dat")
    return {
        "case": name,
        "split": "train",
        "sign": NAMES[name],
        "source_restart_u_sha256": cfg["source_restart_u_sha256"],
        "source_restart_p_sha256": cfg["source_restart_p_sha256"],
        "action_metrics": cfg["action_metrics"],
        "front_force": window_stats(front, START_TIME, END_TIME),
        "rear_force": window_stats(rear, START_TIME, END_TIME),
        "numerical_health": health,
        "raw_case_audit": raw,
    }


def compare(rows: list[dict]) -> dict:
    if len(rows) != 2 or {row["sign"] for row in rows} != {-1.0, 1.0}:
        raise ValueError("incomplete signed train pair")
    if len({(row["source_restart_u_sha256"], row["source_restart_p_sha256"]) for row in rows}) != 1:
        raise ValueError("signed train cases did not share source restart")
    if any(not math.isfinite(row["rear_force"]["cl_rms"]) for row in rows):
        raise ValueError("nonfinite force summary")
    return {
        "status": "SIGNED_PULSE_REAL_CFD_TRAIN_PAIR_OK",
        "dataset": "tandem_control_gap_targeted_train_v4",
        "split": "train_only_not_validation_or_frozen_test",
        "source_restart": "uncontrolled t=82",
        "time_window": [START_TIME, END_TIME],
        "cases": rows,
        "scientific_scope": "CFD quality/provenance gate only; not yet curated HDF5, improved FNO, active learning efficiency or control benefit",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    result = compare([validate_one(name) for name in NAMES])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "cases": [x["case"] for x in result["cases"]]}))


if __name__ == "__main__":
    main()
