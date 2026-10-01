#!/usr/bin/env python3
"""Audit a fixed nonzero CFD action schedule against a matched zero-action run."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from fluid_control.openfoam_observation import observation_at  # noqa: E402


def compare(cases_root: Path, action_result: Path, zero_result: Path) -> dict:
    action = json.loads(action_result.read_text(encoding="utf-8"))
    zero = json.loads(zero_result.read_text(encoding="utf-8"))
    if action.get("status") != "REAL_OPENFOAM_DIAGNOSTIC_ACTION_SCHEDULE_OK":
        raise ValueError("action CFD result incomplete")
    if zero.get("status") != "REAL_OPENFOAM_SEGMENTED_ZERO_OBSERVATION_SMOKE_OK":
        raise ValueError("zero CFD reference incomplete")
    if action["source_restart_case"] != zero["source_restart_case"]:
        raise ValueError("different restart source")
    if len(action["rows"]) != len(zero["rows"]) or len(action["rows"]) < 3:
        raise ValueError("different or insufficient run lengths")
    if action["actions"][0] != 0 or not any(omega != 0 for omega in action["actions"][1:]):
        raise ValueError("not a warm-up followed by nonzero action")
    rows = []
    for expected_step, (a, z) in enumerate(zip(action["rows"], zero["rows"]), 1):
        if a["step"] != expected_step or z["step"] != expected_step:
            raise ValueError("nonconsecutive steps")
        if not math.isclose(a["end_time"], z["end_time"], abs_tol=1e-8):
            raise ValueError("unmatched comparison times")
        if a["applied_omega"] != action["actions"][expected_step - 1] or z["applied_omega"] != 0:
            raise ValueError("wrong action or zero reference")
        time = a["end_time"]
        action_obs, _ = observation_at(cases_root / action["case"], time, a["applied_omega"])
        zero_obs, _ = observation_at(cases_root / zero["case"], time, 0.0)
        rows.append({
            "step": expected_step,
            "time": time,
            "applied_omega": a["applied_omega"],
            "probe_max_abs_difference": float(np.max(np.abs(action_obs[:64] - zero_obs[:64]))),
            "rear_cd_difference": float(action_obs[64] - zero_obs[64]),
            "rear_cl_difference": float(action_obs[65] - zero_obs[65]),
            "action_max_courant": a["solver"]["max_courant"],
            "action_max_abs_global_continuity_per_step": a["solver"]["max_abs_global_continuity_per_step"],
        })
    warmup_matched = (
        rows[0]["probe_max_abs_difference"] <= 1e-5
        and abs(rows[0]["rear_cd_difference"]) <= 1e-5
        and abs(rows[0]["rear_cl_difference"]) <= 1e-5
    )
    action_detected = any(
        row["probe_max_abs_difference"] > 1e-7
        and abs(row["rear_cl_difference"]) > 1e-4
        for row in rows[1:]
    )
    return {
        "status": "NONZERO_ACTION_REACHES_REAL_CFD" if warmup_matched and action_detected else "ACTION_DIAGNOSTIC_FAILED",
        "scientific_scope": "short_fixed_schedule_boundary_actuation_only_not_rl_or_control_benefit",
        "action_case": action["case"],
        "zero_reference_case": zero["case"],
        "source_restart_case": action["source_restart_case"],
        "warmup_matched": warmup_matched,
        "action_detected_in_probe_and_rear_lift": action_detected,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-root", type=Path, required=True)
    parser.add_argument("--action-result", type=Path, required=True)
    parser.add_argument("--zero-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = compare(args.cases_root, args.action_result, args.zero_result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "NONZERO_ACTION_REACHES_REAL_CFD":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
