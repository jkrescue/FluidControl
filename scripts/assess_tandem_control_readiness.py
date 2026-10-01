#!/usr/bin/env python3
"""Conservative offline PhysicsNeMo surrogate gate, not a CFD control result."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from validate_tandem_fno_stage import HORIZONS, verify_evaluation


def number(row: dict, key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"missing numeric {key}")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid {key}: {value}")
    return value


def assess(observed_path: Path, zero_path: Path) -> dict:
    observed_meta = verify_evaluation(observed_path, "observed")
    zero_meta = verify_evaluation(zero_path, "zero")
    observed = json.loads(observed_path.read_text(encoding="utf-8"))
    zero = json.loads(zero_path.read_text(encoding="utf-8"))
    cases = [item["case"] for item in observed["cases"]]
    if observed_meta["checkpoint_epoch"] != zero_meta["checkpoint_epoch"]:
        raise ValueError("checkpoint epoch mismatch")
    if cases != [item["case"] for item in zero["cases"]] or len(set(cases)) != 4:
        raise ValueError("held-out case mismatch or duplicate")
    if observed.get("segment_stride") != zero.get("segment_stride"):
        raise ValueError("segment stride mismatch")
    if observed.get("action_scale") != zero.get("action_scale"):
        raise ValueError("action scale mismatch")
    checks = {}
    for horizon in HORIZONS:
        actual = observed["summary"][horizon]
        altered = zero["summary"][horizon]
        if actual["segments"] != altered["segments"]:
            raise ValueError(f"segment count mismatch at {horizon}")
        field = number(actual, "state_mae_physical_units")
        force = number(actual, "rear_force_mae")
        reference_field = number(actual, "persistence_state_mae_physical_units")
        reference_force = number(actual, "persistence_rear_force_mae")
        zero_field = number(altered, "state_mae_physical_units")
        zero_force = number(altered, "rear_force_mae")
        checks[f"{horizon}step_field_vs_persistence"] = {
            "passed": field < reference_field, "mae": field, "reference_mae": reference_field}
        checks[f"{horizon}step_force_vs_persistence"] = {
            "passed": force < reference_force, "mae": force, "reference_mae": reference_force}
        checks[f"{horizon}step_force_action_sensitivity"] = {
            "passed": force < zero_force, "mae": force, "zero_input_mae": zero_force}
        if horizon == "50":
            checks["50step_field_action_sensitivity"] = {
                "passed": field < zero_field, "mae": field, "zero_input_mae": zero_field}
    passed = all(item["passed"] for item in checks.values())
    return {
        "status": "CANDIDATE_SURROGATE_SCREEN_PASS" if passed else "CANDIDATE_SURROGATE_SCREEN_FAIL",
        "scientific_scope": "offline_same_geometry_heldout_screen_not_control_benefit",
        "checkpoint_epoch": observed_meta["checkpoint_epoch"],
        "heldout_cases": cases,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observed", type=Path)
    parser.add_argument("zero", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    result = assess(args.observed, args.zero)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
