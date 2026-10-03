#!/usr/bin/env python3
"""Validation-only H100 force and matched-start action-difference gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

MAX_H100_POOLED_TOTAL_CD_NRMSE = 0.10
MAX_STRICT_ZERO_RELATIVE_DELTA_CD_MAE = 0.023
MIN_STRICT_ZERO_RELATIVE_SIGN_ACCURACY = 1.0
MIN_STRICT_CROSS_ACTION_ORDERING_ACCURACY = 1.0
PREDECLARATION_SHA256 = "d7ff174ef10194a8739357376335ca13ff9b45c8079970846bb71f15a715d24b"
ACTIONS = {"m075": -0.75, "m0375": -0.375, "zero": 0.0, "p0375": 0.375, "p075": 0.75}
CASE = re.compile(r"matched_start_acquisition_validation_b(01|05)_(m075|m0375|zero|p0375|p075)")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def sign(value: float, tolerance: float = 1e-12) -> int:
    return int(value > tolerance) - int(value < -tolerance)


def validate_predeclaration(path: Path) -> dict[str, tuple[str, float]]:
    if sha256(path) != PREDECLARATION_SHA256:
        raise ValueError("full40 predeclaration SHA differs")
    cases = load(path).get("cases", {})
    expected = {}
    for name, row in cases.items():
        match = CASE.fullmatch(name)
        if not match:
            continue
        phase, action_name = match.groups()
        action = ACTIONS[action_name]
        if row.get("split") != "validation" or float(row.get("action_target")) != action:
            raise ValueError(f"predeclared validation identity differs: {name}")
        expected[name] = (phase, action)
    if len(expected) != 10:
        raise ValueError("expected exactly ten b01/b05 validation cases")
    for phase in ("01", "05"):
        phase_rows = [name for name, (found, _) in expected.items() if found == phase]
        source_states = {
            json.dumps(cases[name].get("source_state_sha256"), sort_keys=True)
            for name in phase_rows
        }
        if len(phase_rows) != 5 or len(source_states) != 1:
            raise ValueError(f"phase b{phase} is not a five-action matched start")
    return expected


def pooled_h100(report: dict, expected: dict[str, tuple[str, float]]) -> dict:
    if report.get("split") != "validation" or report.get("action_mode") != "observed":
        raise ValueError("report must be observed-action validation only")
    rows = []
    for case in report.get("cases", []):
        name = case.get("case")
        if name not in expected:
            raise ValueError(f"unexpected validation case: {name}")
        metric = case.get("horizons", {}).get("100", {})
        segments = metric.get("segments")
        if not isinstance(segments, int) or segments < 1:
            raise ValueError(f"invalid H100 segments: {name}")
        if metric.get("stable") is not True or metric.get("failed_segments") != 0:
            raise ValueError(f"unstable H100 rollout: {name}")
        rows.append(
            {
                "case": name,
                "segments": segments,
                "rmse": finite(metric.get("total_drag_rmse"), "total drag RMSE"),
                "target_rms": finite(metric.get("total_drag_target_rms"), "target RMS"),
                "front_cl_mae": finite(metric.get("front_cl_mae"), "front Cl MAE"),
                "rear_cl_mae": finite(metric.get("rear_cl_mae"), "rear Cl MAE"),
            }
        )
    if len(rows) != 10 or {row["case"] for row in rows} != set(expected):
        raise ValueError("report does not contain exactly validation10")
    error_ss = sum(row["segments"] * row["rmse"] ** 2 for row in rows)
    target_ss = sum(row["segments"] * row["target_rms"] ** 2 for row in rows)
    pooled = math.sqrt(error_ss / target_ss)
    count = sum(row["segments"] for row in rows)
    return {
        "pooled_total_cd_nrmse": pooled,
        "macro_total_cd_nrmse": sum(row["rmse"] / row["target_rms"] for row in rows) / 10,
        "worst_case_total_cd_nrmse": max(row["rmse"] / row["target_rms"] for row in rows),
        "front_cl_mae": sum(row["segments"] * row["front_cl_mae"] for row in rows) / count,
        "rear_cl_mae": sum(row["segments"] * row["rear_cl_mae"] for row in rows) / count,
        "segments": count,
        "passes_fixed_10pct_gate": pooled <= MAX_H100_POOLED_TOTAL_CD_NRMSE,
    }


def strict_start0_differences(segments: dict, expected: dict[str, tuple[str, float]]) -> dict:
    if segments.get("split") != "validation" or segments.get("action_mode") != "observed":
        raise ValueError("segments must be observed-action validation only")
    rows = {}
    for row in segments.get("segments", []):
        if row.get("horizon") != 100 or row.get("start") != 0:
            continue
        name = row.get("case")
        if name not in expected or name in rows:
            raise ValueError(f"unexpected or duplicate strict start0 case: {name}")
        rows[name] = {
            "predicted": finite(row.get("predicted_total_drag"), "predicted total Cd"),
            "target": finite(row.get("target_total_drag"), "target total Cd"),
        }
    if set(rows) != set(expected):
        raise ValueError("strict start0 does not contain validation10")
    phases = {}
    all_pairs = []
    all_ordering_pairs = []
    for phase in ("01", "05"):
        phase_cases = sorted(
            (expected[name][1], name) for name in rows if expected[name][0] == phase
        )
        pairs = []
        zero_name = next(name for action, name in phase_cases if action == 0.0)
        for action, name in phase_cases:
            if action == 0.0:
                continue
            true_delta = rows[name]["target"] - rows[zero_name]["target"]
            predicted_delta = rows[name]["predicted"] - rows[zero_name]["predicted"]
            pairs.append(
                {
                    "action_relative_to_zero": action,
                    "true_delta_cd": true_delta,
                    "predicted_delta_cd": predicted_delta,
                    "absolute_delta_error": abs(predicted_delta - true_delta),
                    "sign_correct": None if sign(true_delta) == 0 else sign(true_delta) == sign(predicted_delta),
                }
            )
        ordering_pairs = []
        for left_index, (left_action, left_name) in enumerate(phase_cases):
            for right_action, right_name in phase_cases[left_index + 1 :]:
                true_difference = rows[right_name]["target"] - rows[left_name]["target"]
                predicted_difference = (
                    rows[right_name]["predicted"] - rows[left_name]["predicted"]
                )
                ordering_pairs.append(
                    {
                        "actions": [left_action, right_action],
                        "true_cd_difference": true_difference,
                        "predicted_cd_difference": predicted_difference,
                        "ordering_correct": None
                        if sign(true_difference) == 0
                        else sign(true_difference) == sign(predicted_difference),
                    }
                )
        scale = 0.02 * abs(rows[zero_name]["target"])
        phases[f"b{phase}"] = {
            "zero_relative_pairs": pairs,
            "cross_action_ordering_pairs": ordering_pairs,
            "two_percent_zero_cd_scale": scale,
        }
        all_pairs.extend(pairs)
        all_ordering_pairs.extend(ordering_pairs)
    evaluable = [row for row in all_pairs if row["sign_correct"] is not None]
    evaluable_ordering = [
        row for row in all_ordering_pairs if row["ordering_correct"] is not None
    ]
    if not evaluable or not evaluable_ordering:
        raise ValueError("all strict action differences are exact true ties")
    mean_scale = sum(row["two_percent_zero_cd_scale"] for row in phases.values()) / 2
    delta_mae = sum(row["absolute_delta_error"] for row in all_pairs) / len(all_pairs)
    sign_accuracy = sum(bool(row["sign_correct"]) for row in evaluable) / len(evaluable)
    ordering_accuracy = sum(
        bool(row["ordering_correct"]) for row in evaluable_ordering
    ) / len(evaluable_ordering)
    return {
        "scope": "H100 instantaneous endpoint; exactly start=0 common initial states",
        "strict_pairs": len(all_pairs),
        "pairwise_delta_cd_mae": delta_mae,
        "zero_relative_sign_accuracy": sign_accuracy,
        "true_ties_excluded": len(all_pairs) - len(evaluable),
        "cross_action_ordering_pairs": len(all_ordering_pairs),
        "cross_action_ordering_accuracy": ordering_accuracy,
        "cross_action_true_ties_excluded": len(all_ordering_pairs)
        - len(evaluable_ordering),
        "mean_two_percent_zero_cd_scale": mean_scale,
        "predeclared_delta_cd_mae_maximum": MAX_STRICT_ZERO_RELATIVE_DELTA_CD_MAE,
        "predeclared_sign_accuracy_minimum": MIN_STRICT_ZERO_RELATIVE_SIGN_ACCURACY,
        "predeclared_cross_action_ordering_accuracy_minimum": MIN_STRICT_CROSS_ACTION_ORDERING_ACCURACY,
        "passes_delta_cd_mae_gate": delta_mae <= MAX_STRICT_ZERO_RELATIVE_DELTA_CD_MAE,
        "passes_sign_ranking_gate": sign_accuracy >= MIN_STRICT_ZERO_RELATIVE_SIGN_ACCURACY,
        "passes_cross_action_ordering_gate": ordering_accuracy
        >= MIN_STRICT_CROSS_ACTION_ORDERING_ACCURACY,
        "resolves_two_percent_drag_scale_diagnostic": delta_mae <= mean_scale,
        "phases": phases,
        "start_gt_zero_policy": "excluded: action-diverged states are not strict counterfactuals",
    }


def audit(report_path: Path, segments_path: Path, predeclaration: Path) -> dict:
    expected = validate_predeclaration(predeclaration)
    force = pooled_h100(load(report_path), expected)
    action = strict_start0_differences(load(segments_path), expected)
    joint = (
        force["passes_fixed_10pct_gate"]
        and action["passes_delta_cd_mae_gate"]
        and action["passes_sign_ranking_gate"]
        and action["passes_cross_action_ordering_gate"]
    )
    return {
        "status": "FULL40_VALIDATION_SURROGATE_READINESS_PASS" if joint else "FULL40_VALIDATION_SURROGATE_READINESS_FAIL",
        "split": "validation",
        "frozen_test_accessed": False,
        "h100_force_gate": force,
        "h100_start0_action_difference": action,
        "joint_terminal_readiness": joint,
        "hydrogym_policy": "even joint terminal readiness permits only the next validation/window-lift study; it does not prove PPO or physical control benefit",
        "lift_interpretation": "front/rear Cl endpoint MAE is reported but has no invented pass threshold; Cl-prime <=1.05 and mean-Cl bias <=0.10 require a separate validation time-window fidelity audit and final real CFD",
        "report_sha256": sha256(report_path),
        "segments_sha256": sha256(segments_path),
        "predeclaration_sha256": sha256(predeclaration),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--predeclaration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = audit(args.report, args.segments, args.predeclaration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], **result["h100_force"]}, indent=2))


if __name__ == "__main__":
    main()
