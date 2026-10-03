#!/usr/bin/env python3
"""Audit H100 FNO ranking of paired -0.75/+0.75 validation actions.

The two inputs are segment exports for a fixed parent and candidate model.  The
script reads no CFD files and makes no zero-control or closed-loop claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

HORIZON = 100
ACTION_MAGNITUDE = 0.75
ACTION_ATOL = 1.0e-6
VALUE_ATOL = 1.0e-7
RANKING_ATOL = 1.0e-12
ROLE_TOKENS = {"negative": "_m_phase94_", "positive": "_p_phase94_"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def sign(value: float) -> int:
    if value > RANKING_ATOL:
        return 1
    if value < -RANKING_ATOL:
        return -1
    return 0


def preferred_action(delta_negative_minus_positive: float) -> str:
    direction = sign(delta_negative_minus_positive)
    if direction < 0:
        return "negative_-0.75"
    if direction > 0:
        return "positive_+0.75"
    return "tie"


def load_h100(path: Path) -> dict[str, dict[int, dict]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError(f"{path}: expected JSON object")
    if document.get("split") != "validation":
        raise ValueError(f"{path}: split must be validation")
    if document.get("action_mode") != "observed":
        raise ValueError(f"{path}: action_mode must be observed")
    segments = document.get("segments")
    if not isinstance(segments, list):
        raise TypeError(f"{path}: segments must be a list")

    grouped: dict[str, dict[int, dict]] = {role: {} for role in ROLE_TOKENS}
    cases: dict[str, set[str]] = {role: set() for role in ROLE_TOKENS}
    action_maxima: dict[str, list[float]] = {role: [] for role in ROLE_TOKENS}
    for index, row in enumerate(segments):
        if not isinstance(row, dict):
            raise TypeError(f"{path}: segment {index} is not an object")
        if row.get("horizon") != HORIZON:
            continue
        case = row.get("case")
        if not isinstance(case, str):
            raise TypeError(f"{path}: segment {index} case is invalid")
        matches = [role for role, token in ROLE_TOKENS.items() if token in case]
        if len(matches) != 1:
            raise ValueError(f"{path}: cannot identify action role for {case}")
        role = matches[0]
        start = row.get("start")
        if isinstance(start, bool) or not isinstance(start, int) or start < 0:
            raise ValueError(f"{path}: invalid start for {case}")
        if start in grouped[role]:
            raise ValueError(f"{path}: duplicate {role} start {start}")
        predicted = finite(row.get("predicted_total_drag"), "predicted_total_drag")
        target = finite(row.get("target_total_drag"), "target_total_drag")
        absolute_error = finite(
            row.get("total_drag_absolute_error"), "total_drag_absolute_error"
        )
        if not math.isclose(
            absolute_error, abs(predicted - target), rel_tol=1.0e-6, abs_tol=1.0e-7
        ):
            raise ValueError(f"{path}: inconsistent total-drag error at {case}/{start}")
        max_abs_omega = finite(row.get("max_abs_omega"), "max_abs_omega")
        if max_abs_omega < 0 or max_abs_omega > ACTION_MAGNITUDE + ACTION_ATOL:
            raise ValueError(f"{path}: action support exceeds 0.75 at {case}/{start}")
        cases[role].add(case)
        action_maxima[role].append(max_abs_omega)
        grouped[role][start] = {
            "case": case,
            "start": start,
            "predicted_total_cd": predicted,
            "true_total_cd": target,
            "prediction_error": predicted - target,
        }

    for role in ROLE_TOKENS:
        if len(cases[role]) != 1 or not grouped[role]:
            raise ValueError(f"{path}: expected exactly one nonempty {role} H100 case")
        if not math.isclose(
            max(action_maxima[role]),
            ACTION_MAGNITUDE,
            rel_tol=0.0,
            abs_tol=ACTION_ATOL,
        ):
            raise ValueError(f"{path}: {role} case never reaches action magnitude 0.75")
    if set(grouped["negative"]) != set(grouped["positive"]):
        raise ValueError(f"{path}: positive/negative H100 start sets differ")
    return grouped


def audit_model(grouped: dict[str, dict[int, dict]]) -> dict:
    pairs = []
    for start in sorted(grouped["negative"]):
        negative = grouped["negative"][start]
        positive = grouped["positive"][start]
        predicted_delta = (
            negative["predicted_total_cd"] - positive["predicted_total_cd"]
        )
        true_delta = negative["true_total_cd"] - positive["true_total_cd"]
        true_sign = sign(true_delta)
        predicted_sign = sign(predicted_delta)
        ranking_correct = None if true_sign == 0 else predicted_sign == true_sign
        pairs.append(
            {
                "start": start,
                "negative_-0.75": negative,
                "positive_+0.75": positive,
                "predicted_cd_negative_minus_positive": predicted_delta,
                "true_cd_negative_minus_positive": true_delta,
                "pairwise_delta_error": predicted_delta - true_delta,
                "predicted_preferred_action": preferred_action(predicted_delta),
                "true_preferred_action": preferred_action(true_delta),
                "ranking_correct": ranking_correct,
            }
        )
    evaluable = [row for row in pairs if row["ranking_correct"] is not None]
    if not evaluable:
        raise ValueError("all true action pairs are ties; ranking accuracy is undefined")
    strict_initial = next((row for row in pairs if row["start"] == 0), None)
    if strict_initial is None:
        raise ValueError("strict common-initial-state pair at start=0 is missing")
    return {
        "horizon_steps": HORIZON,
        "strict_common_initial_state_start": 0,
        "strict_common_initial_state_pair": strict_initial,
        "strict_common_initial_pairwise_absolute_error": abs(
            strict_initial["pairwise_delta_error"]
        ),
        "strict_common_initial_ranking_correct": strict_initial["ranking_correct"],
        "all_starts_interpretation": (
            "matched elapsed time across action-diverged trajectories; starts >0 "
            "are different physical states and are not counterfactual action rankings"
        ),
        "paired_starts": len(pairs),
        "ranking_evaluable_starts": len(evaluable),
        "true_tie_starts_excluded_from_ranking": len(pairs) - len(evaluable),
        "pairwise_delta_mae": sum(
            abs(row["pairwise_delta_error"]) for row in pairs
        )
        / len(pairs),
        "matched_elapsed_time_sign_ranking_accuracy_diagnostic": sum(
            bool(row["ranking_correct"]) for row in evaluable
        )
        / len(evaluable),
        "negative_action_prediction_mae": sum(
            abs(row["negative_-0.75"]["prediction_error"]) for row in pairs
        )
        / len(pairs),
        "positive_action_prediction_mae": sum(
            abs(row["positive_+0.75"]["prediction_error"]) for row in pairs
        )
        / len(pairs),
        "pairs": pairs,
    }


def verify_common_truth(
    parent: dict[str, dict[int, dict]], candidate: dict[str, dict[int, dict]]
) -> None:
    for role in ROLE_TOKENS:
        if set(parent[role]) != set(candidate[role]):
            raise ValueError(f"parent/candidate {role} start sets differ")
        for start in parent[role]:
            left = parent[role][start]["true_total_cd"]
            right = candidate[role][start]["true_total_cd"]
            if not math.isclose(left, right, rel_tol=0.0, abs_tol=VALUE_ATOL):
                raise ValueError(
                    f"parent/candidate CFD target differs for {role} start {start}"
                )


def analyze(parent_path: Path, candidate_path: Path) -> dict:
    parent_rows = load_h100(parent_path)
    candidate_rows = load_h100(candidate_path)
    verify_common_truth(parent_rows, candidate_rows)
    parent = audit_model(parent_rows)
    candidate = audit_model(candidate_rows)
    return {
        "status": "LOW_ACTION_PAIRWISE_FNO_H100_AUDIT_COMPLETE",
        "split": "validation",
        "actions": [-ACTION_MAGNITUDE, ACTION_MAGNITUDE],
        "delta_definition": "Cd_total(-0.75) - Cd_total(+0.75); lower is better",
        "parent_segments": str(parent_path),
        "parent_segments_sha256": sha256(parent_path),
        "candidate_segments": str(candidate_path),
        "candidate_segments_sha256": sha256(candidate_path),
        "models": {"v3_parent": parent, "v4_candidate": candidate},
        "candidate_minus_parent": {
            "pairwise_delta_mae": (
                candidate["pairwise_delta_mae"] - parent["pairwise_delta_mae"]
            ),
            "matched_elapsed_time_sign_ranking_accuracy_diagnostic": (
                candidate["matched_elapsed_time_sign_ranking_accuracy_diagnostic"]
                - parent["matched_elapsed_time_sign_ranking_accuracy_diagnostic"]
            ),
        },
        "scientific_scope": (
            "validation-only H100 endpoint pairwise FNO diagnostic for two observed "
            "open-loop actions; not zero-control benefit and not closed-loop success"
        ),
        "limitations": [
            "No zero-action trajectory is read, so this cannot establish drag reduction.",
            "The metric is the H100 endpoint total Cd, not a 100-step time-window mean.",
            "For this phase-94 profile, start=0 H100 is an instantaneous endpoint near t=104; it cannot be compared with the [114,174] mean-Cd 2% physical-control gate.",
            "Matching start is elapsed-frame pairing across the two action trajectories; after actuation their physical states are distinct.",
            "Only start=0 is a strict same-initial-state -0.75 versus +0.75 action comparison; all-start sign agreement is descriptive only.",
            "Common t94 restart provenance and action sign are external dataset contracts not contained in these segment JSON files.",
            "Frozen test data are not accessed and this validation diagnostic must not select a model.",
        ],
    }


def write_exclusive(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-segments", type=Path, required=True)
    parser.add_argument("--candidate-segments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    result = analyze(args.parent_segments, args.candidate_segments)
    write_exclusive(args.output, result)
    print(
        json.dumps(
            {
                label: {
                    key: result["models"][label][key]
                    for key in (
                        "pairwise_delta_mae",
                        "matched_elapsed_time_sign_ranking_accuracy_diagnostic",
                        "strict_common_initial_ranking_correct",
                    )
                }
                for label in ("v3_parent", "v4_candidate")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
