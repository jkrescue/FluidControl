#!/usr/bin/env python3
"""Summarize all 20 predeclared train branches against same-phase zero CFD.

Only train cases are opened. Validation and frozen-test case results and
receipts are neither enumerated nor read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from summarize_full40_train_increment_physics import (
    EXPECTED_ANALYSIS_SAMPLES,
    FORCE_DT,
    FULL40_AUTHORIZATION_SHA256,
    FULL40_PREDECLARATION_SHA256,
    PHASE_MANIFEST_SHA256,
    branch_metrics,
    comparison,
    read_force_rows,
    read_json,
    select_exact_window,
    sha256,
    write_json_exclusive_atomic,
)

PHASES = (0, 2, 4, 6)
ACTIONS = (-0.75, -0.375, 0.0, 0.375, 0.75)
CONTROL_ACTIONS = tuple(action for action in ACTIONS if action != 0.0)
ACTION_LABELS = {
    -0.75: "m075",
    -0.375: "m0375",
    0.0: "zero",
    0.375: "p0375",
    0.75: "p075",
}
NINE_QC_STATUS = "MATCHED_START_9_CASE_COMMISSIONING_QC_PASS"


def case_name(phase: int, action: float) -> str:
    return f"matched_start_acquisition_train_b{phase:02d}_{ACTION_LABELS[action]}"


EXACT_TRAIN_CASES = tuple(
    case_name(phase, action) for phase in PHASES for action in ACTIONS
)
EXISTING_NINE = {
    case_name(phase, action)
    for phase in (0, 2, 4)
    for action in (-0.75, 0.0, 0.75)
}
NEW_TRAIN_ELEVEN = set(EXACT_TRAIN_CASES) - EXISTING_NINE


def select_train_matrix(predeclaration: dict) -> dict[str, dict]:
    cases = predeclaration.get("cases", {})
    train = {name: row for name, row in cases.items() if row.get("split") == "train"}
    if set(train) != set(EXACT_TRAIN_CASES) or len(train) != 20:
        raise ValueError("predeclaration train matrix is not the exact 4x5 panel")
    expected = {
        case_name(phase, action): (phase, action)
        for phase in PHASES
        for action in ACTIONS
    }
    for name, (phase, action) in expected.items():
        row = train[name]
        expected_disposition = (
            "existing_nine_case_commissioning"
            if name in EXISTING_NINE
            else "planned_new_remainder_case"
        )
        if (
            row.get("phase_bin") != phase
            or float(row.get("action_target")) != action
            or row.get("disposition") != expected_disposition
        ):
            raise ValueError(f"predeclared train identity differs: {name}")
    return train


def validate_receipt(
    repo: Path,
    name: str,
    row: dict,
    config: dict,
    existing_receipt_dir: Path,
    extension_receipt_dir: Path,
) -> Path:
    existing = name in EXISTING_NINE
    receipt_path = (
        existing_receipt_dir if existing else extension_receipt_dir
    ) / f"{name}.json"
    receipt = read_json(receipt_path)
    expected_status = "RAW_TRANSFER_VERIFIED" if existing else "FULL40_RAW_TRANSFER_VERIFIED"
    if (
        receipt.get("status") != expected_status
        or receipt.get("case") != name
    ):
        raise ValueError(f"strict train receipt differs: {name}")
    if existing and receipt.get("phase_manifest_sha256") != PHASE_MANIFEST_SHA256:
        raise ValueError(f"existing-nine receipt phase provenance differs: {name}")
    if not existing and (
        receipt.get("split") != "train"
        or receipt.get("phase_bin") != row["phase_bin"]
        or float(receipt.get("action_target")) != float(row["action_target"])
        or receipt.get("full40_predeclaration_sha256")
        != FULL40_PREDECLARATION_SHA256
        or receipt.get("full40_extension_authorization_sha256")
        != FULL40_AUTHORIZATION_SHA256
        or receipt.get("source_state_sha256") != row.get("source_state_sha256")
    ):
        raise ValueError(f"full40 extension train receipt differs: {name}")
    manifest = repo / str(receipt.get("worker_raw_manifest", ""))
    if (
        not manifest.is_file()
        or sha256(manifest) != receipt.get("worker_raw_manifest_sha256")
        or config.get("source_state_sha256") != row.get("source_state_sha256")
    ):
        raise ValueError(f"raw manifest or matched-start provenance differs: {name}")
    return receipt_path


def load_train_panel(
    repo: Path,
    cases_dir: Path,
    predeclaration_path: Path,
    nine_aggregate_path: Path,
    existing_receipt_dir: Path,
    extension_receipt_dir: Path,
) -> tuple[dict[str, tuple[Path, dict]], dict]:
    if sha256(predeclaration_path) != FULL40_PREDECLARATION_SHA256:
        raise ValueError("full40 predeclaration SHA-256 differs")
    predeclaration = read_json(predeclaration_path)
    if predeclaration.get("phase_manifest_sha256") != PHASE_MANIFEST_SHA256:
        raise ValueError("phase manifest SHA differs in full40 predeclaration")
    matrix = select_train_matrix(predeclaration)

    nine = read_json(nine_aggregate_path)
    nine_names = {item.get("case") for item in nine.get("cases", [])}
    if (
        nine.get("status") != NINE_QC_STATUS
        or nine.get("complete_nine_case_panel") is not True
        or nine.get("phase_manifest_sha256") != PHASE_MANIFEST_SHA256
        or nine_names != EXISTING_NINE
        or set(nine.get("transfer", {})) != EXISTING_NINE
    ):
        raise ValueError("existing nine-case aggregate QC differs")

    panel = {}
    for name in EXACT_TRAIN_CASES:
        row = matrix[name]
        case = cases_dir / name
        config = read_json(case / "case_config.json")
        begin, end = (float(value) for value in config.get("analysis_window", []))
        if (
            config.get("case") != name
            or config.get("split") != "train"
            or config.get("phase_bin") != row["phase_bin"]
            or float(config.get("action_target")) != float(row["action_target"])
            or config.get("phase_manifest_sha256") != PHASE_MANIFEST_SHA256
            or config.get("source_state_sha256") != row["source_state_sha256"]
            or config.get("action_points") != row["action_points"]
            or [float(config.get("start_time")), float(config.get("end_time"))]
            != [float(value) for value in row["run_window"]]
            or [begin, end] != [float(value) for value in row["analysis_window"]]
            or not np.isclose(end - begin, 60.0)
        ):
            raise ValueError(f"raw train config differs from predeclaration: {name}")
        receipt_path = validate_receipt(
            repo,
            name,
            row,
            config,
            existing_receipt_dir,
            extension_receipt_dir,
        )
        panel[name] = (case, config, receipt_path)
    return panel, matrix


def _macro_and_worst(rows: list[dict]) -> dict:
    drag_key = "total_drag_reduction_fraction_positive_is_better"
    fluctuation_key = "rear_cl_fluctuation_rms_ratio_to_zero"
    bias_key = "abs_mean_rear_cl_over_zero_fluctuation_rms"
    return {
        "comparison_count": len(rows),
        "macro_mean_total_drag_reduction_fraction": float(
            np.mean([row[drag_key] for row in rows])
        ),
        "macro_mean_rear_cl_fluctuation_ratio": float(
            np.mean([row[fluctuation_key] for row in rows])
        ),
        "macro_mean_abs_rear_cl_bias_ratio": float(
            np.mean([row[bias_key] for row in rows])
        ),
        "joint_pass_count": sum(row["canonical_joint_diagnostic_pass"] for row in rows),
        "worst_drag_reduction": min(rows, key=lambda row: row[drag_key]),
        "worst_rear_cl_fluctuation_ratio": max(
            rows, key=lambda row: row[fluctuation_key]
        ),
        "worst_abs_rear_cl_bias_ratio": max(rows, key=lambda row: row[bias_key]),
    }


def build_report(
    repo: Path,
    cases_dir: Path,
    predeclaration_path: Path,
    nine_aggregate_path: Path,
    existing_receipt_dir: Path,
    extension_receipt_dir: Path,
) -> dict:
    panel, _ = load_train_panel(
        repo,
        cases_dir,
        predeclaration_path,
        nine_aggregate_path,
        existing_receipt_dir,
        extension_receipt_dir,
    )
    phases = {}
    by_action = {ACTION_LABELS[action]: [] for action in CONTROL_ACTIONS}
    all_comparisons = []
    for phase in PHASES:
        zero_name = case_name(phase, 0.0)
        zero_case, zero_config, _ = panel[zero_name]
        begin, end = (float(value) for value in zero_config["analysis_window"])
        zero = branch_metrics(
            select_exact_window(read_force_rows(zero_case, "forceFront"), begin, end),
            select_exact_window(read_force_rows(zero_case, "forceRear"), begin, end),
        )
        controls = {}
        phase_rows = []
        for action in CONTROL_ACTIONS:
            name = case_name(phase, action)
            case, config, receipt_path = panel[name]
            control_begin, control_end = map(float, config["analysis_window"])
            if (control_begin, control_end) != (begin, end):
                raise ValueError(f"same-phase analysis windows differ: {name}")
            metrics = branch_metrics(
                select_exact_window(
                    read_force_rows(case, "forceFront"), control_begin, control_end
                ),
                select_exact_window(
                    read_force_rows(case, "forceRear"), control_begin, control_end
                ),
            )
            result = {"case": name, "phase_bin": phase, "action_target": action}
            result.update(comparison(metrics, zero))
            controls[ACTION_LABELS[action]] = {
                "case": name,
                "receipt": str(receipt_path),
                "metrics": metrics,
                "same_phase_zero_comparison": result,
            }
            phase_rows.append(result)
            by_action[ACTION_LABELS[action]].append(result)
            all_comparisons.append(result)
        phases[f"b{phase:02d}"] = {
            "split": "train",
            "analysis_window": [begin, end],
            "same_phase_zero": {"case": zero_name, "metrics": zero},
            "control_branches": controls,
            "phase_summary": _macro_and_worst(phase_rows),
        }

    return {
        "status": "FULL40_TRAIN20_OPEN_LOOP_PHYSICS_SUMMARY",
        "scope": {
            "split": "train",
            "phase_bins": list(PHASES),
            "case_count_including_zero": 20,
            "same_phase_control_comparison_count": 16,
            "excluded_splits": ["validation", "frozen_test"],
            "validation_or_frozen_results_read": False,
        },
        "provenance": {
            "full40_predeclaration": str(predeclaration_path),
            "full40_predeclaration_sha256": FULL40_PREDECLARATION_SHA256,
            "phase_manifest_sha256": PHASE_MANIFEST_SHA256,
            "existing_nine_aggregate_qc": str(nine_aggregate_path),
            "existing_nine_aggregate_qc_sha256": sha256(nine_aggregate_path),
            "strict_existing_receipt_count": 9,
            "strict_extension_train_receipt_count": 11,
        },
        "metric_contract": {
            "analysis_window": "predeclared final 60D/U for every branch",
            "force_sample_interval": FORCE_DT,
            "samples_per_branch": EXPECTED_ANALYSIS_SAMPLES,
            "reference": "same-phase zero-action branch only",
            "total_drag": "mean(Cd_front + Cd_rear)",
            "canonical_joint_gate": (
                "drag reduction >=2%; rear Cl-prime RMS <=1.05x same-phase zero; "
                "abs(mean rear Cl) <=0.10x same-phase zero Cl-prime RMS"
            ),
        },
        "phases": phases,
        "per_action_macro_and_worst": {
            label: _macro_and_worst(rows) for label, rows in by_action.items()
        },
        "all_train_control_macro_and_worst": _macro_and_worst(all_comparisons),
        "interpretation_guards": [
            "All results are prescribed open-loop train-split CFD diagnostics.",
            "No validation or frozen-test result or receipt was opened.",
            "Train macro/worst summaries are not generalization or closed-loop evidence.",
            "These summaries must not tune thresholds or select on frozen-test outcomes.",
        ],
    }


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases-dir", type=Path, default=repo / "cfd/tandem_cylinders/cases"
    )
    parser.add_argument(
        "--predeclaration",
        type=Path,
        default=(
            repo
            / "artifacts/tandem_cylinders/matched_start_full40_predeclared_20261003.json"
        ),
    )
    parser.add_argument(
        "--nine-aggregate",
        type=Path,
        default=repo / "artifacts/matched_start_acquisition/aggregate_qc/result.json",
    )
    parser.add_argument(
        "--existing-receipts",
        type=Path,
        default=repo / "artifacts/matched_start_acquisition/transfer_verified",
    )
    parser.add_argument(
        "--extension-receipts",
        type=Path,
        default=repo / "artifacts/matched_start_full40_extension/transfer_verified",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            repo
            / "artifacts/matched_start_full40_extension/train20_physics_summary.json"
        ),
    )
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to((repo / "artifacts").resolve()):
        parser.error("output must be inside the repository artifacts directory")
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    report = build_report(
        repo,
        args.cases_dir,
        args.predeclaration,
        args.nine_aggregate,
        args.existing_receipts,
        args.extension_receipts,
    )
    write_json_exclusive_atomic(args.output, report)
    summary = report["all_train_control_macro_and_worst"]
    print(
        json.dumps(
            {
                "status": report["status"],
                "comparisons": summary["comparison_count"],
                "joint_pass_count": summary["joint_pass_count"],
                "macro_drag_reduction": summary[
                    "macro_mean_total_drag_reduction_fraction"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
