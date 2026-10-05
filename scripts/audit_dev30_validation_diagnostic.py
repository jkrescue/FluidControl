#!/usr/bin/env python3
"""Audit frozen-blind dev30 FNO validation as a development diagnostic only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

PROFILE = "matched_start_full40_v1"
RELEASE_KIND = "immutable_development_train20_validation10"
PREDECLARATION_SHA256 = "d7ff174ef10194a8739357376335ca13ff9b45c8079970846bb71f15a715d24b"
FORCE_CHANNELS = ["front_cd", "front_cl", "rear_cd", "rear_cl"]
HORIZONS = (1, 10, 50, 100)
P013_KIND = "fcp013_independent_force_dual_fno"
ACTIONS = {"m075": -0.75, "m0375": -0.375, "zero": 0.0, "p0375": 0.375, "p075": 0.75}
CASE = re.compile(
    r"matched_start_acquisition_validation_b(01|05)_"
    r"(m075|m0375|zero|p0375|p075)"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def sign(value: float, tolerance: float = 1.0e-12) -> int:
    return int(value > tolerance) - int(value < -tolerance)


def validate_release(data: Path) -> tuple[dict[str, tuple[str, float]], dict]:
    manifest_path = data / "manifest.json"
    manifest = load(manifest_path)
    if (
        manifest.get("profile") != PROFILE
        or manifest.get("release_kind") != RELEASE_KIND
        or manifest.get("full40_predeclaration_sha256") != PREDECLARATION_SHA256
        or manifest.get("materialized_trajectory_counts")
        != {"train": 20, "validation": 10}
        or manifest.get("frozen_test_materialized") is not False
        or manifest.get("frozen_test_directory_present") is not False
        or not math.isclose(float(manifest.get("max_abs_omega", -1.0)), 0.75)
    ):
        raise ValueError("dev30 release identity differs")
    if (data / "frozen_test").exists():
        raise ValueError("dev30 release unexpectedly exposes frozen_test")
    declared = manifest.get("split_manifests", {}).get("validation", {})
    relative = Path(declared.get("path", ""))
    split_path = data / relative
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not split_path.is_file()
        or sha256(split_path) != declared.get("sha256")
    ):
        raise ValueError("validation split manifest differs")
    split = load(split_path)
    names = split.get("cases")
    hashes = split.get("hdf5_sha256")
    if (
        split.get("split") != "validation"
        or not isinstance(names, list)
        or len(names) != 10
        or names != sorted(names)
        or not isinstance(hashes, dict)
        or set(hashes) != set(names)
    ):
        raise ValueError("validation10 split contract differs")
    expected: dict[str, tuple[str, float]] = {}
    for name in names:
        match = CASE.fullmatch(name)
        if match is None:
            raise ValueError(f"unexpected validation case name: {name}")
        phase, action_name = match.groups()
        expected[name] = (phase, ACTIONS[action_name])
        path = data / "validation" / f"{name}.h5"
        if not path.is_file() or sha256(path) != hashes[name]:
            raise ValueError(f"validation HDF SHA differs: {name}")
    for phase in ("01", "05"):
        actions = {action for found, action in expected.values() if found == phase}
        if actions != set(ACTIONS.values()):
            raise ValueError(f"validation phase b{phase} action panel differs")
    normalization = data / "normalization.json"
    if not normalization.is_file() or sha256(normalization) != manifest.get(
        "normalization_sha256"
    ):
        raise ValueError("dev30 normalization SHA differs")
    return expected, {
        "profile": PROFILE,
        "release_kind": RELEASE_KIND,
        "data_manifest_sha256": sha256(manifest_path),
        "validation_split_manifest_sha256": sha256(split_path),
        "normalization_sha256": sha256(normalization),
        "validation_cases": names,
    }


def validate_report_contract(
    report: dict, expected: dict[str, tuple[str, float]], *,
    candidate_kind: str | None = None, checkpoint_dir: Path | None = None,
) -> None:
    checkpoint_alias = "/workspace/checkpoint"
    if candidate_kind == P013_KIND:
        if checkpoint_dir is None:
            raise ValueError("P013 diagnostic requires the actual dual checkpoint")
        from fluid_control.dual_fno import validate_dual_fno_manifest
        identity = validate_dual_fno_manifest(
            checkpoint_dir.resolve().parent / "dual_model_manifest.json"
        )
        metadata = report.get("checkpoint_metadata")
        if (identity.aerodynamic.directory != checkpoint_dir.resolve()
                or report.get("checkpoint_epoch") != 1
                or not isinstance(metadata, dict)
                or metadata.get("dual_fno") is not True):
            raise ValueError("P013 diagnostic checkpoint role/epoch differs")
        expected_metadata = {
            "manifest_sha256": identity.manifest_sha256,
            "flow_model_sha256": identity.flow.model_sha256,
            "flow_state_sha256": identity.flow.state_sha256,
            "aerodynamic_model_sha256": identity.aerodynamic.model_sha256,
            "aerodynamic_state_sha256": identity.aerodynamic.state_sha256,
        }
        if any(metadata.get(key) != value for key, value in expected_metadata.items()):
            raise ValueError("P013 diagnostic report dual identity differs")
        checkpoint_alias = "/workspace/dual/aerodynamic"
    if (
        report.get("split") != "validation"
        or report.get("action_mode") != "observed"
        or report.get("evaluation_data") != "/workspace/devdata"
        or report.get("normalization_data") != "/workspace/devdata"
        or report.get("checkpoint_dir") != checkpoint_alias
        or report.get("force_channels") != FORCE_CHANNELS
        or float(report.get("action_scale", -1.0)) != 0.75
        or float(report.get("evaluation_action_limit", -1.0)) != 0.75
    ):
        raise ValueError("evaluation runtime contract differs")
    cases = report.get("cases")
    if not isinstance(cases, list) or len(cases) != 10:
        raise ValueError("evaluation must contain validation10")
    if {row.get("case") for row in cases} != set(expected):
        raise ValueError("evaluation case names differ from validation10 manifest")
    for row in cases:
        if set(row.get("horizons", {})) != {str(value) for value in HORIZONS}:
            raise ValueError(f"evaluation horizons differ: {row.get('case')}")


def pooled_metrics(report: dict, expected: dict[str, tuple[str, float]]) -> dict:
    results = {}
    for horizon in HORIZONS:
        rows = []
        for case in report["cases"]:
            metric = case["horizons"][str(horizon)]
            segments = metric.get("segments")
            if (
                not isinstance(segments, int)
                or isinstance(segments, bool)
                or segments < 1
                or metric.get("stable") is not True
                or metric.get("failed_segments") != 0
            ):
                raise ValueError(
                    f"unstable or empty H{horizon}: {case.get('case')}"
                )
            rows.append(
                {
                    "case": case["case"],
                    "segments": segments,
                    "rmse": finite(metric.get("total_drag_rmse"), "total Cd RMSE"),
                    "target_rms": finite(
                        metric.get("total_drag_target_rms"), "total Cd target RMS"
                    ),
                    "mae": finite(metric.get("total_drag_mae"), "total Cd MAE"),
                    "persistence_mae": finite(
                        metric.get("persistence_total_drag_mae"),
                        "persistence total Cd MAE",
                    ),
                    "front_cl_mae": finite(metric.get("front_cl_mae"), "front Cl MAE"),
                    "rear_cl_mae": finite(metric.get("rear_cl_mae"), "rear Cl MAE"),
                }
            )
        if {row["case"] for row in rows} != set(expected):
            raise ValueError(f"H{horizon} cases differ")
        count = sum(row["segments"] for row in rows)
        error_ss = sum(row["segments"] * row["rmse"] ** 2 for row in rows)
        target_ss = sum(
            row["segments"] * row["target_rms"] ** 2 for row in rows
        )

        def weighted(
            key: str, metric_rows: list[dict] = rows, segment_count: int = count
        ) -> float:
            return (
                sum(row["segments"] * row[key] for row in metric_rows)
                / segment_count
            )

        results[str(horizon)] = {
            "segments": count,
            "pooled_total_cd_nrmse": math.sqrt(error_ss / target_ss),
            "macro_total_cd_nrmse": sum(
                row["rmse"] / row["target_rms"] for row in rows
            )
            / len(rows),
            "worst_case_total_cd_nrmse": max(
                row["rmse"] / row["target_rms"] for row in rows
            ),
            "pooled_total_cd_mae": weighted("mae"),
            "pooled_persistence_total_cd_mae": weighted("persistence_mae"),
            "beats_persistence": weighted("mae") < weighted("persistence_mae"),
            "front_cl_mae": weighted("front_cl_mae"),
            "rear_cl_mae": weighted("rear_cl_mae"),
        }
    return results


def strict_start0_ranking(
    segments: dict, expected: dict[str, tuple[str, float]]
) -> dict:
    if segments.get("split") != "validation" or segments.get("action_mode") != "observed":
        raise ValueError("segments must be observed-action validation")
    rows = {}
    for row in segments.get("segments", []):
        if row.get("horizon") != 100 or row.get("start") != 0:
            continue
        name = row.get("case")
        if name not in expected or name in rows:
            raise ValueError(f"unexpected or duplicate H100 start0 case: {name}")
        rows[name] = {
            "predicted": finite(row.get("predicted_total_drag"), "predicted total Cd"),
            "target": finite(row.get("target_total_drag"), "target total Cd"),
        }
    if set(rows) != set(expected):
        raise ValueError("H100 start0 records do not contain validation10")
    phase_results = {}
    zero_pairs = []
    ranking_pairs = []
    for phase in ("01", "05"):
        panel = sorted(
            (action, name)
            for name, (found, action) in expected.items()
            if found == phase
        )
        zero_name = next(name for action, name in panel if action == 0.0)
        phase_zero = []
        for action, name in panel:
            if action == 0.0:
                continue
            true_delta = rows[name]["target"] - rows[zero_name]["target"]
            predicted_delta = rows[name]["predicted"] - rows[zero_name]["predicted"]
            item = {
                "action_relative_to_zero": action,
                "true_delta_cd": true_delta,
                "predicted_delta_cd": predicted_delta,
                "absolute_delta_error": abs(predicted_delta - true_delta),
                "sign_correct": None
                if sign(true_delta) == 0
                else sign(true_delta) == sign(predicted_delta),
            }
            phase_zero.append(item)
            zero_pairs.append(item)
        phase_rank = []
        for left_index, (left_action, left_name) in enumerate(panel):
            for right_action, right_name in panel[left_index + 1 :]:
                true_delta = rows[right_name]["target"] - rows[left_name]["target"]
                predicted_delta = (
                    rows[right_name]["predicted"] - rows[left_name]["predicted"]
                )
                item = {
                    "actions": [left_action, right_action],
                    "true_cd_difference": true_delta,
                    "predicted_cd_difference": predicted_delta,
                    "ordering_correct": None
                    if sign(true_delta) == 0
                    else sign(true_delta) == sign(predicted_delta),
                }
                phase_rank.append(item)
                ranking_pairs.append(item)
        phase_results[f"b{phase}"] = {
            "zero_relative_pairs": phase_zero,
            "cross_action_ordering_pairs": phase_rank,
        }
    sign_rows = [row for row in zero_pairs if row["sign_correct"] is not None]
    order_rows = [row for row in ranking_pairs if row["ordering_correct"] is not None]
    if not sign_rows or not order_rows:
        raise ValueError("all action comparisons are true ties")
    return {
        "scope": "H100 instantaneous endpoint at start=0 common initial state only",
        "strict_start0_cases": 10,
        "zero_relative_pairs": len(zero_pairs),
        "pairwise_delta_cd_mae": sum(
            row["absolute_delta_error"] for row in zero_pairs
        )
        / len(zero_pairs),
        "zero_relative_sign_accuracy": sum(
            bool(row["sign_correct"]) for row in sign_rows
        )
        / len(sign_rows),
        "cross_action_pairs": len(ranking_pairs),
        "cross_action_ordering_accuracy": sum(
            bool(row["ordering_correct"]) for row in order_rows
        )
        / len(order_rows),
        "true_ties_excluded": {
            "zero_relative": len(zero_pairs) - len(sign_rows),
            "cross_action": len(ranking_pairs) - len(order_rows),
        },
        "phases": phase_results,
        "start_gt_zero_policy": (
            "excluded from action ranking: elapsed times after start=0 have action-diverged states"
        ),
    }


def validate_checkpoint(
    report: dict,
    checkpoint_dir: Path,
    *,
    allow_calibrated_epoch_zero: bool = False,
    expected_calibrated_model_sha256: str | None = None,
    expected_calibrated_state_sha256: str | None = None,
    expected_calibrated_kind: str = "FC_P008_TRAIN_ONLY_FORCE_ROW_CANDIDATE",
) -> dict:
    epoch = report.get("checkpoint_epoch")
    from fluid_control.calibrated_checkpoint import validate_calibrated_epoch_zero

    identity = validate_calibrated_epoch_zero(
        checkpoint_dir,
        epoch,
        allow=allow_calibrated_epoch_zero,
        expected_model_sha256=expected_calibrated_model_sha256,
        expected_state_sha256=expected_calibrated_state_sha256,
        expected_kind=expected_calibrated_kind,
    )
    models = sorted(checkpoint_dir.glob("FNO.*.mdlus"))
    if len(models) != 1:
        raise ValueError("checkpoint must contain exactly one PhysicsNeMo FNO model")
    result = {
        "checkpoint_epoch": epoch,
        "checkpoint_model_file": models[0].name,
        "checkpoint_sha256": sha256(models[0]),
    }
    if identity["calibrated_epoch_zero"]:
        result.update(identity)
    return result


def audit(
    report_path: Path,
    segments_path: Path,
    data: Path,
    checkpoint_dir: Path,
    candidate_kind: str,
    *,
    allow_calibrated_epoch_zero: bool = False,
    expected_calibrated_model_sha256: str | None = None,
    expected_calibrated_state_sha256: str | None = None,
    expected_calibrated_kind: str = "FC_P008_TRAIN_ONLY_FORCE_ROW_CANDIDATE",
) -> dict:
    expected, release = validate_release(data)
    report = load(report_path)
    validate_report_contract(report, expected, candidate_kind=candidate_kind,
                             checkpoint_dir=checkpoint_dir)
    horizons = pooled_metrics(report, expected)
    ranking = strict_start0_ranking(load(segments_path), expected)
    return {
        "status": "DEV30_VALIDATION_DIAGNOSTIC_COMPLETE",
        "scope": "development validation10 only; frozen test unavailable and not accessed",
        "split": "validation",
        "formal_gate": False,
        "formal_gate_authorized": False,
        "ppo_authorized": False,
        "candidate_kind": candidate_kind,
        "stage_candidate_only": candidate_kind
        == "dev30_quickscreen_h20_stage_candidate",
        "frozen_test_accessed_or_mounted": False,
        "horizons": horizons,
        "h100_start0_action_ranking": ranking,
        **validate_checkpoint(
            report,
            checkpoint_dir,
            allow_calibrated_epoch_zero=allow_calibrated_epoch_zero,
            expected_calibrated_model_sha256=expected_calibrated_model_sha256,
            expected_calibrated_state_sha256=expected_calibrated_state_sha256,
            expected_calibrated_kind=expected_calibrated_kind,
        ),
        **release,
        "interpretation": (
            "These terminal-force diagnostics support model development only. They are not "
            "the formal full40 Gate, do not establish time-window reward fidelity, and do not "
            "prove real-CFD closed-loop benefit."
        ),
        "report_sha256": sha256(report_path),
        "segments_sha256": sha256(segments_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--candidate-kind",
        choices=(
            "dev30_free_ar_development",
            "dev30_h20_development",
            "dev30_quickscreen_h20_stage_candidate",
            "fc_p008_force_row_calibrated_epoch0",
            "fc_p009_joint_force_row_calibrated_epoch0",
            P013_KIND,
        ),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-calibrated-epoch-zero", action="store_true")
    parser.add_argument("--expected-calibrated-model-sha256")
    parser.add_argument("--expected-calibrated-state-sha256")
    parser.add_argument("--expected-calibrated-kind", choices=("FC_P008_TRAIN_ONLY_FORCE_ROW_CANDIDATE", "FC_P009_TRAIN_ONLY_JOINT_FORCE_ROW_CANDIDATE"), default="FC_P008_TRAIN_ONLY_FORCE_ROW_CANDIDATE")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    calibrated = {}
    if args.allow_calibrated_epoch_zero:
        calibrated = {
            "allow_calibrated_epoch_zero": True,
            "expected_calibrated_model_sha256": args.expected_calibrated_model_sha256,
            "expected_calibrated_state_sha256": args.expected_calibrated_state_sha256,
            "expected_calibrated_kind": args.expected_calibrated_kind,
        }
    result = audit(
        args.report,
        args.segments,
        args.data,
        args.checkpoint_dir,
        args.candidate_kind,
        **calibrated,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "horizons": result["horizons"]}, indent=2))


if __name__ == "__main__":
    main()
