#!/usr/bin/env python3
"""Train-only trailing-window force-statistic diagnostic for FC-P009."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_fcp009_cache_result import (
    EXPECTED_IMPLEMENTATION_SHA,
    EXPECTED_STATUS,
    atomic_json,
    oof_predictions,
    sha256,
    validate_cache_schema,
    validate_normalization_binding,
)
from analyze_fcp009_joint_readout import fit_joint_coefficients, joint_oof_predictions
from build_fcp008_force_readout_candidate import MODEL_SHA, predict

P008_CACHE_SHA = "22d6c8b29bf8df27917b3e14d791ade07a8c18eeaefec12c7301d1a6b1bf2bff"
FORMAL_RECEIPT_SHA = "ac5c0dd047c90fddba884b77cd82bbe4f5147123d0f2f4455fb1b938607e231c"
WINDOW_COUNT = 1368
STEPS_PER_WINDOW = 100
TRAILING_FIRST_STEP = 39


def validate_formal_receipt(formal: dict[str, object], receipt_sha: str) -> None:
    """Bind the completed P009 formal bundle without reusing its science status."""
    if (
        receipt_sha != FORMAL_RECEIPT_SHA
        or formal.get("status") != "FC_P009_POSTEVAL_COMPLETE"
        or formal.get("ppo_auto_launched") is not False
        or formal.get("frozen_test_accessed") is not False
        or formal.get("ppo_authorized") is True
    ):
        raise ValueError("FC-P009 formal receipt differs")


def validate_window_layout(cache: dict[str, np.ndarray]) -> list[dict[str, object]]:
    """Return one identity record per exact 100-step training window."""
    rows = WINDOW_COUNT * STEPS_PER_WINDOW
    required = ("families", "case_names", "phases", "window_starts", "relative_steps")
    if any(np.asarray(cache[key]).shape != (rows,) for key in required):
        raise ValueError("window identity array shape differs")
    windows = []
    for index in range(WINDOW_COUNT):
        begin = index * STEPS_PER_WINDOW
        end = begin + STEPS_PER_WINDOW
        relative = np.asarray(cache["relative_steps"])[begin:end]
        if not np.array_equal(relative, np.arange(1, 101)):
            raise ValueError("window relative-step grid differs")
        identity = {}
        for key in ("families", "case_names", "phases", "window_starts"):
            values = np.asarray(cache[key])[begin:end]
            if np.any(values != values[0]):
                raise ValueError(f"{key} changes within a window")
            identity[key] = values[0].item() if hasattr(values[0], "item") else values[0]
        identity["window_index"] = index
        windows.append(identity)
    return windows


def force_window_statistics(force: np.ndarray) -> dict[str, float]:
    """Compute the prescribed trailing steps 39..100 physical statistics."""
    force = np.asarray(force, dtype=np.float64)
    if force.shape != (STEPS_PER_WINDOW, 4) or not np.isfinite(force).all():
        raise ValueError("physical force window differs")
    trailing = force[TRAILING_FIRST_STEP - 1 :]
    rear_cl = trailing[:, 3]
    return {
        "mean_total_cd": float(np.mean(trailing[:, 0] + trailing[:, 2])),
        "mean_rear_cl": float(np.mean(rear_cl)),
        "rear_cl_fluctuation_rms": float(np.sqrt(np.mean(np.square(rear_cl - np.mean(rear_cl))))),
    }


def window_error_rows(
    prediction: np.ndarray,
    target: np.ndarray,
    force_mean: np.ndarray,
    force_std: np.ndarray,
    windows: list[dict[str, object]],
) -> list[dict[str, object]]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if (
        prediction.shape != target.shape
        or target.shape != (len(windows) * STEPS_PER_WINDOW, 4)
    ):
        raise ValueError("prediction/target row contract differs")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("prediction/target contains nonfinite values")
    prediction = prediction * force_std + force_mean
    target = target * force_std + force_mean
    result = []
    for identity in windows:
        begin = int(identity["window_index"]) * STEPS_PER_WINDOW
        truth = force_window_statistics(target[begin : begin + STEPS_PER_WINDOW])
        estimate = force_window_statistics(prediction[begin : begin + STEPS_PER_WINDOW])
        result.append(
            {
                **identity,
                "truth": truth,
                "prediction": estimate,
                "signed_error": {key: estimate[key] - truth[key] for key in truth},
                "absolute_error": {key: abs(estimate[key] - truth[key]) for key in truth},
            }
        )
    return result


def aggregate_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    metrics = ("mean_total_cd", "mean_rear_cl", "rear_cl_fluctuation_rms")
    grouped: dict[str, dict[str, list[dict[str, object]]]] = {
        "pooled": {"all": rows},
        "family": defaultdict(list),
        "phase": defaultdict(list),
        "case": defaultdict(list),
    }
    for row in rows:
        grouped["family"][str(row["families"])].append(row)
        grouped["phase"][str(row["phases"])].append(row)
        grouped["case"][str(row["case_names"])].append(row)
    output = {}
    for group_name, members in grouped.items():
        output[group_name] = {}
        for label, subset in members.items():
            output[group_name][label] = {
                "window_count": len(subset),
                "unique_case_count": len({str(row["case_names"]) for row in subset}),
                "windows_are_not_independent_physical_samples": True,
                "metrics": {},
            }
            for metric in metrics:
                signed = np.asarray([row["signed_error"][metric] for row in subset], dtype=np.float64)
                output[group_name][label]["metrics"][metric] = {
                    "mean_absolute_error": float(np.mean(np.abs(signed))),
                    "root_mean_squared_error": float(np.sqrt(np.mean(np.square(signed)))),
                    "mean_bias": float(np.mean(signed)),
                    "maximum_absolute_error": float(np.max(np.abs(signed))),
                }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("result", "cache", "p008-cache", "normalization", "formal-receipt", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("exclusive output required")
    result = json.loads(args.result.read_text())
    formal = json.loads(args.formal_receipt.read_text())
    validate_formal_receipt(formal, sha256(args.formal_receipt))
    if (
        result.get("status") != EXPECTED_STATUS
        or result.get("feature_cache_sha256") != sha256(args.cache)
        or result.get("parent_model_sha256") != MODEL_SHA
        or result.get("input_sha256", {}).get("implementation") != EXPECTED_IMPLEMENTATION_SHA
        or result.get("candidate_saved") is not False
    ):
        raise ValueError("FC-P009 evidence binding differs")
    force_std = validate_normalization_binding(result, args.normalization).reshape(4)
    normalization = json.loads(args.normalization.read_text())
    force_mean = np.asarray(normalization["all_force_mean"], dtype=np.float64)
    if force_mean.shape != (4,) or force_std.shape != (4,) or not np.all(force_std > 0):
        raise ValueError("force normalization differs")

    cache = dict(np.load(args.cache, allow_pickle=False))
    free_ar, h1, targets, phases, _ = validate_cache_schema(cache)
    windows = validate_window_layout(cache)
    if sha256(args.p008_cache) != P008_CACHE_SHA:
        raise ValueError("P008 H1 cache SHA differs")
    p008 = dict(np.load(args.p008_cache, allow_pickle=False))
    indices = np.asarray(cache["matched_weight_h1_source_indices"], dtype=np.int64)
    parent_h1 = np.asarray(p008["parent_native_normalized"], dtype=np.float64)[indices]
    parent_ar = np.asarray(cache["parent_native_normalized"], dtype=np.float64)

    all_rows = np.ones(len(targets), dtype=bool)
    joint_coefficients, joint_fit = fit_joint_coefficients(free_ar, h1, targets, all_rows)
    joint_oof, joint_oof_fits = joint_oof_predictions(free_ar, h1, targets, phases)
    specialist_oof = oof_predictions(free_ar, h1, targets, phases)
    predictions = {
        "parent_native_free_ar": parent_ar,
        "parent_native_h1": parent_h1,
        "free_ar_specialist_full_fit_on_free_ar": predict(free_ar, cache["free_ar_alpha0_coefficients"]),
        "h1_specialist_full_fit_on_h1": predict(h1, cache["matched_weight_h1_alpha0_coefficients"]),
        "joint_full_fit_on_free_ar": predict(free_ar, joint_coefficients),
        "joint_full_fit_on_h1": predict(h1, joint_coefficients),
        "free_ar_specialist_phase_oof_on_free_ar": specialist_oof["fit_free_ar_predict_free_ar"],
        "h1_specialist_phase_oof_on_h1": specialist_oof["fit_h1_predict_h1"],
        "joint_phase_oof_on_free_ar": joint_oof["joint_fit_predict_free_ar"],
        "joint_phase_oof_on_h1": joint_oof["joint_fit_predict_h1"],
    }
    reports = {}
    for name, prediction in predictions.items():
        rows = window_error_rows(prediction, targets, force_mean, force_std, windows)
        reports[name] = {"aggregate": aggregate_rows(rows), "windows": rows}

    payload = {
        "status": "FC_P009_TRAIN_WINDOW_STATISTIC_DIAGNOSTIC_COMPLETE",
        "scope": "CPU-only train-cache mechanism diagnostic; no selection, candidate, validation fit, admission, or control evidence",
        "window_definition": {
            "window_count": WINDOW_COUNT,
            "steps_per_window": STEPS_PER_WINDOW,
            "trailing_relative_steps_inclusive": [TRAILING_FIRST_STEP, 100],
            "trailing_sample_count": 62,
            "rear_cl_rms_ddof": 0,
            "windows_are_not_independent_physical_samples": True,
        },
        "source_sha256": {
            "cache_result": sha256(args.result),
            "cache": sha256(args.cache),
            "p008_h1_cache": sha256(args.p008_cache),
            "normalization": sha256(args.normalization),
            "formal_receipt": sha256(args.formal_receipt),
        },
        "fit_contract": {"alpha": 0.0, "domain_mix": {"free_ar": 0.5, "matched_weight_h1": 0.5}},
        "joint_full_fit": joint_fit,
        "joint_phase_oof_fits": joint_oof_fits,
        "reports": reports,
        "gpu_used": False,
        "optimizer_steps": 0,
        "candidate_saved": False,
        "validation_used_for_fit_or_selection": False,
        "frozen_test_accessed": False,
        "ppo_executed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, payload)
    print(payload["status"], flush=True)


if __name__ == "__main__":
    main()
