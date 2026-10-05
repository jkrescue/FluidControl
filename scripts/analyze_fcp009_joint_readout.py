#!/usr/bin/env python3
"""CPU-only fixed 50/50 H1/free-AR joint-readout diagnostic for FC-P009."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_fcp009_cache_result import (
    EXPECTED_IMPLEMENTATION_SHA,
    EXPECTED_STATUS,
    HORIZONS,
    atomic_json,
    oof_predictions,
    sha256,
    validate_cache_schema,
    validate_normalization_binding,
)
from build_fcp008_force_readout_candidate import MODEL_SHA, PHASES, fit_weighted_ridge, predict

P008_CACHE_SHA = "22d6c8b29bf8df27917b3e14d791ade07a8c18eeaefec12c7301d1a6b1bf2bff"


def fit_joint_coefficients(free_ar, h1, targets, row_mask):
    """Give each original row total weight 1/N, split 0.5 across domains."""
    free_ar = np.asarray(free_ar, dtype=np.float64)
    h1 = np.asarray(h1, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    row_mask = np.asarray(row_mask, dtype=bool)
    if (
        free_ar.shape != h1.shape
        or free_ar.ndim != 2
        or free_ar.shape[1] != 128
        or targets.shape != (len(free_ar), 4)
        or row_mask.shape != (len(free_ar),)
        or not all(np.isfinite(value).all() for value in (free_ar, h1, targets))
    ):
        raise ValueError("joint domain row/target contract differs")
    count = int(row_mask.sum())
    if count < 1:
        raise ValueError("empty joint fit")
    features = np.concatenate((free_ar[row_mask], h1[row_mask]))
    duplicated_targets = np.concatenate((targets[row_mask], targets[row_mask]))
    weights = np.full(2 * count, 0.5 / count, dtype=np.float64)
    coefficients, fit = fit_weighted_ridge(features, duplicated_targets, weights, 0.0)
    fit["original_train_rows"] = count
    fit["joint_design_rows"] = 2 * count
    fit["per_original_row_total_weight"] = 1.0 / count
    fit["per_domain_copy_weight"] = 0.5 / count
    fit["domain_mix"] = {"free_ar": 0.5, "matched_weight_h1": 0.5}
    return coefficients, fit


def joint_oof_predictions(free_ar, h1, targets, phases):
    phases = np.asarray(phases).astype(str)
    if len(phases) != len(targets) or set(np.unique(phases)) != set(PHASES):
        raise ValueError("joint phase fold contract differs")
    predictions = {
        "joint_fit_predict_free_ar": np.empty_like(targets, dtype=np.float64),
        "joint_fit_predict_h1": np.empty_like(targets, dtype=np.float64),
    }
    fits = {}
    for phase in PHASES:
        train = phases != phase
        held = ~train
        if not train.any() or not held.any():
            raise ValueError("empty joint phase fold")
        coefficients, fit = fit_joint_coefficients(free_ar, h1, targets, train)
        predictions["joint_fit_predict_free_ar"][held] = predict(free_ar[held], coefficients)
        predictions["joint_fit_predict_h1"][held] = predict(h1[held], coefficients)
        fits[phase] = fit
    return predictions, fits


def metric_block(error, force_std, mask):
    normalized = error[mask]
    physical = normalized * force_std
    normalized_values = {"rear_cd": normalized[:, 2], "rear_cl": normalized[:, 3]}
    physical_values = {
        "rear_cd": physical[:, 2],
        "rear_cl": physical[:, 3],
        "total_cd": physical[:, 0] + physical[:, 2],
    }

    def reduce(values):
        return {
            name: {
                "count": len(value),
                "mae": float(np.mean(np.abs(value))),
                "rmse": float(np.sqrt(np.mean(np.square(value)))),
                "bias": float(np.mean(value)),
            }
            for name, value in values.items()
        }

    return {"normalized": reduce(normalized_values), "physical": reduce(physical_values)}


def report_predictions(predictions, targets, phases, steps, force_std):
    reports = {}
    for name, prediction in predictions.items():
        error = prediction - targets
        reports[name] = {}
        for phase in (*PHASES, "pooled"):
            phase_mask = np.ones(len(phases), dtype=bool) if phase == "pooled" else phases == phase
            reports[name][phase] = {
                "all_steps": metric_block(error, force_std, phase_mask),
                "relative_horizons": {
                    f"H{horizon}": metric_block(
                        error, force_std, phase_mask & (steps == horizon)
                    )
                    for horizon in HORIZONS
                },
            }
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("result", "cache", "p008-cache", "normalization", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("exclusive output required")
    result = json.loads(args.result.read_text())
    if (
        result.get("status") != EXPECTED_STATUS
        or result.get("feature_cache_sha256") != sha256(args.cache)
        or result.get("parent_model_sha256") != MODEL_SHA
        or result.get("input_sha256", {}).get("implementation") != EXPECTED_IMPLEMENTATION_SHA
        or result.get("candidate_saved") is not False
        or any(result.get(key) is not False for key in ("validation_accessed", "frozen_test_accessed", "ppo_executed"))
    ):
        raise ValueError("completed FC-P009 cache receipt differs")
    force_std = validate_normalization_binding(result, args.normalization)
    cache = dict(np.load(args.cache, allow_pickle=False))
    free_ar, h1, targets, phases, steps = validate_cache_schema(cache)
    if sha256(args.p008_cache) != P008_CACHE_SHA:
        raise ValueError("P008 H1 cache SHA differs")
    p008 = dict(np.load(args.p008_cache, allow_pickle=False))
    indices = np.asarray(cache["matched_weight_h1_source_indices"], dtype=np.int64)
    if indices.shape != (136800,) or indices.min() < 0 or indices.max() >= 19648:
        raise ValueError("matched H1 source indices differ")
    parent_h1 = np.asarray(p008["parent_native_normalized"], dtype=np.float64)[indices]
    parent_ar = np.asarray(cache["parent_native_normalized"], dtype=np.float64)
    if parent_h1.shape != targets.shape or parent_ar.shape != targets.shape:
        raise ValueError("parent native prediction shape differs")
    if not np.isfinite(parent_h1).all() or not np.isfinite(parent_ar).all():
        raise ValueError("parent native prediction is nonfinite")

    joint_predictions, joint_fits = joint_oof_predictions(free_ar, h1, targets, phases)
    comparison = oof_predictions(free_ar, h1, targets, phases)
    comparison.update(joint_predictions)
    comparison["parent_predict_free_ar"] = parent_ar
    comparison["parent_predict_h1"] = parent_h1
    payload = {
        "status": "FC_P009_FIXED_50_50_JOINT_READOUT_DIAGNOSTIC_COMPLETE",
        "scope": "CPU-only fixed-alpha0 train-phase diagnostic; not selection, candidate, admission, or control evidence",
        "source_result_sha256": sha256(args.result),
        "source_cache_sha256": sha256(args.cache),
        "p008_h1_cache_sha256": sha256(args.p008_cache),
        "normalization_sha256": sha256(args.normalization),
        "alpha": 0.0,
        "alpha_selection_performed": False,
        "domain_mix": {"free_ar": 0.5, "matched_weight_h1": 0.5},
        "joint_fold_fits": joint_fits,
        "reports": report_predictions(comparison, targets, phases, steps, force_std),
        "gpu_used": False,
        "optimizer_steps": 0,
        "candidate_saved": False,
        "validation_accessed": False,
        "frozen_test_accessed": False,
        "ppo_executed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, payload)
    print(payload["status"], flush=True)


if __name__ == "__main__":
    main()
