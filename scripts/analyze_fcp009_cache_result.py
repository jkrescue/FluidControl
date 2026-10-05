#!/usr/bin/env python3
"""Summarize the completed FC-P009 cache without selecting a model or parameter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from build_fcp008_force_readout_candidate import (
    MODEL_SHA,
    NORM_SHA,
    PHASES,
    fit_weighted_ridge,
    predict,
)

EXPECTED_STATUS = "FC_P009_TRAIN_ONLY_FREE_AR_FEATURE_CACHE_COMPLETE_NOT_ADMISSION"
EXPECTED_IMPLEMENTATION_SHA = "a2c83846714b3d5575840dcd89c0c8e955fc40f217dad011f3e74d277807bfdf"
HORIZONS = (1, 10, 50, 100)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def oof_predictions(free_ar, h1, targets, phases):
    predictions = {
        "fit_free_ar_predict_free_ar": np.empty_like(targets, dtype=np.float64),
        "fit_free_ar_predict_h1": np.empty_like(targets, dtype=np.float64),
        "fit_h1_predict_free_ar": np.empty_like(targets, dtype=np.float64),
        "fit_h1_predict_h1": np.empty_like(targets, dtype=np.float64),
    }
    for phase in PHASES:
        train = phases != phase
        held = ~train
        weights = np.full(int(train.sum()), 1.0 / int(train.sum()), dtype=np.float64)
        ar_coefficients, _ = fit_weighted_ridge(free_ar[train], targets[train], weights, 0.0)
        h1_coefficients, _ = fit_weighted_ridge(h1[train], targets[train], weights, 0.0)
        predictions["fit_free_ar_predict_free_ar"][held] = predict(free_ar[held], ar_coefficients)
        predictions["fit_free_ar_predict_h1"][held] = predict(h1[held], ar_coefficients)
        predictions["fit_h1_predict_free_ar"][held] = predict(free_ar[held], h1_coefficients)
        predictions["fit_h1_predict_h1"][held] = predict(h1[held], h1_coefficients)
    return predictions


def physical_metrics(error, force_std, mask):
    physical = error[mask] * force_std
    values = {
        "rear_cd": physical[:, 2],
        "rear_cl": physical[:, 3],
        "total_cd": physical[:, 0] + physical[:, 2],
    }
    return {
        name: {
            "count": len(value),
            "mae": float(np.mean(np.abs(value))),
            "rmse": float(np.sqrt(np.mean(np.square(value)))),
            "bias": float(np.mean(value)),
        }
        for name, value in values.items()
    }


def summarize(cache, force_std):
    free_ar = np.asarray(cache["features"], dtype=np.float64)
    h1 = np.asarray(cache["matched_weight_h1_features"], dtype=np.float64)
    targets = np.asarray(cache["targets_normalized"], dtype=np.float64)
    phases = np.asarray(cache["phases"]).astype(str)
    steps = np.asarray(cache["relative_steps"], dtype=np.int64)
    if (
        free_ar.shape != (136800, 128)
        or h1.shape != free_ar.shape
        or targets.shape != (136800, 4)
        or set(np.unique(phases)) != set(PHASES)
        or set(np.unique(steps)) != set(range(1, 101))
        or not all(np.isfinite(value).all() for value in (free_ar, h1, targets))
    ):
        raise ValueError("FC-P009 cache schema differs")
    expected_phase_rows = {"b00": 40200, "b02": 40200, "b04": 28200, "b06": 28200}
    for phase, count in expected_phase_rows.items():
        if int((phases == phase).sum()) != count:
            raise ValueError("FC-P009 phase row count differs")
        if any(int(((phases == phase) & (steps == step)).sum()) != count // 100 for step in range(1, 101)):
            raise ValueError("FC-P009 phase/relative-step count differs")
    predictions = oof_predictions(free_ar, h1, targets, phases)
    reports = {}
    for direction, prediction in predictions.items():
        error = prediction - targets
        reports[direction] = {}
        for phase in (*PHASES, "pooled"):
            phase_mask = np.ones(len(phases), dtype=bool) if phase == "pooled" else phases == phase
            reports[direction][phase] = {
                "all_steps": physical_metrics(error, force_std, phase_mask),
                "relative_horizons": {
                    f"H{horizon}": physical_metrics(
                    error, force_std, phase_mask & (steps == horizon)
                    )
                    for horizon in HORIZONS
                },
            }
    return reports


def atomic_json(path: Path, payload: dict):
    temporary = path.with_suffix(".tmp")
    with temporary.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("exclusive output required")
    result = json.loads(args.result.read_text())
    if (
        result.get("status") != EXPECTED_STATUS
        or result.get("feature_cache_sha256") != sha256(args.cache)
        or result.get("candidate_saved") is not False
        or result.get("validation_accessed") is not False
        or result.get("frozen_test_accessed") is not False
        or result.get("ppo_executed") is not False
        or result.get("parent_model_sha256") != MODEL_SHA
        or result.get("input_sha256", {}).get("implementation") != EXPECTED_IMPLEMENTATION_SHA
    ):
        raise ValueError("completed FC-P009 cache receipt differs")
    normalization = json.loads(args.normalization.read_text())
    force_std = np.asarray(normalization.get("all_force_std"), dtype=np.float64)
    if (
        sha256(args.normalization) != NORM_SHA
        or result.get("input_sha256", {}).get("normalization") != NORM_SHA
        or force_std.shape != (4,)
        or not np.isfinite(force_std).all()
        or not (force_std > 0).all()
    ):
        raise ValueError("force normalization contract differs")
    cache = dict(np.load(args.cache, allow_pickle=False))
    payload = {
        "status": "FC_P009_CACHE_ONLY_CROSS_DOMAIN_ANALYSIS_COMPLETE",
        "scope": "CPU-only fixed-alpha0 held-phase analysis; not selection, candidate, admission, or control evidence",
        "source_result_sha256": sha256(args.result),
        "source_cache_sha256": sha256(args.cache),
        "normalization_sha256": sha256(args.normalization),
        "reports": summarize(cache, force_std),
        "alpha": 0.0,
        "selection_performed": False,
        "gpu_used": False,
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
