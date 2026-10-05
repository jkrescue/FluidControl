#!/usr/bin/env python3
"""Train-only phase-blocked ridge stability audit for the fixed FNO features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from itertools import combinations
from pathlib import Path

import numpy as np

CACHE_SHA = "947309d2b38671953c3186abb3be215d064aa5c6775e912d082901d3cb6f23ab"
SOURCE_RESULT_SHA = "44920594e3265ce9c9615220bca4f9b0877ec9bc423eb252bd24ad9fe7d7654d"
ALPHAS = (0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1.0)
PHASES = ("b00", "b02", "b04", "b06")
PROFILES = ("multisine", "prbs")
CHANNELS = ("front_cd", "front_cl", "rear_cd", "rear_cl")
PREFIX = "prefix_targets_1_100"
LATE = "late_targets_101_200"
CONSTANT_STD_ATOL = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_key(panel: str, branch: str, identity: str, kind: str) -> str:
    return f"{panel}__{branch}__{identity.replace(':', '_')}__{kind}"


def branch_arrays(cache, panel: str, branch: str, identity: str) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(cache[cache_key(panel, branch, identity, "features")], dtype=np.float64)
    targets = np.asarray(
        cache[cache_key(panel, branch, identity, "targets_normalized")], dtype=np.float64
    )
    if features.shape != (100, 128) or targets.shape != (100, 4):
        raise ValueError("cached feature/target shape differs")
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError("cached feature/target must be finite")
    return features, targets


def symmetric_rows(cache, panel: str, phases: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Return action and same-phase zero rows with exact symmetric weighting."""
    features, targets = [], []
    for phase in phases:
        for profile in PROFILES:
            action_x, action_y = branch_arrays(cache, panel, "action", f"{phase}:{profile}")
            zero_x, zero_y = branch_arrays(cache, panel, "zero", phase)
            features.extend((action_x, zero_x))
            targets.extend((action_y, zero_y))
    x, y = np.concatenate(features), np.concatenate(targets)
    expected = 400 * len(phases)
    if x.shape != (expected, 128) or y.shape != (expected, 4):
        raise ValueError("symmetric row count differs")
    return x, y


def fit_standardized_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    alpha: float,
    *,
    constant_atol: float = CONSTANT_STD_ATOL,
) -> tuple[np.ndarray, dict]:
    """Fit mean-MSE + alpha*||W||^2; intercept is never penalized.

    Standardization statistics use these rows only. Constant columns are set to
    zero in standardized space and receive a zero coefficient.
    """
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if (
        x.ndim != 2
        or x.shape[1] != 128
        or y.shape != (x.shape[0], 4)
        or x.shape[0] <= 129
        or alpha not in ALPHAS
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("invalid ridge fit contract")
    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=0)
    active = std > constant_atol
    if not active.any():
        raise ValueError("all fixed features are constant")
    xz = (x[:, active] - mean[active]) / std[active]
    ymean = y.mean(axis=0)
    yc = y - ymean
    if alpha == 0.0:
        weights_active, _, rank, singular = np.linalg.lstsq(xz, yc, rcond=1e-10)
    else:
        gram = xz.T @ xz / xz.shape[0]
        cross = xz.T @ yc / xz.shape[0]
        weights_active = np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), cross)
        singular = np.linalg.svd(xz, compute_uv=False)
        rank = int(np.linalg.matrix_rank(xz))
    weights = np.zeros((128, 4), dtype=np.float64)
    weights[active] = weights_active / std[active, None]
    intercept = ymean - mean @ weights
    coefficients = np.concatenate((weights, intercept[None]), axis=0)
    if not np.isfinite(coefficients).all():
        raise ValueError("nonfinite ridge coefficients")
    return coefficients, {
        "alpha": alpha,
        "objective": "mean normalized-force MSE + alpha*sum(standardized_feature_weights^2); intercept unpenalized",
        "row_count": int(x.shape[0]),
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "constant_feature_indices": np.flatnonzero(~active).tolist(),
        "active_feature_count": int(active.sum()),
        "standardized_design_rank": int(rank),
        "standardized_design_singular_max": float(singular[0]),
        "standardized_design_singular_min": float(singular[-1]),
        "original_coordinate_channel_coefficient_l2": [
            float(np.linalg.norm(coefficients[:, index])) for index in range(4)
        ],
    }


def predict(features: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return np.asarray(features, dtype=np.float64) @ coefficients[:128] + coefficients[128]


def error_summary(error: np.ndarray) -> dict:
    error = np.asarray(error, dtype=np.float64)
    if error.ndim != 2 or error.shape[1] != 4 or error.shape[0] == 0 or not np.isfinite(error).all():
        raise ValueError("finite [N,4] error required")
    return {
        channel: {
            "count": int(error.shape[0]),
            "mae": float(np.mean(np.abs(error[:, index]))),
            "rmse": float(np.sqrt(np.mean(np.square(error[:, index])))),
            "bias": float(np.mean(error[:, index])),
        }
        for index, channel in enumerate(CHANNELS)
    }


def evaluate_panel(cache, panel: str, coefficients: np.ndarray, phases=PHASES) -> dict:
    actions, zeros, deltas, per_pair = [], [], [], {}
    zero_by_phase = {}
    for phase in phases:
        zero_x, zero_y = branch_arrays(cache, panel, "zero", phase)
        zero_error = predict(zero_x, coefficients) - zero_y
        zeros.append(zero_error)
        zero_by_phase[phase] = error_summary(zero_error)
        for profile in PROFILES:
            pair_id = f"{phase}:{profile}"
            action_x, action_y = branch_arrays(cache, panel, "action", pair_id)
            action_error = predict(action_x, coefficients) - action_y
            delta_error = (predict(action_x, coefficients) - predict(zero_x, coefficients)) - (
                action_y - zero_y
            )
            actions.append(action_error)
            deltas.append(delta_error)
            per_pair[pair_id] = {
                "action": error_summary(action_error),
                "delta": error_summary(delta_error),
            }
    return {
        "action": error_summary(np.concatenate(actions)),
        "unique_zero": error_summary(np.concatenate(zeros)),
        "delta": error_summary(np.concatenate(deltas)),
        "unique_zero_by_phase": zero_by_phase,
        "per_pair": per_pair,
        "endpoint_counts": {
            "action": 200 * len(phases),
            "unique_zero": 100 * len(phases),
            "delta": 200 * len(phases),
        },
    }


def selection_mse(cache, held_phase: str, coefficients: np.ndarray) -> tuple[float, list[float]]:
    x, y = symmetric_rows(cache, PREFIX, (held_phase,))
    squared = np.square(predict(x, coefficients) - y)
    per_channel = squared.mean(axis=0)
    return float(per_channel.mean()), [float(value) for value in per_channel]


def select_alpha(scores: dict[float, float]) -> float:
    if set(scores) != set(ALPHAS) or any(not math.isfinite(value) for value in scores.values()):
        raise ValueError("complete finite fixed-alpha scores required")
    # Tuple ordering implements exact-score ties in favor of the larger alpha.
    return min(ALPHAS, key=lambda alpha: (scores[alpha], -alpha))


def coefficient_stability(coefficients: list[np.ndarray]) -> dict:
    stack = np.stack(coefficients)
    output = {}
    for index, channel in enumerate(CHANNELS):
        rows = stack[:, :, index]
        norms = np.linalg.norm(rows, axis=1)
        cosines = []
        for left, right in combinations(rows, 2):
            denominator = np.linalg.norm(left) * np.linalg.norm(right)
            cosines.append(float(left @ right / denominator) if denominator > 0 else None)
        output[channel] = {
            "fold_coefficient_l2": [float(value) for value in norms],
            "coefficient_l2_mean": float(norms.mean()),
            "coefficient_l2_std": float(norms.std(ddof=0)),
            "pairwise_cosine_defined": all(value is not None for value in cosines),
            "pairwise_cosine_min": min(cosines) if all(value is not None for value in cosines) else None,
            "pairwise_cosine_mean": float(np.mean(cosines))
            if all(value is not None for value in cosines)
            else None,
        }
    return output


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--source-result", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or sha256(args.cache) != CACHE_SHA or sha256(args.source_result) != SOURCE_RESULT_SHA:
        raise ValueError("exclusive output and fixed source evidence required")
    source = json.loads(args.source_result.read_text())
    if (
        source.get("status") != "FCP003C_FIXED_FEATURE_FORCE_READOUT_DIAGNOSTIC_COMPLETE"
        or source.get("cache", {}).get("sha256") != CACHE_SHA
        or source.get("optimizer_steps") != 0
        or source.get("validation_accessed")
        or source.get("frozen_test_accessed")
        or source.get("ppo_executed")
    ):
        raise ValueError("source diagnostic contract differs")

    cache = np.load(args.cache, allow_pickle=False)
    folds = {}
    alpha_scores = {alpha: [] for alpha in ALPHAS}
    alpha_coefficients = {alpha: [] for alpha in ALPHAS}
    for held_phase in PHASES:
        train_phases = tuple(phase for phase in PHASES if phase != held_phase)
        train_x, train_y = symmetric_rows(cache, PREFIX, train_phases)
        fold = {"training_phases": list(train_phases), "held_phase": held_phase, "alphas": {}}
        for alpha in ALPHAS:
            coefficients, fit = fit_standardized_ridge(train_x, train_y, alpha)
            score, per_channel = selection_mse(cache, held_phase, coefficients)
            alpha_scores[alpha].append(score)
            alpha_coefficients[alpha].append(coefficients)
            fold["alphas"][str(alpha)] = {
                "selection_normalized_mse_equal_four_channel": score,
                "selection_normalized_mse_by_channel": dict(zip(CHANNELS, per_channel, strict=True)),
                "held_prefix_metrics_normalized": evaluate_panel(cache, PREFIX, coefficients, (held_phase,)),
                "fit": fit,
            }
        folds[held_phase] = fold
    aggregate_scores = {alpha: float(np.mean(values)) for alpha, values in alpha_scores.items()}
    selected_alpha = select_alpha(aggregate_scores)
    all_prefix_x, all_prefix_y = symmetric_rows(cache, PREFIX, PHASES)
    final_coefficients, final_fit = fit_standardized_ridge(all_prefix_x, all_prefix_y, selected_alpha)
    alpha0_coefficients, _ = fit_standardized_ridge(all_prefix_x, all_prefix_y, 0.0)
    cached_ols = np.asarray(cache["fitted_coefficients"], dtype=np.float64)
    prefix_features = np.concatenate(
        [branch_arrays(cache, PREFIX, "action", f"{phase}:{profile}")[0] for phase in PHASES for profile in PROFILES]
    )
    alpha0_reference = {
        "cached_ols_coefficient_max_abs_difference": float(np.max(np.abs(alpha0_coefficients - cached_ols))),
        "cached_ols_prefix_action_prediction_max_abs_difference": float(
            np.max(np.abs(predict(prefix_features, alpha0_coefficients) - predict(prefix_features, cached_ols)))
        ),
    }
    result = {
        "status": "FCP003C_PHASE_BLOCKED_RIDGE_CACHE_DIAGNOSTIC_COMPLETE",
        "scope": "train-only cached-feature diagnostic; no model, checkpoint, validation, frozen test, PPO, or admission",
        "fixed_alpha_grid": list(ALPHAS),
        "selection": {
            "protocol": "four leave-one-phase-out prefix folds; equal normalized-force channel MSE; exact ties choose larger alpha",
            "aggregate_held_phase_scores": {str(alpha): score for alpha, score in aggregate_scores.items()},
            "selected_alpha": selected_alpha,
            "late_metrics_used_for_selection": False,
        },
        "folds": folds,
        "coefficient_stability_by_alpha": {
            str(alpha): coefficient_stability(alpha_coefficients[alpha]) for alpha in ALPHAS
        },
        "selected_full_prefix_fit": final_fit,
        "selected_prefix_metrics_normalized": evaluate_panel(cache, PREFIX, final_coefficients),
        "selected_late_metrics_normalized_single_check": evaluate_panel(cache, LATE, final_coefficients),
        "alpha0_reference": alpha0_reference,
        "interpretation_limits": [
            "This diagnostic can support or weaken coefficient-instability and finite-prefix-coverage hypotheses; it cannot establish either as the unique cause.",
            "Poor performance for every alpha does not distinguish coverage limits from fixed-feature information limits.",
            "Late targets 101..200 were used by regular training and are train-internal, not independent validation.",
            "No late-alpha curve is emitted or used for selection.",
        ],
        "input_sha256": {"feature_cache": CACHE_SHA, "source_result": SOURCE_RESULT_SHA},
        "implementation_sha256": sha256(Path(__file__)),
        "optimizer_steps": 0,
        "checkpoint_saved": False,
        "gpu_used": False,
        "validation_accessed": False,
        "frozen_test_accessed": False,
        "ppo_executed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    print(result["status"], selected_alpha, aggregate_scores[selected_alpha])


if __name__ == "__main__":
    main()
