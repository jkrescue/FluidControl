#!/usr/bin/env python3
"""CPU-only FC-P010 tail-RMS readout mechanism diagnostic."""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_fcp009_cache_result import atomic_json, sha256, validate_cache_schema
from analyze_fcp009_joint_readout import fit_joint_coefficients
from analyze_fcp009_window_statistics import aggregate_rows, validate_window_layout, window_error_rows
from build_fcp008_force_readout_candidate import CHANNELS, NORM_SHA, PHASES, predict

CACHE_SHA = "fc1b84fdd5a43d2b29e1f068531940bd6c12f320556d767a2540dfb5a238ca84"
CACHE_RESULT_SHA = "1321c30a1e12172b85b269405c208981390f952bdf6f03a7fe2dd21f3bb91daf"
MODEL_SHA = "dc41fc91d42476e052970b39fc66aed22fa72aa8b6f218a341a3abb095f42e31"
APPROVAL_SHA = "641fcd3e5205c9a88218f08b9a1949e78fca0cc137393cd92b494e785460d677"
MAX_ITER = 200


def tail_rms(values):
    """Centered ddof=0 RMS over relative steps 39..100."""
    import torch

    trailing = values[:, 38:]
    centered = trailing - trailing.mean(dim=1, keepdim=True)
    return torch.sqrt(centered.square().mean(dim=1) + 1.0e-12)


def joint_standardization(free_ar, h1, row_mask):
    free_ar = np.asarray(free_ar, dtype=np.float64)
    h1 = np.asarray(h1, dtype=np.float64)
    row_mask = np.asarray(row_mask, dtype=bool)
    if free_ar.shape != h1.shape or row_mask.shape != (len(free_ar),) or not row_mask.any():
        raise ValueError("joint standardization rows differ")
    values = np.concatenate((free_ar[row_mask], h1[row_mask]))
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=0)
    active = std > 1.0e-12
    if not active.any() or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("joint standardization is nonfinite or constant")
    return mean, std, active


def optimize_rear_cl(
    free_ar,
    h1,
    targets,
    phases,
    force_std_rear_cl,
    init_coefficients,
    held_phase=None,
):
    """Run the one fixed pointwise-plus-tail objective on train phases only."""
    import torch

    free_ar = np.asarray(free_ar, dtype=np.float64)
    h1 = np.asarray(h1, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    phases = np.asarray(phases).astype(str)
    init_coefficients = np.asarray(init_coefficients, dtype=np.float64)
    train = np.ones(len(targets), dtype=bool) if held_phase is None else phases != held_phase
    if (
        free_ar.shape != h1.shape
        or free_ar.shape[1] != 128
        or targets.shape != (len(free_ar), 4)
        or init_coefficients.shape != (129,)
        or len(targets) % 100
        or not np.isfinite(force_std_rear_cl)
        or force_std_rear_cl <= 0
    ):
        raise ValueError("FC-P010 optimization input differs")
    window_train = train.reshape(-1, 100)[:, 0]
    if not np.all(train.reshape(-1, 100) == window_train[:, None]):
        raise ValueError("phase changes within a H100 window")
    mean, std, active = joint_standardization(free_ar, h1, train)
    z_ar = (free_ar[train][:, active] - mean[active]) / std[active]
    z_h1 = (h1[train][:, active] - mean[active]) / std[active]
    target = targets[train, 3]
    truth_window = (targets[:, 3] * force_std_rear_cl).reshape(-1, 100)[window_train]
    truth_rms = np.sqrt(
        np.mean(
            np.square(
                truth_window[:, 38:]
                - truth_window[:, 38:].mean(axis=1, keepdims=True)
            ),
            axis=1,
        )
    )
    scale = float(np.sqrt(np.mean(np.square(truth_rms))))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("fold-train RMS scale is nonfinite or nonpositive")
    beta0 = np.concatenate(
        (init_coefficients[:128][active] * std[active], [mean @ init_coefficients[:128] + init_coefficients[128]])
    )
    tensors = [torch.as_tensor(value, dtype=torch.float64) for value in (z_ar, z_h1, target)]
    z_ar_t, z_h1_t, target_t = tensors
    beta = torch.nn.Parameter(torch.as_tensor(beta0, dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [beta],
        lr=1.0,
        max_iter=MAX_ITER,
        history_size=20,
        tolerance_grad=1.0e-10,
        tolerance_change=1.0e-12,
        line_search_fn="strong_wolfe",
    )

    def components():
        pred_ar = z_ar_t @ beta[:-1] + beta[-1]
        pred_h1 = z_h1_t @ beta[:-1] + beta[-1]
        step = 0.5 * ((pred_ar - target_t).square().mean() + (pred_h1 - target_t).square().mean())
        pred_window = (pred_ar * force_std_rear_cl).reshape(-1, 100)
        rms = ((tail_rms(pred_window) - torch.as_tensor(truth_rms, dtype=torch.float64)) / scale).square().mean()
        return step, rms, 0.5 * step + 0.5 * rms

    def closure():
        optimizer.zero_grad(set_to_none=True)
        total = components()[2]
        if not torch.isfinite(total):
            raise ValueError("FC-P010 objective became nonfinite")
        total.backward()
        return total

    optimizer.step(closure)
    step, rms, total = components()
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    state = optimizer.state[beta]
    iterations = int(state.get("n_iter", 0))
    function_evaluations = int(state.get("func_evals", 0))
    if iterations < 0 or iterations > MAX_ITER or function_evaluations < 1:
        raise ValueError("LBFGS runtime counters differ")
    coefficient = np.zeros(129, dtype=np.float64)
    beta_np = beta.detach().numpy()
    coefficient[:128][active] = beta_np[:-1] / std[active]
    coefficient[128] = beta_np[-1] - mean @ coefficient[:128]
    grad_max = float(beta.grad.abs().max())
    if not np.isfinite(coefficient).all() or not np.isfinite(grad_max):
        raise ValueError("FC-P010 coefficients/gradient are nonfinite")
    return coefficient, {
        "held_phase": held_phase,
        "train_row_count": int(train.sum()),
        "train_window_count": int(window_train.sum()),
        "rms_scale_physical": scale,
        "iterations": iterations,
        "function_evaluations": function_evaluations,
        "termination": {
            "stopped_before_iteration_limit": iterations < MAX_ITER,
            "reached_iteration_limit": iterations >= MAX_ITER,
            "final_gradient_below_tolerance": grad_max <= 1.0e-10,
            "specific_pytorch_stop_reason_available": False,
        },
        "final_gradient_max_abs_standardized": grad_max,
        "initial_original_coordinate_l2": float(np.linalg.norm(init_coefficients)),
        "final_original_coordinate_l2": float(np.linalg.norm(coefficient)),
        "final_step_loss": float(step.detach()),
        "final_scaled_tail_rms_loss": float(rms.detach()),
        "final_total_loss": float(total.detach()),
        "finite": True,
    }


def load_actual_head(model_path: Path) -> np.ndarray:
    """Read the actual saved P009 four-force rows without loading a model."""
    import torch

    with zipfile.ZipFile(model_path) as archive:
        state = torch.load(io.BytesIO(archive.read("model.pt")), map_location="cpu", weights_only=True)
    weight = bias = None
    for name, value in state.items():
        if name.endswith("decoder_net.final_layer.linear.weight"):
            weight = value.detach().cpu().numpy()
        elif name.endswith("decoder_net.final_layer.linear.bias"):
            bias = value.detach().cpu().numpy()
    if weight is None or bias is None or weight.shape != (7, 128) or bias.shape != (7,):
        raise ValueError("actual P009 final head is absent")
    return np.concatenate((weight[3:7].T, bias[3:7][None]), axis=0).astype(np.float64)


def profile(case: str) -> str:
    if case.endswith("_zero"):
        return "static_zero"
    if "dynamic_train8" in case:
        return "train8_prbs" if case.endswith("_prbs") else "train8_multisine"
    if "directppo" in case:
        return "train16_ppo"
    return "base20_static_action"


def replace_rear_cl(base_coefficients, rear_cl_coefficients):
    """Change exactly the rear-Cl affine column and preserve the other forces."""
    base = np.asarray(base_coefficients, dtype=np.float64)
    rear = np.asarray(rear_cl_coefficients, dtype=np.float64)
    if base.shape != (129, 4) or rear.shape != (129,) or not np.isfinite(rear).all():
        raise ValueError("rear-Cl replacement shape differs")
    result = base.copy()
    result[:, 3] = rear
    return result


def subset_windows(cache, row_mask, windows):
    window_mask = np.asarray(row_mask, dtype=bool).reshape(-1, 100)[:, 0]
    indices = np.flatnonzero(window_mask)
    row_indices = np.concatenate([np.arange(index * 100, index * 100 + 100) for index in indices])
    selected = []
    for new_index, old_index in enumerate(indices):
        item = dict(windows[int(old_index)])
        item["window_index"] = new_index
        selected.append(item)
    return row_indices, selected


def step_groups(error, cache, row_mask):
    error = np.asarray(error, dtype=np.float64)[row_mask]
    if error.ndim != 2 or error.shape[1] != 4:
        raise ValueError("step-force error shape differs")
    keys = {
        "pooled": np.repeat("all", len(error)),
        "family": np.asarray(cache["families"])[row_mask].astype(str),
        "phase": np.asarray(cache["phases"])[row_mask].astype(str),
        "case": np.asarray(cache["case_names"])[row_mask].astype(str),
        "profile": np.asarray([profile(str(value)) for value in np.asarray(cache["case_names"])[row_mask]]),
    }
    result = {}
    for group, labels in keys.items():
        result[group] = {}
        for label in np.unique(labels):
            values = error[labels == label]
            result[group][str(label)] = {
                "count": len(values),
                "channels": {
                    channel: {
                        "mae": float(np.mean(np.abs(values[:, index]))),
                        "rmse": float(np.sqrt(np.mean(np.square(values[:, index])))),
                        "bias": float(np.mean(values[:, index])),
                    }
                    for index, channel in enumerate(CHANNELS)
                },
            }
    return result


def report(cache, features, targets, coefficients, row_mask, force_mean, force_std, windows):
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.shape != (129, 4):
        raise ValueError("report coefficient shape differs")
    complete = predict(features, coefficients)
    row_indices, selected_windows = subset_windows(cache, row_mask, windows)
    rows = window_error_rows(
        complete[row_indices], targets[row_indices], force_mean, force_std, selected_windows
    )
    tail = aggregate_rows(rows)
    profile_rows = defaultdict(list)
    for row in rows:
        profile_rows[profile(str(row["case_names"]))].append(row)
    tail["profile"] = {
        name: aggregate_rows(values)["pooled"]["all"]
        for name, values in profile_rows.items()
    }
    return {
        "step_force_physical": step_groups(
            (complete - targets) * force_std, cache, row_mask
        ),
        "tail_window": tail,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache-result", "cache", "normalization", "candidate-model", "approval", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("exclusive output required")
    if (
        sha256(args.cache_result) != CACHE_RESULT_SHA
        or sha256(args.cache) != CACHE_SHA
        or sha256(args.normalization) != NORM_SHA
        or sha256(args.candidate_model) != MODEL_SHA
        or sha256(args.approval) != APPROVAL_SHA
    ):
        raise ValueError("FC-P010 fixed input SHA differs")
    cache = dict(np.load(args.cache, allow_pickle=False))
    free_ar, h1, targets, phases, _steps = validate_cache_schema(cache)
    windows = validate_window_layout(cache)
    normalization = json.loads(args.normalization.read_text())
    force_mean = np.asarray(normalization["all_force_mean"], dtype=np.float64)
    force_std = np.asarray(normalization["all_force_std"], dtype=np.float64)
    actual = load_actual_head(args.candidate_model)
    all_rows = np.ones(len(targets), dtype=bool)
    full_baseline, _ = fit_joint_coefficients(free_ar, h1, targets, all_rows)
    rounding = actual - full_baseline
    if float(np.max(np.abs(rounding))) > 2.0e-6:
        raise ValueError("actual P009 rear-Cl head differs beyond float32 rounding")

    fits = {}
    for held_phase in PHASES:
        train = phases != held_phase
        held = ~train
        baseline, _ = fit_joint_coefficients(free_ar, h1, targets, train)
        optimized, diagnostic = optimize_rear_cl(
            free_ar, h1, targets, phases, force_std[3], baseline[:, 3], held_phase
        )
        optimized_full = replace_rear_cl(baseline, optimized)
        train_window = train.reshape(-1, 100)[:, 0]
        held_window = held.reshape(-1, 100)[:, 0]
        window_families = np.asarray(cache["families"])[::100].astype(str)
        diagnostic["train_window_counts_by_family"] = {
            family: int(np.sum(train_window & (window_families == family)))
            for family in np.unique(window_families)
        }
        diagnostic["held_window_counts_by_family"] = {
            family: int(np.sum(held_window & (window_families == family)))
            for family in np.unique(window_families)
        }
        fits[held_phase] = {
            "optimization": diagnostic,
            "held_row_count": int(held.sum()),
            "held_window_count": int(held.reshape(-1, 100)[:, 0].sum()),
            "baseline": {
                "free_ar": report(cache, free_ar, targets, baseline, held, force_mean, force_std, windows),
                "h1": report(cache, h1, targets, baseline, held, force_mean, force_std, windows),
            },
            "tail_rms_supervised": {
                "free_ar": report(cache, free_ar, targets, optimized_full, held, force_mean, force_std, windows),
                "h1": report(cache, h1, targets, optimized_full, held, force_mean, force_std, windows),
            },
        }
        print(json.dumps({"event": "fcp010_fold_complete", "held_phase": held_phase, **diagnostic}), flush=True)

    full_optimized, full_diagnostic = optimize_rear_cl(
        free_ar, h1, targets, phases, force_std[3], actual[:, 3], None
    )
    full_optimized_matrix = replace_rear_cl(actual, full_optimized)
    print(json.dumps({"event": "fcp010_full_fit_complete", **full_diagnostic}), flush=True)
    payload = {
        "status": "FC_P010_TAIL_RMS_SUPERVISION_CPU_DIAGNOSTIC_COMPLETE",
        "scope": "train-cache mechanism diagnostic only; not a candidate, gate, formal evaluation, or control result",
        "objective": {
            "pointwise_rear_cl_mse_weight": 0.5,
            "free_ar_tail62_centered_rms_mse_weight": 0.5,
            "truth_rms_epsilon": 0.0,
            "prediction_rms_differentiability_epsilon_inside_sqrt": 1.0e-12,
            "alpha": 0.0,
            "parameter_search_performed": False,
            "optimizer": "torch.float64.LBFGS",
            "max_iterations_per_fit": MAX_ITER,
        },
        "actual_float32_vs_full_alpha0_rounding": {
            "max_abs": float(np.max(np.abs(rounding))),
            "root_mean_squared": float(np.sqrt(np.mean(np.square(rounding)))),
            "tolerance": 2.0e-6,
        },
        "phase_oof": fits,
        "full_fit": {
            "optimization": full_diagnostic,
            "baseline_actual_p009": {
                "free_ar": report(cache, free_ar, targets, actual, all_rows, force_mean, force_std, windows),
                "h1": report(cache, h1, targets, actual, all_rows, force_mean, force_std, windows),
            },
            "tail_rms_supervised": {
                "free_ar": report(cache, free_ar, targets, full_optimized_matrix, all_rows, force_mean, force_std, windows),
                "h1": report(cache, h1, targets, full_optimized_matrix, all_rows, force_mean, force_std, windows),
            },
        },
        "input_sha256": {
            "cache_result": CACHE_RESULT_SHA,
            "cache": CACHE_SHA,
            "normalization": NORM_SHA,
            "candidate_model": MODEL_SHA,
            "approval": APPROVAL_SHA,
            "implementation": sha256(Path(__file__)),
        },
        "gpu_used": False,
        "readout_coefficient_optimization_performed": True,
        "fno_training_performed": False,
        "model_checkpoint_saved": False,
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
