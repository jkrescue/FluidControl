#!/usr/bin/env python3
"""Measure fixed-feature linear force-readout representability for FC-P003C.

This is a train-only diagnostic.  It never changes or saves the model.  The
existing final-layer inputs are spatially averaged with the model mask, an
affine four-force readout is fit on targets 1..100, and targets 101..200 are a
strictly later-time check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from diagnose_fcp003c_train_vs_validation_true_state_h1 import (
    BASE_SHA,
    CONFIG_SHA,
    MODEL_SHA,
    NORM_SHA,
    PAIR_SHA,
    STATE_SHA,
    TRAIN8_SHA,
    _load_hdf,
    _manifest,
    _normalized_pair,
    sha256,
)

CHANNELS = ("front_cd", "front_cl", "rear_cd", "rear_cl")
PANELS = {"prefix_targets_1_100": range(1, 101), "late_targets_101_200": range(101, 201)}
RCOND = 1.0e-10
REPRODUCTION_ATOL = 2.0e-5


def configure_fixed_highest_fp32(torch_module) -> tuple[dict, dict]:
    before = {
        "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        "cuda_matmul_allow_tf32": bool(torch_module.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch_module.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch_module.get_float32_matmul_precision(),
    }
    torch_module.backends.cuda.matmul.allow_tf32 = False
    torch_module.backends.cudnn.allow_tf32 = False
    torch_module.set_float32_matmul_precision("highest")
    effective = {
        "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        "cuda_matmul_allow_tf32": bool(torch_module.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch_module.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch_module.get_float32_matmul_precision(),
    }
    if effective != {
        "NVIDIA_TF32_OVERRIDE": before["NVIDIA_TF32_OVERRIDE"],
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
    }:
        raise ValueError("could not establish fixed highest-FP32 diagnostic precision")
    return before, effective


def summarize_error(error: np.ndarray) -> dict:
    error = np.asarray(error, dtype=np.float64)
    if error.ndim != 2 or error.shape[1] != 4 or error.size == 0:
        raise ValueError("error must have shape [N,4]")
    if not np.isfinite(error).all():
        raise ValueError("error must be finite")
    return {
        channel: {
            "count": int(error.shape[0]),
            "mae": float(np.mean(np.abs(error[:, index]))),
            "rmse": float(np.sqrt(np.mean(np.square(error[:, index])))),
            "bias": float(np.mean(error[:, index])),
        }
        for index, channel in enumerate(CHANNELS)
    }


def fit_affine_readout(
    features: np.ndarray, targets: np.ndarray, *, rcond: float = RCOND
) -> tuple[np.ndarray, dict]:
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if (
        features.ndim != 2
        or features.shape[1] != 128
        or targets.shape != (features.shape[0], 4)
        or features.shape[0] <= 129
        or not np.isfinite(features).all()
        or not np.isfinite(targets).all()
        or rcond != RCOND
    ):
        raise ValueError("invalid fixed-feature least-squares contract")
    design = np.concatenate(
        (features, np.ones((features.shape[0], 1), dtype=np.float64)), axis=1
    )
    coefficients, _, rank, singular = np.linalg.lstsq(design, targets, rcond=rcond)
    if singular.shape != (129,) or not np.isfinite(singular).all():
        raise ValueError("unexpected singular-value result")
    cutoff = float(rcond * singular[0])
    retained_count = int(np.count_nonzero(singular > cutoff))
    if retained_count != int(rank):
        raise ValueError("lstsq rank/cutoff accounting differs")
    condition_defined = bool(rank > 0 and singular[int(rank) - 1] > 0.0)
    diagnostics = {
        "rcond": rcond,
        "rank": int(rank),
        "singular_values": [float(value) for value in singular],
        "singular_value_cutoff": cutoff,
        "retained_singular_value_count": retained_count,
        "raw_smallest_singular_value": float(singular[-1]),
        "condition_number_retained": float(singular[0] / singular[int(rank) - 1])
        if condition_defined
        else None,
        "condition_number_defined": condition_defined,
        "coefficient_l2": float(np.linalg.norm(coefficients)),
    }
    return coefficients, diagnostics


def predict_affine(features: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != 128 or coefficients.shape != (129, 4):
        raise ValueError("affine shapes differ")
    return features @ coefficients[:128] + coefficients[128]


def weighted_fit_arrays(panel: dict, pair_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Repeat each phase-zero exactly once for each of its two action pairs."""
    xrows, yrows = [], []
    for pair_id in pair_ids:
        phase = pair_id.split(":", 1)[0]
        for branch, key in (("action", pair_id), ("zero", phase)):
            xrows.append(panel[f"{branch}_features"][key])
            yrows.append(panel[f"{branch}_targets_normalized"][key])
    features, targets = np.concatenate(xrows), np.concatenate(yrows)
    if features.shape != (1600, 128) or targets.shape != (1600, 4):
        raise ValueError("symmetric fit weighting differs")
    return features, targets


def evaluate_panel(panel: dict, pair_ids: list[str], coefficients: np.ndarray) -> dict:
    action_errors, zero_errors, delta_errors = [], [], []
    per_pair = {}
    zero_by_phase = {}
    for pair_id in pair_ids:
        phase = pair_id.split(":", 1)[0]
        action_prediction = predict_affine(panel["action_features"][pair_id], coefficients)
        zero_prediction = predict_affine(panel["zero_features"][phase], coefficients)
        action_target = panel["action_targets_normalized"][pair_id]
        zero_target = panel["zero_targets_normalized"][phase]
        action_error = action_prediction - action_target
        delta_error = (action_prediction - zero_prediction) - (action_target - zero_target)
        action_errors.append(action_error)
        delta_errors.append(delta_error)
        per_pair[pair_id] = {
            "action": summarize_error(action_error * panel["force_std"]),
            "delta": summarize_error(delta_error * panel["force_std"]),
        }
    for phase in sorted(panel["zero_features"]):
        error = (
            predict_affine(panel["zero_features"][phase], coefficients)
            - panel["zero_targets_normalized"][phase]
        )
        zero_errors.append(error)
        zero_by_phase[phase] = summarize_error(error * panel["force_std"])
    return {
        "action": summarize_error(np.concatenate(action_errors) * panel["force_std"]),
        "unique_zero": summarize_error(np.concatenate(zero_errors) * panel["force_std"]),
        "delta": summarize_error(np.concatenate(delta_errors) * panel["force_std"]),
        "per_pair": per_pair,
        "unique_zero_by_phase": zero_by_phase,
        "endpoint_counts": {"action": 800, "unique_zero": 400, "delta": 800},
    }


def tensor_state_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def local_step_indices(begin: int, count: int) -> list[int]:
    if isinstance(begin, bool) or isinstance(count, bool) or begin < 0 or count <= 0:
        raise ValueError("nonnegative begin and positive count required")
    return list(range(begin, begin + count))


def extract_branch_features(model, layer, state, omega, mask, force, endpoints, chunk):
    import torch
    from fluid_control.paired_step_force import true_state_step_input

    features, original_predictions, targets = [], [], []
    reproduction_max = 0.0
    first_input = None
    first_state_output = None
    weight = layer.linear.weight[3:7]
    bias = layer.linear.bias[3:7]
    for begin in range(0, len(endpoints), chunk):
        local = list(endpoints[begin : begin + chunk])
        inputs = torch.cat(
            [
                true_state_step_input(
                    state[:, 0], state[:, 1:], mask, omega, step
                )
                for step in local_step_indices(begin, len(local))
            ]
        )
        captured = []
        hook = layer.register_forward_pre_hook(lambda _module, args: captured.append(args[0]))
        try:
            with torch.no_grad():
                raw = model(inputs)
        finally:
            hook.remove()
        hidden = captured[0]
        batch = len(local)
        pixels = mask.shape[-2] * mask.shape[-1]
        if hidden.shape != (batch * pixels, 128) or raw.shape[1] != 7:
            raise ValueError("official final-layer topology differs")
        hidden = hidden.reshape(batch, pixels, 128)
        flat_mask = mask.reshape(1, pixels, 1).expand(batch, -1, -1)
        pooled = (hidden * flat_mask).sum(1) / flat_mask.sum(1)
        raw_force = (raw[:, 3:] * mask.expand(batch, -1, -1, -1)).sum((-2, -1)) / mask.sum()
        manual_force = pooled @ weight.T + bias
        reproduction_max = max(
            reproduction_max, float((manual_force - raw_force).abs().max().item())
        )
        if first_input is None:
            # Replay the identical batch shape: changing batch size may change
            # floating-point execution order even in evaluation mode.
            first_input = inputs.detach().clone()
            first_state_output = raw[:, :3].detach().clone()
        features.append(pooled.cpu().numpy().astype(np.float32))
        original_predictions.append(raw_force.cpu().numpy().astype(np.float32))
        targets.append(force[np.asarray(local)].astype(np.float32))
    return {
        "features": np.concatenate(features),
        "original_predictions_normalized": np.concatenate(original_predictions),
        "targets_physical": np.concatenate(targets),
        "reproduction_max_abs": reproduction_max,
        "first_input": first_input,
        "first_state_output": first_state_output,
    }


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train8", "base", "pair-manifest", "normalization", "config", "checkpoint-dir", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=10)
    args = parser.parse_args()
    if args.output_dir.exists() or isinstance(args.chunk_size, bool) or not 1 <= args.chunk_size <= 100:
        raise ValueError("new output directory and chunk-size [1,100] required")

    train_manifest = _manifest(args.train8 / "manifest.json", TRAIN8_SHA)
    _manifest(args.base / "manifest.json", BASE_SHA)
    pair_manifest = _manifest(args.pair_manifest, PAIR_SHA)
    if sha256(args.normalization) != NORM_SHA or sha256(args.config) != CONFIG_SHA:
        raise ValueError("normalization/config SHA differs")
    if train_manifest.get("trajectory_counts") != {"train": 8, "validation": 0, "frozen_test": 0}:
        raise ValueError("requires train8 only")
    if train_manifest.get("max_abs_omega") != 0.75:
        raise ValueError("action scale differs")
    pairs = sorted(pair_manifest["pairs"], key=lambda item: (item["phase"], item["profile"]))
    pair_ids = [f"{item['phase']}:{item['profile']}" for item in pairs]
    expected_ids = [f"b{i:02d}:{profile}" for i in (0, 2, 4, 6) for profile in ("multisine", "prbs")]
    if pair_ids != expected_ids:
        raise ValueError("dynamic8 identities differ")
    models = list(args.checkpoint_dir.glob("FNO.0.*.mdlus"))
    states = list(args.checkpoint_dir.glob("checkpoint.0.*.pt"))
    if len(models) != 1 or sha256(models[0]) != MODEL_SHA or len(states) != 1 or sha256(states[0]) != STATE_SHA:
        raise ValueError("fixed C checkpoint differs")

    import torch
    from evaluate_tandem_fno import load_composed_config
    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils import load_checkpoint
    from train_tandem_fno import build_model, configured_force_indices

    DistributedManager.initialize()
    dist = DistributedManager()
    if dist.distributed or not dist.cuda:
        raise RuntimeError("single-GPU CUDA diagnostic required")
    torch.cuda.set_per_process_memory_fraction(0.15, device=dist.device)
    precision_before, precision_effective = configure_fixed_highest_fp32(torch)
    config = load_composed_config(args.config)
    if tuple(configured_force_indices(config)) != (0, 1, 2, 3):
        raise ValueError("four force outputs required")
    model = build_model(config).to(dist.device)
    if load_checkpoint(args.checkpoint_dir, models=model, device=dist.device) != 2:
        raise ValueError("expected C epoch2")
    model.eval()
    final_layer = model.decoder_net.final_layer
    if (
        not isinstance(final_layer.linear, torch.nn.Linear)
        or tuple(final_layer.linear.weight.shape) != (7, 128)
        or tuple(final_layer.linear.bias.shape) != (7,)
        or len(getattr(final_layer.linear, "parametrizations", ())) != 0
    ):
        raise ValueError("official final affine projection differs")
    state_sha_before = tensor_state_sha256(model)
    stats = json.loads(args.normalization.read_text())
    if tuple(stats["all_force_channels"]) != CHANNELS:
        raise ValueError("force order differs")
    force_mean = np.asarray(stats["all_force_mean"], dtype=np.float64).reshape(1, 4)
    force_std = np.asarray(stats["all_force_std"], dtype=np.float64).reshape(1, 4)

    records = {panel: {"actions": {}, "zeros": {}} for panel in PANELS}
    zero_sources = {}
    first_check = None
    reproduction_max = 0.0
    for item in pairs:
        pair_id = f"{item['phase']}:{item['profile']}"
        action = _load_hdf(args.train8 / item["action_file"], item["action_hdf_sha256"], 201)
        zero = _load_hdf(args.base / item["zero_file"], item["zero_hdf_sha256"], 801)
        prior = zero_sources.setdefault(item["phase"], (item["zero_file"], item["zero_hdf_sha256"]))
        if prior != (item["zero_file"], item["zero_hdf_sha256"]):
            raise ValueError("same-phase zero source differs")
        for panel, endpoint_range in PANELS.items():
            endpoints = list(endpoint_range)
            normalized = _normalized_pair(action, zero, endpoints, stats, dist.device)
            action_result = extract_branch_features(
                model, final_layer, normalized["action_state"], normalized["action_omega"],
                normalized["mask"], action["force"], endpoints, args.chunk_size
            )
            records[panel]["actions"][pair_id] = action_result
            reproduction_max = max(reproduction_max, action_result["reproduction_max_abs"])
            if first_check is None:
                first_check = (action_result["first_input"], action_result["first_state_output"])
            if item["phase"] not in records[panel]["zeros"]:
                zero_result = extract_branch_features(
                    model, final_layer, normalized["zero_state"], normalized["zero_omega"],
                    normalized["mask"], zero["force"], endpoints, args.chunk_size
                )
                records[panel]["zeros"][item["phase"]] = zero_result
                reproduction_max = max(reproduction_max, zero_result["reproduction_max_abs"])
            print(
                json.dumps(
                    {
                        "event": "feature_extraction_progress",
                        "panel": panel,
                        "pair_id": pair_id,
                        "endpoints": [endpoints[0], endpoints[-1]],
                        "finite": True,
                        "reproduction_max_abs": reproduction_max,
                    }
                ),
                flush=True,
            )

    if reproduction_max > REPRODUCTION_ATOL:
        raise ValueError("manual affine readout does not reproduce model force output")
    with torch.no_grad():
        repeated_state = model(first_check[0])[:, :3]
    field_repeat_bitwise = torch.equal(repeated_state, first_check[1])
    state_sha_after = tensor_state_sha256(model)
    if not field_repeat_bitwise or state_sha_before != state_sha_after:
        raise ValueError("fixed model/state changed during diagnostic")

    panels = {}
    cache = {}
    for panel, branches in records.items():
        structured = {
            "action_features": {}, "zero_features": {},
            "action_targets_normalized": {}, "zero_targets_normalized": {},
            "force_std": force_std,
        }
        for pair_id, value in branches["actions"].items():
            structured["action_features"][pair_id] = value["features"]
            structured["action_targets_normalized"][pair_id] = (value["targets_physical"] - force_mean) / force_std
        for phase, value in branches["zeros"].items():
            structured["zero_features"][phase] = value["features"]
            structured["zero_targets_normalized"][phase] = (value["targets_physical"] - force_mean) / force_std
        xfit, yfit = weighted_fit_arrays(structured, pair_ids) if panel.startswith("prefix") else (None, None)
        if xfit is not None:
            coefficients, least_squares = fit_affine_readout(xfit, yfit)
        for branch_name, branch in (("action", structured["action_features"]), ("zero", structured["zero_features"])):
            for identity, array in branch.items():
                cache[f"{panel}__{branch_name}__{identity.replace(':', '_')}__features"] = array
                cache[
                    f"{panel}__{branch_name}__{identity.replace(':', '_')}__targets_normalized"
                ] = structured[f"{branch_name}_targets_normalized"][identity]
                source = records[panel][f"{branch_name}s"][identity]
                cache[
                    f"{panel}__{branch_name}__{identity.replace(':', '_')}__original_predictions_normalized"
                ] = source["original_predictions_normalized"]
        panels[panel] = structured
    fitted = {panel: evaluate_panel(value, pair_ids, coefficients) for panel, value in panels.items()}

    original_weight = final_layer.linear.weight[3:7].detach().cpu().numpy().astype(np.float64).T
    original_bias = final_layer.linear.bias[3:7].detach().cpu().numpy().astype(np.float64)
    original_coefficients = np.concatenate((original_weight, original_bias[None]), axis=0)
    original = {panel: evaluate_panel(value, pair_ids, original_coefficients) for panel, value in panels.items()}
    cache["fitted_coefficients"] = coefficients.astype(np.float64)
    cache["original_coefficients"] = original_coefficients.astype(np.float64)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    cache_path = args.output_dir / "fixed_features.npz"
    np.savez_compressed(cache_path, **cache)
    result = {
        "status": "FCP003C_FIXED_FEATURE_FORCE_READOUT_DIAGNOSTIC_COMPLETE",
        "scope": "train-only representability diagnostic; not a new model, admission, or control result",
        "checkpoint_epoch": 2,
        "model_sha256": MODEL_SHA,
        "training_state_sha256": STATE_SHA,
        "fit_window": "targets 1..100 inclusive",
        "strict_late_check": "targets 101..200 inclusive",
        "fit_weighting": {
            "unique_rows": 1200,
            "action_rows": 800,
            "unique_zero_rows": 400,
            "same_phase_zero_rows_repeated_per_pair": 800,
            "weighted_rows": 1600,
        },
        "unique_extraction_counts": {"action_endpoints": 1600, "zero_endpoints": 800, "total": 2400},
        "least_squares": least_squares,
        "fitted_readout_metrics_physical": fitted,
        "original_readout_metrics_physical": original,
        "original_output_reproduction_max_abs_normalized": reproduction_max,
        "original_output_reproduction_atol": REPRODUCTION_ATOL,
        "numerical_protocol": {
            "precision_before": precision_before,
            "precision_effective": precision_effective,
            "scope": "diagnostic-only fixed highest-FP32; differs from the historical C training/formal-evaluation default TF32 protocol",
            "formal_c_bitwise_equivalence_claimed": False,
        },
        "field_repeat_bitwise_identical": field_repeat_bitwise,
        "model_tensor_state_sha256_before": state_sha_before,
        "model_tensor_state_sha256_after": state_sha_after,
        "cache": {"path": cache_path.name, "sha256": sha256(cache_path)},
        "input_sha256": {
            "train8_manifest": TRAIN8_SHA, "base_manifest": BASE_SHA,
            "dynamic_pair_manifest": PAIR_SHA, "normalization": NORM_SHA,
            "config": CONFIG_SHA, "model": MODEL_SHA, "training_state": STATE_SHA,
        },
        "implementation_sha256": sha256(Path(__file__)),
        "force_channels": list(CHANNELS),
        "rcond": RCOND,
        "optimizer_steps": 0,
        "diagnostic_coefficients_saved": True,
        "deployable_checkpoint_saved": False,
        "candidate_saved": False,
        "validation_accessed": False,
        "frozen_test_accessed": False,
        "ppo_executed": False,
    }
    _atomic_json(args.output_dir / "result.json", result)
    print(result["status"], least_squares["rank"], reproduction_max)


if __name__ == "__main__":
    main()
