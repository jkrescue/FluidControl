#!/usr/bin/env python3
"""Bounded no-grad native-vs-ideal FC-P009 train-window diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_fcp009_cache_result import atomic_json, sha256, validate_cache_schema
from analyze_fcp009_window_statistics import force_window_statistics
from build_fcp008_force_readout_candidate import CONFIG_SHA, NORM_SHA, model_tensor_sha256
from build_fcp009_free_ar_force_readout_candidate import (
    FAMILY_STRIDES,
    default_precision_contract,
    free_ar_batch,
)

KIND = "FC_P009_TRAIN_ONLY_JOINT_FORCE_ROW_CANDIDATE"
MODEL_SHA = "dc41fc91d42476e052970b39fc66aed22fa72aa8b6f218a341a3abb095f42e31"
STATE_SHA = "4998e534d4b82b17393c217357ed18220fb8e739166a88147483bb9cc5fb771e"
CACHE_SHA = "fc1b84fdd5a43d2b29e1f068531940bd6c12f320556d767a2540dfb5a238ca84"
CACHE_RESULT_SHA = "1321c30a1e12172b85b269405c208981390f952bdf6f03a7fe2dd21f3bb91daf"
MANIFEST_SHA = {
    "base20": "5213c7bb07c824c6e601636c3cfa974b80ec7c051633b0c8368a045cea41ddd2",
    "train8": "a0bd0e3b79d4d43ace407292c31a595d552dbd6e1a5de1fd149b1ca446f53f35",
    "train16": "7c62dab94e317442cecf7ea3be547cf6e03c594a2c7ec2a1a695caba2445cf5b",
}
SELECTED = (
    {"family": "base20", "case": "matched_start_acquisition_train_b00_zero", "start": 320, "global_window_index": 160, "local_dataset_index": 160, "role": "zero_control"},
    {"family": "train8", "case": "dynamic_train8_b00_prbs", "start": 90, "global_window_index": 816, "local_dataset_index": 96, "role": "high_ideal_rms_error_prbs_b00"},
    {"family": "train8", "case": "dynamic_train8_b02_prbs", "start": 100, "global_window_index": 923, "local_dataset_index": 203, "role": "high_ideal_rms_error_prbs_b02"},
    {"family": "train16", "case": "direct_cfd_directppo2048_v1_env0_ep0009_b00", "start": 0, "global_window_index": 1233, "local_dataset_index": 105, "role": "high_ideal_rms_error_train16_ep9"},
)


def validate_selected_cache(cache: dict[str, np.ndarray]) -> None:
    """Bind each predeclared window to the immutable 1368-window cache."""
    for selected in SELECTED:
        begin = selected["global_window_index"] * 100
        end = begin + 100
        expected = {
            "families": selected["family"],
            "case_names": selected["case"],
            "window_starts": selected["start"],
        }
        for key, value in expected.items():
            actual = np.asarray(cache[key])[begin:end]
            if actual.shape != (100,) or np.any(actual != value):
                raise ValueError(f"predeclared {key} differs for {selected['case']}")
        if not np.array_equal(np.asarray(cache["relative_steps"])[begin:end], np.arange(1, 101)):
            raise ValueError("predeclared relative-step grid differs")


def actual_force_head(weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Return the actual saved four-force affine coefficients in [129,4] form."""
    weight = np.asarray(weight)
    bias = np.asarray(bias)
    if weight.shape != (7, 128) or bias.shape != (7,) or not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise ValueError("saved final affine shape differs")
    return np.concatenate((weight[3:7].T, bias[3:7][None]), axis=0)


def validate_runtime_batch_context(
    selected: dict[str, object],
    metadata_items: list[dict[str, object]],
    cache: dict[str, np.ndarray],
) -> None:
    """Prove all four runtime batch members match the original cached batch4."""
    if len(metadata_items) != 4:
        raise ValueError("runtime batch4 metadata differs")
    batch_begin = (int(selected["local_dataset_index"]) // 4) * 4
    family_global_offset = int(selected["global_window_index"]) - int(selected["local_dataset_index"])
    for offset, metadata in enumerate(metadata_items):
        global_window = family_global_offset + batch_begin + offset
        row = global_window * 100
        expected = {
            "case": str(np.asarray(cache["case_names"])[row]),
            "step": int(np.asarray(cache["window_starts"])[row]),
            "rollout_steps": 100,
            "split": "train",
        }
        if metadata != expected or str(np.asarray(cache["families"])[row]) != selected["family"]:
            raise ValueError("runtime batch4 cache identity differs")


def physical_window_comparison(
    truth_normalized: np.ndarray,
    ideal_normalized: np.ndarray,
    native_normalized: np.ndarray,
    force_mean: np.ndarray,
    force_std: np.ndarray,
) -> dict[str, object]:
    """Compare the same fixed head under pooled-affine and native execution."""
    arrays = tuple(np.asarray(value, dtype=np.float64) for value in (truth_normalized, ideal_normalized, native_normalized))
    if any(value.shape != (100, 4) or not np.isfinite(value).all() for value in arrays):
        raise ValueError("window force arrays differ")
    truth, ideal, native = (value * force_std + force_mean for value in arrays)

    def error_summary(prediction: np.ndarray, target: np.ndarray) -> dict[str, object]:
        error = prediction - target
        return {
            "channel_mae": np.mean(np.abs(error), axis=0).tolist(),
            "channel_rmse": np.sqrt(np.mean(np.square(error), axis=0)).tolist(),
            "channel_bias": np.mean(error, axis=0).tolist(),
            "channel_max_abs": np.max(np.abs(error), axis=0).tolist(),
            "per_step_physical_error": error.tolist(),
        }

    truth_stats = force_window_statistics(truth)
    ideal_stats = force_window_statistics(ideal)
    native_stats = force_window_statistics(native)
    return {
        "truth_window": truth_stats,
        "ideal_window": ideal_stats,
        "native_window": native_stats,
        "ideal_absolute_statistic_error": {key: abs(ideal_stats[key] - truth_stats[key]) for key in truth_stats},
        "native_absolute_statistic_error": {key: abs(native_stats[key] - truth_stats[key]) for key in truth_stats},
        "native_minus_ideal_statistic": {key: native_stats[key] - ideal_stats[key] for key in truth_stats},
        "ideal_minus_truth": error_summary(ideal, truth),
        "native_minus_truth": error_summary(native, truth),
        "native_minus_ideal": error_summary(native, ideal),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "train8", "train16", "normalization", "config", "checkpoint-dir", "cache-result", "cache", "approval", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise ValueError("exclusive output required")
    expected = {
        args.normalization: NORM_SHA,
        args.config: CONFIG_SHA,
        args.cache_result: CACHE_RESULT_SHA,
        args.cache: CACHE_SHA,
    }
    if any(not path.is_file() or sha256(path) != digest for path, digest in expected.items()):
        raise ValueError("fixed diagnostic input SHA differs")
    model_path = args.checkpoint_dir / "FNO.0.0.mdlus"
    state_path = args.checkpoint_dir / "checkpoint.0.0.pt"
    if sha256(model_path) != MODEL_SHA or sha256(state_path) != STATE_SHA:
        raise ValueError("fixed P009 checkpoint differs")
    approval = json.loads(args.approval.read_text())
    if (
        approval.get("status") != "FC_P009_NATIVE_IDEAL_WINDOW_DIAGNOSTIC_EXECUTION_APPROVED"
        or approval.get("implementation_sha256") != sha256(Path(__file__))
        or approval.get("candidate_kind") != KIND
        or approval.get("candidate_model_sha256") != MODEL_SHA
        or approval.get("candidate_state_sha256") != STATE_SHA
        or approval.get("selected_windows") != list(SELECTED)
        or approval.get("optimizer_steps") != 0
        or approval.get("validation_or_frozen_accessed") is not False
        or approval.get("ppo_executed") is not False
    ):
        raise ValueError("execution approval differs")
    normalization = json.loads(args.normalization.read_text())
    force_mean = np.asarray(normalization["all_force_mean"], dtype=np.float64)
    force_std = np.asarray(normalization["all_force_std"], dtype=np.float64)
    if force_mean.shape != (4,) or force_std.shape != (4,) or not np.all(force_std > 0):
        raise ValueError("force normalization differs")
    cache = dict(np.load(args.cache, allow_pickle=False))
    free_ar, h1, targets, _phases, _steps = validate_cache_schema(cache)
    validate_selected_cache(cache)

    import torch
    from evaluate_tandem_fno import load_composed_config
    from fluid_control.calibrated_checkpoint import validate_calibrated_epoch_zero
    from fluid_control.tandem_datapipe import TandemRolloutDataset
    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils import load_checkpoint
    from train_tandem_fno import build_model

    DistributedManager.initialize()
    dist = DistributedManager()
    if dist.distributed or not dist.cuda:
        raise RuntimeError("single-GPU CUDA execution required")
    torch.cuda.set_per_process_memory_fraction(0.15, device=dist.device)
    precision = default_precision_contract(torch)
    config = load_composed_config(args.config)
    model = build_model(config).to(dist.device)
    metadata = {}
    loaded_epoch = load_checkpoint(args.checkpoint_dir, models=model, metadata_dict=metadata, device=dist.device)
    validate_calibrated_epoch_zero(
        args.checkpoint_dir,
        loaded_epoch,
        allow=True,
        expected_model_sha256=MODEL_SHA,
        expected_state_sha256=STATE_SHA,
        expected_kind=KIND,
    )
    model.eval()
    tensor_sha_before = model_tensor_sha256(model)
    layer = model.decoder_net.final_layer.linear
    saved_coefficients = actual_force_head(
        layer.weight.detach().cpu().numpy(), layer.bias.detach().cpu().numpy()
    )
    head_sha256 = hashlib.sha256(saved_coefficients.tobytes()).hexdigest()

    def saved_pooled_affine(features: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            value = torch.nn.functional.linear(
                torch.as_tensor(features, dtype=layer.weight.dtype, device=dist.device),
                layer.weight[3:7],
                layer.bias[3:7],
            )
        return value.detach().cpu().numpy()

    roots = {"base20": args.base, "train8": args.train8, "train16": args.train16}
    if any(sha256(root / "manifest.json") != MANIFEST_SHA[family] for family, root in roots.items()):
        raise ValueError("train manifest differs")
    reports = []
    for selected in SELECTED:
        dataset = TandemRolloutDataset(
            roots[selected["family"]],
            "train",
            100,
            stride=FAMILY_STRIDES[selected["family"]],
            num_workers=1,
            force_indices=(0, 1, 2, 3),
        )
        try:
            batch_begin = (selected["local_dataset_index"] // 4) * 4
            loaded = [dataset._load(index) for index in range(batch_begin, batch_begin + 4)]
            samples = [item[0] for item in loaded]
            metadata_items = [item[1] for item in loaded]
            validate_runtime_batch_context(selected, metadata_items, cache)
            selected_offset = selected["local_dataset_index"] - batch_begin
            metadata_item = metadata_items[selected_offset]
            batch = {
                key: torch.stack([sample[key] for sample in samples])
                for key in samples[0].keys()
            }
            native = free_ar_batch(model, model.decoder_net.final_layer, batch, dist.device, collect_features=True)
        finally:
            dataset.close()
        begin = selected["global_window_index"] * 100
        cached_features = free_ar[begin : begin + 100]
        runtime_features = native["features"][selected_offset]
        cached_saved_pooled = saved_pooled_affine(cached_features)
        runtime_saved_pooled = saved_pooled_affine(runtime_features)
        comparison = physical_window_comparison(
            targets[begin : begin + 100], runtime_saved_pooled, native["native"][selected_offset], force_mean, force_std
        )
        cached_comparison = physical_window_comparison(
            targets[begin : begin + 100], cached_saved_pooled, native["native"][selected_offset], force_mean, force_std
        )
        feature_difference = runtime_features.astype(np.float64) - cached_features.astype(np.float64)
        reports.append(
            {
                **selected,
                "runtime_metadata": metadata_item,
                "runtime_batch_size": 4,
                "runtime_batch_dataset_indices": list(range(batch_begin, batch_begin + 4)),
                "runtime_selected_batch_offset": selected_offset,
                "native_pointwise_wiring_max_abs": native["wiring_max_abs"],
                "runtime_feature_vs_cache": {
                    "max_abs": float(np.max(np.abs(feature_difference))),
                    "root_mean_squared": float(np.sqrt(np.mean(np.square(feature_difference)))),
                },
                "same_runtime_feature_native_vs_ideal": comparison,
                "cached_feature_saved_pooled_vs_runtime_native": cached_comparison,
            }
        )
        print(json.dumps({"event": "fcp009_native_ideal_window", "case": selected["case"], "start": selected["start"], "finite": True}), flush=True)
    tensor_sha_after = model_tensor_sha256(model)
    if tensor_sha_after != tensor_sha_before:
        raise ValueError("diagnostic changed model tensors")
    payload = {
        "status": "FC_P009_NATIVE_IDEAL_TRAIN_WINDOW_DIAGNOSTIC_COMPLETE",
        "scope": "four predeclared train windows; native-vs-ideal mechanism diagnostic only",
        "selected_windows": list(SELECTED),
        "precision_protocol": precision,
        "window_definition": {"rollout_steps": 100, "trailing_relative_steps_inclusive": [39, 100], "sample_count": 62, "rear_cl_rms_ddof": 0},
        "actual_saved_force_head": {
            "coefficient_shape": list(saved_coefficients.shape),
            "coefficient_dtype": str(saved_coefficients.dtype),
            "sha256": head_sha256,
        },
        "reports": reports,
        "candidate_tensor_sha256_before": tensor_sha_before,
        "candidate_tensor_sha256_after": tensor_sha_after,
        "input_sha256": {
            "candidate_model": MODEL_SHA,
            "candidate_state": STATE_SHA,
            "cache_result": CACHE_RESULT_SHA,
            "cache": CACHE_SHA,
            "normalization": NORM_SHA,
            "resolved_config": CONFIG_SHA,
            "execution_approval": sha256(args.approval),
            "implementation": sha256(Path(__file__)),
        },
        "gpu_used": True,
        "no_grad": True,
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
