#!/usr/bin/env python3
"""Cache FC-P009 train-only free-AR H100 features and fixed-alpha0 diagnostics.

PhysicsNeMo supplies the pinned FNO, DataPipe, and checkpoint APIs.  The
free-autoregressive feature extraction and force-row least-squares calibration
are project methods.  No validation, frozen, PPO, or new CFD data are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from build_fcp008_force_readout_candidate import (
    CHANNELS,
    CONFIG_SHA,
    MAPPING_SHA,
    MODEL_SHA,
    NORM_SHA,
    PHASES,
    STATE_SHA,
    Trajectory,
    aggregate_metrics,
    assign_source_phases,
    atomic_json,
    default_precision_contract,
    discover_trajectories,
    fit_weighted_ridge,
    grouped_metrics,
    inventory_payload,
    model_tensor_sha256,
    predict,
    sha256,
    validate_mapping_artifact,
)

ROLLOUT_STEPS = 100
FAMILY_STRIDES = {"base20": 20, "train8": 2, "train16": 2}
FAMILY_WINDOWS = {"base20": 720, "train8": 408, "train16": 240}
FAMILY_UNIQUE_ENDPOINTS = {"base20": 16000, "train8": 1600, "train16": 2048}
TOTAL_WINDOWS = 1368
TOTAL_ROWS = TOTAL_WINDOWS * ROLLOUT_STEPS
TOTAL_UNIQUE_CFD_ENDPOINTS = sum(FAMILY_UNIQUE_ENDPOINTS.values())
WIRING_ATOL = 2e-5


@dataclass(frozen=True)
class WindowRecord:
    family: str
    case: str
    source_phase: str
    start: int


def regular_window_records(records: list[Trajectory]) -> list[WindowRecord]:
    """Enumerate the exact regular H100 windows without claiming independence."""
    windows = []
    for record in records:
        stride = FAMILY_STRIDES[record.family]
        windows.extend(
            WindowRecord(record.family, record.name, record.source_phase, start)
            for start in range(0, record.frames - ROLLOUT_STEPS, stride)
        )
    counts = {
        family: sum(window.family == family for window in windows)
        for family in FAMILY_WINDOWS
    }
    phase_counts = {
        phase: sum(window.source_phase == phase for window in windows)
        for phase in PHASES
    }
    if counts != FAMILY_WINDOWS or phase_counts != {
        "b00": 402,
        "b02": 402,
        "b04": 282,
        "b06": 282,
    }:
        raise ValueError(f"regular H100 window inventory differs: {counts}, {phase_counts}")
    return windows


def window_inventory_payload(records: list[Trajectory]) -> dict:
    windows = regular_window_records(records)
    multiplicity = {}
    for window in windows:
        for endpoint in range(window.start + 1, window.start + ROLLOUT_STEPS + 1):
            key = (window.family, window.case, endpoint)
            multiplicity[key] = multiplicity.get(key, 0) + 1
    if len(multiplicity) != TOTAL_UNIQUE_CFD_ENDPOINTS:
        raise ValueError("unique CFD endpoint count differs")
    by_family = {}
    for family in FAMILY_WINDOWS:
        values = [count for key, count in multiplicity.items() if key[0] == family]
        by_family[family] = {
            "windows": FAMILY_WINDOWS[family],
            "window_step_rows": FAMILY_WINDOWS[family] * ROLLOUT_STEPS,
            "unique_cfd_endpoints": len(values),
            "endpoint_multiplicity_min": min(values),
            "endpoint_multiplicity_max": max(values),
        }
    return {
        "status": "FC_P009_REGULAR_H100_WINDOW_INVENTORY_PASS",
        "window_count": len(windows),
        "window_step_row_count": len(windows) * ROLLOUT_STEPS,
        "unique_cfd_endpoint_count": len(multiplicity),
        "rows_are_independent_physical_samples": False,
        "weighting": "each of 1368 regular H100 windows and each of its 100 rollout steps has equal weight",
        "family": by_family,
        "source_phase_window_counts": {
            phase: sum(window.source_phase == phase for window in windows)
            for phase in PHASES
        },
        "validation_accessed": False,
        "frozen_test_accessed": False,
    }


def _metadata_item(metadata: dict, name: str, index: int):
    if isinstance(metadata, (list, tuple)) and metadata and isinstance(metadata[0], dict):
        return metadata[index][name]
    value = metadata[name]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return value[index]
    if index != 0:
        raise ValueError(f"uncollated metadata for batch index {index}")
    return value


def canonical_force_targets(record, start, stats):
    """Match P008's documented float64-normalize then float32-label formula."""
    import h5py

    with h5py.File(record.path, "r") as hdf:
        physical = np.asarray(
            hdf["force"][start + 1 : start + ROLLOUT_STEPS + 1], dtype=np.float32
        )
    if physical.shape != (ROLLOUT_STEPS, 4):
        raise ValueError("physical force endpoint slice differs")
    return (
        (
            physical.astype(np.float64)
            - np.asarray(stats["all_force_mean"], dtype=np.float64)
        )
        / np.asarray(stats["all_force_std"], dtype=np.float64)
    ).astype(np.float32), physical


def free_ar_batch(model, layer, batch, device, *, collect_features: bool = True):
    """Roll one batch for H100 and capture the existing final-layer features."""
    import torch

    state = batch["state"].to(device).float()
    target_force = batch["target_force"].to(device).float()
    omega = batch["omega"].to(device).float()
    mask = batch["mask"].to(device).float()
    if (
        state.ndim != 4
        or state.shape[1] != 3
        or target_force.shape != (len(state), ROLLOUT_STEPS, 4)
        or omega.shape != (len(state), ROLLOUT_STEPS + 1, 1)
        or mask.shape[:2] != (len(state), 1)
    ):
        raise ValueError("regular rollout batch shape differs")
    batch_size, _, height, width = state.shape
    pixels = height * width
    features, native = [], []
    state_digest = hashlib.sha256()
    wiring_max = 0.0
    captured_input, captured_output = [], []

    def pre_hook(_module, args):
        captured_input.append(args[0])

    def post_hook(_module, _args, output):
        captured_output.append(output)

    pre = layer.register_forward_pre_hook(pre_hook)
    post = layer.register_forward_hook(post_hook)
    try:
        with torch.no_grad():
            for step in range(ROLLOUT_STEPS):
                captured_input.clear()
                captured_output.clear()
                now = omega[:, step].reshape(batch_size, 1, 1, 1).expand(-1, 1, height, width)
                following = omega[:, step + 1].reshape(batch_size, 1, 1, 1).expand(-1, 1, height, width)
                inputs = torch.cat((state, mask, now, following), dim=1)
                raw = model(inputs)
                if len(captured_input) != 1 or len(captured_output) != 1:
                    raise ValueError("official final layer was not called exactly once")
                hidden = captured_input[0]
                point_output = captured_output[0]
                if hidden.shape != (batch_size * pixels, 128) or point_output.shape != (batch_size * pixels, 7):
                    raise ValueError("official final-layer topology differs")
                grid_output = point_output.reshape(batch_size, height, width, 7).permute(0, 3, 1, 2)
                wiring_max = max(wiring_max, float((grid_output - raw).abs().max().item()))
                force = (raw[:, 3:] * mask).sum((-2, -1)) / mask.sum((-2, -1)).clamp_min(1)
                native.append(force.detach().cpu())
                state_digest.update(
                    raw[:, :3].detach().cpu().contiguous().numpy().tobytes()
                )
                if collect_features:
                    hidden = hidden.reshape(batch_size, pixels, 128)
                    flat_mask = mask.permute(0, 2, 3, 1).reshape(batch_size, pixels, 1)
                    features.append(((hidden * flat_mask).sum(1) / flat_mask.sum(1)).detach().cpu())
                state = (state + raw[:, :3]) * mask
    finally:
        pre.remove()
        post.remove()
    if wiring_max > WIRING_ATOL:
        raise ValueError(f"native pointwise-head wiring differs: {wiring_max}")
    result = {
        "features": torch.stack(features, dim=1).numpy() if collect_features else None,
        "targets": target_force.detach().cpu().numpy(),
        "native": torch.stack(native, dim=1).numpy(),
        "state_sha256": state_digest.hexdigest(),
        "wiring_max_abs": wiring_max,
    }
    if any(not np.isfinite(value).all() for value in result.values() if isinstance(value, np.ndarray)):
        raise ValueError("nonfinite free-AR batch result")
    return result


def phase_oof_alpha_zero(features, targets, phases):
    """Report fixed-alpha0 phase-held diagnostics without selecting anything."""
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    phases = np.asarray(phases)
    weights = np.full(len(features), 1.0 / len(features), dtype=np.float64)
    predictions = np.empty_like(targets)
    folds = {}
    for phase in PHASES:
        train = phases != phase
        held = ~train
        coefficients, fit = fit_weighted_ridge(features[train], targets[train], weights[train], 0.0)
        predictions[held] = predict(features[held], coefficients)
        folds[phase] = fit
    error = predictions - targets
    return {
        "alpha": 0.0,
        "selection_performed": False,
        "global_equal_window_step_metrics": aggregate_metrics(error, weights),
        "fold_fits": folds,
    }


def cross_domain_phase_oof(
    free_ar_features, h1_features, targets, phases, relative_steps, force_std=None
):
    """Fit both train domains and score their held phases on both domains."""
    free_ar_features = np.asarray(free_ar_features, dtype=np.float64)
    h1_features = np.asarray(h1_features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    phases = np.asarray(phases).astype(str)
    relative_steps = np.asarray(relative_steps)
    if free_ar_features.shape != h1_features.shape or targets.shape != (len(phases), 4):
        raise ValueError("cross-domain feature/target shapes differ")
    predictions = {
        "fit_free_ar_predict_free_ar": np.empty_like(targets),
        "fit_free_ar_predict_h1": np.empty_like(targets),
        "fit_h1_predict_free_ar": np.empty_like(targets),
        "fit_h1_predict_h1": np.empty_like(targets),
    }
    fold_fits = {}
    for phase in PHASES:
        train = phases != phase
        held = ~train
        weights = np.full(int(train.sum()), 1.0 / int(train.sum()), dtype=np.float64)
        ar_coefficients, ar_fit = fit_weighted_ridge(
            free_ar_features[train], targets[train], weights, 0.0
        )
        h1_coefficients, h1_fit = fit_weighted_ridge(
            h1_features[train], targets[train], weights, 0.0
        )
        predictions["fit_free_ar_predict_free_ar"][held] = predict(free_ar_features[held], ar_coefficients)
        predictions["fit_free_ar_predict_h1"][held] = predict(h1_features[held], ar_coefficients)
        predictions["fit_h1_predict_free_ar"][held] = predict(free_ar_features[held], h1_coefficients)
        predictions["fit_h1_predict_h1"][held] = predict(h1_features[held], h1_coefficients)
        fold_fits[phase] = {"free_ar": ar_fit, "matched_weight_h1": h1_fit}
    if force_std is None:
        force_std = np.ones((1, 4), dtype=np.float64)
    force_std = np.asarray(force_std, dtype=np.float64).reshape(1, 4)
    reports = {}
    uniform = np.full(len(targets), 1.0 / len(targets), dtype=np.float64)
    for name, values in predictions.items():
        error = values - targets
        horizons = {}
        for horizon in (1, 10, 50, 100):
            mask = relative_steps == horizon
            if not mask.any():
                raise ValueError(f"missing relative H{horizon} rows")
            local = np.full(int(mask.sum()), 1.0 / int(mask.sum()), dtype=np.float64)
            horizons[f"H{horizon}"] = aggregate_metrics(error[mask], local)
        reports[name] = {
            "normalized": {
                "all_steps": aggregate_metrics(error, uniform),
                "relative_horizons": horizons,
            },
            "physical": {
                "all_steps": aggregate_metrics(error * force_std, uniform),
                "relative_horizons": {
                    key: aggregate_metrics(
                        error[relative_steps == int(key[1:])] * force_std,
                        np.full(
                            int((relative_steps == int(key[1:])).sum()),
                            1.0 / int((relative_steps == int(key[1:])).sum()),
                            dtype=np.float64,
                        ),
                    )
                    for key in horizons
                },
            },
        }
    return {"alpha": 0.0, "selection_performed": False, "fold_fits": fold_fits, "reports": reports}


def expand_matched_h1_cache(cache, case_names, window_starts, relative_steps, targets, phases, families):
    """Expand P008 unique H1 endpoints to the exact FC-P009 row weighting."""
    required = {
        "features", "targets_normalized", "phases", "families", "trajectory_names"
    }
    if not required.issubset(cache):
        raise ValueError("P008 cache keys differ")
    features = np.asarray(cache["features"])
    source_targets = np.asarray(cache["targets_normalized"])
    source_phases = np.asarray(cache["phases"]).astype(str)
    source_families = np.asarray(cache["families"]).astype(str)
    source_names = np.asarray(cache["trajectory_names"]).astype(str)
    if (
        features.ndim != 2
        or features.shape[1] != 128
        or source_targets.shape != (len(features), 4)
        or any(len(value) != len(features) for value in (source_phases, source_families, source_names))
    ):
        raise ValueError("P008 H1 cache shape differs")
    endpoint_counter = {}
    lookup = {}
    for index, name in enumerate(source_names):
        endpoint = endpoint_counter.get(name, 0) + 1
        endpoint_counter[name] = endpoint
        key = (name, endpoint)
        if key in lookup:
            raise ValueError("P008 endpoint key is duplicated")
        lookup[key] = index
    case_names = np.asarray(case_names).astype(str)
    endpoints = np.asarray(window_starts, dtype=np.int64) + np.asarray(relative_steps, dtype=np.int64)
    indices = np.asarray([lookup.get((case, int(endpoint)), -1) for case, endpoint in zip(case_names, endpoints, strict=True)])
    if (indices < 0).any():
        raise ValueError("free-AR row is absent from P008 H1 cache")
    if not np.array_equal(source_targets[indices], np.asarray(targets)):
        raise ValueError("matched H1 and free-AR physical force targets differ")
    if not np.array_equal(source_phases[indices], np.asarray(phases).astype(str)):
        raise ValueError("matched H1 physical phases differ")
    if not np.array_equal(source_families[indices], np.asarray(families).astype(str)):
        raise ValueError("matched H1 source families differ")
    return {
        "features": features[indices],
        "indices": indices,
        "unique_source_rows": int(np.unique(indices).size),
    }


def validate_train_roots(base, train8, train16, normalization):
    if sha256(normalization) != NORM_SHA:
        raise ValueError("normalization SHA differs")
    expected = normalization.read_bytes()
    for root in (base, train8, train16):
        if (root / "normalization.json").read_bytes() != expected:
            raise ValueError(f"training root normalization differs: {root}")
        manifest = json.loads((root / "manifest.json").read_text())
        if float(manifest.get("max_abs_omega", -1)) != 0.75:
            raise ValueError(f"training root action scale differs: {root}")


def verify_loader_first_batches(datasets, p008_cache, record_by_key):
    """Prove endpoint identity while preserving both documented FP32 formulas."""
    import torch
    from physicsnemo.datapipes import DataLoader

    reports = []
    for family, dataset in datasets.items():
        loader = DataLoader(
            dataset, batch_size=1, shuffle=False, collate_metadata=True,
            prefetch_factor=0, use_streams=False,
        )
        batch, metadata = next(iter(loader))
        case = str(_metadata_item(metadata, "case", 0))
        start = int(_metadata_item(metadata, "step", 0))
        record = record_by_key[(family, case)]
        canonical, physical_force = canonical_force_targets(
            record, start, dataset.stats
        )
        expanded = expand_matched_h1_cache(
            p008_cache,
            np.repeat(case, ROLLOUT_STEPS),
            np.repeat(start, ROLLOUT_STEPS),
            np.arange(1, ROLLOUT_STEPS + 1),
            canonical,
            np.repeat(record.source_phase, ROLLOUT_STEPS),
            np.repeat(family, ROLLOUT_STEPS),
        )
        if expanded["features"].shape != (ROLLOUT_STEPS, 128):
            raise ValueError("official loader/P008 H1 alignment differs")
        official_expected = (
            torch.as_tensor(physical_force).float() - dataset.force_mean[None]
        ) / dataset.force_std[None]
        official_actual = batch["target_force"].reshape(ROLLOUT_STEPS, 4)
        if not torch.equal(official_actual, official_expected):
            raise ValueError("official loader target differs from its FP32 normalization formula")
        reports.append(
            {
                "family": family,
                "case": case,
                "start": start,
                "physical_target_endpoints": [start + 1, start + ROLLOUT_STEPS],
                "p008_float64_then_float32_targets_exact": True,
                "official_loader_float32_formula_targets_exact": True,
                "normalization_arithmetic_max_abs_difference": float(
                    np.max(np.abs(canonical - official_actual.numpy()))
                ),
            }
        )
    return reports


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "base", "train8", "train16", "normalization", "config",
        "checkpoint-dir", "source-phase-mapping", "p008-cache", "approval", "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.exists() or args.batch_size != 4 or not 0 <= args.workers <= 2:
        raise ValueError("exclusive output, batch4, and bounded workers required")
    records = assign_source_phases(discover_trajectories(args.base, args.train8, args.train16))
    source_inventory = inventory_payload(records)
    window_inventory = window_inventory_payload(records)
    validate_mapping_artifact(records, args.source_phase_mapping)
    approval = json.loads(args.approval.read_text())
    if (
        approval.get("status") != "FC_P009_FREE_AR_CACHE_EXECUTION_APPROVED"
        or approval.get("implementation_sha256") != sha256(Path(__file__))
        or approval.get("parent_model_sha256") != MODEL_SHA
        or approval.get("alpha") != 0.0
    ):
        raise ValueError("FC-P009 execution approval differs")
    validate_train_roots(args.base, args.train8, args.train16, args.normalization)
    if sha256(args.config) != CONFIG_SHA:
        raise ValueError("normalization/config differs")
    models = list(args.checkpoint_dir.glob("FNO.0.*.mdlus"))
    states = list(args.checkpoint_dir.glob("checkpoint.0.*.pt"))
    if len(models) != 1 or sha256(models[0]) != MODEL_SHA or len(states) != 1 or sha256(states[0]) != STATE_SHA:
        raise ValueError("fixed C epoch2 parent differs")
    if sha256(args.p008_cache) != "22d6c8b29bf8df27917b3e14d791ade07a8c18eeaefec12c7301d1a6b1bf2bff":
        raise ValueError("P008 H1 cache SHA differs")
    p008_cache = dict(np.load(args.p008_cache, allow_pickle=False))
    if len(p008_cache.get("features", ())) != TOTAL_UNIQUE_CFD_ENDPOINTS:
        raise ValueError("P008 H1 cache endpoint count differs")

    import torch
    from evaluate_tandem_fno import load_composed_config
    from fluid_control.tandem_datapipe import TandemRolloutDataset
    from physicsnemo.datapipes import DataLoader
    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils import load_checkpoint
    from train_tandem_fno import build_model, configured_force_indices

    record_by_key = {(record.family, record.name): record for record in records}
    roots = {"base20": args.base, "train8": args.train8, "train16": args.train16}
    datasets = {
        family: TandemRolloutDataset(
            root, "train", ROLLOUT_STEPS, stride=FAMILY_STRIDES[family],
            num_workers=args.workers, force_indices=(0, 1, 2, 3),
        )
        for family, root in roots.items()
    }
    if {family: len(dataset) for family, dataset in datasets.items()} != FAMILY_WINDOWS:
        raise ValueError("runtime regular window counts differ")
    loader_preflight = verify_loader_first_batches(datasets, p008_cache, record_by_key)
    for dataset in datasets.values():
        dataset.close()
    datasets = {
        family: TandemRolloutDataset(
            root, "train", ROLLOUT_STEPS, stride=FAMILY_STRIDES[family],
            num_workers=args.workers, force_indices=(0, 1, 2, 3),
        )
        for family, root in roots.items()
    }

    DistributedManager.initialize()
    dist = DistributedManager()
    if dist.distributed or not dist.cuda:
        raise RuntimeError("single-GPU CUDA execution required")
    torch.cuda.set_per_process_memory_fraction(0.15, device=dist.device)
    precision = default_precision_contract(torch)
    config = load_composed_config(args.config)
    if tuple(configured_force_indices(config)) != (0, 1, 2, 3):
        raise ValueError("four-force output contract differs")
    model = build_model(config).to(dist.device)
    if load_checkpoint(args.checkpoint_dir, models=model, device=dist.device) != 2:
        raise ValueError("expected C epoch2")
    model.eval()
    parent_tensor_before = model_tensor_sha256(model)
    stats = json.loads(args.normalization.read_text())
    if tuple(stats.get("all_force_channels", ())) != CHANNELS:
        raise ValueError("force-channel order differs")
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "source_inventory.json", source_inventory)
    atomic_json(args.output / "window_inventory.json", window_inventory)

    feature_parts, target_parts, native_parts = [], [], []
    phase_parts, family_parts, case_parts, start_parts, step_parts = [], [], [], [], []
    state_digest = hashlib.sha256()
    wiring_max = 0.0
    processed_windows = 0
    layer = model.decoder_net.final_layer
    for family, dataset in datasets.items():
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, collate_metadata=True,
            prefetch_factor=0, use_streams=False,
        )
        for batch, metadata in loader:
            extracted = free_ar_batch(model, layer, batch, dist.device)
            batch_size = len(extracted["features"])
            feature_parts.append(extracted["features"].reshape(-1, 128))
            native_parts.append(extracted["native"].reshape(-1, 4))
            state_digest.update(extracted["state_sha256"].encode())
            wiring_max = max(wiring_max, extracted["wiring_max_abs"])
            for index in range(batch_size):
                case = str(_metadata_item(metadata, "case", index))
                start = int(_metadata_item(metadata, "step", index))
                record = record_by_key.get((family, case))
                if record is None:
                    raise ValueError(f"runtime case is absent from inventory: {family}/{case}")
                canonical_targets, _physical = canonical_force_targets(
                    record, start, stats
                )
                target_parts.append(canonical_targets)
                phase_parts.extend([record.source_phase] * ROLLOUT_STEPS)
                family_parts.extend([family] * ROLLOUT_STEPS)
                case_parts.extend([case] * ROLLOUT_STEPS)
                start_parts.extend([start] * ROLLOUT_STEPS)
                step_parts.extend(range(1, ROLLOUT_STEPS + 1))
            processed_windows += batch_size
            print(json.dumps({"event": "fcp009_extract_progress", "family": family, "windows": processed_windows, "rows": processed_windows * 100, "finite": True}), flush=True)
        dataset.close()
    features = np.concatenate(feature_parts).astype(np.float32)
    targets = np.concatenate(target_parts).astype(np.float32)
    parent_native = np.concatenate(native_parts).astype(np.float32)
    phases = np.asarray(phase_parts)
    families = np.asarray(family_parts)
    if features.shape != (TOTAL_ROWS, 128) or targets.shape != (TOTAL_ROWS, 4) or processed_windows != TOTAL_WINDOWS:
        raise ValueError("free-AR extraction total differs")
    equal_weights = np.full(TOTAL_ROWS, 1.0 / TOTAL_ROWS, dtype=np.float64)
    free_ar_oof = phase_oof_alpha_zero(features, targets, phases)
    free_ar_coefficients, free_ar_fit = fit_weighted_ridge(features, targets, equal_weights, 0.0)
    free_ar_ideal = predict(features, free_ar_coefficients)
    matched = expand_matched_h1_cache(
        p008_cache, case_parts, start_parts, step_parts, targets, phases, families
    )
    if matched["unique_source_rows"] != TOTAL_UNIQUE_CFD_ENDPOINTS:
        raise ValueError("matched H1 expansion does not cover all physical endpoints")
    h1_oof = phase_oof_alpha_zero(matched["features"], targets, phases)
    h1_coefficients, h1_fit = fit_weighted_ridge(
        matched["features"], targets, equal_weights, 0.0
    )
    h1_ideal = predict(matched["features"], h1_coefficients)
    cross_domain = cross_domain_phase_oof(
        features,
        matched["features"],
        targets,
        phases,
        np.asarray(step_parts),
        np.asarray(stats["all_force_std"], dtype=np.float64),
    )
    parent_tensor_after = model_tensor_sha256(model)
    if parent_tensor_after != parent_tensor_before:
        raise ValueError("parent parameters/buffers changed during cache extraction")
    h1_mask = np.asarray(step_parts) == 1
    h1_feature_difference = (
        features[h1_mask].astype(np.float64)
        - matched["features"][h1_mask].astype(np.float64)
    )

    np.savez_compressed(
        args.output / "train_free_ar_features.npz",
        features=features,
        matched_weight_h1_features=matched["features"].astype(np.float32),
        matched_weight_h1_source_indices=matched["indices"].astype(np.int32),
        targets_normalized=targets,
        parent_native_normalized=parent_native,
        phases=phases,
        families=families,
        case_names=np.asarray(case_parts),
        window_starts=np.asarray(start_parts, dtype=np.int16),
        relative_steps=np.asarray(step_parts, dtype=np.int16),
        equal_row_weights=equal_weights,
        free_ar_alpha0_coefficients=free_ar_coefficients,
        matched_weight_h1_alpha0_coefficients=h1_coefficients,
    )
    parent_error = parent_native.astype(np.float64) - targets
    free_ar_ideal_error = free_ar_ideal - targets
    h1_ideal_error = h1_ideal - targets
    result = {
        "status": "FC_P009_TRAIN_ONLY_FREE_AR_FEATURE_CACHE_COMPLETE_NOT_ADMISSION",
        "scope": "train-only free-AR feature cache and fixed-alpha0 comparison; no candidate, checkpoint, validation, PPO, or control evidence",
        "precision_protocol": precision,
        "source_inventory": source_inventory,
        "window_inventory": window_inventory,
        "official_loader_first_batch_preflight": loader_preflight,
        "source_phase_mapping_sha256": MAPPING_SHA,
        "parent_model_sha256": MODEL_SHA,
        "parent_training_state_sha256": STATE_SHA,
        "parent_checkpoint_epoch": 2,
        "parent_tensor_sha256_before": parent_tensor_before,
        "parent_tensor_sha256_after": parent_tensor_after,
        "alpha": 0.0,
        "alpha_selection_performed": False,
        "free_ar_fit": free_ar_fit,
        "free_ar_phase_held_alpha0_diagnostic": free_ar_oof,
        "matched_weight_h1_fit": h1_fit,
        "matched_weight_h1_phase_held_alpha0_diagnostic": h1_oof,
        "cross_domain_phase_held_alpha0_diagnostic": cross_domain,
        "matched_weight_h1_unique_source_rows": matched["unique_source_rows"],
        "comparison_isolates": "hidden feature source at identical 136800 row identities, targets, phase folds, and equal row weights",
        "relative_h1_feature_numerical_difference": {
            "count": int(h1_mask.sum()),
            "max_abs": float(np.max(np.abs(h1_feature_difference))),
            "rms": float(np.sqrt(np.mean(np.square(h1_feature_difference)))),
            "scope": "observational batch/layout numerical difference; not a gate or rollout-error attribution",
        },
        "train_free_ar_parent_native_normalized": aggregate_metrics(parent_error, equal_weights),
        "train_free_ar_ideal_normalized": aggregate_metrics(free_ar_ideal_error, equal_weights),
        "train_free_ar_ideal_by_phase_family": grouped_metrics(free_ar_ideal_error, phases, families, equal_weights),
        "matched_weight_h1_ideal_normalized": aggregate_metrics(h1_ideal_error, equal_weights),
        "matched_weight_h1_ideal_by_phase_family": grouped_metrics(h1_ideal_error, phases, families, equal_weights),
        "free_ar_state_output_sha256": state_digest.hexdigest(),
        "feature_cache_sha256": sha256(args.output / "train_free_ar_features.npz"),
        "input_sha256": {
            "base_manifest": sha256(args.base / "manifest.json"),
            "base_train_split": sha256(args.base / "splits/train.json"),
            "train8_manifest": sha256(args.train8 / "manifest.json"),
            "train16_manifest": sha256(args.train16 / "manifest.json"),
            "normalization": sha256(args.normalization),
            "resolved_config": sha256(args.config),
            "source_phase_mapping": sha256(args.source_phase_mapping),
            "parent_model": sha256(models[0]),
            "parent_training_state": sha256(states[0]),
            "p008_h1_cache": sha256(args.p008_cache),
            "execution_approval": sha256(args.approval),
            "implementation": sha256(Path(__file__)),
        },
        "optimizer_steps": 0,
        "architecture_changed": False,
        "calibration_fit_performed": True,
        "candidate_saved": False,
        "validation_accessed": False,
        "frozen_test_accessed": False,
        "ppo_executed": False,
        "formal_evaluation_authorized": False,
    }
    atomic_json(args.output / "result.json", result)
    print(result["status"], result["feature_cache_sha256"], flush=True)


if __name__ == "__main__":
    main()
