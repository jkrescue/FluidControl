#!/usr/bin/env python3
"""Build the bounded FC-P009 joint force-row candidate from frozen train caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_fcp009_cache_result import validate_cache_schema
from analyze_fcp009_joint_readout import fit_joint_coefficients
from build_fcp008_force_readout_candidate import (
    BASE_MANIFEST_SHA,
    BASE_SPLIT_SHA,
    CONFIG_SHA,
    MODEL_SHA,
    NORM_SHA,
    STATE_SHA,
    WIRING_ATOL,
    atomic_json,
    default_precision_contract,
    model_tensor_sha256,
    sha256,
    verify_force_row_only,
    zip_member_sha256,
)

KIND = "FC_P009_TRAIN_ONLY_JOINT_FORCE_ROW_CANDIDATE"
STATUS = f"{KIND}_COMPLETE_NOT_ADMISSION"
P009_RESULT_SHA = "1321c30a1e12172b85b269405c208981390f952bdf6f03a7fe2dd21f3bb91daf"
P009_CACHE_SHA = "fc1b84fdd5a43d2b29e1f068531940bd6c12f320556d767a2540dfb5a238ca84"
JOINT_ANALYSIS_SHA = "931fcd2ddd6901ddfbb3ecdbfe9774d7b1e5fe6d17479c87c44ccaf51ff2b0bc"
DOMAIN_MIX = {"free_ar": 0.5, "matched_weight_h1": 0.5}


def candidate_metadata(parent_model_sha256=MODEL_SHA):
    return {
        "status": KIND,
        "candidate_checkpoint_epoch": 0,
        "parent_checkpoint_epoch": 2,
        "calibration_generation": 1,
        "parent_model_sha256": parent_model_sha256,
        "alpha": 0.0,
        "domain_mix": DOMAIN_MIX,
        "calibration_fit_performed": True,
        "optimizer_training_performed": False,
        "validation_accessed": False,
        "frozen_test_accessed": False,
        "ppo_executed": False,
    }


def fit_from_cache(cache):
    free_ar, h1, targets, _phases, _steps = validate_cache_schema(cache)
    coefficients, fit = fit_joint_coefficients(
        free_ar, h1, targets, np.ones(len(targets), dtype=bool)
    )
    if coefficients.shape != (129, 4) or not np.isfinite(coefficients).all():
        raise ValueError("joint coefficient contract differs")
    return coefficients, fit


def native_h1_sanity(parent, candidate, layer_parent, layer_candidate, batch, device):
    """One deterministic base20 batch4 H1 check; no rollout or selection."""
    import torch

    state = batch["state"].to(device).float()
    omega = batch["omega"].to(device).float()
    mask = batch["mask"].to(device).float()
    if state.shape[0] != 4 or state.ndim != 4 or omega.shape[:2] != (4, 2):
        raise ValueError("native sanity batch differs")
    height, width = state.shape[-2:]
    now = omega[:, 0].reshape(4, 1, 1, 1).expand(-1, 1, height, width)
    following = omega[:, 1].reshape(4, 1, 1, 1).expand(-1, 1, height, width)
    inputs = torch.cat((state, mask, now, following), dim=1)

    def run(model, layer):
        captured = []
        handle = layer.register_forward_pre_hook(lambda _module, args: captured.append(args[0]))
        try:
            with torch.no_grad():
                raw = model(inputs)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise ValueError("final affine invocation count differs")
        hidden = captured[0]
        point = layer(hidden)
        replay = point.reshape(4, height, width, 7).permute(0, 3, 1, 2)
        wiring = float((replay - raw).abs().max().item())
        force = (raw[:, 3:] * mask).sum((-2, -1)) / mask.sum((-2, -1)).clamp_min(1)
        flat = hidden.reshape(4, height * width, 128)
        flat_mask = mask.permute(0, 2, 3, 1).reshape(4, height * width, 1)
        pooled = (flat * flat_mask).sum(1) / flat_mask.sum(1)
        pooled_force = pooled @ layer.weight[3:].T + layer.bias[3:]
        return raw, force, wiring, float((pooled_force - force).abs().max().item())

    parent_raw, parent_force, parent_wiring, parent_pooled = run(parent, layer_parent)
    candidate_raw, candidate_force, candidate_wiring, candidate_pooled = run(candidate, layer_candidate)
    if not torch.equal(parent_raw[:, :3], candidate_raw[:, :3]):
        raise ValueError("candidate changed H1 state output")
    if max(parent_wiring, candidate_wiring) > WIRING_ATOL:
        raise ValueError("captured pointwise head replay differs")
    if not torch.isfinite(parent_force).all() or not torch.isfinite(candidate_force).all():
        raise ValueError("native H1 force is nonfinite")
    return {
        "batch_size": 4,
        "rollout_horizon": 1,
        "parent_candidate_state_bitwise_equal": True,
        "parent_force_finite": True,
        "candidate_force_finite": True,
        "parent_pointwise_head_wiring_max_abs": parent_wiring,
        "candidate_pointwise_head_wiring_max_abs": candidate_wiring,
        "parent_pooled_affine_vs_native_max_abs_observational": parent_pooled,
        "candidate_pooled_affine_vs_native_max_abs_observational": candidate_pooled,
        "candidate_force_normalized": candidate_force.detach().cpu().numpy().tolist(),
    }


def metadata_values(metadata, key):
    if isinstance(metadata, (list, tuple)) and metadata and isinstance(metadata[0], dict):
        return [item[key] for item in metadata]
    value = metadata[key]
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value) if isinstance(value, (list, tuple)) else [value]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "base", "normalization", "config", "checkpoint-dir", "p009-result",
        "p009-cache", "joint-analysis", "approval", "output",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.exists():
        raise ValueError("exclusive output required")
    expected = {
        args.p009_result: P009_RESULT_SHA,
        args.p009_cache: P009_CACHE_SHA,
        args.joint_analysis: JOINT_ANALYSIS_SHA,
        args.normalization: NORM_SHA,
        args.config: CONFIG_SHA,
    }
    if any(not path.is_file() or sha256(path) != digest for path, digest in expected.items()):
        raise ValueError("fixed input SHA differs")
    source_result = json.loads(args.p009_result.read_text())
    joint = json.loads(args.joint_analysis.read_text())
    approval = json.loads(args.approval.read_text())
    if (
        source_result.get("status")
        != "FC_P009_TRAIN_ONLY_FREE_AR_FEATURE_CACHE_COMPLETE_NOT_ADMISSION"
        or joint.get("status") != "FC_P009_FIXED_50_50_JOINT_READOUT_DIAGNOSTIC_COMPLETE"
        or joint.get("domain_mix") != DOMAIN_MIX
        or joint.get("alpha") != 0.0
        or approval.get("status") != "FC_P009_JOINT_FORCE_ROW_CANDIDATE_EXECUTION_APPROVED"
        or approval.get("implementation_sha256") != sha256(Path(__file__))
        or approval.get("candidate_kind") != KIND
        or approval.get("candidate_build_authorized") is not True
        or approval.get("p009_result_sha256") != P009_RESULT_SHA
        or approval.get("p009_cache_sha256") != P009_CACHE_SHA
        or approval.get("joint_analysis_sha256") != JOINT_ANALYSIS_SHA
        or approval.get("parent_model_sha256") != MODEL_SHA
        or approval.get("parent_training_state_sha256") != STATE_SHA
        or approval.get("resolved_config_sha256") != CONFIG_SHA
        or approval.get("normalization_sha256") != NORM_SHA
        or approval.get("alpha") != 0.0
        or approval.get("domain_mix") != DOMAIN_MIX
        or approval.get("validation_or_frozen_accessed") is not False
        or approval.get("ppo_executed") is not False
    ):
        raise ValueError("source result/joint analysis/execution approval differs")
    models = list(args.checkpoint_dir.glob("FNO.0.*.mdlus"))
    states = list(args.checkpoint_dir.glob("checkpoint.0.*.pt"))
    if len(models) != 1 or len(states) != 1 or sha256(models[0]) != MODEL_SHA or sha256(states[0]) != STATE_SHA:
        raise ValueError("fixed C epoch2 parent differs")
    if sha256(args.base / "manifest.json") != BASE_MANIFEST_SHA:
        raise ValueError("base20 manifest differs")
    if (
        sha256(args.base / "normalization.json") != NORM_SHA
        or sha256(args.base / "splits/train.json") != BASE_SPLIT_SHA
    ):
        raise ValueError("base20 normalization/train split differs")
    cache = dict(np.load(args.p009_cache, allow_pickle=False))
    coefficients, fit = fit_from_cache(cache)

    import torch
    from evaluate_tandem_fno import load_composed_config
    from fluid_control.tandem_datapipe import TandemRolloutDataset
    from physicsnemo.datapipes import DataLoader
    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils import load_checkpoint, save_checkpoint
    from train_tandem_fno import build_model, configured_force_indices

    DistributedManager.initialize()
    dist = DistributedManager()
    if dist.distributed or not dist.cuda:
        raise RuntimeError("single-GPU CUDA execution required")
    torch.cuda.set_per_process_memory_fraction(0.15, device=dist.device)
    precision = default_precision_contract(torch)
    config = load_composed_config(args.config)
    if tuple(configured_force_indices(config)) != (0, 1, 2, 3):
        raise ValueError("four-force output contract differs")
    parent = build_model(config).to(dist.device)
    if load_checkpoint(args.checkpoint_dir, models=parent, device=dist.device) != 2:
        raise ValueError("expected fixed C epoch2")
    parent.eval()
    before = {name: value.detach().cpu().numpy().copy() for name, value in parent.state_dict().items()}
    candidate = build_model(config).to(dist.device)
    if load_checkpoint(args.checkpoint_dir, models=candidate, device=dist.device) != 2:
        raise ValueError("expected fixed C epoch2 candidate base")
    layer = candidate.decoder_net.final_layer.linear
    weight_key = next(name for name, value in candidate.named_parameters() if value is layer.weight)
    bias_key = next(name for name, value in candidate.named_parameters() if value is layer.bias)
    with torch.no_grad():
        layer.weight[3:7].copy_(torch.as_tensor(coefficients[:128].T, dtype=layer.weight.dtype, device=dist.device))
        layer.bias[3:7].copy_(torch.as_tensor(coefficients[128], dtype=layer.bias.dtype, device=dist.device))
    assigned = {name: value.detach().cpu().numpy().copy() for name, value in candidate.state_dict().items()}
    changed = verify_force_row_only(before, assigned, weight_key, bias_key, coefficients.astype(np.float32))
    assigned_sha = model_tensor_sha256(candidate)
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.output / "candidate"
    save_checkpoint(checkpoint, models=candidate, epoch=0, metadata=candidate_metadata())
    model_path, state_path = checkpoint / "FNO.0.0.mdlus", checkpoint / "checkpoint.0.0.pt"
    training_state = torch.load(state_path, map_location="cpu", weights_only=False)
    if training_state != {"metadata": candidate_metadata()}:
        raise ValueError("official candidate state metadata differs")
    fresh = build_model(config).to(dist.device)
    load_return = load_checkpoint(checkpoint, models=fresh, device=dist.device)
    fresh.eval()
    fresh_arrays = {name: value.detach().cpu().numpy().copy() for name, value in fresh.state_dict().items()}
    verify_force_row_only(before, fresh_arrays, weight_key, bias_key, coefficients.astype(np.float32))
    if load_return != 0 or model_tensor_sha256(fresh) != assigned_sha:
        raise ValueError("fresh official epoch0 reload differs")
    from fluid_control.calibrated_checkpoint import validate_calibrated_epoch_zero
    epoch_zero_identity = validate_calibrated_epoch_zero(
        checkpoint,
        0,
        allow=True,
        expected_model_sha256=sha256(model_path),
        expected_state_sha256=sha256(state_path),
        expected_kind=KIND,
    )
    dataset = TandemRolloutDataset(args.base, "train", 1, stride=1, num_workers=1, force_indices=(0, 1, 2, 3))
    try:
        batch, metadata = next(iter(DataLoader(dataset, batch_size=4, shuffle=False, collate_metadata=True, prefetch_factor=0, use_streams=False)))
        cases = [str(value) for value in metadata_values(metadata, "case")]
        steps = [int(value) for value in metadata_values(metadata, "step")]
        if cases != ["matched_start_acquisition_train_b00_m0375"] * 4 or steps != [0, 1, 2, 3]:
            raise ValueError("deterministic base20 native-sanity identities differ")
        sanity = native_h1_sanity(parent, fresh, parent.decoder_net.final_layer.linear, fresh.decoder_net.final_layer.linear, batch, dist.device)
        sanity["split"] = "train"
        sanity["cases"] = cases
        sanity["window_starts"] = steps
    finally:
        dataset.close()
    result = {
        "status": STATUS,
        "candidate_kind": KIND,
        "scope": "train-only cached fixed-50/50 alpha0 force-row calibration; not admission or control evidence",
        "parent_checkpoint_epoch": 2,
        "calibration_generation": 1,
        "candidate_checkpoint_epoch": 0,
        "alpha": 0.0,
        "domain_mix": DOMAIN_MIX,
        "precision_protocol": precision,
        "joint_fit": fit,
        "changed_parameter_names": changed,
        "parent_tensor_sha256": model_tensor_sha256(parent),
        "candidate_tensor_sha256": model_tensor_sha256(fresh),
        "fresh_official_load_checkpoint_return_epoch": load_return,
        "calibrated_epoch_zero_identity": epoch_zero_identity,
        "state_rows_0_3_byte_identical": True,
        "all_other_tensors_byte_identical": True,
        "native_h1_sanity": sanity,
        "candidate_model_sha256": sha256(model_path),
        "candidate_state_sha256": sha256(state_path),
        "candidate_model_archive_member_sha256": zip_member_sha256(model_path),
        "candidate_training_state_metadata": training_state["metadata"],
        "input_sha256": {
            "p009_result": P009_RESULT_SHA,
            "p009_cache": P009_CACHE_SHA,
            "joint_analysis": JOINT_ANALYSIS_SHA,
            "normalization": NORM_SHA,
            "resolved_config": CONFIG_SHA,
            "parent_model": MODEL_SHA,
            "parent_training_state": STATE_SHA,
            "execution_approval": sha256(args.approval),
            "implementation": sha256(Path(__file__)),
        },
        "calibration_fit_performed": True,
        "optimizer_training_performed": False,
        "optimizer_steps": 0,
        "architecture_changed": False,
        "validation_accessed": False,
        "frozen_test_accessed": False,
        "ppo_executed": False,
        "formal_evaluation_authorized": False,
    }
    atomic_json(args.output / "result.json", result)
    print(STATUS, result["candidate_model_sha256"], flush=True)


if __name__ == "__main__":
    main()
