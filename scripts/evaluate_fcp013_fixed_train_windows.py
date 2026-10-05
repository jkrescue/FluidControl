#!/usr/bin/env python3
"""Project diagnostic: unchanged P011 physical metrics on fixed train windows.

Read-only comparison of the immutable P009 parent and the P013 terminal dual
official FNO system. This is not admission, selection, PPO or CFD control.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from train_fcp013_independent_force_fno import (
    CONFIG_SHA, PARENT_KIND, PARENT_MODEL_SHA, PARENT_STATE_SHA,
    checkpoint_pair, load_frozen_trainer, sha256, tensor_state_sha256,
)


def check_training_result(result: dict, manifest_sha: str) -> None:
    required = {
        "status": "FC_P013_INDEPENDENT_FORCE_FNO_TRAINING_COMPLETE_NOT_ADMISSION",
        "optimizer_steps": 1368,
        "dual_fresh_reload_verified": True,
        "dual_fresh_reload_manifest_sha256": manifest_sha,
        "dual_model_manifest_sha256": manifest_sha,
        "validation_accessed": False,
        "frozen_test_accessed": False,
        "ppo_executed": False,
        "selection_performed": False,
    }
    if any(result.get(key) != value for key, value in required.items()):
        raise ValueError("training terminal contract differs")
    rows = result.get("records", [])
    if len(rows) != 1368 or [r.get("step") for r in rows] != list(range(1, 1369)):
        raise ValueError("actual optimizer record sequence differs")
    for row in rows:
        if row.get("optimizer_steps") != 1:
            raise ValueError("window must contain one optimizer update")
        identity = row.get("identity", {})
        if identity.get("split") != "train" or identity.get("rollout_steps") != 100:
            raise ValueError("training split or horizon differs")
        for key in ("total", "h1_balanced", "ar_balanced", "preclip_gradient_norm"):
            value = row.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("training metric missing or nonfinite")
    before = result.get("flow_tensor_sha256_before")
    if not before or before != result.get("flow_tensor_sha256_after"):
        raise ValueError("frozen flow changed")


def check_diagnostic_comparison(parent: dict, candidate: dict) -> None:
    """Require exact panel identity and complete finite physical metrics."""
    left, right = parent.get("windows", []), candidate.get("windows", [])
    if len(left) != 6 or len(right) != 6:
        raise ValueError("six fixed windows required")
    vector_metrics = (
        "free_ar_field_relative_l2_uvp", "true_state_h1_field_relative_l2_uvp",
        "initial_endpoint_h1_force_absolute_error", "true_state_h1_force_mae",
        "free_ar_force_mae",
    )
    scalar_metrics = (
        "tail62_mean_rear_cl_absolute_error", "tail62_rear_cl_rms_absolute_error",
        "true_state_h1_tail62_mean_rear_cl_absolute_error",
        "true_state_h1_tail62_rear_cl_rms_absolute_error",
    )
    for a, b in zip(left, right, strict=True):
        for key in ("global_index", "family", "identity"):
            if key not in a or a[key] != b.get(key):
                raise ValueError("diagnostic panel identity mismatch")
        for row in (a, b):
            for key in vector_metrics:
                values = row.get(key)
                expected = 3 if key.endswith("uvp") else 4
                if not isinstance(values, list) or len(values) != expected:
                    raise ValueError("diagnostic vector shape differs")
                if not all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in values):
                    raise ValueError("invalid diagnostic vector")
            for key in scalar_metrics:
                value = row.get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError("invalid diagnostic scalar")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--frozen-trainer", type=Path, required=True)
    parser.add_argument("--dual-manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--expected-training-result-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha256(args.config) != CONFIG_SHA:
        raise ValueError("config differs")
    if sha256(args.training_result) != args.expected_training_result_sha256:
        raise ValueError("training result identity differs")
    check_training_result(json.loads(args.training_result.read_text()), args.expected_manifest_sha256)
    trainer = load_frozen_trainer(args.frozen_trainer)
    checkpoint_pair(args.parent)

    import torch
    from omegaconf import OmegaConf
    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils import load_checkpoint
    from fluid_control.augmented_datapipe import compose_training_data
    from fluid_control.calibrated_checkpoint import validate_calibrated_epoch_zero
    from fluid_control.dual_fno import load_dual_fno, validate_dual_runtime_files
    from fluid_control.tandem_datapipe import TandemRolloutDataset
    from train_tandem_fno import build_model, configured_force_indices, predict
    from train_tandem_fno_paired_stats import rollout

    cfg = OmegaConf.load(args.config)
    input_sha = trainer.validate_data_contract(cfg)
    DistributedManager.initialize()
    dist = DistributedManager()
    if dist.distributed or not dist.cuda:
        raise RuntimeError("single CUDA device required")
    torch.cuda.set_per_process_memory_fraction(0.15, dist.device)
    force_indices = configured_force_indices(cfg)
    if tuple(force_indices) != (0, 1, 2, 3):
        raise ValueError("force channel order differs")
    base = TandemRolloutDataset(
        cfg.data.root, "train", 100, stride=int(cfg.training.train_stride),
        num_workers=cfg.training.workers, force_indices=force_indices,
    )
    train, _ = compose_training_data(
        base, [Path(v) for v in cfg.data.additional_train_roots], rollout_steps=100,
        stride=int(cfg.training.additional_train_stride), workers=cfg.training.workers,
        force_indices=force_indices,
    )
    try:
        items = trainer.diagnostic_windows(train)
        parent = build_model(cfg).to(dist.device)
        epoch = load_checkpoint(args.parent, models=parent, device=dist.device)
        validate_calibrated_epoch_zero(
            args.parent, epoch, allow=True, expected_model_sha256=PARENT_MODEL_SHA,
            expected_state_sha256=PARENT_STATE_SHA, expected_kind=PARENT_KIND,
        )
        parent.eval().requires_grad_(False)
        dual, identity = load_dual_fno(
            args.dual_manifest, cfg, dist.device, build_model=build_model,
            expected_manifest_sha256=args.expected_manifest_sha256,
        )
        validate_dual_runtime_files(identity, config_path=args.config,
                                   normalization_path=Path(cfg.data.root) / "normalization.json")
        before = {"parent": tensor_state_sha256(parent), "dual": tensor_state_sha256(dual)}
        panels = {}
        for label, model in (("p009_parent", parent), ("p013_terminal", dual)):
            panels[label] = trainer.evaluate_diagnostics(
                model, items, dist.device, base.state_mean.to(dist.device),
                base.state_std.to(dist.device), base.force_mean.to(dist.device),
                base.force_std.to(dist.device), rollout, predict,
            )
            print(json.dumps({"event": "fixed_train_panel_complete", "model": label,
                              "windows": 6}), flush=True)
        after = {"parent": tensor_state_sha256(parent), "dual": tensor_state_sha256(dual)}
        if before != after:
            raise RuntimeError("read-only diagnostics changed tensors")
        check_diagnostic_comparison(panels["p009_parent"], panels["p013_terminal"])
        result = {
            "status": "FC_P013_FIXED_TRAIN_DIAGNOSTICS_COMPLETE_NOT_ADMISSION",
            "panels": panels, "input_sha256": input_sha,
            "dual_manifest_sha256": identity.manifest_sha256,
            "training_result_sha256": args.expected_training_result_sha256,
            "frozen_numerical_source_sha256": sha256(args.frozen_trainer),
            "config_sha256": CONFIG_SHA, "tensor_sha256_before": before,
            "tensor_sha256_after": after, "optimizer_created": False,
            "optimizer_steps": 0, "candidate_saved": False, "selection_performed": False,
            "validation_accessed": False, "frozen_test_accessed": False, "ppo_executed": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    finally:
        train.close()


if __name__ == "__main__":
    main()
