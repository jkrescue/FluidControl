#!/usr/bin/env python3
"""Train-only FC-P012 decoder gradient decomposition (no update or save)."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

TRAINER_SHA = "9c761cfcb3d4f18dbe35aed1b3defe0614b29db867006055dae63d9fa94485a5"
P009_MODEL_SHA = "dc41fc91d42476e052970b39fc66aed22fa72aa8b6f218a341a3abb095f42e31"
P009_STATE_SHA = "4998e534d4b82b17393c217357ed18220fb8e739166a88147483bb9cc5fb771e"
B_MODEL_SHA = "5102e83e00276994ad85b88c59721d52915400a037fc29f321d593fa69b800d8"
B_STATE_SHA = "c3c8c92ea792ed7332da75af269143b10ea0d0541b84273b7424e2922d9ae6b0"
MODEL_CONTRACTS = {
    "p009_parent": (P009_MODEL_SHA, P009_STATE_SHA, 0),
    "fcp011_decoder_tail": (B_MODEL_SHA, B_STATE_SHA, 1),
}
GROUPS = ("hidden", "rear_cl_row", "complete")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_trainer(path: Path):
    if sha256(path) != TRAINER_SHA:
        raise ValueError("FC-P011 trainer SHA differs")
    spec = importlib.util.spec_from_file_location("fcp011_frozen_trainer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen FC-P011 trainer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mask_component_gradients(
    values: Sequence[Any], names: Sequence[str], trainer: Any
) -> list[Any]:
    """Apply the exact post-backward final-row mask used by FC-P011."""
    import torch

    if len(values) != 4 or len(names) != 4:
        raise ValueError("expected exact four decoder-tail tensors")
    expected = {
        trainer.HIDDEN_WEIGHT,
        trainer.HIDDEN_BIAS,
        trainer.FINAL_WEIGHT,
        trainer.FINAL_BIAS,
    }
    if set(names) != expected:
        raise ValueError("decoder-tail parameter identities differ")
    result = []
    for value, name in zip(values, names, strict=True):
        if value is None:
            raise RuntimeError(f"missing gradient graph for {name}")
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"non-finite gradient for {name}")
        copy = value.detach().clone()
        if name in {trainer.FINAL_WEIGHT, trainer.FINAL_BIAS}:
            if copy.shape[0] != 7:
                raise ValueError("official final decoder output shape differs")
            copy[: trainer.REAR_CL_INDEX].zero_()
        result.append(copy)
    return result


def select_group(values: Sequence[Any], names: Sequence[str], group: str, trainer: Any):
    if group not in GROUPS:
        raise ValueError("unknown gradient group")
    selected = []
    for value, name in zip(values, names, strict=True):
        hidden = name in {trainer.HIDDEN_WEIGHT, trainer.HIDDEN_BIAS}
        rear = name in {trainer.FINAL_WEIGHT, trainer.FINAL_BIAS}
        if group == "complete" or (group == "hidden" and hidden) or (group == "rear_cl_row" and rear):
            selected.append(value)
    if not selected:
        raise RuntimeError("empty gradient group")
    return selected


def vector_stats(values: Sequence[Any]) -> dict[str, Any]:
    import torch

    count = sum(value.numel() for value in values)
    finite = sum(int(torch.isfinite(value).sum()) for value in values)
    if finite != count:
        raise FloatingPointError("non-finite gradient vector")
    squared = sum(float(value.detach().double().abs().square().sum()) for value in values)
    return {
        "l2": math.sqrt(squared),
        "linf": max(float(value.detach().abs().max()) for value in values),
        "element_count": count,
        "finite_count": finite,
        "nonzero_count": sum(int(torch.count_nonzero(value)) for value in values),
    }


def compare_vectors(left: Sequence[Any], right: Sequence[Any]) -> dict[str, Any]:
    if len(left) != len(right):
        raise ValueError("gradient vector lengths differ")
    ls, rs = vector_stats(left), vector_stats(right)
    dot = sum(
        float(a.detach().double().flatten().dot(b.detach().double().flatten()))
        for a, b in zip(left, right, strict=True)
    )
    denominator = ls["l2"] * rs["l2"]
    return {
        "left_l2": ls["l2"],
        "right_l2": rs["l2"],
        "cosine": dot / denominator if denominator else None,
        "cosine_defined": denominator > 0,
        "left_to_right_norm_ratio": ls["l2"] / rs["l2"] if rs["l2"] else None,
    }


def residual_stats(total: Sequence[Any], parts: Sequence[Any]) -> dict[str, Any]:
    residual = [a - b for a, b in zip(total, parts, strict=True)]
    total_stats, residual_summary = vector_stats(total), vector_stats(residual)
    relative = (
        residual_summary["l2"] / total_stats["l2"] if total_stats["l2"] else None
    )
    return {
        "absolute_l2": residual_summary["l2"],
        "relative_l2": relative,
        "interpretation": "observational_only_no_predeclared_tolerance",
    }


def component_gradients(loss, parameters, names, trainer, *, retain_graph: bool):
    import torch

    raw = torch.autograd.grad(
        loss, parameters, retain_graph=retain_graph, allow_unused=False
    )
    return mask_component_gradients(raw, names, trainer)


def decompose(losses, parameters, names, trainer) -> dict[str, Any]:
    field = component_gradients(
        losses["field"], parameters, names, trainer, retain_graph=True
    )
    weighted_force = component_gradients(
        0.2 * losses["balanced_force"], parameters, names, trainer, retain_graph=True
    )
    total = component_gradients(
        losses["total"], parameters, names, trainer, retain_graph=False
    )
    summed = [a + b for a, b in zip(field, weighted_force, strict=True)]
    groups = {}
    for group in GROUPS:
        f = select_group(field, names, group, trainer)
        w = select_group(weighted_force, names, group, trainer)
        t = select_group(total, names, group, trainer)
        s = select_group(summed, names, group, trainer)
        groups[group] = {
            "field": vector_stats(f),
            "weighted_balanced_force": vector_stats(w),
            "total_direct": vector_stats(t),
            "field_vs_weighted_force": compare_vectors(f, w),
            "total_direct_vs_component_sum": residual_stats(t, s),
        }
    return groups


def checkpoint_pair(root: Path, model_sha: str, state_sha: str) -> tuple[Path, Path]:
    models, states = sorted(root.glob("FNO.0.*.mdlus")), sorted(root.glob("checkpoint.0.*.pt"))
    if len(models) != 1 or len(states) != 1:
        raise ValueError("checkpoint pair differs")
    if sha256(models[0]) != model_sha or sha256(states[0]) != state_sha:
        raise ValueError("checkpoint SHA differs")
    return models[0], states[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--p009-parent", type=Path, required=True)
    parser.add_argument("--decoder-tail", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    trainer = load_trainer(args.trainer)

    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils import load_checkpoint
    from train_tandem_fno import build_model, configured_force_indices
    from train_tandem_fno_paired_stats import rollout

    from fluid_control.augmented_datapipe import compose_training_data
    from fluid_control.calibrated_checkpoint import validate_calibrated_epoch_zero
    from fluid_control.tandem_datapipe import TandemRolloutDataset

    if sha256(args.config) != trainer.CONFIG_SHA:
        raise ValueError("resolved config SHA differs")
    cfg = OmegaConf.load(args.config)
    if int(cfg.training.rollout_steps) != 100 or int(cfg.training.batch_size) != 1:
        raise ValueError("H100/batch-one contract differs")
    if not 0 < float(cfg.training.gpu_memory_fraction) <= 0.45:
        raise ValueError("GPU allocator fraction exceeds approved bound")
    input_sha = trainer.validate_data_contract(cfg)
    checkpoints = {
        "p009_parent": args.p009_parent,
        "fcp011_decoder_tail": args.decoder_tail,
    }
    for label, root in checkpoints.items():
        checkpoint_pair(root, *MODEL_CONTRACTS[label][:2])

    DistributedManager.initialize()
    dist = DistributedManager()
    if dist.distributed or not dist.cuda:
        raise RuntimeError("FC-P012 requires exactly one CUDA device")
    torch.cuda.set_per_process_memory_fraction(
        float(cfg.training.gpu_memory_fraction), dist.device
    )
    if (
        torch.get_float32_matmul_precision() != "high"
        or not torch.backends.cuda.matmul.allow_tf32
        or not torch.backends.cudnn.allow_tf32
    ):
        raise RuntimeError("default-TF32/high protocol differs")
    seed = int(cfg.training.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    force_indices = configured_force_indices(cfg)
    if tuple(force_indices) != (0, 1, 2, 3):
        raise ValueError("force channel order differs")
    base = TandemRolloutDataset(
        cfg.data.root, "train", 100, stride=int(cfg.training.train_stride),
        num_workers=cfg.training.workers, force_indices=force_indices,
    )
    train, _ = compose_training_data(
        base, [Path(value) for value in cfg.data.additional_train_roots],
        rollout_steps=100, stride=int(cfg.training.additional_train_stride),
        workers=cfg.training.workers, force_indices=force_indices,
    )
    windows = trainer.diagnostic_windows(train)
    results = []
    model_evidence = []
    for label, root in checkpoints.items():
        model = build_model(cfg).to(dist.device)
        metadata: dict[str, Any] = {}
        epoch = load_checkpoint(root, models=model, metadata_dict=metadata, device=dist.device)
        expected_epoch = MODEL_CONTRACTS[label][2]
        if label == "p009_parent":
            validate_calibrated_epoch_zero(
                root, epoch, allow=True,
                expected_model_sha256=P009_MODEL_SHA,
                expected_state_sha256=P009_STATE_SHA,
                expected_kind=trainer.PARENT_KIND,
            )
        elif epoch != expected_epoch or metadata.get("scope") != "decoder_tail":
            raise ValueError("decoder-tail checkpoint identity differs")
        trainer.configure_trainable_scope(model, "decoder_tail")
        named = dict(model.named_parameters())
        names = (
            trainer.HIDDEN_WEIGHT, trainer.HIDDEN_BIAS,
            trainer.FINAL_WEIGHT, trainer.FINAL_BIAS,
        )
        parameters = tuple(named[name] for name in names)
        expected_shapes = ((128, 128), (128,), (7, 128), (7,))
        if tuple(tuple(value.shape) for value in parameters) != expected_shapes:
            raise ValueError("official decoder parameter shapes differ")
        before = trainer.tensor_state_sha256(model)
        model.train()
        for item in windows:
            started = time.monotonic()
            batch = {key: value[None].to(dist.device) for key, value in item["sample"].items()}
            predicted_state, predicted_force = rollout(
                model, batch["state"], batch["mask"], batch["omega"],
                batch["target_state"], 0.0,
            )
            losses = trainer.balanced_decoder_objective(
                predicted_state, predicted_force, batch["target_state"],
                batch["target_force"], batch["mask"],
            )
            groups = decompose(losses, parameters, names, trainer)
            row = {
                "model": label,
                "model_sha256": MODEL_CONTRACTS[label][0],
                "identity": item["identity"],
                "global_index": item["global_index"],
                "family": item["family"],
                "losses": {
                    "field": float(losses["field"].detach()),
                    "balanced_force": float(losses["balanced_force"].detach()),
                    "weighted_balanced_force": float((0.2 * losses["balanced_force"]).detach()),
                    "total": float(losses["total"].detach()),
                },
                "gradient_groups": groups,
                "elapsed_seconds": time.monotonic() - started,
            }
            results.append(row)
            print(
                json.dumps(
                    {
                        "event": "fcp012_gradient_window_complete",
                        "completed_rows": len(results),
                        "total_rows": 12,
                        "model": label,
                        "identity": item["identity"],
                        "elapsed_seconds": row["elapsed_seconds"],
                    },
                    allow_nan=False,
                ),
                flush=True,
            )
            del predicted_state, predicted_force, losses, groups
        after = trainer.tensor_state_sha256(model)
        if after != before:
            raise RuntimeError("diagnostic changed model tensors")
        model_evidence.append(
            {
                "model": label,
                "checkpoint_model_sha256": MODEL_CONTRACTS[label][0],
                "checkpoint_state_sha256": MODEL_CONTRACTS[label][1],
                "tensor_sha256_before": before,
                "tensor_sha256_after": after,
                "tensor_state_unchanged": True,
                "parameter_names": list(names),
            }
        )
        del model
        torch.cuda.empty_cache()
    train.close()
    if len(results) != 12:
        raise RuntimeError("FC-P012 requires exactly twelve rows")
    result = {
        "status": "FC_P012_TRAIN_ONLY_DECODER_GRADIENT_DIAGNOSTIC_COMPLETE",
        "scope": "train-only diagnostic; no optimizer, clipping, update, checkpoint save, validation, frozen, or PPO",
        "trainer_sha256": TRAINER_SHA,
        "config_sha256": trainer.CONFIG_SHA,
        "input_sha256": input_sha,
        "arithmetic": "default TF32 with float32 matmul precision high; float64 summaries only",
        "component_sum_residual_policy": (
            "observational only: no unvalidated hard tolerance; any material residual must be "
            "reported as numerically uncertain before interpretation"
        ),
        "model_evidence": model_evidence,
        "rows": results,
        "optimizer_created": False,
        "optimizer_steps": 0,
        "clipping_applied": False,
        "candidate_saved": False,
        "validation_accessed": False,
        "frozen_test_accessed": False,
        "ppo_executed": False,
    }
    args.output.mkdir(parents=True)
    temporary = args.output / "result.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output / "result.json")


if __name__ == "__main__":
    main()
