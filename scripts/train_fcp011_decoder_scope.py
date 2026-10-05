#!/usr/bin/env python3
"""Train the two FC-P011 decoder-scope arms on one fixed train-only pass.

The official PhysicsNeMo FNO architecture is unchanged.  Project code limits
updates to either the rear-Cl output row alone or that row plus the final
existing decoder hidden layer.  Validation, frozen data, selection, and PPO
are intentionally absent from this entry point.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Mapping


PARENT_MODEL_SHA = "dc41fc91d42476e052970b39fc66aed22fa72aa8b6f218a341a3abb095f42e31"
PARENT_STATE_SHA = "4998e534d4b82b17393c217357ed18220fb8e739166a88147483bb9cc5fb771e"
PARENT_KIND = "FC_P009_TRAIN_ONLY_JOINT_FORCE_ROW_CANDIDATE"
CONFIG_SHA = "07e55fd11df8030313338cef0344490c3453e515aae9c6b6122e997bad5085d9"
BASE_MANIFEST_SHA = "5213c7bb07c824c6e601636c3cfa974b80ec7c051633b0c8368a045cea41ddd2"
TRAIN8_MANIFEST_SHA = "a0bd0e3b79d4d43ace407292c31a595d552dbd6e1a5de1fd149b1ca446f53f35"
TRAIN16_MANIFEST_SHA = "7c62dab94e317442cecf7ea3be547cf6e03c594a2c7ec2a1a695caba2445cf5b"
NORMALIZATION_SHA = "f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1"
EXPECTED_ORDER_SHA = "177ebd9523cde918eb0c1fb8026286dac7e95f3d228ae0e349757a2a9a288f9f"
SCOPES = ("head_only", "decoder_tail")
TOTAL_STEPS = 1368
DIAGNOSTIC_STEPS = (0, 32, 128, 512, 1368)
DIAGNOSTIC_WINDOWS = (
    (160, "base20", "matched_start_acquisition_train_b00_zero", 320),
    (816, "train8", "dynamic_train8_b00_prbs", 90),
    (923, "train8", "dynamic_train8_b02_prbs", 100),
    (975, "train8", "dynamic_train8_b04_prbs", 0),
    (1077, "train8", "dynamic_train8_b06_prbs", 0),
    (1233, "train16", "direct_cfd_directppo2048_v1_env0_ep0009_b00", 0),
)
REAR_CL_INDEX = 6
FINAL_WEIGHT = "decoder_net.final_layer.linear.weight"
FINAL_BIAS = "decoder_net.final_layer.linear.bias"
HIDDEN_WEIGHT = "decoder_net.layers.1.linear.weight"
HIDDEN_BIAS = "decoder_net.layers.1.linear.bias"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_state_sha256(module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def metadata_rows(metadata: Any) -> list[dict[str, Any]]:
    """Normalize official DataLoader metadata without guessing identities."""
    if isinstance(metadata, list) and all(isinstance(row, dict) for row in metadata):
        return metadata
    if isinstance(metadata, Mapping):
        lengths = {len(value) for value in metadata.values() if isinstance(value, (list, tuple))}
        if lengths == {1}:
            return [{key: value[0] if isinstance(value, (list, tuple)) else value for key, value in metadata.items()}]
    raise ValueError("expected one-item list-of-dicts or collated mapping metadata")


def training_identity(metadata: Any) -> dict[str, Any]:
    rows = metadata_rows(metadata)
    if len(rows) != 1:
        raise ValueError("FC-P011 requires batch size one")
    row = rows[0]
    result = {
        "case": str(row.get("case", row.get("name", ""))),
        "start": int(row.get("start", row.get("step", -1))),
        "dataset_index": int(row.get("dataset_index", -1)),
        "split": str(row.get("split", "")),
        "rollout_steps": int(row.get("rollout_steps", -1)),
    }
    if (
        not result["case"]
        or result["start"] < 0
        or result["dataset_index"] not in (0, 1, 2)
        or result["split"] != "train"
        or result["rollout_steps"] != 100
    ):
        raise ValueError("training metadata identity is incomplete")
    return result


def sequence_sha(values: list[int]) -> str:
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def identity_index(dataset) -> dict[tuple[str, int, int], int]:
    """Map exact official MultiDataset metadata to its global sampler index."""
    children = getattr(dataset, "_datasets", None)
    if not isinstance(children, list) or len(children) != 3:
        raise ValueError("expected exact three-family official MultiDataset")
    result: dict[tuple[str, int, int], int] = {}
    offset = 0
    for dataset_index, child in enumerate(children):
        for local_index, (file_index, step) in enumerate(child.index):
            key = (child.paths[file_index].stem, int(step), dataset_index)
            if key in result:
                raise ValueError("training identity is not unique")
            result[key] = offset + local_index
        offset += len(child)
    if len(result) != TOTAL_STEPS:
        raise ValueError("training identity map must contain 1368 windows")
    return result


def validate_data_contract(cfg) -> dict[str, str]:
    roots = [Path(cfg.data.root), *(Path(value) for value in cfg.data.additional_train_roots)]
    if len(roots) != 3:
        raise ValueError("FC-P011 requires base20/train8/train16 roots")
    expected_manifests = (BASE_MANIFEST_SHA, TRAIN8_MANIFEST_SHA, TRAIN16_MANIFEST_SHA)
    for root, expected in zip(roots, expected_manifests, strict=True):
        if sha256(root / "manifest.json") != expected:
            raise ValueError("training manifest differs")
        if sha256(root / "normalization.json") != NORMALIZATION_SHA:
            raise ValueError("training normalization differs")
        if any((root / name).exists() for name in ("validation", "frozen_test", "test")):
            raise ValueError("runtime training view exposes validation/frozen data")
    return {
        "base_manifest": BASE_MANIFEST_SHA,
        "train8_manifest": TRAIN8_MANIFEST_SHA,
        "train16_manifest": TRAIN16_MANIFEST_SHA,
        "normalization": NORMALIZATION_SHA,
    }


def balanced_decoder_objective(
    predicted_state,
    predicted_force,
    target_state,
    target_force,
    mask,
    *,
    force_loss_weight: float = 0.2,
):
    """Return the fixed FC-P011 objective and auditable raw contributions."""
    import torch

    if predicted_state.shape != target_state.shape or predicted_force.shape != target_force.shape:
        raise ValueError("prediction and target shapes differ")
    if predicted_force.ndim != 3 or predicted_force.shape[-1] != 4:
        raise ValueError("force tensors must be [B,T,4]")
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("mask must be [B,1,H,W]")
    if not isinstance(force_loss_weight, float) or force_loss_weight != 0.2:
        raise ValueError("FC-P011 fixes force_loss_weight at 0.2")
    steps = predicted_state.shape[1]
    weights = torch.full(
        (steps,), 1.0 / steps, dtype=predicted_state.dtype, device=predicted_state.device
    )
    field_by_step = ((predicted_state - target_state).square() * mask[:, None]).sum(
        dim=(2, 3, 4)
    ) / (mask.sum(dim=(1, 2, 3)).clamp_min(1)[:, None] * 3)
    force_channel_by_step = (predicted_force - target_force).square()
    field = (field_by_step * weights[None]).sum(dim=1).mean()
    channel_mse = (force_channel_by_step * weights[None, :, None]).sum(dim=1).mean(dim=0)
    equal_four = channel_mse.mean()
    rear_cl = channel_mse[3]
    balanced_force = 0.5 * equal_four + 0.5 * rear_cl
    total = field + force_loss_weight * balanced_force
    values = (total, field, equal_four, rear_cl, balanced_force, channel_mse)
    if not all(torch.isfinite(value).all() for value in values):
        raise FloatingPointError("non-finite FC-P011 objective")
    return {
        "total": total,
        "field": field,
        "equal_four_force": equal_four,
        "rear_cl": rear_cl,
        "balanced_force": balanced_force,
        "force_channel_mse": channel_mse,
    }


def configure_trainable_scope(model, scope: str) -> dict[str, Any]:
    """Freeze the official FNO and expose only the exact approved tensors."""
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}")
    named = dict(model.named_parameters())
    required = {FINAL_WEIGHT, FINAL_BIAS, HIDDEN_WEIGHT, HIDDEN_BIAS}
    if not required.issubset(named):
        raise ValueError("official decoder parameter names differ")
    for parameter in named.values():
        parameter.requires_grad_(False)
    named[FINAL_WEIGHT].requires_grad_(True)
    named[FINAL_BIAS].requires_grad_(True)
    if scope == "decoder_tail":
        named[HIDDEN_WEIGHT].requires_grad_(True)
        named[HIDDEN_BIAS].requires_grad_(True)
    optimizer_names = sorted(name for name, value in named.items() if value.requires_grad)
    expected = [FINAL_BIAS, FINAL_WEIGHT]
    effective = 129
    if scope == "decoder_tail":
        expected += [HIDDEN_BIAS, HIDDEN_WEIGHT]
        effective += named[HIDDEN_WEIGHT].numel() + named[HIDDEN_BIAS].numel()
    if optimizer_names != sorted(expected):
        raise RuntimeError("trainable parameter scope differs")
    return {
        "optimizer_parameter_names": optimizer_names,
        "optimizer_tensor_elements": sum(named[name].numel() for name in optimizer_names),
        "effective_trainable_coefficients": effective,
    }


def capture_frozen_final_rows(model) -> dict[str, Any]:
    layer = model.decoder_net.final_layer.linear
    return {
        "weight": layer.weight[:REAR_CL_INDEX].detach().clone(),
        "bias": layer.bias[:REAR_CL_INDEX].detach().clone(),
    }


def mask_gradients_and_measure(model, gradient_clip_norm: float = 1.0) -> dict[str, float]:
    import torch

    if gradient_clip_norm != 1.0:
        raise ValueError("FC-P011 fixes global gradient clipping at one")
    layer = model.decoder_net.final_layer.linear
    if layer.weight.grad is None or layer.bias.grad is None:
        raise RuntimeError("rear-Cl output tensors have no gradient")
    layer.weight.grad[:REAR_CL_INDEX].zero_()
    layer.bias.grad[:REAR_CL_INDEX].zero_()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters or any(parameter.grad is None for parameter in parameters):
        raise RuntimeError("trainable parameter gradient is missing")
    if any(not torch.isfinite(parameter.grad).all() for parameter in parameters):
        raise FloatingPointError("non-finite FC-P011 gradient")
    rear_squared = layer.weight.grad[REAR_CL_INDEX].double().square().sum()
    rear_squared += layer.bias.grad[REAR_CL_INDEX].double().square()
    hidden = [
        parameter
        for name, parameter in model.named_parameters()
        if name in {HIDDEN_WEIGHT, HIDDEN_BIAS} and parameter.requires_grad
    ]
    hidden_squared = sum(
        (parameter.grad.double().square().sum() for parameter in hidden),
        torch.zeros((), dtype=torch.float64, device=layer.weight.device),
    )
    preclip = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip_norm)
    if not torch.isfinite(preclip):
        raise FloatingPointError("non-finite FC-P011 gradient norm")
    clip_scale = min(1.0, gradient_clip_norm / (float(preclip.detach()) + 1e-12))
    return {
        "preclip_gradient_norm": float(preclip.detach()),
        "rear_cl_row_gradient_norm": float(torch.sqrt(rear_squared).detach()),
        "decoder_hidden_gradient_norm": float(torch.sqrt(hidden_squared).detach()),
        "gradient_clip_scale": clip_scale,
    }


def mask_and_step(model, optimizer, gradient_clip_norm: float = 1.0) -> dict[str, float]:
    """Apply AdamW while restoring non-rear final rows byte-for-byte."""
    layer = model.decoder_net.final_layer.linear
    frozen = capture_frozen_final_rows(model)
    metrics = mask_gradients_and_measure(model, gradient_clip_norm)
    optimizer.step()
    import torch
    with torch.no_grad():
        layer.weight[:REAR_CL_INDEX].copy_(frozen["weight"])
        layer.bias[:REAR_CL_INDEX].copy_(frozen["bias"])
    return metrics


def assert_scope_confinement(before: Mapping[str, Any], after: Mapping[str, Any], scope: str) -> list[str]:
    """Return changed tensor names after exact scope/slice confinement checks."""
    import torch

    if set(before) != set(after):
        raise ValueError("state dictionaries differ")
    changed = []
    allowed_whole = {HIDDEN_WEIGHT, HIDDEN_BIAS} if scope == "decoder_tail" else set()
    for name in sorted(before):
        left, right = before[name], after[name]
        if torch.equal(left, right):
            continue
        changed.append(name)
        if name in allowed_whole:
            continue
        if name == FINAL_WEIGHT and torch.equal(left[:REAR_CL_INDEX], right[:REAR_CL_INDEX]):
            continue
        if name == FINAL_BIAS and torch.equal(left[:REAR_CL_INDEX], right[:REAR_CL_INDEX]):
            continue
        raise ValueError(f"tensor changed outside approved scope: {name}")
    return changed


def diagnostic_windows(dataset) -> list[dict[str, Any]]:
    items = []
    for global_index, family, case, start in DIAGNOSTIC_WINDOWS:
        sample, metadata = dataset[global_index]
        identity = training_identity([metadata])
        if identity["case"] != case or identity["start"] != start:
            raise ValueError("predeclared diagnostic window identity differs")
        items.append({"global_index": global_index, "family": family, "identity": identity, "sample": sample})
    return items


def evaluate_diagnostics(
    model,
    items,
    device,
    state_mean,
    state_std,
    force_mean,
    force_std,
    rollout_fn,
    predict_fn,
) -> dict[str, Any]:
    """Evaluate fixed train-only H1/free-AR windows without changing RNG or weights."""
    import torch

    was_training = model.training
    model.eval()
    rows = []
    with torch.no_grad():
        for item in items:
            batch = {key: value[None].to(device) for key, value in item["sample"].items()}
            predicted_state, predicted_force = rollout_fn(
                model, batch["state"], batch["mask"], batch["omega"]
            )
            target_state, target_force = batch["target_state"], batch["target_force"]
            true_state_predictions, true_state_forces = [], []
            for begin in range(0, 100, 10):
                inputs = []
                current_states = []
                for step in range(begin, begin + 10):
                    current = batch["state"] if step == 0 else target_state[:, step - 1]
                    height, width = current.shape[-2:]
                    omega_now = batch["omega"][:, step].reshape(-1, 1, 1, 1).expand(-1, 1, height, width)
                    omega_next = batch["omega"][:, step + 1].reshape(-1, 1, 1, 1).expand(-1, 1, height, width)
                    inputs.append(torch.cat((current, batch["mask"], omega_now, omega_next), dim=1))
                    current_states.append(current)
                inputs = torch.cat(inputs, dim=0)
                masks = batch["mask"].expand(10, -1, -1, -1)
                deltas, forces = predict_fn(model, inputs, masks)
                current_states = torch.cat(current_states, dim=0)
                true_state_predictions.append((current_states + deltas) * masks)
                true_state_forces.append(forces)
            true_state_prediction = torch.cat(true_state_predictions, dim=0)[None]
            true_state_force = torch.cat(true_state_forces, dim=0)[None]
            state_error = (predicted_state - target_state) * state_std[None, None]
            h1_state_error = (true_state_prediction - target_state) * state_std[None, None]
            target_physical = target_state * state_std[None, None] + state_mean[None, None]
            mask = batch["mask"][:, None]
            field_relative, h1_field_relative = [], []
            for channel in range(3):
                numerator = (state_error[:, :, channel : channel + 1].square() * mask).sum()
                h1_numerator = (h1_state_error[:, :, channel : channel + 1].square() * mask).sum()
                denominator = (target_physical[:, :, channel : channel + 1].square() * mask).sum().clamp_min(1e-30)
                field_relative.append(float(torch.sqrt(numerator / denominator)))
                h1_field_relative.append(float(torch.sqrt(h1_numerator / denominator)))
            physical_error = (predicted_force - target_force) * force_std[None, None]
            h1_physical_error = (true_state_force - target_force) * force_std[None, None]
            predicted_physical = predicted_force * force_std[None, None] + force_mean[None, None]
            h1_predicted_physical = true_state_force * force_std[None, None] + force_mean[None, None]
            target_force_physical = target_force * force_std[None, None] + force_mean[None, None]
            tail_pred = predicted_physical[0, 38:, 3]
            h1_tail_pred = h1_predicted_physical[0, 38:, 3]
            tail_true = target_force_physical[0, 38:, 3]
            pred_rms = torch.sqrt(((tail_pred - tail_pred.mean()).square()).mean())
            h1_pred_rms = torch.sqrt(((h1_tail_pred - h1_tail_pred.mean()).square()).mean())
            true_rms = torch.sqrt(((tail_true - tail_true.mean()).square()).mean())
            rows.append(
                {
                    "global_index": item["global_index"],
                    "family": item["family"],
                    "identity": item["identity"],
                    "free_ar_field_relative_l2_uvp": field_relative,
                    "true_state_h1_field_relative_l2_uvp": h1_field_relative,
                    "initial_endpoint_h1_force_absolute_error": physical_error[0, 0].abs().tolist(),
                    "true_state_h1_force_mae": h1_physical_error.abs().mean(dim=(0, 1)).tolist(),
                    "free_ar_force_mae": physical_error.abs().mean(dim=(0, 1)).tolist(),
                    "tail62_mean_rear_cl_absolute_error": float((tail_pred.mean() - tail_true.mean()).abs()),
                    "tail62_rear_cl_rms_absolute_error": float((pred_rms - true_rms).abs()),
                    "true_state_h1_tail62_mean_rear_cl_absolute_error": float(
                        (h1_tail_pred.mean() - tail_true.mean()).abs()
                    ),
                    "true_state_h1_tail62_rear_cl_rms_absolute_error": float(
                        (h1_pred_rms - true_rms).abs()
                    ),
                    "prediction_nonrear_sha256": hashlib.sha256(
                        torch.cat((predicted_state.flatten(), predicted_force[..., :3].flatten()))
                        .detach().cpu().contiguous().numpy().tobytes()
                    ).hexdigest(),
                }
            )
    model.train(was_training)
    return {"windows": rows}


def run_resource_probe(model, batch, identity, scope: str, device, rollout_fn) -> dict[str, Any]:
    """Run one bounded H100 backward without constructing an optimizer or saving."""
    import torch

    before = tensor_state_sha256(model)
    model.train()
    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    predicted_state, predicted_force = rollout_fn(
        model, batch["state"], batch["mask"], batch["omega"], batch["target_state"], 0.0
    )
    losses = balanced_decoder_objective(
        predicted_state,
        predicted_force,
        batch["target_state"],
        batch["target_force"],
        batch["mask"],
    )
    losses["total"].backward()
    gradients = mask_gradients_and_measure(model)
    after = tensor_state_sha256(model)
    if after != before:
        raise RuntimeError("resource probe changed model tensors")
    return {
        "status": "FC_P011_DECODER_SCOPE_RESOURCE_PROBE_COMPLETE",
        "scope": scope,
        "identity": identity,
        "loss": float(losses["total"].detach()),
        "gradient_metrics": gradients,
        "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "cuda_peak_reserved_bytes": (
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
        ),
        "elapsed_seconds": time.monotonic() - started,
        "model_tensor_sha256_before": before,
        "model_tensor_sha256_after": after,
        "optimizer_created": False,
        "optimizer_steps": 0,
        "model_saved": False,
        "validation_accessed": False,
        "frozen_test_accessed": False,
        "ppo_executed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=SCOPES, required=True)
    parser.add_argument("--resource-probe", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    import numpy as np
    import torch
    from fluid_control.augmented_datapipe import compose_training_data
    from fluid_control.calibrated_checkpoint import validate_calibrated_epoch_zero
    from fluid_control.tandem_datapipe import TandemRolloutDataset
    from omegaconf import OmegaConf
    from physicsnemo.datapipes import DataLoader
    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils import load_checkpoint, save_checkpoint
    from train_tandem_fno import build_model, configured_force_indices, predict
    from train_tandem_fno_paired_stats import rollout

    if sha256(args.config) != CONFIG_SHA:
        raise ValueError("resolved config SHA differs")
    cfg = OmegaConf.load(args.config)
    required = {
        "rollout_steps": 100,
        "batch_size": 1,
        "seed": 20261003,
        "learning_rate": 1e-5,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
        "teacher_forcing_start": 0.0,
        "teacher_forcing_end": 0.0,
        "expected_regular_batches": TOTAL_STEPS,
    }
    if any(cfg.training.get(key) != value for key, value in required.items()):
        raise ValueError("resolved FC-P011 training contract differs")
    expected_model = {
        "in_channels": 6,
        "out_channels": 7,
        "latent_channels": 48,
        "num_fno_layers": 5,
        "num_fno_modes": [32, 32],
        "decoder_layers": 2,
        "decoder_layer_size": 128,
        "padding": 8,
        "coord_features": True,
    }
    if {key: cfg.model.get(key) for key in expected_model} != expected_model:
        raise ValueError("official FNO architecture differs")
    input_sha = validate_data_contract(cfg)
    model_files = sorted(args.parent.glob("FNO.0.*.mdlus"))
    state_files = sorted(args.parent.glob("checkpoint.0.*.pt"))
    if len(model_files) != 1 or len(state_files) != 1:
        raise ValueError("P009 parent pair differs")
    if sha256(model_files[0]) != PARENT_MODEL_SHA or sha256(state_files[0]) != PARENT_STATE_SHA:
        raise ValueError("P009 parent SHA differs")

    DistributedManager.initialize()
    dist = DistributedManager()
    if dist.distributed or not dist.cuda:
        raise RuntimeError("FC-P011 requires exactly one CUDA device")
    torch.cuda.set_per_process_memory_fraction(float(cfg.training.gpu_memory_fraction), dist.device)
    if (
        torch.get_float32_matmul_precision() != "high"
        or not torch.backends.cuda.matmul.allow_tf32
        or not torch.backends.cudnn.allow_tf32
    ):
        raise RuntimeError("FC-P011 requires the existing default-TF32/high protocol")
    seed = int(cfg.training.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    force_indices = configured_force_indices(cfg)
    if tuple(force_indices) != (0, 1, 2, 3):
        raise ValueError("FC-P011 requires all four force channels")
    base = TandemRolloutDataset(
        cfg.data.root,
        "train",
        100,
        stride=int(cfg.training.train_stride),
        num_workers=cfg.training.workers,
        force_indices=force_indices,
    )
    train, sources = compose_training_data(
        base,
        [Path(value) for value in cfg.data.additional_train_roots],
        rollout_steps=100,
        stride=int(cfg.training.additional_train_stride),
        workers=cfg.training.workers,
        force_indices=force_indices,
    )
    loader = DataLoader(
        train,
        batch_size=1,
        shuffle=True,
        collate_metadata=True,
        prefetch_factor=int(cfg.data.prefetch_factor),
        num_streams=int(cfg.data.num_streams),
        use_streams=True,
        seed=seed,
    )
    if len(loader) != TOTAL_STEPS:
        raise RuntimeError("FC-P011 requires exactly 1368 regular windows")
    index_by_identity = identity_index(train)
    audit_loader = DataLoader(train, batch_size=1, shuffle=True, prefetch_factor=0, use_streams=False, seed=seed)
    expected_order = list(iter(audit_loader.sampler))
    if sequence_sha(expected_order) != EXPECTED_ORDER_SHA:
        raise RuntimeError("official sampler order differs")
    fixed_diagnostics = diagnostic_windows(train)

    model = build_model(cfg).to(dist.device)
    loaded = load_checkpoint(args.parent, models=model, device=dist.device)
    parent_identity = validate_calibrated_epoch_zero(
        args.parent,
        loaded,
        allow=True,
        expected_model_sha256=PARENT_MODEL_SHA,
        expected_state_sha256=PARENT_STATE_SHA,
        expected_kind=PARENT_KIND,
    )
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    before_sha = tensor_state_sha256(model)
    scope = configure_trainable_scope(model, args.scope)
    if scope["effective_trainable_coefficients"] != (129 if args.scope == "head_only" else 16641):
        raise RuntimeError("effective trainable coefficient count differs")
    state_mean = base.state_mean.to(dist.device)
    state_std = base.state_std.to(dist.device)
    force_mean = base.force_mean.to(dist.device)
    force_std = base.force_std.to(dist.device)
    diagnostics = {
        "0": evaluate_diagnostics(
            model,
            fixed_diagnostics,
            dist.device,
            state_mean,
            state_std,
            force_mean,
            force_std,
            rollout,
            predict,
        )
    }

    if args.resource_probe:
        batch, metadata = next(iter(loader))
        identity = training_identity(metadata)
        batch = {key: value.to(dist.device) for key, value in batch.items()}
        result = run_resource_probe(model, batch, identity, args.scope, dist.device, rollout)
        args.output.mkdir(parents=True)
        result["input_sha256"] = input_sha
        (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        train.close()
        return
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
    )
    model.train()
    identities, observed_order, records = [], [], []
    for step, (batch, metadata) in enumerate(loader, start=1):
        if step > TOTAL_STEPS:
            raise RuntimeError("training loader exceeded fixed pass")
        identity = training_identity(metadata)
        identities.append(identity)
        key = (identity["case"], identity["start"], identity["dataset_index"])
        if key not in index_by_identity:
            raise RuntimeError("observed training identity is absent from exact inventory")
        observed_order.append(index_by_identity[key])
        batch = {key: value.to(dist.device, non_blocking=True) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        predicted_state, predicted_force = rollout(
            model,
            batch["state"],
            batch["mask"],
            batch["omega"],
            batch["target_state"],
            0.0,
        )
        losses = balanced_decoder_objective(
            predicted_state,
            predicted_force,
            batch["target_state"],
            batch["target_force"],
            batch["mask"],
        )
        losses["total"].backward()
        step_result = mask_and_step(model, optimizer)
        row = {
            "step": step,
            "identity": identity,
            "total_loss": float(losses["total"].detach()),
            "field_loss": float(losses["field"].detach()),
            "equal_four_force_mse": float(losses["equal_four_force"].detach()),
            "rear_cl_mse": float(losses["rear_cl"].detach()),
            "balanced_force_loss": float(losses["balanced_force"].detach()),
            "force_channel_mse": [float(value) for value in losses["force_channel_mse"].detach()],
            **step_result,
        }
        if not all(np.isfinite(value) for key, value in row.items() if isinstance(value, float)):
            raise FloatingPointError("non-finite FC-P011 training record")
        records.append(row)
        if step in DIAGNOSTIC_STEPS:
            diagnostics[str(step)] = evaluate_diagnostics(
                model,
                fixed_diagnostics,
                dist.device,
                state_mean,
                state_std,
                force_mean,
                force_std,
                rollout,
                predict,
            )
        if step % 8 == 0 or step in DIAGNOSTIC_STEPS:
            print(json.dumps(row, allow_nan=False), flush=True)
    family_counts = Counter(row["dataset_index"] for row in identities)
    if (
        len(records) != TOTAL_STEPS
        or len({(row["case"], row["start"], row["dataset_index"]) for row in identities}) != TOTAL_STEPS
        or family_counts != Counter({0: 720, 1: 408, 2: 240})
        or observed_order != expected_order
    ):
        raise RuntimeError("FC-P011 did not consume one unique complete train pass")

    after = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    changed = assert_scope_confinement(before, after, args.scope)
    if FINAL_WEIGHT not in changed or FINAL_BIAS not in changed:
        raise RuntimeError("rear-Cl output row did not change")
    if args.scope == "decoder_tail" and not {HIDDEN_WEIGHT, HIDDEN_BIAS}.issubset(changed):
        raise RuntimeError("decoder-tail tensors did not change")
    if args.scope == "head_only":
        before_nonrear = [row["prediction_nonrear_sha256"] for row in diagnostics["0"]["windows"]]
        after_nonrear = [row["prediction_nonrear_sha256"] for row in diagnostics[str(TOTAL_STEPS)]["windows"]]
        if before_nonrear != after_nonrear:
            raise RuntimeError("head-only arm changed field or non-rear force predictions")
    args.output.mkdir(parents=True)
    final_dir = args.output / "final"
    save_checkpoint(
        final_dir,
        models=model,
        optimizer=optimizer,
        epoch=1,
        metadata={
            "status": "FC_P011_DECODER_SCOPE_TERMINAL_CHECKPOINT",
            "scope": args.scope,
            "optimizer_steps": TOTAL_STEPS,
            "parent_model_sha256": PARENT_MODEL_SHA,
            "parent_state_sha256": PARENT_STATE_SHA,
            "selection_performed": False,
            "validation_accessed": False,
            "frozen_test_accessed": False,
            "ppo_executed": False,
        },
    )
    final_models = list(final_dir.glob("FNO.0.*.mdlus"))
    final_states = list(final_dir.glob("checkpoint.0.*.pt"))
    if len(final_models) != 1 or len(final_states) != 1:
        raise RuntimeError("terminal official checkpoint pair differs")
    fresh = build_model(cfg).to(dist.device)
    fresh_metadata: dict[str, Any] = {}
    if load_checkpoint(final_dir, models=fresh, metadata_dict=fresh_metadata, device=dist.device) != 1:
        raise RuntimeError("fresh official terminal checkpoint load failed")
    required_metadata = {
        "status": "FC_P011_DECODER_SCOPE_TERMINAL_CHECKPOINT",
        "scope": args.scope,
        "optimizer_steps": TOTAL_STEPS,
        "parent_model_sha256": PARENT_MODEL_SHA,
        "parent_state_sha256": PARENT_STATE_SHA,
        "selection_performed": False,
        "validation_accessed": False,
        "frozen_test_accessed": False,
        "ppo_executed": False,
    }
    if any(fresh_metadata.get(key) != value for key, value in required_metadata.items()):
        raise RuntimeError("fresh terminal checkpoint metadata differs")
    fresh_state = {name: value.detach().cpu().clone() for name, value in fresh.state_dict().items()}
    if any(not torch.equal(after[name], fresh_state[name]) for name in after):
        raise RuntimeError("fresh terminal checkpoint tensors differ from reviewed in-memory model")
    assert_scope_confinement(before, fresh_state, args.scope)
    result = {
        "status": "FC_P011_DECODER_SCOPE_TRAINING_COMPLETE_NOT_ADMISSION",
        "scope": args.scope,
        "parent_identity": parent_identity,
        "parent_model_sha256": PARENT_MODEL_SHA,
        "parent_state_sha256": PARENT_STATE_SHA,
        "model_state_before_sha256": before_sha,
        "model_state_after_sha256": tensor_state_sha256(model),
        "train_order_sha256": sequence_sha(observed_order),
        "expected_train_order_sha256": EXPECTED_ORDER_SHA,
        "family_window_counts": {"base20": 720, "train8": 408, "train16": 240},
        "train_identities": identities,
        "training_records": records,
        "training_sources": sources,
        "trainable_scope": scope,
        "changed_tensor_names": changed,
        "optimizer_steps": TOTAL_STEPS,
        "batch_size": 1,
        "loss_contract": {
            "field_weight": 1.0,
            "force_weight": 0.2,
            "equal_four_share": 0.5,
            "rear_cl_share": 0.5,
            "effective_force_channel_weights": [0.125, 0.125, 0.125, 0.625],
            "rollout_steps": 100,
            "step_weights": "uniform",
            "teacher_forcing": 0.0,
        },
        "diagnostic_steps_predeclared": list(DIAGNOSTIC_STEPS),
        "diagnostic_windows_predeclared": [
            {"global_index": index, "family": family, "case": case, "start": start}
            for index, family, case, start in DIAGNOSTIC_WINDOWS
        ],
        "train_only_diagnostics": diagnostics,
        "selection_performed": False,
        "validation_accessed": False,
        "frozen_test_accessed": False,
        "ppo_executed": False,
        "terminal_model_sha256": sha256(final_models[0]),
        "terminal_state_sha256": sha256(final_states[0]),
        "fresh_reload_tensor_sha256": tensor_state_sha256(fresh),
        "config_sha256": sha256(args.config),
        "input_sha256": input_sha,
        "implementation_sha256": sha256(Path(__file__)),
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    train.close()


if __name__ == "__main__":
    main()
