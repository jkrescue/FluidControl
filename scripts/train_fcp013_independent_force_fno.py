#!/usr/bin/env python3
"""Train an independent official FNO for forces while freezing flow rollout."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any

TRAINER_SHA = "9c761cfcb3d4f18dbe35aed1b3defe0614b29db867006055dae63d9fa94485a5"
CONFIG_SHA = "07e55fd11df8030313338cef0344490c3453e515aae9c6b6122e997bad5085d9"
NORMALIZATION_SHA = "f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1"
PARENT_MODEL_SHA = "dc41fc91d42476e052970b39fc66aed22fa72aa8b6f218a341a3abb095f42e31"
PARENT_STATE_SHA = "4998e534d4b82b17393c217357ed18220fb8e739166a88147483bb9cc5fb771e"
PARENT_KIND = "FC_P009_TRAIN_ONLY_JOINT_FORCE_ROW_CANDIDATE"
TOTAL_STEPS = 1368
ROLLOUT_STEPS = 100
CHUNK_SIZE = 10
FORCE_CHANNELS = ("front_cd", "front_cl", "rear_cd", "rear_cl")
OFFICIAL_FROZEN_PARAMETER_NAMES = (
    "spec_encoder.lift_network.0.conv.bias",
    "spec_encoder.lift_network.2.conv.bias",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_state_sha256(module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def load_frozen_trainer(path: Path):
    if sha256(path) != TRAINER_SHA:
        raise ValueError("FC-P011 trainer SHA differs")
    spec = importlib.util.spec_from_file_location("fcp011_frozen_trainer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen FC-P011 trainer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def balanced_force_objective(predicted, target):
    """P011's exact balanced normalized four-force loss, without field loss."""
    import torch

    if predicted.shape != target.shape or predicted.ndim != 3 or predicted.shape[-1] != 4:
        raise ValueError("force tensors must have identical [B,T,4] shapes")
    channel_mse = (predicted - target).square().mean(dim=(0, 1))
    equal_four = channel_mse.mean()
    rear_cl = channel_mse[3]
    balanced = 0.5 * equal_four + 0.5 * rear_cl
    if not all(torch.isfinite(value).all() for value in (channel_mse, equal_four, rear_cl, balanced)):
        raise FloatingPointError("non-finite force objective")
    return {
        "balanced": balanced,
        "equal_four": equal_four,
        "rear_cl": rear_cl,
        "channel_mse": channel_mse,
    }


def make_inputs(states, mask, omega_now, omega_next):
    height, width = states.shape[-2:]
    now = omega_now.reshape(-1, 1, 1, 1).expand(-1, 1, height, width)
    nxt = omega_next.reshape(-1, 1, 1, 1).expand(-1, 1, height, width)
    return __import__("torch").cat((states, mask, now, nxt), dim=1)


def frozen_flow_states(flow_model, state, mask, omega, predict_fn):
    """Return causal current states using only frozen-flow residual updates."""
    import torch

    if omega.shape[1] != ROLLOUT_STEPS + 1:
        raise ValueError("FC-P013 requires 101 action endpoints")
    if any(parameter.requires_grad for parameter in flow_model.parameters()):
        raise ValueError("flow model must be frozen")
    current = state
    states = []
    with torch.no_grad():
        for step in range(ROLLOUT_STEPS):
            states.append(current)
            inputs = make_inputs(current, mask, omega[:, step], omega[:, step + 1])
            delta, _ = predict_fn(flow_model, inputs, mask)
            current = (current + delta) * mask
    return torch.stack(states, dim=1)


def true_state_inputs(state, target_state):
    import torch

    if target_state.shape[1] != ROLLOUT_STEPS:
        raise ValueError("FC-P013 requires H100 targets")
    return torch.cat((state[:, None], target_state[:, :-1]), dim=1)


def chunk_force_objective(
    aerodynamic_model,
    flow_states,
    h1_states,
    mask,
    omega,
    target_force,
    predict_fn,
    *,
    chunk_size: int = CHUNK_SIZE,
    backward: bool,
):
    """Accumulate the exact 0.5 H1 + 0.5 AR force objective by time chunk."""
    import torch

    if chunk_size < 1 or ROLLOUT_STEPS % chunk_size:
        raise ValueError("chunk size must divide 100")
    if flow_states.shape != h1_states.shape or flow_states.shape[1] != ROLLOUT_STEPS:
        raise ValueError("H1/AR state sequences differ")
    if target_force.shape[1:] != (ROLLOUT_STEPS, 4):
        raise ValueError("target force alignment differs")
    totals = {
        "h1_balanced": torch.zeros((), device=flow_states.device),
        "ar_balanced": torch.zeros((), device=flow_states.device),
        "total": torch.zeros((), device=flow_states.device),
    }
    channel_h1 = torch.zeros(4, device=flow_states.device)
    channel_ar = torch.zeros(4, device=flow_states.device)
    batch = flow_states.shape[0]
    for begin in range(0, ROLLOUT_STEPS, chunk_size):
        end = begin + chunk_size
        length = end - begin
        h1 = h1_states[:, begin:end].reshape(batch * length, *h1_states.shape[2:])
        ar = flow_states[:, begin:end].reshape(batch * length, *flow_states.shape[2:])
        states = torch.cat((h1, ar), dim=0)
        masks = mask[:, None].expand(-1, length, -1, -1, -1).reshape(
            batch * length, *mask.shape[1:]
        )
        masks = torch.cat((masks, masks), dim=0)
        now = omega[:, begin:end].reshape(-1)
        nxt = omega[:, begin + 1 : end + 1].reshape(-1)
        inputs = make_inputs(states, masks, torch.cat((now, now)), torch.cat((nxt, nxt)))
        _, forces = predict_fn(aerodynamic_model, inputs, masks)
        h1_force, ar_force = forces[: batch * length], forces[batch * length :]
        targets = target_force[:, begin:end].reshape(batch * length, 4)
        h1_loss = balanced_force_objective(h1_force[:, None], targets[:, None])
        ar_loss = balanced_force_objective(ar_force[:, None], targets[:, None])
        weight = length / ROLLOUT_STEPS
        chunk_total = 0.5 * h1_loss["balanced"] + 0.5 * ar_loss["balanced"]
        if backward:
            (weight * chunk_total).backward()
        totals["h1_balanced"] += weight * h1_loss["balanced"].detach()
        totals["ar_balanced"] += weight * ar_loss["balanced"].detach()
        totals["total"] += weight * chunk_total.detach()
        channel_h1 += weight * h1_loss["channel_mse"].detach()
        channel_ar += weight * ar_loss["channel_mse"].detach()
    return {
        **{key: float(value) for key, value in totals.items()},
        "h1_channel_mse": [float(value) for value in channel_h1],
        "ar_channel_mse": [float(value) for value in channel_ar],
        "chunk_size": chunk_size,
        "chunks": ROLLOUT_STEPS // chunk_size,
    }


def run_window(flow_model, aerodynamic_model, batch, predict_fn, *, backward: bool):
    flow_states = frozen_flow_states(
        flow_model, batch["state"], batch["mask"], batch["omega"], predict_fn
    )
    h1_states = true_state_inputs(batch["state"], batch["target_state"])
    return chunk_force_objective(
        aerodynamic_model,
        flow_states,
        h1_states,
        batch["mask"],
        batch["omega"],
        batch["target_force"],
        predict_fn,
        backward=backward,
    )


def optimizer_window_step(
    flow_model,
    aerodynamic_model,
    batch,
    predict_fn,
    optimizer,
    *,
    gradient_clip_norm: float = 1.0,
):
    """Perform exactly one approved aerodynamic update for one H100 window."""
    import torch

    if gradient_clip_norm != 1.0:
        raise ValueError("FC-P013 fixes gradient clipping at one")
    optimizer.zero_grad(set_to_none=True)
    metrics = run_window(
        flow_model, aerodynamic_model, batch, predict_fn, backward=True
    )
    gradient_audit = audit_aerodynamic_gradients(aerodynamic_model)
    trainable = [
        parameter for parameter in aerodynamic_model.parameters() if parameter.requires_grad
    ]
    preclip = torch.nn.utils.clip_grad_norm_(
        trainable, gradient_clip_norm
    )
    if not torch.isfinite(preclip):
        raise FloatingPointError("non-finite aerodynamic gradient norm")
    optimizer.step()
    return {
        **metrics,
        "preclip_gradient_norm": float(preclip.detach()),
        "optimizer_steps": 1,
        "gradient_audit": gradient_audit,
    }


def audit_aerodynamic_gradients(model) -> dict[str, Any]:
    """Require the exact official two frozen biases and 28 finite gradients."""
    import torch

    named = dict(model.named_parameters())
    frozen = sorted(name for name, value in named.items() if not value.requires_grad)
    if frozen != sorted(OFFICIAL_FROZEN_PARAMETER_NAMES):
        raise RuntimeError(f"official frozen parameter identities differ: {frozen}")
    expected_shapes = {
        "spec_encoder.lift_network.0.conv.bias": (24,),
        "spec_encoder.lift_network.2.conv.bias": (48,),
    }
    if {name: tuple(named[name].shape) for name in frozen} != expected_shapes:
        raise RuntimeError("official frozen lifting-bias shapes differ")
    trainable = {name: value for name, value in named.items() if value.requires_grad}
    if len(trainable) != 28:
        raise RuntimeError("official trainable parameter tensor count differs")
    missing = sorted(name for name, value in trainable.items() if value.grad is None)
    nonfinite = sorted(
        name
        for name, value in trainable.items()
        if value.grad is not None and not torch.isfinite(value.grad).all()
    )
    if missing or nonfinite:
        raise RuntimeError(
            f"trainable gradient audit failed: missing={missing}, nonfinite={nonfinite}"
        )
    return {
        "official_frozen_parameter_names": frozen,
        "official_frozen_parameter_shapes": {
            name: list(expected_shapes[name]) for name in frozen
        },
        "trainable_parameter_tensor_count": len(trainable),
        "trainable_gradient_all_finite": True,
    }


def checkpoint_pair(root: Path) -> tuple[Path, Path]:
    models = sorted(root.glob("FNO.0.*.mdlus"))
    states = sorted(root.glob("checkpoint.0.*.pt"))
    if len(models) != 1 or len(states) != 1:
        raise ValueError("parent checkpoint pair differs")
    if sha256(models[0]) != PARENT_MODEL_SHA or sha256(states[0]) != PARENT_STATE_SHA:
        raise ValueError("parent checkpoint SHA differs")
    return models[0], states[0]


def build_dual_manifest(output: Path, aerodynamic_model: Path, aerodynamic_state: Path):
    flow_model = output / "flow" / "FNO.0.0.mdlus"
    flow_state = output / "flow" / "checkpoint.0.0.pt"
    manifest = {
        "schema_version": 1,
        "status": "FC_P013_DUAL_FNO_MANIFEST_VERIFIED",
        "kind": "FC_P013_INDEPENDENT_FORCE_FNO",
        "config_sha256": CONFIG_SHA,
        "normalization_sha256": NORMALIZATION_SHA,
        "precision_protocol": {
            "float32_matmul_precision": "high",
            "cuda_matmul_allow_tf32": True,
            "cudnn_allow_tf32": True,
        },
        "flow_parent_model_sha256": PARENT_MODEL_SHA,
        "flow_parent_state_sha256": PARENT_STATE_SHA,
        "aerodynamic_initial_model_sha256": PARENT_MODEL_SHA,
        "aerodynamic_initial_state_sha256": PARENT_STATE_SHA,
        "architecture": {
            "in_channels": 6,
            "out_channels": 7,
            "latent_channels": 48,
            "num_fno_layers": 5,
            "num_fno_modes": [32, 32],
            "decoder_layers": 2,
            "decoder_layer_size": 128,
            "padding": 8,
            "coord_features": True,
            "force_channels": list(FORCE_CHANNELS),
        },
        "flow": {
            "role": "flow", "frozen": True, "checkpoint_relative_directory": "flow",
            "model_file": flow_model.name, "state_file": flow_state.name,
            "checkpoint_epoch": 0, "model_sha256": sha256(flow_model),
            "state_sha256": sha256(flow_state), "metadata_kind": PARENT_KIND,
        },
        "aerodynamic": {
            "role": "aerodynamic", "frozen": False,
            "checkpoint_relative_directory": "aerodynamic",
            "model_file": aerodynamic_model.name, "state_file": aerodynamic_state.name,
            "checkpoint_epoch": 1, "model_sha256": sha256(aerodynamic_model),
            "state_sha256": sha256(aerodynamic_state),
            "metadata_kind": "FC_P013_INDEPENDENT_FORCE_FNO_AERODYNAMIC_CHECKPOINT",
        },
    }
    if manifest["flow"]["checkpoint_relative_directory"] == manifest["aerodynamic"]["checkpoint_relative_directory"]:
        raise RuntimeError("dual checkpoint directories must differ")
    path = output / "dual_model_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--frozen-trainer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resource-probe", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    trainer = load_frozen_trainer(args.frozen_trainer)

    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from physicsnemo.datapipes import DataLoader
    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils import load_checkpoint, save_checkpoint
    from train_tandem_fno import build_model, configured_force_indices, predict

    from fluid_control.augmented_datapipe import compose_training_data
    from fluid_control.calibrated_checkpoint import validate_calibrated_epoch_zero
    from fluid_control.tandem_datapipe import TandemRolloutDataset

    if sha256(args.config) != CONFIG_SHA:
        raise ValueError("resolved config SHA differs")
    cfg = OmegaConf.load(args.config)
    required = {
        "rollout_steps": 100, "batch_size": 1, "seed": 20261003,
        "learning_rate": 1e-5, "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0, "expected_regular_batches": TOTAL_STEPS,
    }
    if any(cfg.training.get(key) != value for key, value in required.items()):
        raise ValueError("FC-P013 training contract differs")
    if not 0 < float(cfg.training.gpu_memory_fraction) <= 0.45:
        raise ValueError("allocator fraction exceeds approved bound")
    input_sha = trainer.validate_data_contract(cfg)
    parent_model, parent_state = checkpoint_pair(args.parent)

    DistributedManager.initialize()
    dist = DistributedManager()
    if dist.distributed or not dist.cuda:
        raise RuntimeError("FC-P013 requires exactly one CUDA device")
    torch.cuda.set_per_process_memory_fraction(float(cfg.training.gpu_memory_fraction), dist.device)
    if torch.get_float32_matmul_precision() != "high" or not torch.backends.cuda.matmul.allow_tf32 or not torch.backends.cudnn.allow_tf32:
        raise RuntimeError("default-TF32/high protocol differs")
    seed = int(cfg.training.seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    force_indices = configured_force_indices(cfg)
    if tuple(force_indices) != (0, 1, 2, 3):
        raise ValueError("force channel order differs")
    base = TandemRolloutDataset(cfg.data.root, "train", 100, stride=int(cfg.training.train_stride), num_workers=cfg.training.workers, force_indices=force_indices)
    train, _ = compose_training_data(base, [Path(value) for value in cfg.data.additional_train_roots], rollout_steps=100, stride=int(cfg.training.additional_train_stride), workers=cfg.training.workers, force_indices=force_indices)
    loader = DataLoader(train, batch_size=1, shuffle=True, collate_metadata=True, prefetch_factor=int(cfg.data.prefetch_factor), num_streams=int(cfg.data.num_streams), use_streams=True, seed=seed)
    if len(loader) != TOTAL_STEPS:
        raise RuntimeError("FC-P013 requires exactly 1368 windows")
    audit_loader = DataLoader(train, batch_size=1, shuffle=True, prefetch_factor=0, use_streams=False, seed=seed)
    expected_order = list(iter(audit_loader.sampler))
    if trainer.sequence_sha(expected_order) != trainer.EXPECTED_ORDER_SHA:
        raise RuntimeError("official sampler order differs")

    flow_model = build_model(cfg).to(dist.device)
    aerodynamic_model = build_model(cfg).to(dist.device)
    flow_metadata: dict[str, Any] = {}; aero_metadata: dict[str, Any] = {}
    flow_epoch = load_checkpoint(args.parent, models=flow_model, metadata_dict=flow_metadata, device=dist.device)
    aero_epoch = load_checkpoint(args.parent, models=aerodynamic_model, metadata_dict=aero_metadata, device=dist.device)
    for epoch, metadata in ((flow_epoch, flow_metadata), (aero_epoch, aero_metadata)):
        validate_calibrated_epoch_zero(args.parent, epoch, allow=True, expected_model_sha256=PARENT_MODEL_SHA, expected_state_sha256=PARENT_STATE_SHA, expected_kind=PARENT_KIND)
    if flow_model is aerodynamic_model or any(a is b for a, b in zip(flow_model.parameters(), aerodynamic_model.parameters(), strict=True)):
        raise RuntimeError("flow and aerodynamic models are not independent")
    for parameter in flow_model.parameters(): parameter.requires_grad_(False)
    flow_model.eval(); aerodynamic_model.train()
    flow_before = tensor_state_sha256(flow_model); aero_before = tensor_state_sha256(aerodynamic_model)

    if args.resource_probe:
        batch, metadata = next(iter(loader))
        identity = trainer.training_identity(metadata)
        batch = {key: value.to(dist.device) for key, value in batch.items()}
        aerodynamic_model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(dist.device); started = time.monotonic()
        metrics = run_window(flow_model, aerodynamic_model, batch, predict, backward=True)
        gradient_audit = audit_aerodynamic_gradients(aerodynamic_model)
        if tensor_state_sha256(flow_model) != flow_before or tensor_state_sha256(aerodynamic_model) != aero_before:
            raise RuntimeError("resource probe changed model tensors")
        args.output.mkdir(parents=True)
        result = {
            "status": "FC_P013_INDEPENDENT_FORCE_FNO_RESOURCE_PROBE_COMPLETE",
            "identity": identity, "metrics": metrics,
            "gradient_audit": gradient_audit,
            "elapsed_seconds": time.monotonic() - started,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(dist.device),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(dist.device),
            "flow_tensor_sha256_before": flow_before,
            "flow_tensor_sha256_after": tensor_state_sha256(flow_model),
            "aerodynamic_tensor_sha256_before": aero_before,
            "aerodynamic_tensor_sha256_after": tensor_state_sha256(aerodynamic_model),
            "optimizer_created": False, "optimizer_steps": 0, "candidate_saved": False,
            "validation_accessed": False, "frozen_test_accessed": False, "ppo_executed": False,
            "input_sha256": input_sha,
        }
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        train.close(); return

    optimizer = torch.optim.AdamW(
        [parameter for parameter in aerodynamic_model.parameters() if parameter.requires_grad],
        lr=1e-5,
        weight_decay=1e-4,
    )
    records = []; observed_order = []
    identity_map = trainer.identity_index(train)
    for step, (batch, metadata) in enumerate(loader, start=1):
        identity = trainer.training_identity(metadata)
        observed_order.append(identity_map[(identity["case"], identity["start"], identity["dataset_index"])])
        batch = {key: value.to(dist.device, non_blocking=True) for key, value in batch.items()}
        metrics = optimizer_window_step(
            flow_model, aerodynamic_model, batch, predict, optimizer
        )
        if tensor_state_sha256(flow_model) != flow_before: raise RuntimeError("frozen flow changed")
        row = {"step": step, "identity": identity, **metrics}
        records.append(row)
        if step % 8 == 0: print(json.dumps(row, allow_nan=False), flush=True)
    if len(records) != TOTAL_STEPS or observed_order != expected_order:
        raise RuntimeError("FC-P013 did not consume the fixed complete train pass")
    if tensor_state_sha256(aerodynamic_model) == aero_before:
        raise RuntimeError("aerodynamic model did not change")

    args.output.mkdir(parents=True)
    flow_dir = args.output / "flow"; flow_dir.mkdir()
    shutil.copy2(parent_model, flow_dir / parent_model.name); shutil.copy2(parent_state, flow_dir / parent_state.name)
    aero_dir = args.output / "aerodynamic"
    save_checkpoint(aero_dir, models=aerodynamic_model, optimizer=optimizer, epoch=1, metadata={
        "status": "FC_P013_INDEPENDENT_FORCE_FNO_AERODYNAMIC_CHECKPOINT",
        "checkpoint_epoch": 1,
        "optimizer_steps": TOTAL_STEPS,
        "flow_parent_model_sha256": PARENT_MODEL_SHA,
        "flow_parent_state_sha256": PARENT_STATE_SHA,
        "aerodynamic_initial_model_sha256": PARENT_MODEL_SHA,
        "aerodynamic_initial_state_sha256": PARENT_STATE_SHA,
        "selection_performed": False,
        "validation_accessed": False, "frozen_test_accessed": False, "ppo_executed": False,
    })
    aero_models = list(aero_dir.glob("FNO.0.1.mdlus")); aero_states = list(aero_dir.glob("checkpoint.0.1.pt"))
    if len(aero_models) != 1 or len(aero_states) != 1: raise RuntimeError("aerodynamic checkpoint pair differs")
    fresh = build_model(cfg).to(dist.device); fresh_metadata: dict[str, Any] = {}
    if load_checkpoint(aero_dir, models=fresh, metadata_dict=fresh_metadata, device=dist.device) != 1:
        raise RuntimeError("fresh aerodynamic reload failed")
    if fresh_metadata.get("status") != "FC_P013_INDEPENDENT_FORCE_FNO_AERODYNAMIC_CHECKPOINT":
        raise RuntimeError("aerodynamic checkpoint metadata differs")
    if tensor_state_sha256(fresh) != tensor_state_sha256(aerodynamic_model):
        raise RuntimeError("fresh aerodynamic reload tensor SHA differs")
    manifest = build_dual_manifest(args.output, aero_models[0], aero_states[0])
    final_layer = aerodynamic_model.decoder_net.final_layer.linear
    result = {
        "status": "FC_P013_INDEPENDENT_FORCE_FNO_TRAINING_COMPLETE_NOT_ADMISSION",
        "optimizer_steps": TOTAL_STEPS, "records": records,
        "flow_tensor_sha256_before": flow_before, "flow_tensor_sha256_after": tensor_state_sha256(flow_model),
        "aerodynamic_tensor_sha256_before": aero_before, "aerodynamic_tensor_sha256_after": tensor_state_sha256(aerodynamic_model),
        "dual_model_manifest_sha256": sha256(manifest), "input_sha256": input_sha,
        "discarded_aerodynamic_field_rows": [0, 1, 2],
        "discarded_field_rows_may_change_via_adamw_weight_decay": True,
        "discarded_field_weight_gradient_norm_terminal": float(
            final_layer.weight.grad[:3].detach().double().square().sum().sqrt()
        ),
        "discarded_field_bias_gradient_norm_terminal": float(
            final_layer.bias.grad[:3].detach().double().square().sum().sqrt()
        ),
        "selection_performed": False, "validation_accessed": False,
        "frozen_test_accessed": False, "ppo_executed": False,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    train.close()


if __name__ == "__main__":
    main()
