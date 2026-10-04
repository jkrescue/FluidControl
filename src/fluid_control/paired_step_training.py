"""Bounded-memory training step for true-state paired force supervision.

This is project training code, not a PhysicsNeMo API.  It keeps the regular
autoregressive objective unchanged, then adds train-only paired force
gradients in independent true-state time chunks before one clip/optimizer
step.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch

from fluid_control.paired_step_force import (
    paired_step_force_delta_loss,
    true_state_step_input,
)


def paired_objective_kind(configured: Any = None) -> str:
    """Resolve the optional objective switch while preserving the old default."""
    value = "paired_statistics" if configured is None else str(configured)
    if value not in {"paired_statistics", "true_state_step_force"}:
        raise ValueError(f"unsupported paired_objective_kind: {value}")
    return value


def prepare_true_state_training_model(model: torch.nn.Module) -> None:
    """Enter training mode before the caller constructs the regular graph."""
    model.train()
    if not model.training:
        raise RuntimeError("model refused training mode")


def summarize_true_state_step_updates(
    records: list[Mapping[str, Any]],
    *,
    expected_updates: int,
    force_channels: int,
) -> dict[str, Any]:
    """Validate and summarize the exact per-epoch paired update records."""
    if len(records) != expected_updates:
        raise RuntimeError(
            f"true-state update count differs: {len(records)} != {expected_updates}"
        )
    channel_values = []
    weighted_channel_values = []
    for record in records:
        values = record.get("paired_step_force_per_channel_mse")
        weighted_values = record.get(
            "paired_step_force_per_channel_weighted_contribution"
        )
        numeric = (
            record.get("paired_step_force_loss"),
            record.get("paired_step_force_weighted_loss"),
            record.get("preclip_gradient_norm"),
        )
        if (
            record.get("optimizer_steps") != 1
            or not isinstance(values, list)
            or len(values) != force_channels
            or not isinstance(weighted_values, list)
            or len(weighted_values) != force_channels
            or not all(
                torch.isfinite(torch.as_tensor(value))
                for value in (*numeric, *values, *weighted_values)
            )
        ):
            raise RuntimeError("invalid true-state paired update record")
        channel_values.append(values)
        weighted_channel_values.append(weighted_values)
    channels = torch.as_tensor(channel_values, dtype=torch.float64)
    weighted_channels = torch.as_tensor(weighted_channel_values, dtype=torch.float64)
    return {
        "train_true_state_paired_step_force_loss": sum(
            float(record["paired_step_force_loss"]) for record in records
        )
        / expected_updates,
        "train_true_state_paired_step_force_weighted_loss": sum(
            float(record["paired_step_force_weighted_loss"]) for record in records
        )
        / expected_updates,
        "train_true_state_paired_step_force_per_channel_mse": channels.mean(0).tolist(),
        "train_true_state_paired_step_force_per_channel_weighted_contribution": (
            weighted_channels.mean(0).tolist()
        ),
        "train_true_state_paired_step_preclip_gradient_norms": [
            float(record["preclip_gradient_norm"]) for record in records
        ],
        "train_true_state_paired_optimizer_steps": sum(
            int(record["optimizer_steps"]) for record in records
        ),
    }


def _validate_pair(pair: Mapping[str, torch.Tensor], total_steps: int) -> None:
    required = {
        "action_state",
        "zero_state",
        "action_omega",
        "zero_omega",
        "action_force",
        "zero_force",
        "mask",
    }
    if not required.issubset(pair):
        raise ValueError(f"paired batch is missing keys: {sorted(required - set(pair))}")
    if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps < 1:
        raise ValueError("total_steps must be a positive integer")
    action_state = pair["action_state"]
    zero_state = pair["zero_state"]
    if (
        action_state.ndim != 5
        or action_state.shape != zero_state.shape
        or action_state.shape[1] != total_steps + 1
        or action_state.shape[2] != 3
    ):
        raise ValueError("paired states must have shape [B,T+1,3,H,W]")
    batch = action_state.shape[0]
    channels = pair["action_force"].shape[-1]
    expected_force = (batch, total_steps + 1, channels)
    if (
        pair["action_force"].ndim != 3
        or tuple(pair["action_force"].shape) != expected_force
        or tuple(pair["zero_force"].shape) != expected_force
        or tuple(pair["action_omega"].shape) != (batch, total_steps + 1, 1)
        or tuple(pair["zero_omega"].shape) != (batch, total_steps + 1, 1)
        or tuple(pair["mask"].shape)
        != (batch, 1, action_state.shape[-2], action_state.shape[-1])
    ):
        raise ValueError("paired force/action/mask shapes differ from paired states")


def predict_true_state_force_chunk(
    model: torch.nn.Module,
    pair: Mapping[str, torch.Tensor],
    start: int,
    stop: int,
    predict_force: Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict paired forces for endpoints ``start+1`` through ``stop``."""
    action, zero = [], []
    mask = torch.cat((pair["mask"], pair["mask"]), dim=0)
    batch = pair["action_state"].shape[0]
    for step in range(start, stop):
        action_input = true_state_step_input(
            pair["action_state"][:, 0],
            pair["action_state"][:, 1:],
            pair["mask"],
            pair["action_omega"],
            step,
        )
        zero_input = true_state_step_input(
            pair["zero_state"][:, 0],
            pair["zero_state"][:, 1:],
            pair["mask"],
            pair["zero_omega"],
            step,
        )
        force = predict_force(model, torch.cat((action_input, zero_input), dim=0), mask)
        if force.ndim != 2 or force.shape[0] != 2 * batch:
            raise ValueError("predict_force must return [2B,C]")
        action.append(force[:batch])
        zero.append(force[batch:])
    return torch.stack(action, dim=1), torch.stack(zero, dim=1)


def true_state_paired_optimizer_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    base_loss: torch.Tensor,
    pair: Mapping[str, torch.Tensor],
    channel_weights: torch.Tensor,
    paired_weight: float,
    predict_force: Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], torch.Tensor],
    total_steps: int = 100,
    chunk_size: int = 10,
    gradient_clip_norm: float = 1.0,
) -> dict[str, Any]:
    """Apply one regular-plus-paired update with one final gradient clip.

    The already-built regular graph is backwarded and released first.  Each
    true-state chunk then builds and releases an independent graph.  No
    optimizer step occurs when a loss or accumulated gradient is non-finite.
    """
    optimizer.zero_grad(set_to_none=True)
    _validate_pair(pair, total_steps)
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size < 1
        or chunk_size > total_steps
    ):
        raise ValueError("chunk_size must be in [1,total_steps]")
    if not torch.isfinite(torch.as_tensor(paired_weight)) or paired_weight < 0:
        raise ValueError("paired_weight must be finite and non-negative")
    if (
        not torch.isfinite(torch.as_tensor(gradient_clip_norm))
        or gradient_clip_norm <= 0
    ):
        raise ValueError("gradient_clip_norm must be positive and finite")
    if base_loss.ndim != 0:
        raise ValueError("base_loss must be scalar")

    pair_loss_value = 0.0
    channel_sse = torch.zeros(
        pair["action_force"].shape[-1],
        dtype=torch.float64,
        device=pair["action_force"].device,
    )
    batch = pair["action_force"].shape[0]
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    try:
        if not torch.isfinite(base_loss):
            raise FloatingPointError("non-finite regular loss")
        base_loss.backward()
        for start in range(0, total_steps, chunk_size):
            stop = min(total_steps, start + chunk_size)
            predicted_action, predicted_zero = predict_true_state_force_chunk(
                model, pair, start, stop, predict_force
            )
            target_action = pair["action_force"][:, 1 + start : 1 + stop]
            target_zero = pair["zero_force"][:, 1 + start : 1 + stop]
            error = (predicted_action - predicted_zero) - (
                target_action - target_zero
            )
            channel_sse += error.detach().double().square().sum(dim=(0, 1))
            raw_loss = paired_step_force_delta_loss(
                predicted_action,
                predicted_zero,
                target_action,
                target_zero,
                channel_weights,
            )
            scaled_loss = raw_loss * paired_weight * ((stop - start) / total_steps)
            if not torch.isfinite(scaled_loss):
                raise FloatingPointError("non-finite true-state paired chunk loss")
            scaled_loss.backward()
            pair_loss_value += float(raw_loss.detach()) * ((stop - start) / total_steps)

        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
            raise FloatingPointError("missing or non-finite accumulated gradients")
        preclip_norm = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip_norm)
        if not torch.isfinite(preclip_norm):
            raise FloatingPointError("non-finite pre-clip gradient norm")
        optimizer.step()
    except Exception:
        optimizer.zero_grad(set_to_none=True)
        raise

    channel_mse = channel_sse / (batch * total_steps)
    return {
        "loss": float(base_loss.detach()) + paired_weight * pair_loss_value,
        "base_loss": float(base_loss.detach()),
        "paired_step_force_loss": pair_loss_value,
        "paired_step_force_weighted_loss": paired_weight * pair_loss_value,
        "paired_step_force_per_channel_mse": channel_mse.cpu().tolist(),
        "paired_step_force_per_channel_weighted_contribution": (
            channel_mse * channel_weights.detach().double()
        ).cpu().tolist(),
        "preclip_gradient_norm": float(preclip_norm.detach()),
        "optimizer_steps": 1,
        "chunk_size": chunk_size,
        "chunk_count": (total_steps + chunk_size - 1) // chunk_size,
    }
