"""Train-only true-state paired force-response primitives.

These project helpers are intentionally independent of the PhysicsNeMo model
implementation.  They define a possible controlled follow-up objective; they
do not alter the current trainer or authorize an experiment.
"""

from __future__ import annotations

import torch


def true_state_step_input(
    initial_state: torch.Tensor,
    target_state: torch.Tensor,
    mask: torch.Tensor,
    omega: torch.Tensor,
    step: int,
) -> torch.Tensor:
    """Build the causal FNO input for one step using the recorded CFD state.

    ``target_state[:, k]`` is the state at endpoint ``k + 1``.  Therefore step
    zero consumes ``initial_state`` and later step ``k`` consumes
    ``target_state[:, k - 1]``.  The action channels are the recorded endpoint
    pair ``omega[k], omega[k + 1]``.  No predicted state is accepted here.
    """
    if initial_state.ndim != 4 or target_state.ndim != 5:
        raise ValueError("state tensors must have shapes [B,C,H,W] and [B,T,C,H,W]")
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("mask must have shape [B,1,H,W]")
    if omega.ndim != 3 or omega.shape[-1] != 1:
        raise ValueError("omega must have shape [B,T+1,1]")
    batch, channels, height, width = initial_state.shape
    steps = target_state.shape[1]
    if (
        batch < 1
        or steps < 1
        or channels != 3
        or target_state.shape[:3] != (batch, steps, channels)
        or target_state.shape[-2:] != (height, width)
        or mask.shape != (batch, 1, height, width)
        or omega.shape[:2] != (batch, steps + 1)
    ):
        raise ValueError("state, mask, and action sequence shapes differ")
    if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step < steps:
        raise ValueError("step must be an integer in [0, T)")
    state = initial_state if step == 0 else target_state[:, step - 1]
    omega_now = omega[:, step].reshape(batch, 1, 1, 1).expand(-1, 1, height, width)
    omega_next = omega[:, step + 1].reshape(batch, 1, 1, 1).expand(-1, 1, height, width)
    return torch.cat((state, mask, omega_now, omega_next), dim=1)


def paired_step_force_delta_loss(
    predicted_action: torch.Tensor,
    predicted_zero: torch.Tensor,
    target_action: torch.Tensor,
    target_zero: torch.Tensor,
    channel_weights: torch.Tensor,
) -> torch.Tensor:
    """Weighted MSE of per-step action-minus-zero force in normalized units.

    All force tensors use the unchanged train-only force normalization and
    shape ``[B,T,C]``.  The caller must align index ``k`` with the endpoint
    ``k + 1`` predicted from the true state/action input built for step ``k``;
    tensors alone cannot prove that temporal metadata contract.  A common
    force bias cancels by construction.  Channel weights must already be the
    existing positive, sum-one training weights.
    """
    shapes = {
        tuple(predicted_action.shape),
        tuple(predicted_zero.shape),
        tuple(target_action.shape),
        tuple(target_zero.shape),
    }
    if len(shapes) != 1 or predicted_action.ndim != 3:
        raise ValueError("all force tensors must have identical shape [B,T,C]")
    channels = predicted_action.shape[-1]
    if channel_weights.shape != (channels,):
        raise ValueError("channel weights differ from force channels")
    if (
        not torch.isfinite(channel_weights).all()
        or torch.any(channel_weights <= 0)
        or not torch.isclose(
            channel_weights.sum(),
            torch.ones((), dtype=channel_weights.dtype, device=channel_weights.device),
            rtol=1e-6,
            atol=1e-7,
        )
    ):
        raise ValueError("channel weights must be positive, finite, and sum to one")
    values = (predicted_action - predicted_zero) - (target_action - target_zero)
    loss = (values.square() * channel_weights).sum(dim=-1).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite paired step-force loss")
    return loss
