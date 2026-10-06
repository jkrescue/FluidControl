"""Project-owned P029 H10 field/force objective; official FNOs are unchanged."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


HORIZON = 10


def _finite(*values: torch.Tensor) -> None:
    if any(not bool(torch.isfinite(value).all()) for value in values):
        raise FloatingPointError("nonfinite P029 objective value")


def _scale(value: float | torch.Tensor, reference: torch.Tensor, name: str) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if result.ndim != 0 or result.requires_grad or not bool(torch.isfinite(result)) or not bool(result > 0):
        raise ValueError(f"{name} must be a fixed finite positive scalar")
    return result


def control_aware_rollout_losses(
    flow_model: torch.nn.Module,
    aerodynamic_model: torch.nn.Module,
    initial_state: torch.Tensor,
    target_state: torch.Tensor,
    mask: torch.Tensor,
    actions: torch.Tensor,
    target_force: torch.Tensor,
    predict_fn: Callable,
    make_inputs_fn: Callable,
) -> dict[str, torch.Tensor]:
    """Return raw per-step losses with deployed K1 action/target alignment.

    Force at step ``j`` is predicted from current q_(s+j), omega_(s+j), and
    omega_(s+j+1), then compared with F_(s+j+1).  It is evaluated before the
    flow update: force step zero therefore has no flow dependency, while steps
    one through nine consume preceding flow predictions.  qhat_(s+10) remains
    supervised by the field loss only.
    """
    if initial_state.ndim != 4 or initial_state.shape[1] != 3:
        raise ValueError("initial state must be [B,3,H,W]")
    batch, _, height, width = initial_state.shape
    expected = {
        "target_state": ((batch, 100, 3, height, width), target_state),
        "mask": ((batch, 1, height, width), mask),
        "actions": ((batch, 101, 1), actions),
        "target_force": ((batch, 100, 4), target_force),
    }
    for name, (shape, value) in expected.items():
        if value.shape != shape:
            raise ValueError(f"{name} shape differs")
        if value.device != initial_state.device or value.dtype != initial_state.dtype:
            raise ValueError(f"{name} dtype/device differs")
    if any(value.requires_grad for _, value in expected.values()) or initial_state.requires_grad:
        raise ValueError("training data must not require gradients")
    _finite(initial_state, target_state, mask, actions, target_force)
    if (not bool(torch.all((mask == 0) | (mask == 1)))
            or not bool(torch.all(mask.sum((1, 2, 3)) > 0))):
        raise ValueError("finite nonempty binary mask required for every batch member")
    if any(parameter.requires_grad or parameter.grad is not None
           for parameter in aerodynamic_model.parameters()):
        raise ValueError("aerodynamic parameters must be frozen with no gradients")
    if aerodynamic_model.training:
        raise ValueError("aerodynamic model must be eval")

    current = initial_state * mask
    field_losses = []
    force_losses = []
    for step in range(HORIZON):
        inputs = make_inputs_fn(current, mask, actions[:, step], actions[:, step + 1])
        _, force = predict_fn(aerodynamic_model, inputs, mask)
        if force.shape != (batch, 4):
            raise ValueError("aerodynamic force shape differs")
        force_error = force - target_force[:, step]
        force_losses.append(force_error.square().mean(dim=1))

        delta, unused_force = predict_fn(flow_model, inputs, mask)
        if delta.shape != current.shape or unused_force.shape != (batch, 4):
            raise ValueError("official flow prediction shape differs")
        current = (current + delta) * mask
        field_error = (current - target_state[:, step]) * mask
        denominator = mask.sum((1, 2, 3)).clamp_min(1) * 3
        field_losses.append(field_error.square().sum((1, 2, 3)) / denominator)
        _finite(force, delta, unused_force, current, field_losses[-1], force_losses[-1])
    return {
        "field_per_step": torch.stack(field_losses, dim=1),
        "force_per_step": torch.stack(force_losses, dim=1),
        "terminal_state": current,
    }


def field_force_rollout_objective(
    flow_model: torch.nn.Module,
    aerodynamic_model: torch.nn.Module,
    initial_state: torch.Tensor,
    target_state: torch.Tensor,
    mask: torch.Tensor,
    actions: torch.Tensor,
    target_force: torch.Tensor,
    predict_fn: Callable,
    make_inputs_fn: Callable,
    *,
    field_scale: float | torch.Tensor,
    force_scale: float | torch.Tensor,
    backward: bool,
) -> dict[str, Any]:
    """Apply the preregistered 50/50 parent-normalized P029 objective."""
    if type(backward) is not bool:
        raise ValueError("backward must be an exact bool")
    losses = control_aware_rollout_losses(
        flow_model,
        aerodynamic_model,
        initial_state,
        target_state,
        mask,
        actions,
        target_force,
        predict_fn,
        make_inputs_fn,
    )
    raw_field = losses["field_per_step"].mean()
    raw_force = losses["force_per_step"].mean()
    field_denominator = _scale(field_scale, raw_field, "field_scale")
    force_denominator = _scale(force_scale, raw_force, "force_scale")
    field_contribution = 0.5 * raw_field / field_denominator
    force_contribution = 0.5 * raw_force / force_denominator
    total = field_contribution + force_contribution
    _finite(raw_field, raw_force, field_contribution, force_contribution, total)
    if backward:
        total.backward()
    return {
        "total": float(total.detach()),
        "raw_field": float(raw_field.detach()),
        "raw_force": float(raw_force.detach()),
        "normalized_field_contribution": float(field_contribution.detach()),
        "normalized_force_contribution": float(force_contribution.detach()),
        "field_per_step": [float(value) for value in losses["field_per_step"].mean(0).detach()],
        "force_per_step": [float(value) for value in losses["force_per_step"].mean(0).detach()],
        "field_scale": float(field_denominator),
        "force_scale": float(force_denominator),
        "rollout_steps": HORIZON,
        "force_terms_with_flow_gradient": 9,
        "terminal_state_force_supervised": False,
        "full_ten_step_field_gradient": bool(backward),
    }
