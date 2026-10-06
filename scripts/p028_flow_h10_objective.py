"""FC-P028 project H10 flow objective; official FNO architecture is unchanged."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch


HORIZON = 10
ACCUMULATION_WINDOWS = 8


def _finite(*values: torch.Tensor) -> None:
    if any(not torch.isfinite(value).all() for value in values):
        raise FloatingPointError("nonfinite P028 objective input/output")


def flow_rollout_objective(
    flow_model: torch.nn.Module,
    initial_state: torch.Tensor,
    target_state: torch.Tensor,
    mask: torch.Tensor,
    actions: torch.Tensor,
    predict_fn: Callable,
    make_inputs_fn: Callable,
    *,
    backward: bool,
) -> dict[str, Any]:
    """Equal-weight masked field MSE through one non-detached H10 rollout.

    The caller passes the original H100/101-endpoint batch.  Only the first ten
    transitions are consumed, preserving the original 1368-window sampler.
    Force outputs from the seven-channel flow FNO are shape-checked but never
    enter this loss.
    """
    if type(backward) is not bool:
        raise ValueError("backward must be an exact bool")
    if initial_state.ndim != 4 or initial_state.shape[1] != 3:
        raise ValueError("initial state must be [B,3,H,W]")
    batch, _, height, width = initial_state.shape
    if target_state.shape != (batch, 100, 3, height, width):
        raise ValueError("original [B,100,3,H,W] targets required")
    if mask.shape != (batch, 1, height, width):
        raise ValueError("mask shape differs")
    if actions.shape != (batch, 101, 1):
        raise ValueError("original [B,101,1] actions required")
    if any(value.device != initial_state.device or value.dtype != initial_state.dtype
           for value in (target_state, mask, actions)):
        raise ValueError("objective dtype/device differs")
    if any(value.requires_grad for value in (initial_state, target_state, mask, actions)):
        raise ValueError("training data must not require gradients")
    _finite(initial_state, target_state, mask, actions)
    if not torch.all((mask == 0) | (mask == 1)) or not torch.all(mask.sum((1, 2, 3)) > 0):
        raise ValueError("finite nonempty binary masks required")

    current = initial_state * mask
    losses = []
    force_channels = None
    for step in range(HORIZON):
        inputs = make_inputs_fn(current, mask, actions[:, step], actions[:, step + 1])
        delta, unused_force = predict_fn(flow_model, inputs, mask)
        if delta.shape != current.shape or unused_force.shape != (batch, 4):
            raise ValueError("official flow prediction shape differs")
        if force_channels is None:
            force_channels = unused_force.shape[1]
        elif unused_force.shape[1] != force_channels:
            raise ValueError("flow force-output channel count changed")
        _finite(delta, unused_force)
        current = (current + delta) * mask
        target = target_state[:, step]
        error = (current - target) * mask
        denominator = mask.sum((1, 2, 3)).clamp_min(1) * 3
        losses.append(error.square().sum((1, 2, 3)) / denominator)
    per_step = torch.stack(losses, dim=1).mean(dim=0)
    total = torch.stack(losses, dim=1).mean()
    _finite(total, per_step)
    if backward:
        total.backward()
    return {
        "total": float(total.detach()),
        "per_step": [float(value) for value in per_step.detach()],
        "rollout_steps": HORIZON,
        "full_ten_step_gradient": bool(backward),
        "unused_flow_force_channels": int(force_channels),
    }


def accumulate_eight_window_gradients(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    windows: Iterable[Any],
    run_window: Callable[[Any], dict[str, Any]],
) -> dict[str, Any]:
    """Accumulate exactly eight raw H10 gradients and divide by eight.

    This deliberately does not clip or step.  The trainer owns the approved
    force-row/moment masking, clip-one, AdamW step and exact-row restoration.
    """
    optimizer.zero_grad(set_to_none=True)
    records = []
    for window in windows:
        if len(records) == ACCUMULATION_WINDOWS:
            raise ValueError("too many P028 windows in accumulation group")
        record = run_window(window)
        if set(("total", "rollout_steps")) - set(record):
            raise ValueError("P028 window record is incomplete")
        if record["rollout_steps"] != HORIZON or not torch.isfinite(
            torch.as_tensor(record["total"])
        ):
            raise FloatingPointError("invalid P028 window record")
        records.append(record)
    if len(records) != ACCUMULATION_WINDOWS:
        raise ValueError("incomplete P028 accumulation group")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters or any(parameter.grad is None for parameter in parameters):
        raise RuntimeError("missing P028 flow gradient")
    if any(not torch.isfinite(parameter).all() or not torch.isfinite(parameter.grad).all()
           for parameter in parameters):
        raise FloatingPointError("nonfinite P028 parameter/gradient")
    for parameter in parameters:
        parameter.grad.div_(ACCUMULATION_WINDOWS)
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise FloatingPointError("nonfinite P028 AdamW state")
    squared = sum(
        (parameter.grad.detach().abs().double().square().sum() for parameter in parameters),
        torch.zeros((), dtype=torch.float64, device=parameters[0].device),
    )
    return {
        "records": records,
        "windows": ACCUMULATION_WINDOWS,
        "mean_total": sum(float(record["total"]) for record in records)
        / ACCUMULATION_WINDOWS,
        "mean_gradient_norm_before_scope_mask_and_clip": float(torch.sqrt(squared)),
        "optimizer_step_performed": False,
    }
