"""Stateless project glue for explicit K1/K4 dual-FNO inference.

The wrapped models remain official PhysicsNeMo FNO instances.  This module only
assembles causal project inputs and combines the existing raw output channels;
it does not add a learned module, residual update, mask pooling, or force scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

import p026_state_history as history
from fluid_control.dual_fno import combine_dual_raw


HistorySource = Literal[
    "trajectory_observed_with_frame0_left_padding",
    "generated_frame0_reset_padding",
    "autoregressive_prediction",
]


@dataclass(frozen=True)
class HistoryBuffer:
    """Explicit causal history; never stored as mutable adapter state."""

    states: torch.Tensor  # [B,K,3,H,W]
    actions: torch.Tensor  # [B,K]
    padding_mask: torch.Tensor  # [B,K], true only for generated left padding
    source: HistorySource

    @property
    def k(self) -> int:
        return int(self.states.shape[1])


def _validate_buffer(buffer: HistoryBuffer) -> None:
    states, actions, padded = buffer.states, buffer.actions, buffer.padding_mask
    if (
        states.ndim != 5
        or states.shape[1] not in (1, 4)
        or states.shape[2] != 3
        or actions.shape != states.shape[:2]
        or padded.shape != states.shape[:2]
        or padded.dtype != torch.bool
    ):
        raise ValueError("history buffer shape differs")
    if actions.device != states.device or actions.dtype != states.dtype:
        raise ValueError("history action device/dtype differs")
    if padded.device != states.device:
        raise ValueError("history padding-mask device differs")
    if not torch.isfinite(states).all() or not torch.isfinite(actions).all():
        raise FloatingPointError("nonfinite history buffer")


def clone_history(buffer: HistoryBuffer) -> HistoryBuffer:
    """Return a non-aliasing copy suitable for env copy/reset/restore."""

    _validate_buffer(buffer)
    return HistoryBuffer(
        states=buffer.states.clone(),
        actions=buffer.actions.clone(),
        padding_mask=buffer.padding_mask.clone(),
        source=buffer.source,
    )


def trajectory_history(
    states: torch.Tensor,
    actions: torch.Tensor,
    starts: torch.Tensor,
    *,
    k: int,
) -> HistoryBuffer:
    """Gather only observed frames/actions at or before each trajectory start."""

    if states.ndim != 4 or states.shape[1] != 3 or len(states) == 0:
        raise ValueError("trajectory states must be [T,3,H,W]")
    actions = actions.reshape(-1)
    if actions.shape != (len(states),):
        raise ValueError("trajectory actions must have one value per frame")
    if states.device != actions.device or states.dtype != actions.dtype:
        raise ValueError("trajectory state/action device or dtype differs")
    if starts.ndim != 1 or starts.dtype == torch.bool:
        raise ValueError("one-dimensional integer starts required")
    if starts.dtype not in (torch.int32, torch.int64):
        raise ValueError("integer starts required")
    if starts.device != states.device:
        raise ValueError("trajectory starts device differs")
    gathered_states, gathered_actions, padding = [], [], []
    for item in starts.tolist():
        start = int(item)
        if start < 0 or start >= len(states):
            raise IndexError("trajectory start outside available frames")
        indices, padded = history.history_indices(start, k)
        if any(index > start for index in indices):
            raise AssertionError("future observation entered history")
        gathered_states.append(states[indices])
        gathered_actions.append(actions[indices])
        padding.append(padded)
    result = HistoryBuffer(
        states=torch.stack(gathered_states),
        actions=torch.stack(gathered_actions),
        padding_mask=torch.tensor(padding, dtype=torch.bool, device=states.device),
        source="trajectory_observed_with_frame0_left_padding",
    )
    _validate_buffer(result)
    return result


def reset_history(q0: torch.Tensor, omega0: torch.Tensor, *, k: int) -> HistoryBuffer:
    """Generate explicit trajectory-frame-0 padding for a fresh env reset."""

    if q0.ndim != 4 or q0.shape[1] != 3 or q0.shape[0] == 0 or k not in (1, 4):
        raise ValueError("reset state must be nonempty [B,3,H,W] and K1/K4")
    omega0 = omega0.reshape(-1)
    if omega0.shape != (q0.shape[0],):
        raise ValueError("reset action must have one value per batch item")
    if omega0.device != q0.device or omega0.dtype != q0.dtype:
        raise ValueError("reset state/action device or dtype differs")
    states = q0[:, None].expand(-1, k, -1, -1, -1).clone()
    actions = omega0[:, None].expand(-1, k).clone()
    padding = torch.ones((len(q0), k), dtype=torch.bool, device=q0.device)
    padding[:, -1] = False
    result = HistoryBuffer(
        states=states,
        actions=actions,
        padding_mask=padding,
        source="generated_frame0_reset_padding",
    )
    _validate_buffer(result)
    return result


def pack_history_input(
    buffer: HistoryBuffer, mask: torch.Tensor, next_action: torch.Tensor
) -> torch.Tensor:
    """Reuse the reviewed canonical packer for each explicit batch member."""

    _validate_buffer(buffer)
    if mask.shape != (buffer.states.shape[0], 1, *buffer.states.shape[-2:]):
        raise ValueError("history mask shape differs")
    next_action = next_action.reshape(-1)
    if next_action.shape != (buffer.states.shape[0],):
        raise ValueError("next action batch differs")
    if any(value.device != buffer.states.device for value in (mask, next_action)):
        raise ValueError("history pack device differs")
    if any(value.dtype != buffer.states.dtype for value in (mask, next_action)):
        raise ValueError("history pack dtype differs")
    packed = [
        history.build_input(
            buffer.states[index],
            mask[index],
            buffer.actions[index],
            next_action[index],
        )
        for index in range(buffer.states.shape[0])
    ]
    return torch.stack(packed)


def advance_history(
    buffer: HistoryBuffer,
    predicted_next_state: torch.Tensor,
    applied_next_action: torch.Tensor,
) -> HistoryBuffer:
    """Shift in predictions/actions explicitly; no observed future state is accepted."""

    _validate_buffer(buffer)
    if predicted_next_state.shape != (
        buffer.states.shape[0],
        3,
        *buffer.states.shape[-2:],
    ):
        raise ValueError("predicted next-state shape differs")
    applied_next_action = applied_next_action.reshape(-1)
    if applied_next_action.shape != (buffer.states.shape[0],):
        raise ValueError("applied next-action batch differs")
    shifted_states, shifted_actions = [], []
    for index in range(buffer.states.shape[0]):
        state, action = history.shift_history(
            buffer.states[index],
            buffer.actions[index],
            predicted_next_state[index],
            applied_next_action[index],
        )
        shifted_states.append(state)
        shifted_actions.append(action.reshape(-1))
    padding = torch.cat(
        (
            buffer.padding_mask[:, 1:],
            torch.zeros(
                (buffer.states.shape[0], 1),
                dtype=torch.bool,
                device=buffer.states.device,
            ),
        ),
        dim=1,
    )
    result = HistoryBuffer(
        states=torch.stack(shifted_states),
        actions=torch.stack(shifted_actions),
        padding_mask=padding,
        source="autoregressive_prediction",
    )
    _validate_buffer(result)
    return result


def _validate_dual_inputs(
    flow_inputs: torch.Tensor, aerodynamic_inputs: torch.Tensor, k: int
) -> None:
    expected_aero_channels = 4 * k + 2
    if (
        flow_inputs.ndim != 4
        or flow_inputs.shape[1] != 6
        or aerodynamic_inputs.ndim != 4
        or aerodynamic_inputs.shape[1] != expected_aero_channels
        or flow_inputs.shape[0] != aerodynamic_inputs.shape[0]
        or flow_inputs.shape[-2:] != aerodynamic_inputs.shape[-2:]
    ):
        raise ValueError("flow/aerodynamic input shape differs")
    if (
        flow_inputs.device != aerodynamic_inputs.device
        or flow_inputs.dtype != aerodynamic_inputs.dtype
    ):
        raise ValueError("flow/aerodynamic input device or dtype differs")
    current = 3 * (k - 1)
    mask = 3 * k
    current_action = mask + k
    next_action = current_action + 1
    comparisons = (
        (flow_inputs[:, :3], aerodynamic_inputs[:, current : current + 3]),
        (flow_inputs[:, 3:4], aerodynamic_inputs[:, mask : mask + 1]),
        (flow_inputs[:, 4:5], aerodynamic_inputs[:, current_action : current_action + 1]),
        (flow_inputs[:, 5:6], aerodynamic_inputs[:, next_action : next_action + 1]),
    )
    if any(not torch.equal(left, right) for left, right in comparisons):
        raise ValueError("flow/current history mask or action differs")
    if not torch.isfinite(flow_inputs).all() or not torch.isfinite(
        aerodynamic_inputs
    ).all():
        raise FloatingPointError("nonfinite dual input")


def make_history_dual_fno_adapter(flow_model, aerodynamic_model, *, k: int):
    """Wrap official models with stateless project history-routing glue."""

    if k not in (1, 4):
        raise ValueError("K1/K4 only")

    class StatelessHistoryDualFNO(torch.nn.Module):
        history_length = k

        def __init__(self):
            super().__init__()
            self.flow_model = flow_model
            self.aerodynamic_model = aerodynamic_model

        def forward(
            self,
            flow_inputs: torch.Tensor,
            aerodynamic_inputs: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if aerodynamic_inputs is None:
                if k == 4:
                    raise ValueError("K4 requires an explicit 18-channel aerodynamic input")
                aerodynamic_inputs = flow_inputs
            _validate_dual_inputs(flow_inputs, aerodynamic_inputs, k)
            return combine_dual_raw(
                self.flow_model(flow_inputs),
                self.aerodynamic_model(aerodynamic_inputs),
            )

    flow_model.eval().requires_grad_(False)
    aerodynamic_model.eval().requires_grad_(False)
    return StatelessHistoryDualFNO().eval()
