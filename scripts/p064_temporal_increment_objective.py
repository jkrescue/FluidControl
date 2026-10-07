"""Project history force objective; frozen flow, no aerodynamic field feedback.

The caller loads flow and aerodynamic parents separately. This module neither
loads nor changes them. Only the independently frozen flow supplies AR states;
the aerodynamic predictor's first three (field) outputs are discarded.
"""

import hashlib
import importlib.util
from pathlib import Path

import torch

OBJECTIVE_SHA = "f3cf4b9a745cc0cbee39db9385cbfc398e4834e25bc487d8fb2d6eca07b483d7"
TEMPORAL_INCREMENT_WEIGHT = 1.0
CONTROL_DT_D_OVER_U = 0.1


def temporal_increment_objective(prediction, target, original_objective):
    """Adjacent force increments, not causal action contrasts.

    Both inputs are normalized with the unchanged train-only force std.
    No division by dt: this is force change per fixed 0.1 D/U interval.
    Gradients flow through both neighboring predictions; targets stay detached.
    """
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[1] < 2 or prediction.shape[2] != 4:
        raise ValueError("batch x at-least-two x four force chunk required")
    if target.requires_grad:
        raise ValueError("target must not require gradients")
    return original_objective.balanced_force_objective(
        prediction[:, 1:] - prediction[:, :-1],
        target[:, 1:] - target[:, :-1],
    )["balanced"]


def load_original_objective(path):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != OBJECTIVE_SHA:
        raise ValueError("immutable P013 objective differs")
    spec = importlib.util.spec_from_file_location("p026_original_objective", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def chunk_inputs(states, preceding_states, mask, omega, preceding_actions, begin, end):
    """Build chronological histories using only states <= each current index.

    states: [B,100,3,H,W], preceding_states: [B,K-1,3,H,W]. The preceding
    tensors already contain explicit start-of-trajectory padding from the
    history adapter. omega is the existing normalized [B,101,1] command series.
    No future field/force measurement is an input to this function.
    """
    if states.ndim != 5 or states.shape[1:3] != (100, 3):
        raise ValueError("B100x3 state sequence required")
    batch, _, _, height, width = states.shape
    k = preceding_states.shape[1] + 1
    if k not in (1, 4) or preceding_states.shape != (batch, k - 1, 3, height, width):
        raise ValueError("K1/K4 preceding states required")
    if (
        mask.shape != (batch, 1, height, width)
        or omega.shape != (batch, 101, 1)
        or preceding_actions.shape != (batch, k - 1, 1)
    ):
        raise ValueError("history mask/action shapes differ")
    if not 0 <= begin < end <= 100:
        raise ValueError("invalid chunk bounds")
    if any(
        x.requires_grad
        for x in (states, preceding_states, mask, omega, preceding_actions)
    ):
        raise ValueError("all state/action histories must be frozen")
    if any(
        x.device != states.device or x.dtype != states.dtype
        for x in (preceding_states, mask, omega, preceding_actions)
    ):
        raise ValueError("history dtype/device differs")
    sequence = torch.cat((preceding_states, states), dim=1)
    commands = torch.cat((preceding_actions, omega), dim=1)
    length = end - begin
    # Batch-major/time-major flattening matches immutable P013 mixed20 order.
    history = torch.stack([sequence[:, t : t + k] for t in range(begin, end)], dim=1)
    actions = torch.stack(
        [commands[:, t : t + k + 1] for t in range(begin, end)], dim=1
    )
    fields = history.reshape(batch * length, 3 * k, height, width)
    masks = (
        mask[:, None]
        .expand(-1, length, -1, -1, -1)
        .reshape(batch * length, 1, height, width)
    )
    planes = actions.reshape(batch * length, k + 1, 1, 1).expand(-1, -1, height, width)
    return torch.cat((fields, masks, planes), dim=1), masks


def chunk_force_objective(
    aerodynamic_model,
    flow_states,
    h1_states,
    mask,
    omega,
    target_force,
    preceding_states,
    preceding_actions,
    predict_fn,
    original_objective,
    *,
    backward
):
    """Exact ten weighted chunks, each .5 H1 + .5 AR original force objective.

    Chunk backward is exact because all field histories are frozen and no
    predicted aerodynamic force/field is fed into another time step.
    """
    if flow_states.shape != h1_states.shape or target_force.shape != (
        flow_states.shape[0],
        100,
        4,
    ):
        raise ValueError("H1/AR target alignment differs")
    data = (
        flow_states,
        h1_states,
        mask,
        omega,
        target_force,
        preceding_states,
        preceding_actions,
    )
    if any(x.requires_grad for x in data):
        raise ValueError("data/frozen flow must not require gradients")
    if any(not torch.isfinite(x).all() for x in data):
        raise FloatingPointError("nonfinite frozen data")
    totals = {
        key: torch.zeros((), device=flow_states.device)
        for key in ("h1_balanced", "ar_balanced", "total")
    }
    channel_h1 = torch.zeros(4, device=flow_states.device)
    channel_ar = torch.zeros(4, device=flow_states.device)
    batch = flow_states.shape[0]
    captured = {"h1": [], "ar": []}
    temporal_total = torch.zeros((), device=flow_states.device)
    for begin in range(0, 100, 10):
        end = begin + 10
        h1, masks = chunk_inputs(
            h1_states, preceding_states, mask, omega, preceding_actions, begin, end
        )
        ar, ar_masks = chunk_inputs(
            flow_states, preceding_states, mask, omega, preceding_actions, begin, end
        )
        inputs = torch.cat((h1, ar), dim=0)
        masks = torch.cat((masks, ar_masks), dim=0)
        # Deliberately discard aerodynamic field outputs. Never update flow here.
        _, forces = predict_fn(aerodynamic_model, inputs, masks)
        if forces.shape != (2 * batch * 10, 4):
            raise ValueError("mixed H1/AR four-force output shape differs")
        h1_force, ar_force = forces[: batch * 10], forces[batch * 10 :]
        targets = target_force[:, begin:end].reshape(batch * 10, 4)
        hl = original_objective.balanced_force_objective(
            h1_force[:, None], targets[:, None]
        )
        al = original_objective.balanced_force_objective(
            ar_force[:, None], targets[:, None]
        )
        chunk_total = 0.5 * hl["balanced"] + 0.5 * al["balanced"]
        if backward:
            if TEMPORAL_INCREMENT_WEIGHT == 0:
                (0.1 * chunk_total).backward()
            else:
                temporal = temporal_increment_objective(
                    h1_force.reshape(batch, 10, 4),
                    targets.reshape(batch, 10, 4), original_objective)
                (0.1 * chunk_total + TEMPORAL_INCREMENT_WEIGHT * (9 / 99) * temporal).backward()
                temporal_total += (9 / 99) * temporal.detach()
                if begin:
                    # Recompute BOTH endpoints; no detached previous prediction.
                    edge_inputs, edge_masks = chunk_inputs(
                        h1_states, preceding_states, mask, omega,
                        preceding_actions, begin - 1, begin + 1)
                    _, edge_forces = predict_fn(aerodynamic_model, edge_inputs, edge_masks)
                    edge_loss = temporal_increment_objective(
                        edge_forces.reshape(batch, 2, 4),
                        target_force[:, begin - 1:begin + 1], original_objective)
                    (TEMPORAL_INCREMENT_WEIGHT * edge_loss / 99).backward()
                    temporal_total += edge_loss.detach() / 99
        totals["h1_balanced"] += 0.1 * hl["balanced"].detach()
        totals["ar_balanced"] += 0.1 * al["balanced"].detach()
        totals["total"] += 0.1 * chunk_total.detach()
        channel_h1 += 0.1 * hl["channel_mse"].detach()
        channel_ar += 0.1 * al["channel_mse"].detach()
        captured["h1"].append(h1_force.detach().reshape(batch, 10, 4))
        captured["ar"].append(ar_force.detach().reshape(batch, 10, 4))
    return dict(
        temporal_increment_loss=float(temporal_total) if backward else None,
        training_objective=(float(totals["total"] + TEMPORAL_INCREMENT_WEIGHT * temporal_total) if backward else None),
        **{key: float(value) for key, value in totals.items()},
        h1_channel_mse=[float(x) for x in channel_h1],
        ar_channel_mse=[float(x) for x in channel_ar],
        chunk_size=10,
        chunks=10,
        normalized_predictions={
            domain: torch.cat(parts, dim=1) for domain, parts in captured.items()
        },
    )
