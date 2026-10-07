"""Train-only temporal-force residual primitives; no loading or execution."""

from __future__ import annotations

from typing import Any, Callable

import torch


FORCE_ROWS = slice(3, 7)
FINAL_WEIGHT = "decoder_net.final_layer.linear.weight"
FINAL_BIAS = "decoder_net.final_layer.linear.bias"


def zero_force_output_rows(model: torch.nn.Module) -> dict[str, Any]:
    """Make the four force deltas exactly zero without changing field rows."""
    named = dict(model.named_parameters())
    if set((FINAL_WEIGHT, FINAL_BIAS)) - set(named):
        raise ValueError("official seven-output terminal layer is absent")
    weight, bias = named[FINAL_WEIGHT], named[FINAL_BIAS]
    if weight.ndim != 2 or weight.shape[0] != 7 or bias.shape != (7,):
        raise ValueError("official terminal layer shape differs")
    field_weight = weight[:3].detach().clone()
    field_bias = bias[:3].detach().clone()
    with torch.no_grad():
        weight[FORCE_ROWS].zero_()
        bias[FORCE_ROWS].zero_()
    if not torch.equal(weight[:3], field_weight) or not torch.equal(bias[:3], field_bias):
        raise RuntimeError("field-output rows changed during residual initialization")
    if torch.count_nonzero(weight[FORCE_ROWS]) or torch.count_nonzero(bias[FORCE_ROWS]):
        raise RuntimeError("force-output rows did not become exact zero")
    return {
        "weight_name": FINAL_WEIGHT,
        "bias_name": FINAL_BIAS,
        "weight_shape": list(weight.shape),
        "bias_shape": list(bias.shape),
        "zero_force_rows": [3, 4, 5, 6],
        "field_rows_preserved": True,
    }


def causal_current_force(target_force: torch.Tensor, initial_force: torch.Tensor) -> torch.Tensor:
    """Truth-current force for H1: initial f0 then prior endpoints f1..f99."""
    if target_force.ndim != 3 or target_force.shape[1:] != (100, 4):
        raise ValueError("target_force must be [B,100,4]")
    if initial_force.shape != (target_force.shape[0], 4):
        raise ValueError("initial_force must be [B,4]")
    if any(not torch.isfinite(x).all() for x in (target_force, initial_force)):
        raise FloatingPointError("nonfinite force sequence")
    return torch.cat((initial_force[:, None], target_force[:, :-1]), dim=1)


def temporal_residual_objective(
    aerodynamic_model: torch.nn.Module,
    flow_states: torch.Tensor,
    h1_states: torch.Tensor,
    mask: torch.Tensor,
    omega: torch.Tensor,
    target_force: torch.Tensor,
    initial_force: torch.Tensor,
    make_inputs: Callable[..., torch.Tensor],
    predict_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    force_objective: Callable[[torch.Tensor, torch.Tensor], dict[str, torch.Tensor]],
    *,
    backward: bool,
) -> dict[str, Any]:
    """Full H100 residual recurrence with no chunk-boundary detach.

    Network force rows are normalized increments. H1 uses the true force at the
    current endpoint; AR starts at the same measured f0 and thereafter carries
    only its own reconstructed prediction. Loss remains the original absolute
    endpoint-force objective.
    """
    if flow_states.shape != h1_states.shape or flow_states.ndim != 5:
        raise ValueError("H1/AR state sequences differ")
    batch, steps = flow_states.shape[:2]
    if steps != 100 or target_force.shape != (batch, 100, 4):
        raise ValueError("fixed H100 force alignment differs")
    if mask.shape[:2] != (batch, 1) or omega.shape != (batch, 101, 1):
        raise ValueError("mask/action contract differs")
    true_current = causal_current_force(target_force, initial_force)
    ar_current = initial_force
    h1_predictions, ar_predictions = [], []
    for step in range(100):
        states = torch.cat((h1_states[:, step], flow_states[:, step]), dim=0)
        masks = torch.cat((mask, mask), dim=0)
        now = torch.cat((omega[:, step], omega[:, step]), dim=0)
        nxt = torch.cat((omega[:, step + 1], omega[:, step + 1]), dim=0)
        inputs = make_inputs(states, masks, now, nxt)
        _, delta = predict_fn(aerodynamic_model, inputs, masks)
        if delta.shape != (2 * batch, 4) or not torch.isfinite(delta).all():
            raise ValueError("aerodynamic delta output differs")
        h1_next = true_current[:, step] + delta[:batch]
        ar_next = ar_current + delta[batch:]
        h1_predictions.append(h1_next)
        ar_predictions.append(ar_next)
        ar_current = ar_next  # full autograd carry; deliberately never detached
    h1 = torch.stack(h1_predictions, dim=1)
    ar = torch.stack(ar_predictions, dim=1)
    h1_loss = force_objective(h1, target_force)
    ar_loss = force_objective(ar, target_force)
    total = 0.5 * h1_loss["balanced"] + 0.5 * ar_loss["balanced"]
    if not torch.isfinite(total):
        raise FloatingPointError("nonfinite residual objective")
    if backward:
        total.backward()
    return {
        "h1": h1,
        "ar": ar,
        "h1_balanced": h1_loss["balanced"].detach(),
        "ar_balanced": ar_loss["balanced"].detach(),
        "total": total.detach(),
        "aerodynamic_calls": 100,
        "force_recurrence_detaches": 0,
        "supervised_force_points_per_domain": 100,
    }


def exact_recompute_vjp_objective(
    aerodynamic_model: torch.nn.Module,
    flow_states: torch.Tensor,
    h1_states: torch.Tensor,
    mask: torch.Tensor,
    omega: torch.Tensor,
    target_force: torch.Tensor,
    initial_force: torch.Tensor,
    make_inputs: Callable[..., torch.Tensor],
    predict_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    force_objective: Callable[[torch.Tensor, torch.Tensor], dict[str, torch.Tensor]],
    *,
    chunk_size: int = 10,
    backward: bool,
) -> dict[str, Any]:
    """Exact full-recurrence VJP with at most ``2*chunk_size`` model activations.

    The delta network does not consume force, and both field sequences are
    frozen. Therefore each delta is an independent function of parameters.
    A no-grad pass obtains all deltas; a tiny force-only graph computes exact
    full-H100 cotangents (including every future AR loss); model chunks are then
    replayed against those cotangents. No force carry is detached or truncated.
    """
    if chunk_size < 1 or 100 % chunk_size:
        raise ValueError("chunk_size must divide 100")
    if not aerodynamic_model.training:
        raise ValueError("probe replay must use the fixed training-mode model")
    batch = flow_states.shape[0]
    if flow_states.shape != h1_states.shape or flow_states.shape[1] != 100:
        raise ValueError("fixed H100 state sequences differ")

    def chunk_inputs(begin: int, end: int):
        length = end - begin
        h1 = h1_states[:, begin:end].reshape(batch * length, *h1_states.shape[2:])
        ar = flow_states[:, begin:end].reshape(batch * length, *flow_states.shape[2:])
        states = torch.cat((h1, ar), dim=0)
        masks = mask[:, None].expand(-1, length, -1, -1, -1).reshape(
            batch * length, *mask.shape[1:]
        )
        masks = torch.cat((masks, masks), dim=0)
        now = omega[:, begin:end].reshape(batch * length, -1)
        nxt = omega[:, begin + 1 : end + 1].reshape(batch * length, -1)
        inputs = make_inputs(states, masks, torch.cat((now, now)), torch.cat((nxt, nxt)))
        return inputs, masks, batch * length

    saved_h1, saved_ar = [], []
    with torch.no_grad():
        for begin in range(0, 100, chunk_size):
            inputs, masks, count = chunk_inputs(begin, begin + chunk_size)
            _, delta = predict_fn(aerodynamic_model, inputs, masks)
            if delta.shape != (2 * count, 4) or not torch.isfinite(delta).all():
                raise ValueError("aerodynamic delta output differs")
            saved_h1.append(delta[:count].reshape(batch, chunk_size, 4))
            saved_ar.append(delta[count:].reshape(batch, chunk_size, 4))
    h1_leaf = torch.cat(saved_h1, 1).detach().requires_grad_(True)
    ar_leaf = torch.cat(saved_ar, 1).detach().requires_grad_(True)
    true_current = causal_current_force(target_force, initial_force)
    h1_prediction = true_current + h1_leaf
    ar_prediction = initial_force[:, None] + torch.cumsum(ar_leaf, dim=1)
    h1_loss = force_objective(h1_prediction, target_force)
    ar_loss = force_objective(ar_prediction, target_force)
    total = 0.5 * h1_loss["balanced"] + 0.5 * ar_loss["balanced"]
    if not torch.isfinite(total):
        raise FloatingPointError("nonfinite residual objective")
    h1_cotangent, ar_cotangent = torch.autograd.grad(total, (h1_leaf, ar_leaf))
    if backward:
        for begin in range(0, 100, chunk_size):
            end = begin + chunk_size
            inputs, masks, count = chunk_inputs(begin, end)
            _, delta = predict_fn(aerodynamic_model, inputs, masks)
            h1_delta = delta[:count].reshape(batch, chunk_size, 4)
            ar_delta = delta[count:].reshape(batch, chunk_size, 4)
            if not torch.equal(h1_delta.detach(), h1_leaf[:, begin:end].detach()):
                raise RuntimeError("H1 replay differs; stochastic model/precision is unsupported")
            if not torch.equal(ar_delta.detach(), ar_leaf[:, begin:end].detach()):
                raise RuntimeError("AR replay differs; stochastic model/precision is unsupported")
            surrogate = (h1_delta * h1_cotangent[:, begin:end]).sum()
            surrogate = surrogate + (ar_delta * ar_cotangent[:, begin:end]).sum()
            surrogate.backward()
    return {
        "h1": h1_prediction.detach(),
        "ar": ar_prediction.detach(),
        "h1_balanced": h1_loss["balanced"].detach(),
        "ar_balanced": ar_loss["balanced"].detach(),
        "total": total.detach(),
        "aerodynamic_no_grad_calls": 100 // chunk_size,
        "aerodynamic_recompute_calls": 100 // chunk_size if backward else 0,
        "maximum_model_batch": 2 * batch * chunk_size,
        "force_recurrence_detaches": 0,
        "gradient_method": "exact_full_h100_force_cotangent_then_chunked_model_vjp",
        "supervised_force_points_per_domain": 100,
    }


def gradient_inventory(model: torch.nn.Module) -> dict[str, Any]:
    """Report every trainable gradient; zero hidden gradients are valid at step0."""
    rows = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            if parameter.grad is not None:
                raise RuntimeError("frozen parameter received a gradient")
            continue
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"missing/nonfinite gradient: {name}")
        rows[name] = {
            "shape": list(parameter.shape),
            "l2": float(parameter.grad.detach().double().norm()),
            "nonzero": int(torch.count_nonzero(parameter.grad.detach())),
        }
    if len(rows) != 28:
        raise RuntimeError("official aerodynamic trainable tensor count differs")
    head = {FINAL_WEIGHT, FINAL_BIAS}
    hidden_nonzero = sum(row["nonzero"] for name, row in rows.items() if name not in head)
    return {
        "tensors": rows,
        "trainable_tensors": len(rows),
        "hidden_nonzero_elements": hidden_nonzero,
        "zero_head_expected_hidden_zero_at_initial_backward": True,
    }
