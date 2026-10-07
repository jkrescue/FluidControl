"""Isolated CPU-review helpers for P064 physical y-reflection pairing.

This module does not modify the B DataPipe or trainer.  It operates on the
normalized rollout sample returned by the project TandemRolloutDataset and
requires the unchanged train-only statistics explicitly.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Mapping

import torch


STATE_PARITY = (1.0, -1.0, 1.0)  # u, v, p
FORCE_PARITY = (1.0, -1.0, 1.0, -1.0)  # front Cd/Cl, rear Cd/Cl


@dataclass(frozen=True)
class ReflectionStats:
    state_mean: torch.Tensor
    state_std: torch.Tensor
    force_mean: torch.Tensor
    force_std: torch.Tensor

    def on(self, reference: torch.Tensor) -> "ReflectionStats":
        def converted(value: torch.Tensor) -> torch.Tensor:
            return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)

        state_mean = converted(self.state_mean).reshape(3, 1, 1)
        state_std = converted(self.state_std).reshape(3, 1, 1)
        force_mean = converted(self.force_mean).reshape(4)
        force_std = converted(self.force_std).reshape(4)
        if not all(
            bool(torch.isfinite(value).all())
            for value in (state_mean, state_std, force_mean, force_std)
        ):
            raise ValueError("normalization statistics must be finite")
        if torch.any(state_std <= 0) or torch.any(force_std <= 0):
            raise ValueError("normalization standard deviations must be positive")
        return ReflectionStats(state_mean, state_std, force_mean, force_std)


def _flip_y(value: torch.Tensor) -> torch.Tensor:
    if value.ndim < 2:
        raise ValueError("spatial tensor requires y and x dimensions")
    return value.flip(-2)


def _require_symmetric_mask(mask: torch.Tensor) -> None:
    if mask.ndim < 3 or mask.shape[-3] != 1:
        raise ValueError("mask must end in [1,y,x]")
    if not torch.equal(mask, _flip_y(mask)):
        raise ValueError("training mask is not exactly y-reflection symmetric")


def reflect_physical_state(state: torch.Tensor) -> torch.Tensor:
    if state.shape[-3] != 3:
        raise ValueError("state must end in [3,y,x]")
    parity = state.new_tensor(STATE_PARITY).reshape(
        *((1,) * (state.ndim - 3)), 3, 1, 1
    )
    return _flip_y(state) * parity


def reflect_physical_force(force: torch.Tensor) -> torch.Tensor:
    if force.shape[-1] != 4:
        raise ValueError("force must end in four physical components")
    return force * force.new_tensor(FORCE_PARITY)


def _reflect_normalized_state(
    normalized: torch.Tensor,
    mask: torch.Tensor,
    stats: ReflectionStats,
) -> torch.Tensor:
    physical = normalized * stats.state_std + stats.state_mean
    reflected = reflect_physical_state(physical)
    return ((reflected - stats.state_mean) / stats.state_std) * _flip_y(mask)


def _reflect_normalized_force(
    normalized: torch.Tensor,
    stats: ReflectionStats,
) -> torch.Tensor:
    physical = normalized * stats.force_std + stats.force_mean
    reflected = reflect_physical_force(physical)
    return (reflected - stats.force_mean) / stats.force_std


def reflect_normalized_rollout(
    sample: Mapping[str, torch.Tensor], stats: ReflectionStats
) -> dict[str, torch.Tensor]:
    """Reflect a normalized rollout without changing train-only statistics."""

    required = {"state", "target_state", "omega", "target_force", "mask"}
    keys = set(sample.keys())
    if not required.issubset(keys):
        raise ValueError(f"rollout sample is missing {sorted(required - keys)}")
    state = sample["state"]
    mask = sample["mask"]
    if (
        state.ndim != 3
        or sample["target_state"].ndim != 4
        or sample["omega"].ndim != 2
        or sample["target_force"].ndim != 2
        or mask.ndim != 3
        or state.shape[0] != 3
        or sample["target_state"].shape[1] != 3
        or sample["omega"].shape[1] != 1
        or sample["target_force"].shape[1] != 4
    ):
        raise ValueError("unbatched rollout shape contract differs")
    _require_symmetric_mask(mask)
    bound = stats.on(state)
    reflected_mask = _flip_y(mask).clone()
    result = {key: sample[key] for key in sample.keys()}
    result.update(
        state=_reflect_normalized_state(state, mask, bound),
        target_state=_reflect_normalized_state(
            sample["target_state"], mask.unsqueeze(0), bound
        ),
        omega=-sample["omega"],
        target_force=_reflect_normalized_force(sample["target_force"], bound),
        mask=reflected_mask,
    )
    # Never leave normalization offsets in solid cells.
    result["state"] *= reflected_mask
    result["target_state"] *= reflected_mask.unsqueeze(0)
    return result


def reflect_normalized_history(
    states: torch.Tensor,
    actions: torch.Tensor,
    mask: torch.Tensor,
    stats: ReflectionStats,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reflect preceding normalized states/actions; K1 empty history is valid."""

    _require_symmetric_mask(mask)
    if states.shape[-3:] != (3, *mask.shape[-2:]):
        raise ValueError("history state shape differs")
    bound = stats.on(mask)
    reflected = _reflect_normalized_state(states, mask, bound)
    reflected *= _flip_y(mask)
    return reflected, -actions


def physical_roundtrip_report(*values: torch.Tensor) -> dict[str, float]:
    """Check exact involution of the pure physical flip/sign map."""

    maxima: list[float] = []
    for value in values:
        if value.ndim >= 3 and value.shape[-3] == 3:
            twice = reflect_physical_state(reflect_physical_state(value))
        elif value.shape[-1] == 4:
            twice = reflect_physical_force(reflect_physical_force(value))
        else:
            raise ValueError("unsupported physical reflection tensor")
        if not torch.equal(twice.contiguous().view(torch.uint8), value.contiguous().view(torch.uint8)):
            raise ValueError("physical reflection is not an exact involution")
        maxima.append(float((twice - value).abs().max()))
    return {"physical_max_abs_error": max(maxima, default=0.0)}


def normalized_roundtrip_report(
    sample: Mapping[str, torch.Tensor], stats: ReflectionStats
) -> dict[str, float | bool]:
    """Report float32 denormalize/renormalize roundtrip with fixed tolerance."""

    twice = reflect_normalized_rollout(reflect_normalized_rollout(sample, stats), stats)
    eps = torch.finfo(sample["state"].dtype).eps
    max_abs = 0.0
    max_rel = 0.0
    for key in ("state", "target_state", "target_force"):
        original = sample[key]
        error = (twice[key] - original).abs()
        max_abs = max(max_abs, float(error.max()))
        denom = original.abs().clamp_min(1.0)
        max_rel = max(max_rel, float((error / denom).max()))
        physical_scale = max(1.0, float(original.abs().max()))
        if not torch.allclose(
            twice[key], original, atol=8 * eps * physical_scale, rtol=8 * eps
        ):
            raise ValueError(f"normalized reflection roundtrip exceeds tolerance: {key}")
    for key in ("omega", "mask"):
        if not torch.equal(twice[key], sample[key]):
            raise ValueError(f"exact reflection roundtrip differs: {key}")
    zero = twice["mask"] == 0
    masked_zero = bool(torch.all(twice["state"].masked_select(zero.expand_as(twice["state"])) == 0))
    target_zero = zero.unsqueeze(0).expand_as(twice["target_state"])
    masked_zero = masked_zero and bool(
        torch.all(twice["target_state"].masked_select(target_zero) == 0)
    )
    if not masked_zero:
        raise ValueError("masked normalized cells are not exactly zero")
    return {
        "normalized_max_abs_error": max_abs,
        "normalized_max_relative_error": max_rel,
        "atol_multiplier_eps": 8.0,
        "rtol_multiplier_eps": 8.0,
        "masked_cells_exact_zero": True,
    }


@contextmanager
def _scaled_parameter_backward(model: torch.nn.Module, scale: float):
    handles = [
        parameter.register_hook(lambda gradient, scale=scale: gradient * scale)
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def _require_detached_lightweight(value, path="result") -> None:
    if torch.is_tensor(value):
        if value.requires_grad or value.grad_fn is not None:
            raise ValueError(f"branch result retains an autograd graph: {path}")
        if value.numel() > 64:
            raise ValueError(f"branch result retains a large tensor: {path}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_detached_lightweight(item, f"{path}.{key}")
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _require_detached_lightweight(item, f"{path}[{index}]")


def serial_half_pair_backward(
    model: torch.nn.Module,
    original: Mapping[str, torch.Tensor],
    reflected: Mapping[str, torch.Tensor],
    branch_backward: Callable[[Mapping[str, torch.Tensor], str], Mapping],
) -> dict[str, Mapping]:
    """Run two independent branches while scaling every internal backward by 1/2.

    ``branch_backward`` must construct its branch flow rollout from the supplied
    q0/actions. Branch results may contain only detached lightweight audit
    statistics, so no original graph or large prediction tensor is retained
    across the reflected branch.
    """

    with _scaled_parameter_backward(model, 0.5):
        original_result = dict(branch_backward(original, "original"))
    _require_detached_lightweight(original_result, "original_result")
    with _scaled_parameter_backward(model, 0.5):
        reflected_result = dict(branch_backward(reflected, "reflected"))
    _require_detached_lightweight(reflected_result, "reflected_result")
    return {"original": original_result, "reflected": reflected_result}


def optional_reflection_backward(
    enabled: bool,
    model: torch.nn.Module,
    original: Mapping[str, torch.Tensor],
    reflect: Callable[[Mapping[str, torch.Tensor]], Mapping[str, torch.Tensor]],
    branch_backward: Callable[[Mapping[str, torch.Tensor], str], Mapping],
):
    """Explicit seam: disabled calls the unchanged original branch exactly once."""

    if not enabled:
        return branch_backward(original, "original")
    return serial_half_pair_backward(model, original, reflect(original), branch_backward)
