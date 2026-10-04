"""Batch-contract helpers for controlled matched-pair FNO training."""

from __future__ import annotations

import torch


def paired_batch_indices(
    train_batches: int, paired_batches: int, schedule: str = "frontloaded"
) -> tuple[int, ...]:
    """Return deterministic regular-batch indices carrying paired updates."""
    if train_batches < 1:
        raise ValueError("train_batches must be positive")
    if not 1 <= paired_batches <= train_batches:
        raise ValueError("paired_batches must be in [1, train_batches]")
    if schedule == "frontloaded":
        return tuple(range(paired_batches))
    if schedule != "interleaved":
        raise ValueError(f"unsupported paired batch schedule: {schedule}")
    if paired_batches == 1:
        return (0,)
    indices = tuple(
        index * (train_batches - 1) // (paired_batches - 1)
        for index in range(paired_batches)
    )
    if len(set(indices)) != paired_batches:
        raise ValueError("interleaved paired batch indices are not unique")
    return indices


def combine_paired_rollout_batch(pair: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Combine action/zero branches without changing their causal ordering."""
    required = {
        "action_state", "zero_state", "action_omega", "zero_omega",
        "action_force", "zero_force", "mask", "paired_targets",
    }
    if set(pair) != required:
        raise ValueError(f"paired batch keys differ: {sorted(set(pair) ^ required)}")
    action_state, zero_state = pair["action_state"], pair["zero_state"]
    if action_state.shape != zero_state.shape or action_state.ndim != 5:
        raise ValueError("paired states must have identical [batch,time,channel,y,x] shape")
    batch, frames = action_state.shape[:2]
    if frames != 101:
        raise ValueError("paired rollout requires exactly 101 frames")
    if pair["action_omega"].shape != (batch, frames, 1) or pair[
        "zero_omega"
    ].shape != (batch, frames, 1):
        raise ValueError("paired omega shape differs")
    if pair["action_force"].shape != (batch, frames, 4) or pair[
        "zero_force"
    ].shape != (batch, frames, 4):
        raise ValueError("paired force shape differs")
    if pair["mask"].shape != (batch, 1, *action_state.shape[-2:]):
        raise ValueError("paired mask shape differs")
    if pair["paired_targets"].shape != (batch, 3, 3):
        raise ValueError("paired target shape differs")
    if not all(torch.isfinite(value).all() for value in pair.values()):
        raise ValueError("paired batch contains non-finite values")
    return {
        "state": torch.cat((action_state[:, 0], zero_state[:, 0]), dim=0),
        "target_state": torch.cat((action_state[:, 1:], zero_state[:, 1:]), dim=0),
        "omega": torch.cat((pair["action_omega"], pair["zero_omega"]), dim=0),
        "mask": torch.cat((pair["mask"], pair["mask"]), dim=0),
        "action_force": pair["action_force"][:, 1:],
        "zero_force": pair["zero_force"][:, 1:],
        "paired_targets": pair["paired_targets"],
    }
