"""Differentiable train-only paired force-statistic objectives.

This project loss complements, but does not modify, the official PhysicsNeMo
FNO implementation.  It compares action-minus-zero force statistics from
matched-start CFD trajectories without assuming the sign of the response.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


REQUIRED_CHANNELS = ("front_cd", "rear_cd", "rear_cl")


def _channel_indices(channels: Sequence[str]) -> tuple[int, int, int]:
    missing = [name for name in REQUIRED_CHANNELS if name not in channels]
    if missing:
        raise ValueError(f"paired force statistics require channels: {missing}")
    return tuple(channels.index(name) for name in REQUIRED_CHANNELS)


def physical_force_statistics(
    normalized_force: torch.Tensor,
    force_mean: torch.Tensor,
    force_std: torch.Tensor,
    channels: Sequence[str],
    horizons: Sequence[int] = (20, 50, 100),
) -> torch.Tensor:
    """Return mean total-Cd, mean rear-Cl and centered rear-Cl RMS.

    ``normalized_force`` has shape ``[..., time, channel]``.  The returned
    tensor has shape ``[..., horizon, statistic]``.  Horizon ``H`` consumes
    exactly target frames 1..H, matching the causal rollout convention.
    """
    if normalized_force.ndim < 2:
        raise ValueError("normalized_force must include time and channel axes")
    if normalized_force.shape[-1] != len(channels):
        raise ValueError("force channel dimension differs from channel names")
    if force_mean.shape != force_std.shape or force_mean.numel() != len(channels):
        raise ValueError("force normalization shape differs from channels")
    if not horizons or any(int(h) <= 0 for h in horizons):
        raise ValueError("horizons must be positive")
    if max(int(h) for h in horizons) > normalized_force.shape[-2]:
        raise ValueError("force sequence is shorter than requested horizon")
    if torch.any(force_std <= 0) or not torch.isfinite(force_std).all():
        raise ValueError("force_std must be positive and finite")

    front_cd, rear_cd, rear_cl = _channel_indices(tuple(channels))
    physical = normalized_force * force_std + force_mean
    rows = []
    for horizon in horizons:
        window = physical[..., : int(horizon), :]
        total_cd = window[..., front_cd] + window[..., rear_cd]
        lift = window[..., rear_cl]
        centered = lift - lift.mean(dim=-1, keepdim=True)
        rows.append(
            torch.stack(
                (
                    total_cd.mean(dim=-1),
                    lift.mean(dim=-1),
                    torch.linalg.vector_norm(centered, dim=-1)
                    / torch.sqrt(
                        torch.as_tensor(
                            int(horizon), dtype=lift.dtype, device=lift.device
                        )
                    ),
                ),
                dim=-1,
            )
        )
    return torch.stack(rows, dim=-2)


def paired_statistic_loss(
    predicted_action: torch.Tensor,
    predicted_zero: torch.Tensor,
    target_action: torch.Tensor,
    target_zero: torch.Tensor,
    force_mean: torch.Tensor,
    force_std: torch.Tensor,
    channels: Sequence[str],
    horizons: Sequence[int] = (20, 50, 100),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return normalized MSE and predicted/target action-minus-zero stats.

    Statistic errors are scaled solely by train normalization: total-Cd uses
    the root-sum-square of the two Cd standard deviations, while both rear-Cl
    statistics use the rear-Cl standard deviation.  No validation result or
    response sign enters this objective.
    """
    shapes = {
        tuple(predicted_action.shape),
        tuple(predicted_zero.shape),
        tuple(target_action.shape),
        tuple(target_zero.shape),
    }
    if len(shapes) != 1:
        raise ValueError("all paired force sequences must have identical shape")
    stats = [
        physical_force_statistics(value, force_mean, force_std, channels, horizons)
        for value in (predicted_action, predicted_zero, target_action, target_zero)
    ]
    predicted_delta = stats[0] - stats[1]
    target_delta = stats[2] - stats[3]
    front_cd, rear_cd, rear_cl = _channel_indices(tuple(channels))
    scales = torch.stack(
        (
            torch.sqrt(force_std[front_cd].square() + force_std[rear_cd].square()),
            force_std[rear_cl],
            force_std[rear_cl],
        )
    )
    loss = ((predicted_delta - target_delta) / scales).square().mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite paired force-statistic loss")
    return loss, predicted_delta, target_delta
