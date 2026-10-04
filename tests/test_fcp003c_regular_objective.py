from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "paired_stats_trainer", REPO / "scripts/train_tandem_fno_paired_stats.py"
)
assert SPEC and SPEC.loader
TRAINER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAINER)


class ScaleModel(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(float(value)))


def fake_rollout(model, _state, _mask, _omega, target_state, _teacher_forcing):
    target_force = fake_rollout.target_force
    return target_state * model.scale, target_force * model.scale


def legacy_reference(
    network,
    state,
    target_state,
    omega,
    target_force,
    mask,
    channel_weights,
    force_loss_weight,
    rollout_discount,
    teacher_forcing_ratio,
):
    predicted_state, predicted_force = fake_rollout(
        network, state, mask, omega, target_state, teacher_forcing_ratio
    )
    steps = predicted_state.shape[1]
    weights = torch.pow(
        torch.as_tensor(rollout_discount, device=state.device),
        torch.arange(steps, device=state.device, dtype=state.dtype),
    )
    weights = weights / weights.sum()
    field_by_step = ((predicted_state - target_state).square() * mask[:, None]).sum(
        dim=(2, 3, 4)
    ) / (mask.sum(dim=(1, 2, 3)).clamp_min(1)[:, None] * 3)
    force_by_step = (
        (predicted_force - target_force).square() * channel_weights[None, None]
    ).sum(dim=2)
    field_loss = (field_by_step * weights[None]).sum(dim=1).mean()
    force_loss = (force_by_step * weights[None]).sum(dim=1).mean()
    return field_loss + force_loss_weight * force_loss, field_loss, force_loss


class RegularObjectiveExtractionTest(unittest.TestCase):
    def test_extracted_regular_objective_matches_legacy_loss_and_gradient(self):
        generator = torch.Generator().manual_seed(29)
        state = torch.randn((2, 3, 2, 3), generator=generator)
        target_state = torch.randn((2, 4, 3, 2, 3), generator=generator)
        omega = torch.randn((2, 5, 1), generator=generator)
        target_force = torch.randn((2, 4, 4), generator=generator)
        fake_rollout.target_force = target_force
        mask = torch.ones((2, 1, 2, 3))
        channel_weights = torch.tensor([1 / 7, 1 / 7, 4 / 7, 1 / 7])
        arguments = (
            state,
            target_state,
            omega,
            target_force,
            mask,
            channel_weights,
        )
        keywords = {
            "force_loss_weight": 0.2,
            "rollout_discount": 1.0,
            "teacher_forcing_ratio": 0.0,
        }

        legacy_model = ScaleModel(0.73)
        extracted_model = ScaleModel(0.73)
        legacy = legacy_reference(legacy_model, *arguments, **keywords)
        legacy[0].backward()
        with mock.patch.object(TRAINER, "rollout", side_effect=fake_rollout):
            extracted = TRAINER.regular_rollout_objective(
                extracted_model, *arguments, **keywords
            )
        extracted[0].backward()

        for actual, expected in zip(extracted, legacy):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            extracted_model.scale.grad, legacy_model.scale.grad, rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
