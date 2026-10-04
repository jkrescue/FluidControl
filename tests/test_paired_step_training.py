import copy
import unittest
from unittest import mock

import torch

from fluid_control.paired_step_force import (
    paired_step_force_absolute_loss,
    paired_step_force_delta_loss,
)
from fluid_control.paired_step_training import (
    paired_objective_kind,
    prepare_true_state_training_model,
    predict_true_state_force_chunk,
    summarize_true_state_step_updates,
    true_state_paired_optimizer_step,
)
from fluid_control.paired_training import (
    paired_batch_indices,
    validate_paired_identity_passes,
)


class CountingSGD(torch.optim.SGD):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.step_count = 0

    def step(self, closure=None):
        self.step_count += 1
        return super().step(closure)


class TinyForce(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(6, 4)

    def forward(self, inputs):
        features = inputs.mean(dim=(-2, -1)) if inputs.ndim == 4 else inputs
        return self.linear(features)


def predict_force(model, inputs, _mask):
    return model(inputs)


def make_pair(steps=5):
    generator = torch.Generator().manual_seed(17)
    shape = (1, steps + 1, 3, 2, 2)
    return {
        "action_state": torch.randn(shape, generator=generator),
        "zero_state": torch.randn(shape, generator=generator),
        "action_omega": torch.randn((1, steps + 1, 1), generator=generator),
        "zero_omega": torch.zeros((1, steps + 1, 1)),
        "action_force": torch.randn((1, steps + 1, 4), generator=generator),
        "zero_force": torch.randn((1, steps + 1, 4), generator=generator),
        "mask": torch.ones((1, 1, 2, 2)),
    }


class TestTrueStatePairedOptimizerStep(unittest.TestCase):
    def test_absolute_sees_common_bias_that_delta_cancels(self):
        target_action = torch.zeros(1, 2, 4)
        target_zero = torch.ones(1, 2, 4)
        common_bias = torch.tensor([1.0, 2.0, 3.0, 4.0])[None, None]
        predicted_action = target_action + common_bias
        predicted_zero = target_zero + common_bias
        weights = torch.tensor([1 / 7, 1 / 7, 4 / 7, 1 / 7])
        self.assertEqual(
            float(
                paired_step_force_delta_loss(
                    predicted_action,
                    predicted_zero,
                    target_action,
                    target_zero,
                    weights,
                )
            ),
            0.0,
        )
        expected = float((common_bias.square() * weights).sum(-1).mean())
        self.assertAlmostEqual(
            float(
                paired_step_force_absolute_loss(
                    predicted_action,
                    predicted_zero,
                    target_action,
                    target_zero,
                    weights,
                )
            ),
            expected,
        )

    def test_old_objective_remains_default_and_schedule_contract_is_unchanged(self):
        self.assertEqual(paired_objective_kind(), "paired_statistics")
        self.assertEqual(
            paired_objective_kind("true_state_step_force"),
            "true_state_step_force",
        )
        with self.assertRaises(ValueError):
            paired_objective_kind("unknown")
        indices = paired_batch_indices(1368, 16, "interleaved")
        self.assertEqual(indices, tuple(index * 1367 // 15 for index in range(16)))
        identities = {
            f"b{phase:02d}:{profile}"
            for phase in (0, 2, 4, 6)
            for profile in ("prbs", "multisine")
        }
        first = sorted(identities)
        second = list(reversed(first))
        validate_paired_identity_passes([first, second], identities, 2)

    def test_explicitly_restores_training_mode_before_graph_construction(self):
        model = TinyForce()
        model.eval()
        self.assertFalse(model.training)
        prepare_true_state_training_model(model)
        self.assertTrue(model.training)

    def test_chunked_gradients_and_one_step_equal_monolithic(self):
        torch.manual_seed(3)
        chunked = TinyForce()
        monolithic = copy.deepcopy(chunked)
        pair = make_pair(steps=5)
        weights = torch.tensor([1 / 7, 1 / 7, 4 / 7, 1 / 7])
        base_inputs = torch.randn(3, 6)
        base_target = torch.randn(3, 4)
        paired_weight = 2.5
        clip_norm = 0.35

        chunked_optimizer = CountingSGD(chunked.parameters(), lr=0.03)
        original_clip = torch.nn.utils.clip_grad_norm_
        with mock.patch("torch.nn.utils.clip_grad_norm_", wraps=original_clip) as clip:
            result = true_state_paired_optimizer_step(
                model=chunked,
                optimizer=chunked_optimizer,
                base_loss=(chunked(base_inputs) - base_target).square().mean(),
                pair=pair,
                channel_weights=weights,
                paired_weight=paired_weight,
                predict_force=predict_force,
                total_steps=5,
                chunk_size=2,
                gradient_clip_norm=clip_norm,
            )
        self.assertEqual(clip.call_count, 1)
        self.assertEqual(chunked_optimizer.step_count, 1)
        self.assertEqual(result["optimizer_steps"], 1)
        self.assertEqual(result["chunk_count"], 3)

        monolithic_optimizer = CountingSGD(monolithic.parameters(), lr=0.03)
        monolithic_optimizer.zero_grad(set_to_none=True)
        base_loss = (monolithic(base_inputs) - base_target).square().mean()
        base_loss.backward()
        predicted_action, predicted_zero = predict_true_state_force_chunk(
            monolithic, pair, 0, 5, predict_force
        )
        pair_loss = paired_step_force_delta_loss(
            predicted_action,
            predicted_zero,
            pair["action_force"][:, 1:],
            pair["zero_force"][:, 1:],
            weights,
        )
        (paired_weight * pair_loss).backward()
        torch.nn.utils.clip_grad_norm_(monolithic.parameters(), clip_norm)
        monolithic_optimizer.step()
        self.assertEqual(monolithic_optimizer.step_count, 1)

        self.assertAlmostEqual(
            result["paired_step_force_loss"], float(pair_loss.detach()), places=6
        )
        for actual, expected in zip(chunked.parameters(), monolithic.parameters()):
            torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)

    def test_absolute_chunked_gradients_equal_monolithic(self):
        torch.manual_seed(31)
        chunked = TinyForce()
        monolithic = copy.deepcopy(chunked)
        pair = make_pair(steps=5)
        weights = torch.tensor([1 / 7, 1 / 7, 4 / 7, 1 / 7])
        base_inputs = torch.randn(3, 6)
        base_target = torch.randn(3, 4)
        paired_weight, clip_norm = 2.5, 0.35
        chunked_optimizer = CountingSGD(chunked.parameters(), lr=0.03)
        result = true_state_paired_optimizer_step(
            model=chunked,
            optimizer=chunked_optimizer,
            base_loss=(chunked(base_inputs) - base_target).square().mean(),
            pair=pair,
            channel_weights=weights,
            paired_weight=paired_weight,
            predict_force=predict_force,
            total_steps=5,
            chunk_size=2,
            gradient_clip_norm=clip_norm,
            force_objective="absolute",
        )
        monolithic_optimizer = CountingSGD(monolithic.parameters(), lr=0.03)
        monolithic_optimizer.zero_grad(set_to_none=True)
        base_loss = (monolithic(base_inputs) - base_target).square().mean()
        base_loss.backward()
        predicted_action, predicted_zero = predict_true_state_force_chunk(
            monolithic, pair, 0, 5, predict_force
        )
        pair_loss = paired_step_force_absolute_loss(
            predicted_action,
            predicted_zero,
            pair["action_force"][:, 1:],
            pair["zero_force"][:, 1:],
            weights,
        )
        (paired_weight * pair_loss).backward()
        torch.nn.utils.clip_grad_norm_(monolithic.parameters(), clip_norm)
        monolithic_optimizer.step()
        self.assertEqual(result["force_objective"], "absolute")
        self.assertAlmostEqual(
            result["paired_step_force_loss"], float(pair_loss.detach()), places=6
        )
        for actual, expected in zip(chunked.parameters(), monolithic.parameters()):
            torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)

    def test_default_delta_matches_explicit_delta(self):
        torch.manual_seed(37)
        default_model = TinyForce()
        explicit_model = copy.deepcopy(default_model)
        pair = make_pair(steps=4)
        weights = torch.tensor([1 / 7, 1 / 7, 4 / 7, 1 / 7])
        inputs = torch.randn(2, 6)
        target = torch.randn(2, 4)
        results = []
        for model, objective in ((default_model, None), (explicit_model, "delta")):
            kwargs = {} if objective is None else {"force_objective": objective}
            optimizer = CountingSGD(model.parameters(), lr=0.02)
            results.append(
                true_state_paired_optimizer_step(
                    model=model,
                    optimizer=optimizer,
                    base_loss=(model(inputs) - target).square().mean(),
                    pair=pair,
                    channel_weights=weights,
                    paired_weight=3.0,
                    predict_force=predict_force,
                    total_steps=4,
                    chunk_size=2,
                    gradient_clip_norm=1.0,
                    **kwargs,
                )
            )
        self.assertEqual(results[0], results[1])
        for actual, expected in zip(
            default_model.parameters(), explicit_model.parameters()
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_channel_mse_and_endpoint_alignment(self):
        torch.manual_seed(5)
        model = TinyForce()
        pair = make_pair(steps=4)
        weights = torch.tensor([0.1, 0.2, 0.3, 0.4])
        expected_action, expected_zero = predict_true_state_force_chunk(
            model, pair, 0, 4, predict_force
        )
        expected_error = (expected_action - expected_zero) - (
            pair["action_force"][:, 1:] - pair["zero_force"][:, 1:]
        )
        expected_channel_mse = expected_error.square().mean(dim=(0, 1))

        optimizer = CountingSGD(model.parameters(), lr=0.0)
        result = true_state_paired_optimizer_step(
            model=model,
            optimizer=optimizer,
            base_loss=model(torch.zeros(1, 6)).square().mean(),
            pair=pair,
            channel_weights=weights,
            paired_weight=1.0,
            predict_force=predict_force,
            total_steps=4,
            chunk_size=3,
            gradient_clip_norm=100.0,
        )
        torch.testing.assert_close(
            torch.tensor(result["paired_step_force_per_channel_mse"]),
            expected_channel_mse,
        )
        torch.testing.assert_close(
            torch.tensor(result["paired_step_force_per_channel_weighted_contribution"]),
            expected_channel_mse * weights,
        )

    def test_nonfinite_pair_refuses_optimizer_step_and_clears_gradients(self):
        torch.manual_seed(7)
        model = TinyForce()
        before = [parameter.detach().clone() for parameter in model.parameters()]
        pair = make_pair(steps=4)
        pair["action_force"][0, 2, 0] = float("inf")
        optimizer = CountingSGD(model.parameters(), lr=0.1)
        with self.assertRaises(FloatingPointError):
            true_state_paired_optimizer_step(
                model=model,
                optimizer=optimizer,
                base_loss=model(torch.zeros(1, 6)).square().mean(),
                pair=pair,
                channel_weights=torch.full((4,), 0.25),
                paired_weight=10.0,
                predict_force=predict_force,
                total_steps=4,
                chunk_size=2,
                gradient_clip_norm=1.0,
            )
        self.assertEqual(optimizer.step_count, 0)
        for initial, parameter in zip(before, model.parameters()):
            torch.testing.assert_close(initial, parameter)
            self.assertIsNone(parameter.grad)

    def test_rejects_shape_and_numeric_contract_changes(self):
        model = TinyForce()
        pair = make_pair(steps=4)
        optimizer = CountingSGD(model.parameters(), lr=0.1)
        arguments = {
            "model": model,
            "optimizer": optimizer,
            "base_loss": model(torch.zeros(1, 6)).square().mean(),
            "pair": pair,
            "channel_weights": torch.full((4,), 0.25),
            "paired_weight": 10.0,
            "predict_force": predict_force,
            "total_steps": 4,
            "chunk_size": 2,
            "gradient_clip_norm": 1.0,
        }
        for key, value in (
            ("chunk_size", 0),
            ("paired_weight", float("nan")),
            ("gradient_clip_norm", 0.0),
        ):
            changed = dict(arguments)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                true_state_paired_optimizer_step(**changed)
        changed = dict(arguments)
        changed["pair"] = dict(pair)
        changed["pair"].pop("zero_force")
        changed["pair"]["extra_metadata_tensor"] = torch.ones(1)
        with self.assertRaises(ValueError):
            true_state_paired_optimizer_step(**changed)
        for key in ("total_steps", "chunk_size"):
            changed = dict(arguments)
            changed[key] = 2.5
            with self.subTest(key=key), self.assertRaises(ValueError):
                true_state_paired_optimizer_step(**changed)

    def test_prevalidation_failure_clears_residual_gradients(self):
        model = TinyForce()
        optimizer = CountingSGD(model.parameters(), lr=0.1)
        model(torch.ones(1, 6)).sum().backward()
        self.assertTrue(
            any(parameter.grad is not None for parameter in model.parameters())
        )
        pair = make_pair(steps=4)
        pair.pop("zero_force")
        with self.assertRaises(ValueError):
            true_state_paired_optimizer_step(
                model=model,
                optimizer=optimizer,
                base_loss=model(torch.zeros(1, 6)).square().mean(),
                pair=pair,
                channel_weights=torch.full((4,), 0.25),
                paired_weight=10.0,
                predict_force=predict_force,
                total_steps=4,
                chunk_size=2,
                gradient_clip_norm=1.0,
            )
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertEqual(optimizer.step_count, 0)

    def test_epoch_summary_requires_sixteen_finite_single_step_records(self):
        records = [
            {
                "paired_step_force_loss": float(index + 1),
                "paired_step_force_weighted_loss": float(10 * (index + 1)),
                "paired_step_force_per_channel_mse": [1.0, 2.0, 3.0, 4.0],
                "paired_step_force_per_channel_weighted_contribution": [
                    0.1,
                    0.2,
                    1.2,
                    0.4,
                ],
                "preclip_gradient_norm": 0.5,
                "optimizer_steps": 1,
            }
            for index in range(16)
        ]
        summary = summarize_true_state_step_updates(
            records, expected_updates=16, force_channels=4
        )
        self.assertEqual(summary["train_true_state_paired_optimizer_steps"], 16)
        self.assertEqual(
            summary["train_true_state_paired_step_force_per_channel_mse"],
            [1.0, 2.0, 3.0, 4.0],
        )
        self.assertEqual(
            summary[
                "train_true_state_paired_step_force_per_channel_weighted_contribution"
            ],
            [0.1, 0.2, 1.2, 0.4],
        )
        self.assertEqual(
            len(summary["train_true_state_paired_step_preclip_gradient_norms"]), 16
        )
        with self.assertRaises(RuntimeError):
            summarize_true_state_step_updates(
                records[:-1], expected_updates=16, force_channels=4
            )
        records[0]["optimizer_steps"] = 0
        with self.assertRaises(RuntimeError):
            summarize_true_state_step_updates(
                records, expected_updates=16, force_channels=4
            )


if __name__ == "__main__":
    unittest.main()
