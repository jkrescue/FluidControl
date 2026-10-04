import unittest

import torch

from fluid_control.paired_step_force import (
    paired_step_force_delta_loss,
    true_state_step_input,
)


class PairedStepForceTests(unittest.TestCase):
    def test_true_state_step_input_uses_recorded_state_and_causal_actions(self):
        initial = torch.full((1, 3, 2, 2), 10.0)
        targets = torch.stack(
            [torch.full((1, 3, 2, 2), value) for value in (20.0, 30.0, 40.0)],
            dim=1,
        )
        mask = torch.ones(1, 1, 2, 2)
        omega = torch.tensor([[[0.0], [0.1], [0.2], [0.3]]])

        first = true_state_step_input(initial, targets, mask, omega, 0)
        third = true_state_step_input(initial, targets, mask, omega, 2)

        self.assertTrue(torch.equal(first[:, :3], initial))
        self.assertTrue(torch.equal(third[:, :3], targets[:, 1]))
        self.assertTrue(torch.all(first[:, 4] == 0.0))
        self.assertTrue(torch.all(first[:, 5] == 0.1))
        self.assertTrue(torch.all(third[:, 4] == 0.2))
        self.assertTrue(torch.all(third[:, 5] == 0.3))

    def test_step_force_delta_loss_matches_hand_calculation_and_cancels_bias(self):
        weights = torch.tensor([1.0, 1.0, 4.0, 1.0]) / 7.0
        target_zero = torch.zeros(1, 2, 4)
        target_action = torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]]]
        )
        error = torch.tensor(
            [[[1.0, 0.0, -1.0, 2.0], [0.0, 2.0, 1.0, -1.0]]]
        )
        common_bias = torch.full_like(target_action, 9.0)
        predicted_zero = target_zero + common_bias
        predicted_action = target_action + common_bias + error

        actual = paired_step_force_delta_loss(
            predicted_action, predicted_zero, target_action, target_zero, weights
        )
        expected = (error.square() * weights).sum(dim=-1).mean()
        self.assertAlmostEqual(float(actual), float(expected))

        swapped = paired_step_force_delta_loss(
            predicted_zero, predicted_action, target_zero, target_action, weights
        )
        self.assertAlmostEqual(float(swapped), float(actual))

    def test_step_force_delta_loss_has_finite_nonzero_gradient(self):
        weights = torch.ones(4) / 4
        predicted_action = torch.ones(1, 3, 4, requires_grad=True)
        predicted_zero = torch.zeros(1, 3, 4, requires_grad=True)
        target = torch.zeros(1, 3, 4)
        loss = paired_step_force_delta_loss(
            predicted_action, predicted_zero, target, target, weights
        )
        loss.backward()
        self.assertTrue(torch.isfinite(predicted_action.grad).all())
        self.assertTrue(torch.isfinite(predicted_zero.grad).all())
        self.assertEqual(
            int(torch.count_nonzero(predicted_action.grad)), predicted_action.numel()
        )
        self.assertTrue(torch.allclose(predicted_action.grad, -predicted_zero.grad))

    def test_true_state_step_input_rejects_invalid_step(self):
        for bad_step in (-1, 3, True, 1.5):
            with self.subTest(bad_step=bad_step), self.assertRaises(ValueError):
                true_state_step_input(
                    torch.zeros(1, 3, 2, 2),
                    torch.zeros(1, 3, 3, 2, 2),
                    torch.ones(1, 1, 2, 2),
                    torch.zeros(1, 4, 1),
                    bad_step,
                )

    def test_true_state_step_input_rejects_wrong_channels_or_empty_axes(self):
        cases = (
            (
                torch.zeros(1, 2, 2, 2),
                torch.zeros(1, 3, 2, 2, 2),
                torch.ones(1, 1, 2, 2),
                torch.zeros(1, 4, 1),
            ),
            (
                torch.zeros(1, 3, 2, 2),
                torch.zeros(1, 3, 2, 2, 2),
                torch.ones(1, 1, 2, 2),
                torch.zeros(1, 4, 1),
            ),
            (
                torch.zeros(0, 3, 2, 2),
                torch.zeros(0, 3, 3, 2, 2),
                torch.ones(0, 1, 2, 2),
                torch.zeros(0, 4, 1),
            ),
            (
                torch.zeros(1, 3, 2, 2),
                torch.zeros(1, 0, 3, 2, 2),
                torch.ones(1, 1, 2, 2),
                torch.zeros(1, 1, 1),
            ),
        )
        for values in cases:
            with self.subTest(shapes=tuple(tuple(value.shape) for value in values)):
                with self.assertRaises(ValueError):
                    true_state_step_input(*values, 0)

    def test_step_force_delta_loss_rejects_invalid_weights_and_nonfinite(self):
        values = torch.zeros(1, 2, 4)
        with self.assertRaises(ValueError):
            paired_step_force_delta_loss(values, values, values, values, torch.ones(4))
        bad = values.clone()
        bad[0, 0, 0] = torch.nan
        with self.assertRaises(FloatingPointError):
            paired_step_force_delta_loss(
                bad, values, values, values, torch.ones(4) / 4
            )


if __name__ == "__main__":
    unittest.main()
