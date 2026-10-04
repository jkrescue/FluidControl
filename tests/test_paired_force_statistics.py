import unittest

import torch

from fluid_control.paired_force_statistics import paired_statistic_loss, physical_force_statistics

CHANNELS = ("front_cd", "front_cl", "rear_cd", "rear_cl")
MEAN = torch.tensor([1.0, 0.1, 2.0, -0.2])
STD = torch.tensor([2.0, 3.0, 4.0, 5.0])


class PairedForceStatisticsTest(unittest.TestCase):
    def test_statistics_use_causal_prefix_and_centered_rms(self):
        force = torch.zeros(1, 4, 4)
        force[0, :, 0] = torch.tensor([0.0, 1.0, 2.0, 3.0])
        force[0, :, 2] = torch.tensor([0.0, -1.0, -2.0, -3.0])
        force[0, :, 3] = torch.tensor([0.0, 1.0, 2.0, 3.0])
        result = physical_force_statistics(force, MEAN, STD, CHANNELS, (2, 4))
        physical = force * STD + MEAN
        for index, horizon in enumerate((2, 4)):
            total_cd = physical[0, :horizon, 0] + physical[0, :horizon, 2]
            rear_cl = physical[0, :horizon, 3]
            expected = torch.stack((total_cd.mean(), rear_cl.mean(), rear_cl.std(unbiased=False)))
            torch.testing.assert_close(result[0, index], expected)

    def test_exact_prediction_has_zero_loss_and_gradients(self):
        action = torch.randn(2, 100, 4, requires_grad=True)
        zero = torch.randn(2, 100, 4, requires_grad=True)
        loss, predicted, target = paired_statistic_loss(
            action, zero, action.detach(), zero.detach(), MEAN, STD, CHANNELS
        )
        self.assertAlmostEqual(loss.item(), 0.0, places=12)
        torch.testing.assert_close(predicted, target)
        loss.backward()
        self.assertIsNotNone(action.grad)
        self.assertIsNotNone(zero.grad)

    def test_loss_does_not_assume_effect_sign(self):
        zero = torch.zeros(1, 100, 4)
        positive, negative = zero.clone(), zero.clone()
        positive[..., 3], negative[..., 3] = 0.2, -0.2
        exact_positive = paired_statistic_loss(positive, zero, positive, zero, MEAN, STD, CHANNELS)[0]
        exact_negative = paired_statistic_loss(negative, zero, negative, zero, MEAN, STD, CHANNELS)[0]
        wrong_sign = paired_statistic_loss(negative, zero, positive, zero, MEAN, STD, CHANNELS)[0]
        self.assertAlmostEqual(exact_positive.item(), 0.0)
        self.assertAlmostEqual(exact_negative.item(), 0.0)
        self.assertGreater(wrong_sign.item(), 0.0)

    def test_invalid_contract_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "shorter"):
            physical_force_statistics(torch.zeros(1, 100, 4), MEAN, STD, CHANNELS, (101,))
        with self.assertRaisesRegex(ValueError, "require"):
            physical_force_statistics(
                torch.zeros(1, 100, 4), MEAN, STD,
                ("front_cd", "rear_cd", "front_cl", "x"),
            )

    def test_nonfinite_loss_rejected(self):
        action = torch.zeros(1, 100, 4)
        action[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            paired_statistic_loss(
                action, torch.zeros_like(action), torch.zeros_like(action),
                torch.zeros_like(action), MEAN, STD, CHANNELS,
            )


if __name__ == "__main__":
    unittest.main()
