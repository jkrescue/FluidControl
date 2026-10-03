"""Numeric unit fixtures for physical total-drag validation diagnostics."""
import unittest

import torch

from train_tandem_fno_rollout import total_drag_error_sums


class TotalDragMetricsTests(unittest.TestCase):
    def setUp(self):
        self.channels = ["front_cd", "front_cl", "rear_cd", "rear_cl"]
        self.mean = torch.tensor([1.0, 0.0, 2.0, 0.0])[None, None]
        self.std = torch.tensor([2.0, 1.0, 3.0, 1.0])[None, None]

    def test_denormalizes_before_summing_front_and_rear(self):
        target = torch.zeros(1, 2, 4)
        predicted = torch.tensor([[[0.5, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 2.0 / 3.0, 0.0]]])
        sums = total_drag_error_sums(predicted, target, self.mean, self.std, self.channels)
        torch.testing.assert_close(sums, torch.tensor([4., 9., 5., 18.], dtype=torch.float64))

    def test_opposite_force_errors_cancel_in_total_drag(self):
        target = torch.zeros(1, 1, 4)
        predicted = torch.tensor([[[0.5, 0.0, -1.0 / 3.0, 0.0]]])
        sums = total_drag_error_sums(predicted, target, self.mean, self.std, self.channels)
        self.assertLess(float(sums[0]), 1e-12)

    def test_pooled_ratio_is_not_mean_of_sample_ratios(self):
        target = torch.zeros(2, 1, 4)
        target[1, 0, 0] = 1.5
        predicted = target.clone()
        predicted[0, 0, 0] += 0.5
        sums = total_drag_error_sums(predicted, target, self.mean, self.std, self.channels)
        self.assertAlmostEqual(float(torch.sqrt(sums[0] / sums[1])), (1.0 / 45.0) ** 0.5)
        self.assertNotAlmostEqual(float(torch.sqrt(sums[0] / sums[1])), 1.0 / 6.0)

    def test_legacy_rear_only_has_no_total_drag_metric(self):
        self.assertIsNone(total_drag_error_sums(
            torch.zeros(1, 1, 2), torch.zeros(1, 1, 2),
            torch.zeros(1, 1, 2), torch.ones(1, 1, 2), ["rear_cd", "rear_cl"],
        ))


if __name__ == "__main__":
    unittest.main()
