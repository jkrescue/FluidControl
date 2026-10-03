"""Physical flow-field diagnostics: masking, pooling and reference handling."""
import unittest

import numpy as np
import torch

from evaluate_tandem_fno import field_error_sums, relative_field_metrics


class RelativeFieldMetricTests(unittest.TestCase):
    def test_solid_cells_do_not_contribute(self):
        target = torch.tensor([[[[2., 999.]], [[1., 999.]], [[4., 999.]]]])
        error = torch.tensor([[[[1., 999.]], [[2., 999.]], [[2., 999.]]]])
        mask = torch.tensor([[[[1., 0.]]]])
        sums = field_error_sums(error, target, mask)
        torch.testing.assert_close(sums, torch.tensor([[1., 4., 4.], [4., 1., 16.]], dtype=torch.float64))
        metrics = relative_field_metrics(sums.numpy())
        self.assertEqual(metrics["field_relative_l2_u_v_p"], [.5, 2., .5])
        self.assertAlmostEqual(metrics["velocity_relative_l2"], 1.)

    def test_pooling_uses_energy_sums(self):
        first = np.array([[1., 0., 0.], [1., 1., 1.]])
        second = np.array([[1., 0., 0.], [100., 1., 1.]])
        pooled = relative_field_metrics(first + second)["field_relative_l2_u_v_p"][0]
        self.assertAlmostEqual(pooled, np.sqrt(2 / 101))
        self.assertNotAlmostEqual(pooled, .55)

    def test_zero_reference_is_not_claimed_perfect(self):
        self.assertEqual(relative_field_metrics(np.zeros((2, 3)))["field_relative_l2_u_v_p"], [None] * 3)

    def test_exact_prediction_has_zero_error(self):
        metrics = relative_field_metrics([[0., 0., 0.], [1., 2., 3.]])
        self.assertEqual(metrics["velocity_relative_l2"], 0.)


if __name__ == "__main__":
    unittest.main()
