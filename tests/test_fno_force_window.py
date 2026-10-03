import importlib.util
import math
from pathlib import Path
import unittest

import numpy as np


path = Path(__file__).resolve().parents[1] / "scripts/diagnose_fno_force_window.py"
spec = importlib.util.spec_from_file_location("force_window", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class WindowStatisticsTests(unittest.TestCase):
    def test_constant_force_and_exact_window(self):
        times = [i / 10 for i in range(101)]
        result = module.window_statistics(times, [[1, 0, 2, 0.2]] * 101, 10)
        self.assertEqual(result["sample_count"], 62)
        self.assertAlmostEqual(result["first_sample_time"], 3.9)
        self.assertAlmostEqual(result["mean_total_cd"], 3)
        self.assertAlmostEqual(result["rear_cl_fluctuation_rms"], 0)

    def test_future_values_excluded(self):
        times = [i / 10 for i in range(111)]
        forces = [[1, 0, 2, 0] if t <= 10 else [100, 100, 100, 100] for t in times]
        self.assertAlmostEqual(
            module.window_statistics(times, forces, 10)["mean_total_cd"], 3
        )

    def test_fluctuation_excludes_mean(self):
        times = [i / 10 for i in range(101)]
        forces = [[1, 0, 2, 4 + (-1) ** i] for i in range(101)]
        result = module.window_statistics(times, forces, 10)
        self.assertAlmostEqual(result["rear_cl_mean"], 4)
        self.assertAlmostEqual(result["rear_cl_fluctuation_rms"], 1)

    def test_incomplete_and_nonuniform_fail(self):
        with self.assertRaises(ValueError):
            module.window_statistics([0, 1], [[1, 0, 2, 0]] * 2, 1)
        with self.assertRaises(ValueError):
            module.window_statistics([0, 1, 3, 10], [[1, 0, 2, 0]] * 4, 10)

    def test_nonfinite_selected_force_fails(self):
        times = [i / 10 for i in range(101)]
        forces = [[1, 0, 2, 0] for _ in times]
        forces[-1][0] = math.nan
        with self.assertRaises(ValueError):
            module.window_statistics(times, forces, 10)

    def test_float32_time_quantization_is_not_resampled(self):
        times = [130 + i / 10 + (6e-6 if i % 2 else -9e-6) for i in range(101)]
        result = module.window_statistics(times, [[1, 0, 2, 0]] * 101, times[-1])
        self.assertEqual(result["last_sample_time"], times[-1])
        self.assertEqual(result["sample_count"], 62)


class FieldStepStatisticsTests(unittest.TestCase):
    def test_pressure_offset_separates_raw_and_demeaned_error(self):
        truth = np.zeros((3, 2, 3), dtype=float)
        truth[0] = 2.0
        truth[1] = -1.0
        truth[2] = np.asarray([[1.0, -1.0, 2.0], [-2.0, 0.5, -0.5]])
        predicted = truth.copy()
        predicted[2] += 4.0
        predicted_before = predicted.copy()
        result = module.field_step_statistics(predicted, truth, np.ones((1, 2, 3)))
        np.testing.assert_array_equal(predicted, predicted_before)
        self.assertGreater(result["pressure_raw_relative_l2"], 0.0)
        self.assertAlmostEqual(
            result["pressure_demeaned_relative_l2_diagnostic_only"], 0.0
        )
        self.assertAlmostEqual(result["velocity_uv_relative_l2"], 0.0)
        self.assertAlmostEqual(result["predicted_pressure_spatial_mean"], 4.0)
        self.assertAlmostEqual(result["truth_pressure_spatial_mean"], 0.0)

    def test_mask_excludes_solid_cells(self):
        truth = np.zeros((3, 2, 2), dtype=float)
        predicted = truth.copy()
        predicted[:, 1, 1] = 1e9
        mask = np.asarray([[1, 1], [1, 0]])
        result = module.field_step_statistics(predicted, truth, mask)
        self.assertIsNone(result["pressure_raw_relative_l2"])
        self.assertIsNone(result["velocity_uv_relative_l2"])

    def test_zero_reference_pressure_and_velocity_are_undefined(self):
        truth = np.zeros((3, 2, 2), dtype=float)
        predicted = np.ones((3, 2, 2), dtype=float)
        result = module.field_step_statistics(predicted, truth, np.ones((2, 2)))
        self.assertIsNone(result["pressure_raw_relative_l2"])
        self.assertIsNone(result["pressure_demeaned_relative_l2_diagnostic_only"])
        self.assertIsNone(result["velocity_uv_relative_l2"])

    def test_field_diagnostic_rejects_nonfinite_or_shape_mismatch(self):
        field = np.zeros((3, 2, 2), dtype=float)
        with self.assertRaises(ValueError):
            module.field_step_statistics(field[:, :, :1], field, np.ones((2, 2)))
        bad = field.copy()
        bad[2, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            module.field_step_statistics(bad, field, np.ones((2, 2)))

    def test_summary_records_range_mean_and_final(self):
        rows = [
            {
                "predicted_pressure_spatial_mean": value,
                "truth_pressure_spatial_mean": 0.0,
                "pressure_raw_relative_l2": value + 1.0,
                "pressure_demeaned_relative_l2_diagnostic_only": value + 2.0,
                "velocity_uv_relative_l2": value + 3.0,
            }
            for value in (0.0, 1.0, 2.0)
        ]
        result = module.summarize_field_diagnostics(rows)
        self.assertEqual(
            result["predicted_pressure_spatial_mean"],
            {
                "defined_steps": 3,
                "total_steps": 3,
                "minimum": 0.0,
                "maximum": 2.0,
                "mean": 1.0,
                "final": 2.0,
            },
        )
        self.assertEqual(result["velocity_uv_relative_l2"]["final"], 5.0)

    def test_all_null_relative_summary_is_explicit(self):
        rows = [
            {
                "predicted_pressure_spatial_mean": 1.0,
                "truth_pressure_spatial_mean": 0.0,
                "pressure_raw_relative_l2": None,
                "pressure_demeaned_relative_l2_diagnostic_only": None,
                "velocity_uv_relative_l2": None,
            }
            for _ in range(3)
        ]
        result = module.summarize_field_diagnostics(rows)
        expected = {
            "defined_steps": 0,
            "total_steps": 3,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "final": None,
        }
        self.assertEqual(result["pressure_raw_relative_l2"], expected)
        self.assertEqual(result["velocity_uv_relative_l2"], expected)


class FixedSpatialIntegrationTests(unittest.TestCase):
    def test_only_predeclared_horizons_invoke_helper(self):
        calls = []

        def diagnostic(truth, prediction, x, y, mask):
            calls.append((truth, prediction, x, y, mask))
            return {"status": "OBSERVATIONAL_FINITE_ROI_SPATIAL_DIAGNOSTIC"}

        arrays = [np.asarray([value]) for value in range(5)]
        for step in (1, 2, 10, 49, 50, 99, 100):
            result = module.fixed_spatial_snapshot_diagnostic(
                step, *arrays, diagnostic=diagnostic
            )
            if step in (1, 10, 50, 100):
                self.assertEqual(result["step"], step)
                self.assertEqual(
                    result["status"],
                    "OBSERVATIONAL_FINITE_ROI_SPATIAL_DIAGNOSTIC",
                )
            else:
                self.assertIsNone(result)
        self.assertEqual(len(calls), 4)

    def test_integration_preserves_input_arrays(self):
        arrays = [np.asarray([value], dtype=float) for value in range(5)]
        before = [array.copy() for array in arrays]

        def diagnostic(*_args):
            return {"scope": "read_only"}

        module.fixed_spatial_snapshot_diagnostic(
            100, *arrays, diagnostic=diagnostic
        )
        for actual, expected in zip(arrays, before):
            np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
