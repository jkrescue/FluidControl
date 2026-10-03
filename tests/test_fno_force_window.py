import importlib.util
import math
from pathlib import Path
import unittest


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


if __name__ == "__main__":
    unittest.main()
