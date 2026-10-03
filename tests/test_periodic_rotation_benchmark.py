"""Pre-solve bounds and symmetry for periodic open-loop benchmarks."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cfd" / "tandem_cylinders"))
from make_periodic_rotation_benchmark import PANEL, action_points, audit, metrics  # noqa: E402


class PeriodicRotationTests(unittest.TestCase):
    def test_fixed_periods_are_bounded_and_zero_mean(self) -> None:
        self.assertEqual(set(PANEL.values()), {10.0, 20.0})
        for period in PANEL.values():
            points = action_points(period)
            self.assertEqual(len(points), 801)
            self.assertEqual(points[0], (80.0, 0.0))
            self.assertLess(abs(sum(omega for _, omega in points)), 1e-8)
            self.assertLessEqual(metrics(points)["max_abs_domega_dt"], 2.0)
            half_steps = round(period / 2 / 0.1)
            for index in range(1, 801 - half_steps):
                self.assertAlmostEqual(points[index][1], -points[index + half_steps][1], places=10)

    def test_action_pool_and_acceptance_predeclared(self) -> None:
        plan = audit()
        self.assertEqual(plan["status"], "PREDECLARED_NO_CFD_RESULTS")
        self.assertEqual(plan["analysis_window"], [120.0, 160.0])
        self.assertEqual(plan["acceptance_from_RESEARCH_OBJECTIVE_md"]["total_cd_reduction_min"], 0.02)
        with self.assertRaises(ValueError):
            action_points(6.154)


if __name__ == "__main__":
    unittest.main()
