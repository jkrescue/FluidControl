"""Pre-solve checks for the long matched-start rotation panel."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cfd" / "tandem_cylinders"))
from make_control_landscape_long_panel import LONG_PANEL, action_points, audit, validate_points  # noqa: E402


class LongPanelTests(unittest.TestCase):
    def test_all_cases_share_restart_and_bound_actions(self) -> None:
        for spec in LONG_PANEL:
            points = action_points(spec)
            self.assertEqual(len(points), 801)
            self.assertEqual(points[0], (80.0, 0.0))
            self.assertEqual(points[-1][0], 160.0)
            metrics = validate_points(points)
            self.assertLessEqual(metrics["max_abs_omega"], 1.0 + 1e-9)
            self.assertLessEqual(metrics["max_abs_domega_dt"], 0.5 + 1e-9)

    def test_zero_and_opposite_rotation_are_predeclared_validation_only(self) -> None:
        self.assertEqual(sorted(spec.omega_final for spec in LONG_PANEL), [-1.0, 0.0, 1.0])
        plan = audit()
        self.assertEqual(plan["analysis_window"], [120.0, 160.0])
        self.assertEqual(plan["split"], "validation_only_not_training_or_frozen_test")
        self.assertEqual(len(plan["cases"]), 3)


if __name__ == "__main__":
    unittest.main()
