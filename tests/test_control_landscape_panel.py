"""Pre-solve tests for the frozen matched-start action panel."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cfd" / "tandem_cylinders"))
from make_control_landscape_panel import PANEL, action_metrics, action_points, panel_audit  # noqa: E402


class ControlLandscapePanelTests(unittest.TestCase):
    def test_all_actions_start_from_same_state_and_stay_in_training_support(self) -> None:
        for spec in PANEL:
            points = action_points(spec)
            self.assertEqual(len(points), 361)
            self.assertEqual(points[0], (80.0, 0.0))
            self.assertEqual(points[-1][0], 116.0)
            metrics = action_metrics(points)
            self.assertLessEqual(metrics["max_abs_omega"], 1.0 + 1e-12)
            self.assertLessEqual(metrics["max_abs_domega_dt"], 1.6 + 1e-12)

    def test_constant_sign_symmetry_and_zero_reference(self) -> None:
        trajectories = {spec.kind: action_points(spec) for spec in PANEL}
        for (_, zero), (_, positive), (_, negative) in zip(
            trajectories["zero"],
            trajectories["positive_constant"],
            trajectories["negative_constant"],
        ):
            self.assertEqual(zero, 0.0)
            self.assertAlmostEqual(positive, -negative)

    def test_sine_is_nontrivial_and_predeclared_validation_only(self) -> None:
        sine = action_points(next(spec for spec in PANEL if spec.kind == "shedding_period_sine"))
        self.assertTrue(any(abs(value) > 0.9 for _, value in sine))
        self.assertTrue(all(math.isfinite(value) for _, value in sine))
        audit = panel_audit()
        self.assertEqual(audit["status"], "PREDECLARED_ACTIONS_NO_CFD_RESULTS")
        self.assertEqual(audit["split"], "validation_only_not_training_or_frozen_test")
        self.assertEqual(len(audit["cases"]), 4)


if __name__ == "__main__":
    unittest.main()
