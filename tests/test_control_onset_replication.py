"""Pre-solve checks for the matched step-vs-ramp CFD experiment."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cfd" / "tandem_cylinders"))
from make_control_onset_replication import PANEL, action_points, audit  # noqa: E402


class OnsetReplicationTests(unittest.TestCase):
    def test_signed_step_schedules_are_identical_except_sign(self) -> None:
        positive = action_points(1.0)
        negative = action_points(-1.0)
        self.assertEqual(len(positive), 801)
        self.assertEqual(positive[0], (80.0, 1.0))
        self.assertEqual(positive[-1], (160.0, 1.0))
        self.assertEqual([time for time, _ in positive], [time for time, _ in negative])
        self.assertEqual([omega for _, omega in positive], [-omega for _, omega in negative])
        self.assertEqual(set(PANEL.values()), {-1.0, 1.0})

    def test_plan_is_validation_only_and_fixed_before_solving(self) -> None:
        plan = audit()
        self.assertEqual(plan["analysis_window"], [120.0, 160.0])
        self.assertEqual(plan["status"], "PREDECLARED_NO_CFD_RESULTS")
        self.assertEqual(plan["split"], "validation_only_not_training_or_frozen_test")
        with self.assertRaises(ValueError):
            action_points(0.5)


if __name__ == "__main__":
    unittest.main()
