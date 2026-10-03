from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "cfd" / "tandem_cylinders"))
from make_two_phase_alternating_panel import (
    ACTION_PERIOD,
    ANALYSIS_OFFSET,
    DURATION,
    PANEL,
    action_audit,
    action_points,
)


class TwoPhaseAlternatingPanelTests(unittest.TestCase):
    def test_control_schedules_are_bounded_rate_limited_and_zero_mean(self) -> None:
        controlled = [spec for spec in PANEL if spec.controlled]
        self.assertEqual([spec.start for spec in controlled], [90.0, 94.0])
        self.assertEqual((DURATION - ANALYSIS_OFFSET) / ACTION_PERIOD, 3.0)
        for spec in controlled:
            metrics = action_audit(action_points(spec))
            self.assertAlmostEqual(metrics["signed_time_mean_omega"], 0.0)
            self.assertEqual(metrics["max_abs_omega"], 1.0)
            self.assertEqual(metrics["max_abs_domega_dt"], 1.0)

    def test_each_phase_has_a_fresh_zero_case(self) -> None:
        by_start = {}
        for spec in PANEL:
            by_start.setdefault(spec.start, []).append(spec.controlled)
        self.assertEqual(by_start, {90.0: [False, True], 94.0: [False, True]})


if __name__ == "__main__":
    unittest.main()
