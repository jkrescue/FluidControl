from __future__ import annotations

import sys
import unittest
from itertools import pairwise
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "cfd" / "tandem_cylinders"))
from make_phase90_long_signed_panel import (
    ACTION_INTERVAL,
    ANALYSIS_START,
    END_TIME,
    PANEL,
    START_TIME,
    action_points,
)


class Phase90LongSignedPanelTests(unittest.TestCase):
    def test_schedule_and_fixed_windows(self) -> None:
        self.assertEqual((START_TIME, ANALYSIS_START, END_TIME), (90.0, 102.0, 126.0))
        self.assertEqual([spec.target for spec in PANEL], [0.0, 1.0, -1.0])
        for spec in PANEL:
            points = action_points(spec)
            self.assertEqual(len(points), 361)
            self.assertEqual(points[0], (START_TIME, 0.0))
            self.assertEqual(points[-1], (END_TIME, spec.target))
            for first, second in pairwise(points):
                self.assertAlmostEqual(second[0] - first[0], ACTION_INTERVAL)
                self.assertLessEqual(abs(second[1] - first[1]), 0.05 + 1e-12)


if __name__ == "__main__":
    unittest.main()
