from __future__ import annotations

import sys
import unittest
from itertools import pairwise
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "cfd" / "tandem_cylinders"))
from make_crossphase_ranking_panel import (
    ACTION_INTERVAL,
    END_TIME,
    PANEL,
    START_TIME,
    action_points,
)


class CrossPhaseRankingPanelTests(unittest.TestCase):
    def test_three_predeclared_actions_share_grid_and_start_at_zero(self) -> None:
        self.assertEqual([spec.target for spec in PANEL], [0.0, 1.0, -1.0])
        for spec in PANEL:
            points = action_points(spec)
            self.assertEqual(len(points), 21)
            self.assertEqual(points[0], (START_TIME, 0.0))
            self.assertEqual(points[-1], (END_TIME, spec.target))
            for first, second in pairwise(points):
                self.assertAlmostEqual(second[0] - first[0], ACTION_INTERVAL)
                self.assertLessEqual(abs(second[1] - first[1]), 0.1 + 1e-12)


if __name__ == "__main__":
    unittest.main()
