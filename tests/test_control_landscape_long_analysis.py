"""Checks for long paired-force comparison and action bookkeeping."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cfd" / "tandem_cylinders"))
from analyze_control_landscape_long_panel import compare, full_effort  # noqa: E402


class LongPanelAnalysisTests(unittest.TestCase):
    def test_drag_and_lift_tradeoff_are_separate(self) -> None:
        baseline = {"case": "zero", "omega_final": 0.0, "total_cd_mean": 2.0,
                    "rear": {"cl_mean": 0.0, "cl_rms": 1.0, "cl_abs_mean": 1.0}}
        actuated = {"case": "plus", "omega_final": 1.0, "total_cd_mean": 1.8,
                    "rear": {"cl_mean": 1.0, "cl_rms": 0.9, "cl_abs_mean": 1.2}}
        outcome = compare([baseline, actuated])["cases"][1]["relative_to_zero"]
        self.assertAlmostEqual(outcome["total_cd_change_percent"], -10.0)
        self.assertLess(outcome["rear_cl_fluctuation_rms_ratio"], 1.0)
        self.assertGreater(outcome["rear_cl_total_rms_ratio"], 1.0)
        self.assertAlmostEqual(outcome["rear_cl_abs_mean_ratio"], 1.2)

    def test_full_effort_includes_initial_ramp(self) -> None:
        result = full_effort([[80.0, 0.0], [81.0, 0.5], [82.0, 1.0]])
        self.assertGreater(result["domega_dt_rms"], 0)
        self.assertIn("not measured", result["note"])


if __name__ == "__main__":
    unittest.main()
