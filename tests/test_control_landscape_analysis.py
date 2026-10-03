"""Unit checks for paired-force and objective calculations."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cfd" / "tandem_cylinders"))
from analyze_control_landscape_panel import action_effort, compare, total_drag_by_time  # noqa: E402


class ControlLandscapeAnalysisTests(unittest.TestCase):
    def test_total_drag_pairs_both_cylinders_at_same_timestamps(self) -> None:
        front = [(92.0, 1.0, 0.0), (92.005, 2.0, 0.0), (116.0, 3.0, 0.0)]
        rear = [(92.0, 4.0, 0.0), (92.005, 5.0, 0.0), (116.0, 6.0, 0.0)]
        self.assertEqual(total_drag_by_time(front, rear, 92.0, 116.0),
                         [(92.0, 5.0), (92.005, 7.0), (116.0, 9.0)])
        rear[1] = (92.01, 5.0, 0.0)
        with self.assertRaises(ValueError):
            total_drag_by_time(front, rear, 92.0, 116.0)

    def test_control_response_reports_lift_tradeoff(self) -> None:
        baseline = {"case": "zero", "action_kind": "zero", "total_cd_mean": 2.0,
                    "front": {"cl_rms": 0.3}, "rear": {"cl_rms": 1.0}}
        controlled = {"case": "actuated", "action_kind": "positive_constant",
                      "total_cd_mean": 1.8, "front": {"cl_rms": 0.3},
                      "rear": {"cl_rms": 1.5}}
        result = compare([baseline, controlled])
        metrics = result["cases"][1]["relative_to_zero"]
        self.assertAlmostEqual(metrics["total_cd_change_percent"], -10.0)
        self.assertAlmostEqual(metrics["rear_cl_rms_ratio"], 1.5)
        self.assertEqual(result["status"], "EXPLORATORY_MATCHED_CFD_PANEL_OK")

    def test_kinematic_effort_is_not_called_physical_work(self) -> None:
        points = [[92.0, 0.0], [92.1, 1.0], [116.0, 1.0]]
        effort = action_effort(points)
        self.assertGreater(effort["omega_rms"], 0)
        self.assertIn("not actuator torque", effort["note"])


if __name__ == "__main__":
    unittest.main()
