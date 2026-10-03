"""Locked three-criterion periodic-benchmark decision tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cfd" / "tandem_cylinders"))
from analyze_periodic_rotation_benchmark import evaluate  # noqa: E402


def row(case: str, drag: float, rms: float, mean: float) -> dict:
    return {
        "case": case,
        "source_restart_u_sha256": "same",
        "source_restart_p_sha256": "same",
        "total_cd_mean": drag,
        "rear": {"cl_rms": rms, "cl_mean": mean},
    }


class PeriodicAnalysisTests(unittest.TestCase):
    def test_joint_acceptance_requires_all_three(self) -> None:
        zero = row("landscape_long_val_zero_20261003", 2.0, 1.0, 0.0)
        passing = row("periodic_val_p10_20261003", 1.95, 1.04, 0.09)
        failing_mean = row("periodic_val_p20_20261003", 1.9, 0.9, 0.11)
        decisions = evaluate([zero, passing, failing_mean])["decisions"]
        self.assertTrue(decisions[0]["one_phase_coarse_grid_screen_pass"])
        self.assertFalse(decisions[1]["one_phase_coarse_grid_screen_pass"])
        self.assertFalse(decisions[1]["checks"]["abs_mean_rear_cl_at_most_10pct_zero_fluct_rms"])


if __name__ == "__main__":
    unittest.main()
