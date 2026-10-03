from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


def load_module():
    path = (
        Path(__file__).parents[1]
        / "scripts"
        / "audit_two_phase_alternating_panel.py"
    )
    spec = importlib.util.spec_from_file_location("alternating_audit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TwoPhaseAlternatingAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_fluctuation_rms_removes_mean(self) -> None:
        self.assertAlmostEqual(
            self.module.fluctuation_rms(np.asarray([4.0, 6.0, 4.0, 6.0])),
            1.0,
        )

    def test_joint_gate_does_not_hide_lift_failure(self) -> None:
        zero = {"total_cd_mean": 2.0, "rear_cl_fluctuation_rms": 1.0}
        control = {
            "total_cd_mean": 1.9,
            "rear_cl_fluctuation_rms": 1.04,
            "rear_cl_mean": 0.11,
        }
        result = self.module.comparison(control, zero)
        self.assertTrue(result["canonical_checks"]["total_drag_reduction_at_least_2pct"])
        self.assertFalse(
            result["canonical_checks"][
                "rear_mean_lift_at_most_10pct_zero_fluctuation_rms"
            ]
        )
        self.assertFalse(result["canonical_joint_check"])


if __name__ == "__main__":
    unittest.main()
