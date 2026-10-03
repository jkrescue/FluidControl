from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def load_module():
    path = Path(__file__).parents[1] / "scripts" / "audit_low_action_phase94_physics.py"
    spec = importlib.util.spec_from_file_location("low_action_phase94", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LowActionPhase94AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_canonical_gate_keeps_three_checks_separate(self) -> None:
        zero = {
            "mean_total_cd": 2.0,
            "rear_cl_fluctuation_rms": 1.0,
        }
        control = {
            "mean_total_cd": 1.96,
            "rear_cl_fluctuation_rms": 1.05,
            "abs_mean_rear_cl": 0.101,
        }
        result = self.module.canonical_comparison(control, zero)
        self.assertTrue(result["drag_reduction_at_least_2_percent"])
        self.assertTrue(result["rear_cl_fluctuation_within_1p05_zero"])
        self.assertFalse(
            result["abs_mean_rear_cl_within_0p1_zero_cl_fluctuation_rms"]
        )
        self.assertEqual(result["canonical_joint_gate"], "FAIL")


if __name__ == "__main__":
    unittest.main()
