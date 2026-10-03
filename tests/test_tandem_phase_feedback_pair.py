from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


def load_module():
    path = (
        Path(__file__).parents[1]
        / "scripts"
        / "run_tandem_phase_feedback_pair.py"
    )
    spec = importlib.util.spec_from_file_location("run_tandem_phase_feedback_pair", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_torque_module():
    path = (
        Path(__file__).parents[1]
        / "scripts"
        / "audit_tandem_phase_feedback_torque.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_tandem_phase_feedback_torque", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TandemPhaseFeedbackPairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()
        cls.torque = load_torque_module()

    def test_feedback_action_has_fixed_sign_and_rate_limits(self) -> None:
        target, applied = self.module.feedback_action(2.0, 0.0)
        self.assertEqual(target, 1.0)
        self.assertEqual(applied, 0.1)
        target, applied = self.module.feedback_action(-2.0, 0.1)
        self.assertEqual(target, -1.0)
        self.assertEqual(applied, 0.0)
        self.assertLessEqual(abs(applied), self.module.OMEGA_LIMIT)

    def test_force_metrics_and_predeclared_acceptance(self) -> None:
        times = np.arange(4, dtype=float)
        front_zero = np.column_stack((times, np.full(4, 1.4), [-1.0, 1.0, -1.0, 1.0]))
        rear_zero = np.column_stack((times, np.full(4, 1.0), [-2.0, 2.0, -2.0, 2.0]))
        front_control = np.column_stack(
            (times, np.full(4, 1.4), [-0.9, 0.9, -0.9, 0.9])
        )
        rear_control = np.column_stack(
            (times, np.full(4, 0.8), [-1.8, 1.8, -1.8, 1.8])
        )
        zero = self.module.force_metrics(front_zero, rear_zero)
        control = self.module.force_metrics(front_control, rear_control)
        result = self.module.compare_metrics(control, zero)
        self.assertAlmostEqual(result["total_drag_reduction"], 1.0 / 12.0)
        self.assertAlmostEqual(result["rear_cl_total_rms_ratio"], 0.9)
        self.assertTrue(result["short_pilot_joint_check"])
        self.assertTrue(result["canonical_physical_joint_check"])

    def test_original_pilot_guard_uses_total_lift_rms_and_mean_lift(self) -> None:
        zero = {
            "total_cd_mean": 2.4,
            "rear_cl_total_rms": 2.0,
            "rear_cl_fluctuation_rms": 2.0,
            "front_cl_total_rms": 1.0,
        }
        control = {
            "total_cd_mean": 2.2,
            "rear_cl_total_rms": 2.01,
            "rear_cl_fluctuation_rms": 0.1,
            "rear_cl_mean": 2.0,
            "front_cl_total_rms": 1.0,
        }
        result = self.module.compare_metrics(control, zero)
        self.assertFalse(
            result["predeclared_checks"][
                "rear_mean_lift_at_most_10pct_zero_fluctuation_rms"
            ]
        )
        self.assertFalse(result["short_pilot_joint_check"])
        self.assertFalse(result["canonical_physical_joint_check"])

    def test_canonical_gate_requires_2pct_drag_and_fluctuation_rms(self) -> None:
        zero = {
            "total_cd_mean": 2.0,
            "rear_cl_total_rms": 1.0,
            "rear_cl_fluctuation_rms": 1.0,
            "rear_cl_mean": 0.0,
            "front_cl_total_rms": 1.0,
        }
        insufficient_drag = {
            "total_cd_mean": 1.97,
            "rear_cl_total_rms": 1.0,
            "rear_cl_fluctuation_rms": 1.0,
            "rear_cl_mean": 0.0,
            "front_cl_total_rms": 1.0,
        }
        excessive_fluctuation = {
            "total_cd_mean": 1.95,
            "rear_cl_total_rms": 1.0,
            "rear_cl_fluctuation_rms": 1.06,
            "rear_cl_mean": 0.0,
            "front_cl_total_rms": 1.0,
        }

        drag_result = self.module.compare_metrics(insufficient_drag, zero)
        fluctuation_result = self.module.compare_metrics(excessive_fluctuation, zero)
        self.assertFalse(
            drag_result["canonical_physical_checks"][
                "total_drag_reduction_at_least_2pct"
            ]
        )
        self.assertFalse(drag_result["canonical_physical_joint_check"])
        self.assertFalse(
            fluctuation_result["canonical_physical_checks"][
                "rear_cl_fluctuation_rms_increase_at_most_5pct"
            ]
        )
        self.assertFalse(fluctuation_result["canonical_physical_joint_check"])

    def test_torque_power_sign_and_normalization(self) -> None:
        time = np.asarray([0.0, 1.0])
        moment = np.asarray([-0.2, -0.4])
        actions = np.asarray([[0.0, 1.0], [1.0, 1.0]])
        rows, metrics = self.torque.power_metrics(time, moment, actions, 2.0)
        np.testing.assert_allclose(rows[:, 4], [0.1, 0.2])
        self.assertAlmostEqual(
            metrics["mean_positive_only_actuator_power_over_zero_drag_power"],
            0.15,
        )


if __name__ == "__main__":
    unittest.main()
