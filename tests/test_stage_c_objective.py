import unittest

import numpy as np

from fluid_control.stage_c_objective import (
    stage_c_cost_components,
    stage_c_force_ledger,
    validate_stage_c_baseline,
)


class StageCObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.baseline = {
            "total_drag": 3.0,
            "front_lift_rms": 0.1,
            "rear_lift_rms": 0.2,
            "source": "phase_matched_zero_action_fixture",
        }

    def test_force_ledger_uses_system_drag_and_both_lifts(self):
        forces = np.asarray([[1.0, 0.2, 2.0, 0.4], [1.0, -0.2, 2.0, -0.4]])
        ledger = stage_c_force_ledger(forces, self.baseline)
        self.assertAlmostEqual(ledger["total_drag"], 3.0)
        self.assertAlmostEqual(ledger["front_lift_ratio"], 2.0)
        self.assertAlmostEqual(ledger["rear_lift_ratio"], 2.0)
        self.assertAlmostEqual(ledger["drag_improvement"], 0.0)

    def test_cost_components_match_locked_stage_c_weights(self):
        ledger = {
            **stage_c_force_ledger(np.asarray([[0.9, 0.2, 1.8, 0.4]]), self.baseline),
            "window_ready": True,
        }
        terms = stage_c_cost_components(
            ledger,
            omega=2.5,
            delta_omega=0.25,
            action_scale=5.0,
            max_delta_omega=0.5,
        )
        self.assertAlmostEqual(terms["total_drag"], -0.1)
        self.assertAlmostEqual(terms["front_lift_excess"], 0.05)
        self.assertAlmostEqual(terms["rear_lift_excess"], 0.10)
        self.assertAlmostEqual(terms["actuation"], 0.0025)
        self.assertAlmostEqual(terms["rate"], 0.0025)

    def test_incomplete_window_only_charges_actuation(self):
        ledger = {
            **stage_c_force_ledger(np.ones((1, 4)), self.baseline),
            "window_ready": False,
        }
        terms = stage_c_cost_components(
            ledger,
            omega=0.0,
            delta_omega=0.0,
            action_scale=5.0,
            max_delta_omega=0.5,
        )
        self.assertEqual(sum(terms.values()), 0.0)

    def test_baseline_validation_is_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "missing keys"):
            validate_stage_c_baseline({"total_drag": 1.0})
        with self.assertRaisesRegex(ValueError, "positive"):
            validate_stage_c_baseline({**self.baseline, "rear_lift_rms": 0.0})


if __name__ == "__main__":
    unittest.main()
