"""Pre-solve train-only signed-pulse dataset plan tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cfd" / "tandem_cylinders"))
from make_signed_pulse_training_pair import NAMES, action_points, audit, metrics  # noqa: E402


class SignedPulseTrainingTests(unittest.TestCase):
    def test_pair_is_mirrored_and_bounded(self) -> None:
        positive, negative = action_points(1.0), action_points(-1.0)
        self.assertEqual(len(positive), 801)
        self.assertEqual(positive[0], (82.0, 0.0))
        self.assertEqual(positive[-1], (162.0, 0.0))
        self.assertEqual([time for time, _ in positive], [time for time, _ in negative])
        self.assertEqual([omega for _, omega in positive], [-omega for _, omega in negative])
        self.assertLessEqual(metrics(positive)["max_abs_domega_dt"], 0.625 + 1e-9)
        self.assertEqual(set(NAMES.values()), {-1.0, 1.0})

    def test_plan_keeps_validation_out(self) -> None:
        plan = audit()
        self.assertEqual(plan["split"], "train_only")
        self.assertEqual(plan["status"], "PREDECLARED_TRAIN_ONLY_ACTIONS_NO_CFD_LABELS")
        self.assertEqual(plan["time_window"], [82.0, 162.0])
        with self.assertRaises(ValueError):
            action_points(0.5)


if __name__ == "__main__":
    unittest.main()
