"""Regression tests for fail-closed PhysicsNeMo stage result gates."""

from __future__ import annotations

import json
import unittest
from unittest.mock import Mock

from scripts.validate_tandem_fno_stage import (
    EVAL_METRICS,
    HORIZONS,
    TRAIN_METRICS,
    verify_evaluation,
    verify_training,
)


def report_path(payload: object) -> Mock:
    return Mock(read_text=Mock(return_value=json.dumps(payload)))


class StageValidatorTests(unittest.TestCase):
    def test_training_requires_complete_finite_epochs(self) -> None:
        history = [
            {"epoch": 1, **{key: 0.5 for key in TRAIN_METRICS}},
            {"epoch": 2, **{key: 0.4 for key in TRAIN_METRICS}},
        ]
        self.assertEqual(verify_training(report_path(history), 2)["last_epoch"], 2)
        with self.assertRaises(ValueError):
            verify_training(report_path(history), 3)
        history[1]["state_mae_physical_units"] = float("nan")
        with self.assertRaises(ValueError):
            verify_training(report_path(history), 2)

    def test_evaluation_requires_all_four_cases_and_horizons(self) -> None:
        case_horizons = {
            horizon: {"stable": True, "segments": 1} for horizon in HORIZONS
        }
        summary = {
            horizon: {
                "stable": True,
                "segments": 4,
                "failed_segments": 0,
                "state_channel_mae_u_v_p": [0.1, 0.1, 0.1],
                **{key: 0.5 for key in EVAL_METRICS},
            }
            for horizon in HORIZONS
        }
        payload = {
            "split": "test",
            "action_mode": "observed",
            "checkpoint_epoch": 5,
            "cases": [
                {"case": f"expanded_test_{index:02d}", "horizons": case_horizons}
                for index in range(4)
            ],
            "summary": summary,
        }
        result = verify_evaluation(report_path(payload), "observed")
        self.assertEqual(result["cases"], 4)
        with self.assertRaises(ValueError):
            verify_evaluation(report_path(payload), "zero")
        payload["summary"]["50"]["failed_segments"] = 1
        with self.assertRaises(ValueError):
            verify_evaluation(report_path(payload), "observed")


if __name__ == "__main__":
    unittest.main()
