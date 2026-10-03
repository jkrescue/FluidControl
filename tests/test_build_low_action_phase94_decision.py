from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def load_module():
    path = Path(__file__).parents[1] / "scripts" / "build_low_action_phase94_decision.py"
    spec = importlib.util.spec_from_file_location("low_action_decision", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metrics(nrmse: float, mae: float, persistence: float) -> dict:
    return {
        "pooled_total_drag_nrmse": nrmse,
        "total_drag_mae": mae,
        "persistence_total_drag_mae": persistence,
    }


class LowActionDecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_counterfactual_pass_requires_all_three_checks(self) -> None:
        result = self.module.derive_decision(
            metrics(0.09, 0.20, 0.25), metrics(0.08, 0.15, 0.25)
        )
        self.assertEqual(
            result["decision"], "ELIGIBLE_FOR_PAIRED_ACTION_RANKING_AUDIT"
        )
        self.assertTrue(all(result["decision_checks"].values()))
        self.assertIn("does not pass Gate-C", result["reason"])

    def test_nrmse_threshold_failure_blocks_promotion(self) -> None:
        result = self.module.derive_decision(
            metrics(0.12, 0.20, 0.25), metrics(0.11, 0.15, 0.25)
        )
        self.assertEqual(result["decision"], "DO_NOT_PROMOTE_V4_TO_CLOSED_LOOP")
        self.assertFalse(
            result["decision_checks"][
                "v4_h100_pooled_total_drag_nrmse_at_most_10pct"
            ]
        )

    def test_persistence_or_regression_failure_blocks_promotion(self) -> None:
        persistence_failure = self.module.derive_decision(
            metrics(0.09, 0.20, 0.25), metrics(0.08, 0.26, 0.25)
        )
        regression_failure = self.module.derive_decision(
            metrics(0.07, 0.20, 0.25), metrics(0.08, 0.15, 0.25)
        )
        self.assertEqual(
            persistence_failure["decision"], "DO_NOT_PROMOTE_V4_TO_CLOSED_LOOP"
        )
        self.assertEqual(
            regression_failure["decision"], "DO_NOT_PROMOTE_V4_TO_CLOSED_LOOP"
        )


if __name__ == "__main__":
    unittest.main()
