import copy
import unittest

from audit_tandem_gate_b import FORCE_CHANNELS, HORIZONS, audit_gate_b


def report(mode="observed", nrmse=0.05, case_prefix="heldout"):
    summary = {}
    for horizon in HORIZONS:
        summary[horizon] = {
            "stable": True,
            "failed_segments": 0,
            "segments": 4,
            "state_mae_physical_units": 0.01,
            "total_drag_mae": 0.1,
            "persistence_total_drag_mae": 0.2,
            "total_drag_nrmse": nrmse,
        }
    return {
        "checkpoint_epoch": 30,
        "action_mode": mode,
        "normalization_data": "train",
        "force_channels": list(FORCE_CHANNELS),
        "summary": summary,
        "cases": [{"case": f"{case_prefix}_{index}"} for index in range(4)],
    }


class GateBAuditTests(unittest.TestCase):
    def setUp(self):
        self.observed = report()
        self.counterfactuals = {
            mode: report(mode=mode) for mode in ("zero", "sign_flip", "shuffle")
        }
        for candidate in self.counterfactuals.values():
            for row in candidate["summary"].values():
                row["total_drag_mae"] = 0.3
        self.independent = report(case_prefix="phase")

    def test_passes_complete_consistent_evidence(self):
        result = audit_gate_b(self.observed, self.counterfactuals, self.independent)
        self.assertEqual(result["status"], "GATE_B_PASS")
        self.assertEqual(result["failed_checks"], [])

    def test_fails_closed_on_full_period_drift(self):
        self.observed["summary"]["100"]["total_drag_nrmse"] = 0.11
        result = audit_gate_b(self.observed, self.counterfactuals, self.independent)
        self.assertEqual(result["status"], "GATE_B_NEEDS_MULTISTEP_RETRAINING")
        self.assertIn("heldout_full_period_total_drag_nrmse", result["failed_checks"])

    def test_rejects_mismatched_counterfactual_cases(self):
        broken = copy.deepcopy(self.counterfactuals)
        broken["zero"]["cases"][0]["case"] = "wrong"
        with self.assertRaisesRegex(ValueError, "held-out case mismatch"):
            audit_gate_b(self.observed, broken, self.independent)

    def test_rejects_checkpoint_mismatch(self):
        self.independent["checkpoint_epoch"] = 25
        with self.assertRaisesRegex(ValueError, "checkpoint epoch mismatch"):
            audit_gate_b(self.observed, self.counterfactuals, self.independent)


if __name__ == "__main__":
    unittest.main()
