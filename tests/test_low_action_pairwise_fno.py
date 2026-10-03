from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_low_action_pairwise_fno.py"
if not SCRIPT.exists():
    SCRIPT = ROOT / "audit_low_action_pairwise_fno.py"


def load_module():
    spec = importlib.util.spec_from_file_location("low_action_pairwise_fno", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = load_module()


def row(case: str, start: int, predicted: float, target: float) -> dict:
    return {
        "case": case,
        "horizon": 100,
        "start": start,
        "max_abs_omega": 0.75,
        "predicted_total_drag": predicted,
        "target_total_drag": target,
        "total_drag_absolute_error": abs(predicted - target),
    }


def document(predictions: dict[tuple[str, int], float]) -> dict:
    targets = {
        ("negative", 0): 2.0,
        ("positive", 0): 3.0,
        ("negative", 25): 4.0,
        ("positive", 25): 3.0,
    }
    names = {
        "negative": "validation_signed_low_pulse_m_phase94_v1_20261003",
        "positive": "validation_signed_low_pulse_p_phase94_v1_20261003",
    }
    rows = [
        row(names[role], start, predictions[(role, start)], target)
        for (role, start), target in targets.items()
    ]
    rows.append({"case": names["negative"], "horizon": 10, "start": 0})
    return {"split": "validation", "action_mode": "observed", "segments": rows}


class LowActionPairwiseFnoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.parent_path = self.root / "parent.json"
        self.candidate_path = self.root / "candidate.json"
        self.parent = document(
            {
                ("negative", 0): 3.2,
                ("positive", 0): 2.7,
                ("negative", 25): 3.1,
                ("positive", 25): 3.5,
            }
        )
        self.candidate = document(
            {
                ("negative", 0): 2.1,
                ("positive", 0): 2.8,
                ("negative", 25): 4.5,
                ("positive", 25): 3.4,
            }
        )
        self.write_inputs()

    def tearDown(self) -> None:
        self.directory.cleanup()

    def write_inputs(self) -> None:
        self.parent_path.write_text(json.dumps(self.parent), encoding="utf-8")
        self.candidate_path.write_text(json.dumps(self.candidate), encoding="utf-8")

    def test_paired_delta_metrics_and_strict_start_zero_are_separate(self) -> None:
        result = MODULE.analyze(self.parent_path, self.candidate_path)
        parent = result["models"]["v3_parent"]
        candidate = result["models"]["v4_candidate"]
        self.assertAlmostEqual(parent["pairwise_delta_mae"], 1.45)
        self.assertEqual(
            parent["matched_elapsed_time_sign_ranking_accuracy_diagnostic"], 0.0
        )
        self.assertAlmostEqual(candidate["pairwise_delta_mae"], 0.2)
        self.assertEqual(
            candidate["matched_elapsed_time_sign_ranking_accuracy_diagnostic"],
            1.0,
        )
        self.assertTrue(candidate["strict_common_initial_ranking_correct"])
        self.assertEqual(candidate["strict_common_initial_state_start"], 0)
        self.assertIn("different physical states", candidate["all_starts_interpretation"])
        first = candidate["strict_common_initial_state_pair"]
        self.assertAlmostEqual(first["predicted_cd_negative_minus_positive"], -0.7)
        self.assertAlmostEqual(first["true_cd_negative_minus_positive"], -1.0)

    def test_mismatched_action_start_sets_are_rejected(self) -> None:
        self.candidate["segments"] = [
            item
            for item in self.candidate["segments"]
            if not (
                item.get("horizon") == 100
                and "_p_phase94_" in item["case"]
                and item["start"] == 25
            )
        ]
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "start sets differ"):
            MODULE.analyze(self.parent_path, self.candidate_path)

    def test_parent_candidate_cfd_truth_must_be_identical(self) -> None:
        target = next(
            item
            for item in self.candidate["segments"]
            if item.get("horizon") == 100 and item["start"] == 0
        )
        target["target_total_drag"] += 0.01
        target["total_drag_absolute_error"] = abs(
            target["predicted_total_drag"] - target["target_total_drag"]
        )
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "CFD target differs"):
            MODULE.analyze(self.parent_path, self.candidate_path)

    def test_missing_strict_common_initial_pair_is_rejected(self) -> None:
        for document_value in (self.parent, self.candidate):
            document_value["segments"] = [
                item
                for item in document_value["segments"]
                if item.get("horizon") != 100 or item["start"] != 0
            ]
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "start=0"):
            MODULE.analyze(self.parent_path, self.candidate_path)

    def test_output_is_exclusive_and_scope_forbids_control_claim(self) -> None:
        result = MODULE.analyze(self.parent_path, self.candidate_path)
        self.assertIn("not zero-control benefit", result["scientific_scope"])
        self.assertIn("not closed-loop success", result["scientific_scope"])
        self.assertTrue(
            any("instantaneous endpoint near t=104" in item for item in result["limitations"])
        )
        self.assertTrue(any("[114,174]" in item for item in result["limitations"]))
        output = self.root / "audit.json"
        MODULE.write_exclusive(output, result)
        with self.assertRaises(FileExistsError):
            MODULE.write_exclusive(output, result)


if __name__ == "__main__":
    unittest.main()
