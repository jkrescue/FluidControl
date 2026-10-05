from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/derive_d012_fcp008_step_receipts.py"
)
SPEC = importlib.util.spec_from_file_location("derive_d012_fcp008", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
CHECKPOINT = "a" * 64
STATE = "b" * 64


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def refresh_bundle_receipt(receipt: Path) -> None:
    root = receipt.parent
    value = json.loads(receipt.read_text(encoding="utf-8"))
    value["sha256"] = {
        str(path.relative_to(root)): digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != receipt and path.name != "outer.log"
    }
    write_json(receipt, value)


def make_bundle(root: Path) -> tuple[Path, str]:
    lineage = {
        "status": MODULE.LINEAGE_STATUS,
        "candidate_kind": MODULE.CANDIDATE_KIND,
        "checkpoint_relative_directory": "candidate_build/candidate",
        "checkpoint_model_file": "FNO.0.0.mdlus",
        "checkpoint_state_file": "checkpoint.0.0.pt",
        "checkpoint_epoch": 0,
        "parent_checkpoint_epoch": 2,
        "calibration_generation": 1,
        "checkpoint_sha256": CHECKPOINT,
        "checkpoint_state_sha256": STATE,
        "calibration_fit_performed": True,
        "optimizer_training_performed": False,
        "training_performed": False,
        "validation_or_frozen_accessed": False,
        "ppo_auto_launch": False,
        "official_image_id": MODULE.IMAGE_ID,
        "formal_protocol": MODULE.PROTOCOL,
        "precision_protocol": {
            key: value
            for key, value in MODULE.PRECISION.items()
            if key not in {"status", "official_image_id"}
        },
    }
    write_json(root / "lineage.json", lineage)
    lineage_sha = digest(root / "lineage.json")
    documents = {
        "precision.json": MODULE.PRECISION,
        "evidence/formal_evaluation_approval.json": {
            "status": "FC_P008_FORMAL_EVALUATION_APPROVED",
            "candidate_model_sha256": CHECKPOINT,
            "candidate_state_sha256": STATE,
            "formal_evaluation_authorized": True,
            "frozen_test_accessed": False,
            "ppo_auto_launch": False,
        },
        "validation10/evaluation.json": {"rows": 1},
        "validation10/segments.json": {"rows": 1},
        "validation10/diagnostic.json": {"rows": 1},
        "validation10/endpoint_gate.json": {"rows": 1},
        "dynamic6/evaluation.json": {"rows": 606},
        "dynamic6/segments.json": {"rows": 606},
        "dynamic6/diagnostic.json": {"checkpoint_sha256": CHECKPOINT},
        "force_window/result.json": {"model_sha256": CHECKPOINT},
        "development_gate.json": {
            "checkpoint_sha256": CHECKPOINT,
            "status": "DYNAMIC_FNO_DEVELOPMENT_ADMISSION_FAIL",
            "ppo_authorized": False,
        },
    }
    for relative, payload in documents.items():
        write_json(root / relative, payload)
    for step, names in MODULE.STEP_FILES.items():
        write_json(
            root / f"step_receipts/{step}.json",
            {
                "status": MODULE.SOURCE_STEP_STATUS,
                "candidate_kind": MODULE.CANDIDATE_KIND,
                "step": step,
                "checkpoint_epoch": 0,
                "checkpoint_sha256": CHECKPOINT,
                "checkpoint_state_sha256": STATE,
                "lineage_sha256": lineage_sha,
                "precision_sha256": digest(root / "precision.json"),
                "formal_evaluation_approval_sha256": digest(
                    root / "evidence/formal_evaluation_approval.json"
                ),
                "posteval_chain_receipt_sha256": "c" * 64,
                "sha256": {name: digest(root / name) for name in names},
            },
        )
    receipt = root / "receipt.json"
    write_json(
        receipt,
        {
            "status": MODULE.COMPLETE_STATUS,
            "candidate_kind": MODULE.CANDIDATE_KIND,
            "checkpoint_epoch": 0,
            "checkpoint_sha256": CHECKPOINT,
            "checkpoint_state_sha256": STATE,
            "lineage_sha256": lineage_sha,
            "official_image_id": MODULE.IMAGE_ID,
            "protocol": MODULE.PROTOCOL,
            "precision_sha256": digest(root / "precision.json"),
            "formal_evaluation_approval_sha256": digest(
                root / "evidence/formal_evaluation_approval.json"
            ),
            "posteval_chain_receipt_sha256": "c" * 64,
            "frozen_test_accessed": False,
            "ppo_auto_launched": False,
            "sha256": {},
        },
    )
    refresh_bundle_receipt(receipt)
    return receipt, lineage_sha


class DerivedReceiptTests(unittest.TestCase):
    def test_absent_formal_completion_fails_without_writing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dynamic = root / "derived/dynamic.json"
            force = root / "derived/force.json"
            with self.assertRaises(FileNotFoundError):
                MODULE.payloads(
                    root / "posteval/receipt.json",
                    CHECKPOINT,
                    STATE,
                    "c" * 64,
                )
            self.assertFalse(dynamic.exists())
            self.assertFalse(force.exists())

    def test_binds_epoch_zero_model_state_lineage_without_copying_science(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, lineage_sha = make_bundle(Path(directory))
            result = MODULE.payloads(receipt, CHECKPOINT, STATE, lineage_sha)
            self.assertEqual(set(result), {"dynamic6", "force_window"})
            for value in result.values():
                self.assertEqual(value["status"], MODULE.DERIVED_STEP_STATUS)
                self.assertEqual(value["checkpoint_epoch"], 0)
                self.assertEqual(value["checkpoint_state_sha256"], STATE)
                self.assertEqual(value["lineage_sha256"], lineage_sha)
                self.assertFalse(value["scientific_status_reused"])
                self.assertFalse(value["ppo_authorized"])
                self.assertNotIn("PASS", value["status"])
            self.assertEqual(len(result["dynamic6"]["sha256"]), 3)
            self.assertEqual(len(result["force_window"]["sha256"]), 2)

    def test_rejects_missing_tampered_wrong_kind_and_scope(self):
        faults = ("missing", "tamper", "kind", "scope", "precision", "approval")
        for fault in faults:
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                receipt, lineage_sha = make_bundle(root)
                if fault == "missing":
                    (root / "dynamic6/segments.json").unlink()
                elif fault == "tamper":
                    (root / "dynamic6/segments.json").write_text("tampered\n")
                elif fault == "kind":
                    value = json.loads(receipt.read_text())
                    value["candidate_kind"] = "true_state_paired_step_lambda10"
                    write_json(receipt, value)
                elif fault == "scope":
                    value = json.loads(receipt.read_text())
                    value["frozen_test_accessed"] = True
                    write_json(receipt, value)
                elif fault == "precision":
                    precision = root / "precision.json"
                    value = json.loads(precision.read_text())
                    value["float32_matmul_precision"] = "highest"
                    write_json(precision, value)
                    for step in MODULE.STEP_FILES:
                        step_path = root / f"step_receipts/{step}.json"
                        step_value = json.loads(step_path.read_text())
                        step_value["precision_sha256"] = digest(precision)
                        write_json(step_path, step_value)
                    receipt_value = json.loads(receipt.read_text())
                    receipt_value["precision_sha256"] = digest(precision)
                    write_json(receipt, receipt_value)
                    refresh_bundle_receipt(receipt)
                else:
                    approval = root / "evidence/formal_evaluation_approval.json"
                    value = json.loads(approval.read_text())
                    value["formal_evaluation_authorized"] = False
                    write_json(approval, value)
                    for step in MODULE.STEP_FILES:
                        step_path = root / f"step_receipts/{step}.json"
                        step_value = json.loads(step_path.read_text())
                        step_value["formal_evaluation_approval_sha256"] = digest(
                            approval
                        )
                        write_json(step_path, step_value)
                    receipt_value = json.loads(receipt.read_text())
                    receipt_value["formal_evaluation_approval_sha256"] = digest(
                        approval
                    )
                    write_json(receipt, receipt_value)
                    refresh_bundle_receipt(receipt)
                with self.assertRaises(ValueError):
                    MODULE.payloads(receipt, CHECKPOINT, STATE, lineage_sha)

    def test_rejects_wrong_epoch_state_lineage_and_training_claim(self):
        faults = ("epoch", "state", "lineage_sha", "training")
        for fault in faults:
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                receipt, lineage_sha = make_bundle(root)
                if fault == "epoch":
                    value = json.loads(receipt.read_text())
                    value["checkpoint_epoch"] = 2
                    write_json(receipt, value)
                elif fault == "state":
                    value = json.loads(receipt.read_text())
                    value["checkpoint_state_sha256"] = "c" * 64
                    write_json(receipt, value)
                elif fault == "lineage_sha":
                    lineage_sha = "d" * 64
                else:
                    lineage = root / "lineage.json"
                    value = json.loads(lineage.read_text())
                    value["training_performed"] = True
                    write_json(lineage, value)
                    lineage_sha = digest(lineage)
                    receipt_value = json.loads(receipt.read_text())
                    receipt_value["lineage_sha256"] = lineage_sha
                    write_json(receipt, receipt_value)
                    for step in MODULE.STEP_FILES:
                        step_path = root / f"step_receipts/{step}.json"
                        step_value = json.loads(step_path.read_text())
                        step_value["lineage_sha256"] = lineage_sha
                        write_json(step_path, step_value)
                    refresh_bundle_receipt(receipt)
                with self.assertRaises(ValueError):
                    MODULE.payloads(receipt, CHECKPOINT, STATE, lineage_sha)

    def test_rejects_source_step_mismatch_even_if_completion_table_is_refreshed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt, lineage_sha = make_bundle(root)
            step = root / "step_receipts/dynamic6.json"
            value = json.loads(step.read_text())
            value["checkpoint_state_sha256"] = "e" * 64
            write_json(step, value)
            refresh_bundle_receipt(receipt)
            with self.assertRaisesRegex(ValueError, "source step identity"):
                MODULE.payloads(receipt, CHECKPOINT, STATE, lineage_sha)

    def test_rejects_output_inside_bundle_and_escaping_bundle_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            outside = Path(directory) / "outside"
            receipt, lineage_sha = make_bundle(root)
            outside.mkdir()
            with self.assertRaises(ValueError):
                MODULE.validate_output_paths(
                    receipt, root / "derived.json", outside / "force.json"
                )
            escape = Path(directory) / "escape.json"
            escape.write_text("{}\n")
            (root / "escape.json").symlink_to(escape)
            refresh_bundle_receipt(receipt)
            with self.assertRaisesRegex(ValueError, "escapes source root"):
                MODULE.payloads(receipt, CHECKPOINT, STATE, lineage_sha)


if __name__ == "__main__":
    unittest.main()
