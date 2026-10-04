from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/derive_d012_fcp003c_step_receipts.py"
)
SPEC = importlib.util.spec_from_file_location("derive_d012_fcp003c", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
CHECKPOINT = "a" * 64


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def make_bundle(root: Path) -> Path:
    documents = {
        "lineage.json": {
            "status": "FC_P003C_CANDIDATE_LINEAGE_PASS",
            "candidate_kind": MODULE.CANDIDATE_KIND,
            "checkpoint_sha256": CHECKPOINT,
            "training_performed": True,
            "frozen_test_opened_or_enumerated": False,
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
        },
    }
    for relative, payload in documents.items():
        write_json(root / relative, payload)
    for step, names in MODULE.STEP_FILES.items():
        write_json(root / f"step_receipts/{step}.json", {
            "status": MODULE.SOURCE_STEP_STATUS,
            "step": step,
            "checkpoint_sha256": CHECKPOINT,
            "sha256": {
                str((root / name).resolve()): digest(root / name) for name in names
            },
        })
    table = {
        str(path.relative_to(root)): digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    receipt = root / "receipt.json"
    write_json(receipt, {
        "status": MODULE.COMPLETE_STATUS,
        "candidate_kind": MODULE.CANDIDATE_KIND,
        "checkpoint_sha256": CHECKPOINT,
        "frozen_test_accessed": False,
        "ppo_auto_launched": False,
        "sha256": table,
    })
    return receipt


def refresh_bundle_receipt(receipt: Path) -> None:
    root = receipt.parent
    value = json.loads(receipt.read_text())
    value["sha256"] = {
        str(path.relative_to(root)): digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != receipt and path.name != "outer.log"
    }
    write_json(receipt, value)


class DerivedReceiptTests(unittest.TestCase):
    def test_binds_native_steps_without_copying_scientific_status(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = make_bundle(Path(directory))
            result = MODULE.payloads(receipt, CHECKPOINT)
            self.assertEqual(set(result), {"dynamic6", "force_window"})
            self.assertFalse(result["dynamic6"]["scientific_status_reused"])
            self.assertEqual(result["dynamic6"]["status"], MODULE.DERIVED_STEP_STATUS)
            self.assertEqual(len(result["dynamic6"]["sha256"]), 3)
            self.assertEqual(len(result["force_window"]["sha256"]), 2)
            self.assertEqual(
                result["force_window"]["source_posteval_receipt_sha256"],
                digest(receipt),
            )

    def test_rejects_missing_file_modified_sha_wrong_kind_and_wrong_scope(self):
        faults = ("missing", "modified_sha", "wrong_kind", "wrong_trained")
        for fault in faults:
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                receipt = make_bundle(root)
                if fault == "missing":
                    (root / "dynamic6/segments.json").unlink()
                elif fault == "modified_sha":
                    source = root / "step_receipts/dynamic6.json"
                    value = json.loads(source.read_text())
                    value["sha256"][str((root / "dynamic6/segments.json").resolve())] = (
                        "0" * 64
                    )
                    write_json(source, value)
                    refresh_bundle_receipt(receipt)
                elif fault == "wrong_kind":
                    value = json.loads(receipt.read_text())
                    value["candidate_kind"] = "dynamic_paired_interleaved_lambda10"
                    write_json(receipt, value)
                else:
                    lineage = root / "lineage.json"
                    value = json.loads(lineage.read_text())
                    value["training_performed"] = False
                    write_json(lineage, value)
                    refresh_bundle_receipt(receipt)
                with self.assertRaises(ValueError):
                    MODULE.payloads(receipt, CHECKPOINT)

    def test_rejects_mixed_checkpoint_and_fake_completion_status(self):
        for fault in ("checkpoint", "status"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                receipt = make_bundle(root)
                if fault == "checkpoint":
                    diagnostic = root / "dynamic6/diagnostic.json"
                    write_json(diagnostic, {"checkpoint_sha256": "b" * 64})
                    source = root / "step_receipts/dynamic6.json"
                    value = json.loads(source.read_text())
                    value["sha256"][str(diagnostic.resolve())] = digest(diagnostic)
                    write_json(source, value)
                    refresh_bundle_receipt(receipt)
                else:
                    value = json.loads(receipt.read_text())
                    value["status"] = "FC_P003C_POSTEVAL_PASS"
                    write_json(receipt, value)
                with self.assertRaises(ValueError):
                    MODULE.payloads(receipt, CHECKPOINT)

    def test_rejects_output_inside_source_bundle_even_through_symlink_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            outside = Path(directory) / "outside"
            receipt = make_bundle(root)
            outside.mkdir()
            (root / "linked-output").symlink_to(outside, target_is_directory=True)
            for dynamic in (root / "derived.json", root / "linked-output/derived.json"):
                with self.subTest(dynamic=dynamic), self.assertRaises(ValueError):
                    MODULE.validate_output_paths(
                        receipt, dynamic, outside / "force.json"
                    )

    def test_rejects_bundle_symlink_that_resolves_outside_source_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            receipt = make_bundle(root)
            outside = Path(directory) / "outside.json"
            outside.write_text("{}\n", encoding="utf-8")
            (root / "escape.json").symlink_to(outside)
            refresh_bundle_receipt(receipt)
            with self.assertRaises(ValueError):
                MODULE.payloads(receipt, CHECKPOINT)


if __name__ == "__main__":
    unittest.main()
