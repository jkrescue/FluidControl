import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/derive_d012_p003b_step_receipts.py"
SPEC = importlib.util.spec_from_file_location("derive_d012_p003b", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
CHECKPOINT = "a" * 64


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_bundle(root: Path) -> Path:
    documents = {
        "lineage.json": {"checkpoint_sha256": CHECKPOINT},
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
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    table = {
        str(path.relative_to(root)): digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    receipt = root / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "FC_P003B_POSTEVAL_COMPLETE",
                "candidate_kind": "dynamic_paired_interleaved_lambda10",
                "checkpoint_sha256": CHECKPOINT,
                "frozen_test_accessed": False,
                "ppo_auto_launched": False,
                "sha256": table,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return receipt


class DerivedReceiptTests(unittest.TestCase):
    def test_binds_original_files_without_copying_scientific_status(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = make_bundle(Path(directory))
            result = MODULE.payloads(receipt, CHECKPOINT)
            self.assertEqual(set(result), {"dynamic6", "force_window"})
            self.assertFalse(result["dynamic6"]["scientific_status_reused"])
            self.assertEqual(len(result["dynamic6"]["sha256"]), 3)
            self.assertEqual(len(result["force_window"]["sha256"]), 2)
            self.assertEqual(
                result["force_window"]["source_posteval_receipt_sha256"],
                digest(receipt),
            )

    def test_rejects_missing_file_tamper_mixed_checkpoint_and_incomplete_table(self):
        for fault in ("missing", "tamper", "checkpoint", "table"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                receipt = make_bundle(root)
                if fault == "missing":
                    (root / "dynamic6/segments.json").unlink()
                elif fault == "tamper":
                    (root / "force_window/result.json").write_text("{}\n")
                elif fault == "checkpoint":
                    path = root / "dynamic6/diagnostic.json"
                    path.write_text(json.dumps({"checkpoint_sha256": "b" * 64}) + "\n")
                    payload = json.loads(receipt.read_text())
                    payload["sha256"]["dynamic6/diagnostic.json"] = digest(path)
                    receipt.write_text(json.dumps(payload) + "\n")
                else:
                    payload = json.loads(receipt.read_text())
                    payload["sha256"].pop("dynamic6/segments.json")
                    receipt.write_text(json.dumps(payload) + "\n")
                with self.assertRaises(ValueError):
                    MODULE.payloads(receipt, CHECKPOINT)

    def test_rejects_fake_pass_status_and_wrong_scope(self):
        for key, value in (
            ("status", "FC_P003B_POSTEVAL_PASS"),
            ("candidate_kind", "paired_stats_interleaved_lambda10"),
            ("frozen_test_accessed", True),
            ("ppo_auto_launched", True),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                receipt = make_bundle(Path(directory))
                payload = json.loads(receipt.read_text())
                payload[key] = value
                receipt.write_text(json.dumps(payload) + "\n")
                with self.assertRaises(ValueError):
                    MODULE.payloads(receipt, CHECKPOINT)


if __name__ == "__main__":
    unittest.main()
