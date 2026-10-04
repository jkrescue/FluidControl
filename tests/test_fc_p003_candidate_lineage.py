from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load_module(relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class FCP003LineageTests(unittest.TestCase):
    def test_approved_roots_and_single_factor_are_explicit(self) -> None:
        module = load_module("scripts/audit_fc_p003_candidate_lineage.py", "paired_lineage")
        self.assertEqual(module.CANDIDATE, "tandem_fno_paired_stats_interleaved_lambda10_20261005")
        self.assertTrue(module.ORDER_AUDIT.startswith("fab043a7"))
        self.assertTrue(module.PAIR.startswith("15bfa7a4"))
        self.assertTrue(module.PARENT_MODEL.startswith("8466bd47"))

    def test_archive_payload_ignores_zip_timestamp_but_not_payload(self) -> None:
        import tempfile
        import zipfile
        module = load_module("scripts/audit_fc_p003_candidate_lineage.py", "paired_zip")
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a.zip", Path(directory) / "b.zip"
            with zipfile.ZipFile(a, "w") as archive:
                archive.writestr("model.pt", b"same")
                archive.writestr("args.json", b"args")
                archive.writestr("metadata.json", b"metadata")
            with zipfile.ZipFile(b, "w") as archive:
                info = zipfile.ZipInfo("model.pt", date_time=(2025, 1, 1, 0, 0, 0))
                archive.writestr(info, b"same")
                archive.writestr("args.json", b"args")
                archive.writestr("metadata.json", b"metadata")
            self.assertNotEqual(module.sha256(a), module.sha256(b))
            self.assertEqual(module.archive_payload(a), module.archive_payload(b))
            with zipfile.ZipFile(b, "w") as archive:
                archive.writestr("model.pt", b"different")
                archive.writestr("args.json", b"args")
                archive.writestr("metadata.json", b"metadata")
            self.assertNotEqual(module.archive_payload(a), module.archive_payload(b))

    def test_complete_validator_rejects_tamper(self) -> None:
        import tempfile
        module = load_module("scripts/validate_fc_p003_posteval_step.py", "paired_validator")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "result.json"; artifact.write_text("{}\n")
            checkpoint = "a" * 64
            receipt = {
                "status": "FC_P003_POSTEVAL_COMPLETE",
                "checkpoint_sha256": checkpoint,
                "ppo_auto_launched": False,
                "frozen_test_accessed": False,
                "sha256": {"result.json": module.sha256(artifact)},
            }
            (root / "receipt.json").write_text(json.dumps(receipt))
            (root / "outer.log").write_text("mutable transport log\n")
            module.validate_complete(root, checkpoint)
            artifact.write_text("tampered\n")
            with self.assertRaisesRegex(ValueError, "hash table"):
                module.validate_complete(root, checkpoint)

    def test_archive_payload_rejects_unexpected_member_set(self) -> None:
        import tempfile
        import zipfile
        module = load_module("scripts/audit_fc_p003_candidate_lineage.py", "paired_members")
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "bad.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("model.pt", b"model")
                archive.writestr("args.json", b"args")
                archive.writestr("metadata.json", b"metadata")
                archive.writestr("unexpected", b"bad")
            with self.assertRaisesRegex(ValueError, "member set"):
                module.archive_payload(archive_path)


if __name__ == "__main__":
    unittest.main()
