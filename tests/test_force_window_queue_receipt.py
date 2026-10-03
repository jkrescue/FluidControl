"""Test queue provenance checks using synthetic metadata, never CFD fields."""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/run_fno_force_window_after_training.sh"
)


def validator():
    return (
        SCRIPT.read_text()
        .split('python3 - "$training" "$posteval" <<\'PY\'\n', 1)[1]
        .split("\nPY\n", 1)[0]
    )


def fixture(tmp_path):
    train = tmp_path / "training"
    post = train / "posteval_resume_v1"
    (train / "best").mkdir(parents=True)
    model = train / "best/FNO.0.4.mdlus"
    model.write_bytes(b"unit-test metadata placeholder, not a model")
    (train / "training_history.json").write_text(
        json.dumps([{"epoch": i} for i in range(1, 5)])
    )
    hashes = {}
    for suite in ("validation10", "dynamic6"):
        (post / suite).mkdir(parents=True)
        for name in ("evaluation", "segments", "diagnostic"):
            status = (
                "DEV30_VALIDATION_DIAGNOSTIC_COMPLETE"
                if suite == "validation10"
                else "DYNAMIC6_FNO_DIAGNOSTIC_FAIL"
            )
            path = post / suite / f"{name}.json"
            path.write_text(json.dumps({"status": status}))
            hashes[f"{suite}/{name}.json"] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    receipt = {
        "status": "DYNAMIC_FNO_POSTEVAL_RESUME_COMPLETE",
        "checkpoint_epoch": 4,
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "frozen_test_accessed": False,
        "training_performed": False,
        "evaluation_data_mount": "/workspace/devdata",
        "sha256": hashes,
    }
    return train, post, receipt


def run(fixture):
    train, post, receipt = fixture
    (post / "receipt.json").write_text(json.dumps(receipt))
    return subprocess.run(
        [sys.executable, "-", str(train), str(post)],
        input=validator(),
        text=True,
        capture_output=True,
    )


class QueueReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = fixture(Path(self.temp.name))

    def test_complete_diagnostic_can_fail_scientifically(self):
        self.assertEqual(run(self.fixture).returncode, 0)  # No PPO authorization.

    def test_receipt_contract_rejects_changes(self):
        for field, value in [
            ("frozen_test_accessed", True),
            ("training_performed", True),
            ("evaluation_data_mount", "/workspace/base"),
            ("checkpoint_epoch", 3),
        ]:
            with self.subTest(field=field):
                original = self.fixture[2][field]
                self.fixture[2][field] = value
                self.assertNotEqual(run(self.fixture).returncode, 0)
                self.fixture[2][field] = original

    def test_rejects_changed_result(self):
        (self.fixture[1] / "dynamic6/segments.json").write_text("{}")
        self.assertIn("SHA differs", run(self.fixture).stderr)

    def test_rejects_changed_model(self):
        (self.fixture[0] / "best/FNO.0.4.mdlus").write_bytes(b"changed fixture")
        self.assertIn("checkpoint differs", run(self.fixture).stderr)

    def test_rejects_missing_bound_file(self):
        self.fixture[2]["sha256"].pop("dynamic6/evaluation.json")
        self.assertIn("exactly six", run(self.fixture).stderr)

    def test_rejects_incomplete_training(self):
        (self.fixture[0] / "training_history.json").write_text('[{"epoch": 1}]')
        self.assertIn("four epochs", run(self.fixture).stderr)


if __name__ == "__main__":
    unittest.main()
