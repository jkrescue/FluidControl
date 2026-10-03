"""Train-only composition checks; real DataLoader integration is checked separately."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fluid_control.augmented_datapipe import compose_training_data


class AugmentedDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.base_root = root / "base"
        self.base_root.mkdir()
        self.extra = root / "extra"
        (self.extra / "train").mkdir(parents=True)
        normalization = b'{"source":"training-only"}\n'
        (self.base_root / "normalization.json").write_bytes(normalization)
        (self.extra / "normalization.json").write_bytes(normalization)
        self.base = SimpleNamespace(root=self.base_root, action_scale=0.75)
        hashes = {}
        for phase in (0, 2, 4, 6):
            for kind in ("prbs", "multisine"):
                name = f"dynamic_train8_b{phase:02d}_{kind}.h5"
                payload = name.encode()
                (self.extra / "train" / name).write_bytes(payload)
                hashes[name] = hashlib.sha256(payload).hexdigest()
        self.manifest = {
            "status": "DYNAMIC_TRAIN8_TRAIN_ONLY_CURATED",
            "trajectory_counts": {"train": 8, "validation": 0, "frozen_test": 0},
            "max_abs_omega": 0.75, "hdf_sha256": hashes,
        }
        self.save_manifest()

    def save_manifest(self):
        (self.extra / "manifest.json").write_text(json.dumps(self.manifest))

    def compose(self):
        return compose_training_data(self.base, [self.extra], rollout_steps=20,
                                     stride=2, workers=1, force_indices=(0, 1, 2, 3))

    def test_empty_preserves_existing_dataset(self):
        result, records = compose_training_data(self.base, [], rollout_steps=20,
                                               stride=2, workers=1, force_indices=(0, 1, 2, 3))
        self.assertIs(result, self.base)
        self.assertEqual(records, [])

    def test_valid_metadata_composes_official_multi_dataset(self):
        extra = MagicMock()
        extra.__len__.return_value = 728
        with patch("fluid_control.augmented_datapipe.TandemRolloutDataset", return_value=extra), \
                patch("fluid_control.augmented_datapipe.MultiDataset") as multi:
            result, records = self.compose()
            self.assertIs(result, multi.return_value)
            multi.assert_called_once_with(self.base, extra, output_strict=True)
            self.assertEqual(records[0]["windows"], 728)

    def test_rejects_nontraining_manifest(self):
        self.manifest["trajectory_counts"]["validation"] = 1
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "train-only"):
            self.compose()

    def test_rejects_exposed_validation(self):
        (self.extra / "validation").mkdir()
        with self.assertRaisesRegex(ValueError, "non-training"):
            self.compose()

    def test_rejects_changed_normalization(self):
        (self.extra / "normalization.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "normalization"):
            self.compose()

    def test_rejects_changed_hdf(self):
        next((self.extra / "train").glob("*.h5")).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA differs"):
            self.compose()

    def test_rejects_changed_action_scale(self):
        self.manifest["max_abs_omega"] = 1.0
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "action normalization"):
            self.compose()


if __name__ == "__main__":
    unittest.main()
