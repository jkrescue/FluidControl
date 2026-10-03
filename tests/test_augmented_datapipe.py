"""Train-only composition checks; real DataLoader integration is checked separately."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fluid_control.augmented_datapipe import (
    TRAIN16_AUTH_SHA,
    compose_training_data,
    expected_training_files,
)


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
            "max_abs_omega": 0.75,
            "hdf_sha256": hashes,
        }
        self.save_manifest()

    def save_manifest(self):
        (self.extra / "manifest.json").write_text(json.dumps(self.manifest))

    def compose(self):
        return compose_training_data(
            self.base,
            [self.extra],
            rollout_steps=20,
            stride=2,
            workers=1,
            force_indices=(0, 1, 2, 3),
        )

    def test_empty_preserves_existing_dataset(self):
        result, records = compose_training_data(
            self.base,
            [],
            rollout_steps=20,
            stride=2,
            workers=1,
            force_indices=(0, 1, 2, 3),
        )
        self.assertIs(result, self.base)
        self.assertEqual(records, [])

    def test_valid_metadata_composes_official_multi_dataset(self):
        extra = MagicMock()
        extra.__len__.return_value = 728
        with (
            patch(
                "fluid_control.augmented_datapipe.TandemRolloutDataset",
                return_value=extra,
            ),
            patch("fluid_control.augmented_datapipe.MultiDataset") as multi,
        ):
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


class DirectPPOTrain16Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.base_root = root / "base"
        self.base_root.mkdir()
        self.extra = root / "extra"
        (self.extra / "train").mkdir(parents=True)
        normalization = b'{"source":"training-only-unit-test"}\n'
        (self.base_root / "normalization.json").write_bytes(normalization)
        (self.extra / "normalization.json").write_bytes(normalization)
        self.norm_sha = hashlib.sha256(normalization).hexdigest()
        self.base = SimpleNamespace(root=self.base_root, action_scale=0.75)
        hashes = {}
        for env, phase in ((0, 0), (1, 2)):
            for episode in range(2, 10):
                name = f"direct_cfd_directppo2048_v1_env{env}_ep{episode:04d}_b{phase:02d}.h5"
                payload = b"unit-test metadata placeholder, not CFD: " + name.encode()
                (self.extra / "train" / name).write_bytes(payload)
                hashes[name] = hashlib.sha256(payload).hexdigest()
        self.manifest = {
            "status": "DIRECTPPO_TRAIN16_TRAIN_ONLY_CURATED",
            "profile": "directppo_train16_v1",
            "trajectory_counts": {"train": 16, "validation": 0, "frozen_test": 0},
            "frames_per_trajectory": 129,
            "max_abs_omega": 0.75,
            "normalization_sha256": self.norm_sha,
            "curation_authorization_sha256": TRAIN16_AUTH_SHA,
            "validation_or_frozen_accessed": False,
            "vtk_receipt_sha256": {Path(name).stem: "a" * 64 for name in hashes},
            "hdf_sha256": hashes,
        }
        self.save()

    def save(self):
        (self.extra / "manifest.json").write_text(json.dumps(self.manifest))

    def compose(self):
        return compose_training_data(
            self.base,
            [self.extra],
            rollout_steps=100,
            stride=2,
            workers=1,
            force_indices=(0, 1, 2, 3),
        )

    def test_official_multi_dataset_composition(self):
        dataset = MagicMock()
        dataset.__len__.return_value = 240
        with (
            patch(
                "fluid_control.augmented_datapipe.TandemRolloutDataset",
                return_value=dataset,
            ),
            patch("fluid_control.augmented_datapipe.MultiDataset") as multi,
        ):
            _, records = self.compose()
            multi.assert_called_once_with(self.base, dataset, output_strict=True)
            self.assertEqual(records[0]["trajectory_count"], 16)
            self.assertEqual(records[0]["release_status"], self.manifest["status"])

    def test_rejects_provenance_change(self):
        for key, wrong in [
            ("profile", "other"),
            ("frames_per_trajectory", 128),
            ("normalization_sha256", "0" * 64),
            ("curation_authorization_sha256", "0" * 64),
            ("validation_or_frozen_accessed", True),
        ]:
            with self.subTest(key=key):
                original = self.manifest[key]
                self.manifest[key] = wrong
                with self.assertRaisesRegex(ValueError, "provenance"):
                    expected_training_files(self.manifest, self.norm_sha)
                self.manifest[key] = original

    def test_rejects_extra_validation_case(self):
        wrong = self.extra / "train/direct_cfd_directppo2048_b01_eval.h5"
        wrong.write_bytes(b"unit-test placeholder")
        with self.assertRaisesRegex(ValueError, "phase/file set"):
            self.compose()

    def test_rejects_changed_hdf(self):
        next((self.extra / "train").glob("*.h5")).write_bytes(b"changed fixture")
        with self.assertRaisesRegex(ValueError, "SHA differs"):
            self.compose()

    def test_rejects_missing_vtk_receipt(self):
        self.manifest["vtk_receipt_sha256"].pop(
            next(iter(self.manifest["vtk_receipt_sha256"]))
        )
        self.save()
        with self.assertRaisesRegex(ValueError, "VTK receipt"):
            self.compose()

    def test_rejects_unfinalized(self):
        self.manifest["status"] = "DIRECTPPO_TRAIN16_VTK_READY"
        self.save()
        with self.assertRaisesRegex(ValueError, "not finalized"):
            self.compose()


if __name__ == "__main__":
    unittest.main()
