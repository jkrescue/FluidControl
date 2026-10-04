from __future__ import annotations

import copy
import tempfile
import threading
import unittest
from pathlib import Path

import torch

from fluid_control.paired_stat_datapipe import (
    MatchedPairStatDataset,
    physical_pair_targets,
)


def manifest() -> dict:
    pairs = []
    for phase in ("b00", "b02", "b04", "b06"):
        for action in ("m075", "m0375", "p0375", "p075"):
            pairs.append(
                {
                    "phase": phase,
                    "action": action,
                    "split": "train",
                    "start": 0,
                    "action_file": f"case_{phase}_{action}.h5",
                    "zero_file": f"case_{phase}_zero.h5",
                    "action_hdf_sha256": "a" * 64,
                    "zero_hdf_sha256": "b" * 64,
                    "initial_sha256": {"state": phase},
                    "zero_initial_sha256": {"state": phase},
                    "targets": {str(h): {} for h in (20, 50, 100)},
                }
            )
    return {
        "status": "TRAIN20_MATCHED_PAIR_DATAPIPE_READY",
        "split": "train",
        "start": 0,
        "sequence_length": 101,
        "dt": 0.1,
        "horizons": [20, 50, 100],
        "force_channels": ["front_cd", "front_cl", "rear_cd", "rear_cl"],
        "pair_count": 16,
        "validation_or_frozen_accessed": False,
        "max_abs_omega": 0.75,
        "pairs": pairs,
    }


class PairManifestRejectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def dataset(self, value: dict) -> MatchedPairStatDataset:
        dataset = object.__new__(MatchedPairStatDataset)
        dataset._executor = None
        dataset._inflight = set()
        dataset._prefetch_handles = {}
        dataset._readers = {}
        dataset._lock = threading.Lock()
        dataset.root = self.root.resolve()
        dataset.manifest = value
        (self.root / "train").mkdir(exist_ok=True)
        for pair in value["pairs"]:
            for key in ("action_file", "zero_file"):
                (self.root / "train" / pair[key]).touch()
        return dataset

    def test_rejects_validation_pair(self) -> None:
        value = manifest()
        value["pairs"][0]["split"] = "validation"
        with self.assertRaisesRegex(ValueError, "train/start0"):
            self.dataset(value)._validate_manifest(False)

    def test_rejects_wrong_phase_pair(self) -> None:
        value = manifest()
        value["pairs"][0]["phase"] = "b02"
        with self.assertRaisesRegex(ValueError, "duplicate paired identity|exact 4-phase"):
            self.dataset(value)._validate_manifest(False)

    def test_rejects_nonidentical_initial_state(self) -> None:
        value = manifest()
        value["pairs"][0]["zero_initial_sha256"] = {"state": "different"}
        with self.assertRaisesRegex(ValueError, "non-identical matched start"):
            self.dataset(value)._validate_manifest(False)

    def test_rejects_duplicate_pair(self) -> None:
        value = manifest()
        value["pairs"][1] = copy.deepcopy(value["pairs"][0])
        with self.assertRaisesRegex(ValueError, "duplicate paired identity"):
            self.dataset(value)._validate_manifest(False)

    def test_physical_target_tamper_is_detectable(self) -> None:
        zero = torch.zeros(101, 4)
        action = zero.clone()
        action[1:, 0] = 0.25
        targets = physical_pair_targets(action, zero)
        tampered = targets.clone()
        tampered[2, 0] += 0.01
        self.assertFalse(torch.allclose(targets, tampered, rtol=2e-5, atol=2e-6))


if __name__ == "__main__":
    unittest.main()
