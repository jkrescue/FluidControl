from __future__ import annotations

import copy
import tempfile
import threading
import unittest
from pathlib import Path

from fluid_control.dynamic_pair_stat_datapipe import (
    SOURCE_KEYS,
    DynamicMatchedPairStatDataset,
)


def manifest():
    pairs = []
    source = {key: key * 64 for key in SOURCE_KEYS}
    for phase in (0, 2, 4, 6):
        for profile in ("multisine", "prbs"):
            pairs.append({"phase": f"b{phase:02d}", "profile": profile, "split": "train", "start": 0, "horizon": 100,
                          "action_file": f"train/a_{phase}_{profile}.h5", "zero_file": f"train/z_{phase}.h5",
                          "action_hdf_sha256": "a" * 64, "zero_hdf_sha256": "b" * 64,
                          "action_source_state_sha256": source, "zero_source_state_sha256": copy.deepcopy(source), "targets": {}})
    return {"status": "FC_P003_DYNAMIC8_PAIR_CANDIDATE_QC_PASS", "scope": "train-only existing-data candidate; not approved for training",
            "split": "train", "pair_count": 8, "sequence_length": 101, "horizon": 100, "horizons": [20, 50, 100],
            "force_channels": ["front_cd", "front_cl", "rear_cd", "rear_cl"], "state0_tolerance": {"rtol": 0.0, "atol": 3e-7},
            "validation_or_frozen_accessed": False, "unique_initial_restart_count": 4,
            "profiles_per_phase": ["multisine", "prbs"], "pairs": pairs}


def shell(tmp_path, value):
    dataset = object.__new__(DynamicMatchedPairStatDataset)
    dataset._executor = None; dataset._inflight = set(); dataset._prefetch_handles = {}
    dataset._readers = {}; dataset._reader_lock = threading.Lock(); dataset._lock = threading.Lock()
    dataset.action_root = (tmp_path / "action").resolve(); dataset.zero_root = (tmp_path / "zero").resolve(); dataset.manifest = value
    for root in (dataset.action_root, dataset.zero_root): (root / "train").mkdir(parents=True, exist_ok=True)
    for pair in value["pairs"]:
        (dataset.action_root / pair["action_file"]).touch(exist_ok=True)
        (dataset.zero_root / pair["zero_file"]).touch(exist_ok=True)
    return dataset


class DynamicPairManifestTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_rejects_restart_sha_mismatch(self):
        value = manifest()
        value["pairs"][0]["zero_source_state_sha256"]["U"] = "different"
        with self.assertRaisesRegex(ValueError, "source restart SHA"):
            shell(self.root, value)._validate_manifest(False)

    def test_rejects_tolerance_or_pair_set_change(self):
        value = manifest()
        value["state0_tolerance"]["atol"] = 1e-5
        with self.assertRaisesRegex(ValueError, "state0_tolerance"):
            shell(self.root, value)._validate_manifest(False)
        value = manifest()
        value["pairs"].pop()
        with self.assertRaisesRegex(ValueError, "exact 4-phase"):
            shell(self.root, value)._validate_manifest(False)


if __name__ == "__main__":
    unittest.main()
