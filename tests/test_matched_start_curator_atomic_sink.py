from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


def load_module():
    path = Path(__file__).parents[1] / "scripts" / "curate_low_action_phase94_validation.py"
    spec = importlib.util.spec_from_file_location("commissioning_curator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as error:
        if error.name and (
            error.name.startswith("physicsnemo")
            or error.name.startswith("warp")
        ):
            raise unittest.SkipTest(
                "PhysicsNeMo Curator integration test requires .venv-curator-py312"
            ) from error
        raise
    return module


class MatchedStartAtomicSinkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_profile_is_train_only_and_atomic_sink_refuses_overwrite(self) -> None:
        profile = "matched_start_commissioning_train9_v1"
        self.assertEqual(
            self.module.PROFILE_COUNTS[profile],
            {"train": 9, "validation": 0, "test": 0},
        )
        item = {
            "case": "case_a", "split": "train",
            "state": np.zeros((1, 3, 2, 2), np.float32),
            "mask": np.ones((1, 1, 2, 2), np.uint8),
            "omega": np.zeros((1, 1), np.float32),
            "force": np.zeros((1, 4), np.float32),
            "time": np.zeros((1, 1), np.float32),
            "x": np.zeros(2, np.float32), "y": np.zeros(2, np.float32),
            "config": {"split": "train"},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sink = self.module.TrajectoryHDF5Sink(root, atomic_tmp=True)
            paths = sink(iter([item]), 0)
            target = root / "train/case_a.h5"
            self.assertEqual(paths, [str(target)])
            self.assertTrue(target.is_file())
            self.assertFalse(target.with_suffix(".h5.tmp").exists())
            with self.assertRaises(FileExistsError):
                sink(iter([item]), 0)

    def test_generic_finalize_is_blocked_for_commissioning_profile(self) -> None:
        arguments = [
            "curate", "--profile", "matched_start_commissioning_train9_v1",
            "--finalize-only",
        ]
        with (
            mock.patch.object(sys, "argv", arguments),
            self.assertRaises(SystemExit),
        ):
            self.module.main()


if __name__ == "__main__":
    unittest.main()
