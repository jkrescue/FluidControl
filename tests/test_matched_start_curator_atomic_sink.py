from __future__ import annotations

import hashlib
import importlib.util
import json
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

    def build_matched_start_source(self, root: Path) -> tuple[Path, str, dict]:
        cases_root = root / "cases"
        name = "matched_start_acquisition_train_b00_zero"
        case = cases_root / name
        case.mkdir(parents=True)
        source_hashes = {}
        for object_name in self.module.MATCHED_START_FORCE_OBJECTS:
            path = (
                cases_root
                / "tandem_backward_dt005"
                / "postProcessing"
                / object_name
                / "0/coefficient.dat"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# Time Cd Cl\n0 1.0 0.0\n1 1.1 0.1\n")
            source_hashes[object_name] = hashlib.sha256(path.read_bytes()).hexdigest()
        config = {
            "case": name,
            "split": "train",
            "expected_field_frames": 801,
            "source_restart_case": "tandem_backward_dt005",
            "source_force_sha256": source_hashes,
        }
        (case / "case_config.json").write_text(json.dumps(config))
        return cases_root, name, config

    def test_source_preflight_binds_baseline_force_before_vtk_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cases_root, name, config = self.build_matched_start_source(Path(directory))
            records = self.module.case_records(
                cases_root, self.module.MATCHED_START_PROFILE, {name}
            )
            self.assertEqual(records[0]["config"], config)

    def test_source_preflight_rejects_baseline_force_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cases_root, name, _ = self.build_matched_start_source(Path(directory))
            source = (
                cases_root
                / "tandem_backward_dt005/postProcessing/forceRear/0/coefficient.dat"
            )
            source.write_text("# Time Cd Cl\n0 99 99\n1 99 99\n")
            with self.assertRaisesRegex(ValueError, "baseline source-force SHA mismatch"):
                self.module.case_records(
                    cases_root, self.module.MATCHED_START_PROFILE, {name}
                )

    def test_source_force_guard_is_not_applied_to_legacy_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            self.module,
            "validate_matched_start_source_force",
            side_effect=AssertionError("matched-start-only guard was called"),
        ), self.assertRaises(FileNotFoundError):
            self.module.case_records(Path(directory), "stage1")


if __name__ == "__main__":
    unittest.main()
