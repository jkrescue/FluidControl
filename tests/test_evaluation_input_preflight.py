"""Input-only checks; fixtures contain no scientific CFD values."""

import tempfile
import unittest
from pathlib import Path

import h5py

from evaluate_tandem_fno import preflight_evaluation_paths


class EvaluationPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "validation").mkdir()

    def make_hdf(self, frames=201, force_frames=None):
        path = self.root / "validation/case.h5"
        with h5py.File(path, "w") as handle:
            for key in ("state", "mask", "omega", "force", "time"):
                handle.create_dataset(
                    key,
                    shape=(
                        force_frames
                        if key == "force" and force_frames is not None
                        else frames,
                    ),
                )
        return path

    def check(self, **kwargs):
        options = {"horizons": [1, 10, 50, 100], "batch_size": 8, "stride": 1}
        options.update(kwargs)
        return preflight_evaluation_paths(self.root, "validation", **options)

    def test_empty_directory_is_explicit_error(self):
        with self.assertRaisesRegex(FileNotFoundError, "container mounts"):
            self.check()

    def test_broken_host_symlink_is_explicit_error(self):
        (self.root / "validation").rmdir()
        (self.root / "validation").symlink_to(self.root / "missing-host-path")
        with self.assertRaisesRegex(FileNotFoundError, "container mounts"):
            self.check()

    def test_short_time_series_rejected(self):
        self.make_hdf(frames=100)
        with self.assertRaisesRegex(ValueError, "cover requested horizons"):
            self.check()

    def test_mismatched_time_lengths_rejected(self):
        self.make_hdf(force_frames=200)
        with self.assertRaisesRegex(ValueError, "cover requested horizons"):
            self.check()

    def test_incomplete_final_batch_is_valid(self):
        path = self.make_hdf()
        self.assertEqual(self.check(), [path])  # 101 H100 starts, not divisible by 8.

    def test_invalid_runtime_parameters_rejected(self):
        self.make_hdf()
        for options in (
            {"horizons": [0]},
            {"horizons": []},
            {"batch_size": 0},
            {"stride": 0},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.check(**options)


if __name__ == "__main__":
    unittest.main()
