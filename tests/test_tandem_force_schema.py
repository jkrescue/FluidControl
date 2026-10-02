from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


def load_normalization_module():
    path = Path(__file__).parents[1] / "scripts" / "add_all_force_normalization.py"
    spec = importlib.util.spec_from_file_location("add_all_force_normalization", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TandemForceSchemaTests(unittest.TestCase):
    def test_all_force_normalization_uses_train_split_only(self) -> None:
        module = load_normalization_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train"
            test = root / "test"
            train.mkdir()
            test.mkdir()
            first = np.asarray([[1, 2, 3, 4], [3, 4, 5, 6]], dtype=np.float32)
            second = np.asarray([[5, 6, 7, 8]], dtype=np.float32)
            ignored = np.full((10, 4), 1_000, dtype=np.float32)
            for path, values in (
                (train / "a.h5", first),
                (train / "b.h5", second),
                (test / "ignored.h5", ignored),
            ):
                with h5py.File(path, "w") as handle:
                    handle.create_dataset("force", data=values)

            mean, std, count = module.compute(root)
            expected = np.concatenate((first, second), axis=0).astype(np.float64)
            self.assertEqual(count, 3)
            np.testing.assert_allclose(mean, expected.mean(axis=0), rtol=1e-12)
            np.testing.assert_allclose(std, expected.std(axis=0), rtol=1e-12)

    def test_total_drag_config_declares_four_physical_force_channels(self) -> None:
        root = Path(__file__).parents[1]
        text = (root / "conf" / "tandem_fno_total_drag.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn("force_indices: [0, 1, 2, 3]", text)
        self.assertIn("out_channels: 7", text)
        self.assertEqual(
            load_normalization_module().CHANNELS,
            ("front_cd", "front_cl", "rear_cd", "rear_cl"),
        )


if __name__ == "__main__":
    unittest.main()
