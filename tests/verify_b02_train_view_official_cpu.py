"""Actual Python 3.12/PhysicsNeMo CPU fixture for first/last b02 H100 windows."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import h5py
import numpy as np


PROJECT = Path("/workspace/fluid_control")
STAGE = Path("/tmp/p064-b02-controlled-train-conversion-sota-20261007")
sys.path.insert(0, str(PROJECT / "src"))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    view = load(STAGE / "scripts/build_b02_train_view.py", "b02_view_fixture")
    with tempfile.TemporaryDirectory(prefix="b02-view-official-") as directory:
        root = Path(directory) / "view"
        train = root / "train"
        train.mkdir(parents=True)
        hdf = train / "b02_canonical_ppo_train.h5"
        with h5py.File(hdf, "w") as handle:
            handle["state"] = np.arange(801 * 3 * 2 * 3, dtype=np.float32).reshape(801, 3, 2, 3) / 100
            handle["mask"] = np.ones((801, 1, 2, 3), dtype=np.uint8)
            handle["omega"] = np.linspace(-.75, .75, 801, dtype=np.float32)[:, None]
            handle["force"] = np.arange(801 * 4, dtype=np.float32).reshape(801, 4) / 50
            handle["time"] = np.arange(801, dtype=np.float64)[:, None] / 10 + 106
            handle["x"] = np.arange(3, dtype=np.float32)
            handle["y"] = np.arange(2, dtype=np.float32)
        norm = (PROJECT / "data/curated/tandem_cylinders_matched_start_full40_dev30_v1/normalization.json").read_bytes()
        (root / "normalization.json").write_bytes(norm)
        (root / "manifest.json").write_text(json.dumps({"max_abs_omega": .75}))
        checks = view.verify_windows(root, hdf)
        assert [(item["start"], item["endpoint"]) for item in checks] == [(0, 100), (700, 800)]
    print("OFFICIAL_B02_VIEW_FIXTURE_PASS")


if __name__ == "__main__":
    main()
