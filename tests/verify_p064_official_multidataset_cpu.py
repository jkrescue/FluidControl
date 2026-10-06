"""Tiny installed-PhysicsNeMo check; synthetic HDF only, no real data/model/GPU."""

from pathlib import Path
import importlib.util
import json
import sys
import tempfile

import h5py
import numpy as np
import torch
from physicsnemo.datapipes import DataLoader, DatasetBase, MultiDataset
from tensordict import TensorDict

import fluid_control
from fluid_control.tandem_datapipe import TandemRolloutDataset


MODULE_PATH = Path(__file__).resolve().parents[1] / "src/fluid_control/p064_controlled_aero_ab.py"
SPEC = importlib.util.spec_from_file_location(
    "fluid_control.p064_controlled_aero_ab", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
ScheduledABDataset = MODULE.ScheduledABDataset
scheduled_training_identity = MODULE.scheduled_training_identity


class Child(DatasetBase):
    def __init__(self, family, count):
        super().__init__(num_workers=0)
        self.family = family
        self.count = count

    def __len__(self):
        return self.count

    def _load(self, index):
        return TensorDict({"value": torch.tensor(index)}, batch_size=[]), {
            "case": self.family,
            "start": index,
            "split": "train",
            "rollout_steps": 100,
        }


class Reference:
    action_scale = 0.75


def synthetic_b00(root: Path):
    (root / "train").mkdir(parents=True)
    normalization = {
        "state_mean": [0.0, 0.0, 0.0],
        "state_std": [1.0, 1.0, 1.0],
        "all_force_channels": ["front_cd", "front_cl", "rear_cd", "rear_cl"],
        "all_force_mean": [0.0, 0.0, 0.0, 0.0],
        "all_force_std": [1.0, 1.0, 1.0, 1.0],
    }
    (root / "normalization.json").write_text(json.dumps(normalization))
    (root / "manifest.json").write_text(json.dumps({"max_abs_omega": 0.75}))
    path = root / "train" / "synthetic_controlled_b00.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("state", data=np.zeros((801, 3, 2, 3), np.float32))
        handle.create_dataset("mask", data=np.ones((801, 1, 2, 3), np.float32))
        handle.create_dataset(
            "omega", data=np.linspace(0, 0.7, 801, dtype=np.float32)[:, None]
        )
        force = np.arange(801 * 4, dtype=np.float32).reshape(801, 4)
        handle.create_dataset("force", data=force)
        handle.create_dataset(
            "time", data=(np.arange(801, dtype=np.float32) * 0.1)[:, None]
        )
    return TandemRolloutDataset(
        root, "train", 100, stride=1, num_workers=0, force_indices=(0, 1, 2, 3)
    )


def main():
    order = json.loads(Path(__file__).with_name("p064_parent_order.json").read_text())
    children = [Child(f"family{i}", n) for i, n in enumerate((720, 408, 240))]
    original = MultiDataset(*children, output_strict=True)
    adapter = ScheduledABDataset(original, Reference(), order, "A", num_workers=0)
    sample, metadata = adapter._load(0)
    assert not hasattr(original, "_load")
    assert sample["value"] == order[0]
    assert metadata["dataset_index"] == 0
    assert metadata["ab_source"] == "original44"
    loader = DataLoader(
        adapter,
        batch_size=1,
        shuffle=False,
        collate_metadata=True,
        prefetch_factor=0,
        use_streams=False,
        seed=20261003,
    )
    _, collated = next(iter(loader))
    assert scheduled_training_identity(collated)["dataset_index"] == 0
    adapter.close()

    with tempfile.TemporaryDirectory(prefix="p064-official-reader-") as tmp:
        b00 = synthetic_b00(Path(tmp))
        children = [Child(f"family{i}", n) for i, n in enumerate((720, 408, 240))]
        original = MultiDataset(*children, output_strict=True)
        adapter = ScheduledABDataset(
            original, b00, order, "B", b00_dataset=b00, num_workers=0
        )
        first, first_meta = adapter._load(0)
        last_position = next(
            i for i, row in enumerate(adapter.schedule) if row.b00_start == 700
        )
        last, last_meta = adapter._load(last_position)
        assert first["target_state"].shape == last["target_state"].shape == (100, 3, 2, 3)
        assert first_meta["step"] == first_meta["ab_b00_start"] == 0
        assert last_meta["step"] == last_meta["ab_b00_start"] == 700
        assert last_meta["dataset_index"] == 3
        for dataset_index in (0, 1, 2):
            position = next(
                i
                for i, row in enumerate(adapter.schedule)
                if row.source == "original44"
                and (0 if row.original_global_index < 720 else 1 if row.original_global_index < 1128 else 2)
                == dataset_index
            )
            _, row_meta = adapter._load(position)
            assert row_meta["dataset_index"] == dataset_index
        adapter.close()
    print("P064_OFFICIAL_MULTIDATASET_CPU_PASS")


if __name__ == "__main__":
    main()
