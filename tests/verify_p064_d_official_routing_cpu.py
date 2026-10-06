"""Actual PhysicsNeMo CPU routing check for D b02 starts 0 and 700."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import h5py
import numpy as np
import torch
from physicsnemo.datapipes import DataLoader, DatasetBase


PROJECT = Path("/workspace/fluid_control")
STAGE = Path("/tmp/p064-b00-b02-coverage-d-sota-20261007")
sys.path.insert(0, str(PROJECT / "artifacts/p064_controlled_data_dose_c50_source_20261007_immutable/src"))
from fluid_control.tandem_datapipe import TandemRolloutDataset


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


schedule = load(STAGE / "src/fluid_control/p064_controlled_aero_abd.py", "p064_d_schedule")
runner = load(STAGE / "scripts/train_fcp064_controlled_aero_abd.py", "p064_d_runner")
ORDER = json.loads((STAGE / "tests/parent_order.json").read_text())


def make_view(root, offset):
    train = root / "train"
    train.mkdir(parents=True)
    hdf = train / f"controlled_{offset}.h5"
    with h5py.File(hdf, "w") as handle:
        state = np.arange(801 * 3 * 2 * 3, dtype=np.float32).reshape(801, 3, 2, 3)
        handle["state"] = state / 100 + offset
        handle["mask"] = np.ones((801, 1, 2, 3), dtype=np.uint8)
        handle["omega"] = np.linspace(-.75, .75, 801, dtype=np.float32)[:, None]
        handle["force"] = np.arange(801 * 4, dtype=np.float32).reshape(801, 4) / 50
        handle["time"] = np.arange(801, dtype=np.float64)[:, None] / 10 + offset
        handle["x"] = np.arange(3, dtype=np.float32)
        handle["y"] = np.arange(2, dtype=np.float32)
    norm = PROJECT / "data/curated/tandem_cylinders_matched_start_full40_dev30_v1/normalization.json"
    (root / "normalization.json").write_bytes(norm.read_bytes())
    (root / "manifest.json").write_text(json.dumps({"max_abs_omega": .75}))
    return TandemRolloutDataset(root, "train", 100, stride=1, num_workers=0,
                                force_indices=(0, 1, 2, 3))


class Original(DatasetBase):
    def __init__(self, reference):
        super().__init__(num_workers=0)
        self.reference = reference
        self._datasets = [object(), object(), object()]

    def __len__(self):
        return 1368

    def _load(self, index):
        sample, _ = self.reference[index % 701]
        return sample, {"case": "original", "start": index % 701,
                        "dataset_index": 0, "split": "train", "rollout_steps": 100}

    def close(self):
        pass


class History:
    @staticmethod
    def history_indices(start, k):
        assert k == 1
        return [start], [False]


def main():
    with tempfile.TemporaryDirectory(prefix="p064-d-official-") as directory:
        root = Path(directory)
        b00 = make_view(root / "b00", 148)
        b02 = make_view(root / "b02", 106)
        dataset = schedule.ScheduledABCDataset(
            Original(b00), b00, ORDER, "D", b00_dataset=b00, b02_dataset=b02,
            num_workers=0,
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False,
                            collate_metadata=True, prefetch_factor=0,
                            use_streams=False, seed=20261003)
        selected = {}
        try:
            for position, (batch, metadata) in enumerate(loader):
                if position not in (4, 252):
                    if position >= 252:
                        break
                    continue
                identity = schedule.scheduled_training_identity(metadata)
                sample = {key: value[0] for key, value in batch.items()}
                pre, actions, history = runner.preceding(
                    dataset, identity, sample, 1, History
                )
                selected[position] = (identity, sample, pre, actions, history)
                if position >= 252:
                    break
            assert set(selected) == {4, 252}
            first, last = selected[4], selected[252]
            assert first[0]["dataset_index"] == last[0]["dataset_index"] == 4
            assert first[0]["case"] == last[0]["case"] == "controlled_106"
            assert first[0]["start"] == 0 and last[0]["start"] == 700
            assert first[1]["target_state"].shape[0] == last[1]["target_state"].shape[0] == 100
            assert float(first[1]["omega"][0]) == -1.0
            assert float(last[1]["omega"][0]) == .75
            for record in (first, last):
                assert record[2].shape[1] == record[3].shape[1] == 0
                assert record[4]["frame_indices"] == [record[0]["start"]]
                assert record[4]["full_observed_history"] is True
        finally:
            dataset.close()
    print("P064_D_OFFICIAL_ROUTING_HISTORY_PASS")


if __name__ == "__main__":
    main()
