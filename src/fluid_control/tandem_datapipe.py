"""PhysicsNeMo HDF5Reader-backed windows for tandem-cylinder training."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import h5py
import torch
from physicsnemo.datapipes import DatasetBase
from physicsnemo.datapipes.readers.hdf5 import HDF5Reader
from tensordict import TensorDict


class TandemWindowDataset(DatasetBase):
    """PhysicsNeMo map dataset for action-conditioned temporal windows."""

    def __init__(self, root: str | Path, split: str, stride: int = 1,
                 num_workers: int = 2) -> None:
        super().__init__(num_workers=num_workers)
        self.root = Path(root)
        self.split = split
        self.paths = sorted((self.root / split).glob("*.h5"))
        if not self.paths:
            raise FileNotFoundError(f"no curated trajectories for split={split}: {self.root}")
        self.stats = json.loads((self.root / "normalization.json").read_text(encoding="utf-8"))
        self.state_mean = torch.tensor(self.stats["state_mean"], dtype=torch.float32)[:, None, None]
        self.state_std = torch.tensor(self.stats["state_std"], dtype=torch.float32)[:, None, None]
        self.force_mean = torch.tensor(self.stats["force_mean"], dtype=torch.float32)
        self.force_std = torch.tensor(self.stats["force_std"], dtype=torch.float32)
        self.index = []
        for file_index, path in enumerate(self.paths):
            with h5py.File(path, "r") as handle:
                count = len(handle["state"])
            self.index.extend((file_index, step) for step in range(0, count - 1, stride))
        self._readers: dict[int, HDF5Reader] = {}
        self._reader_lock = threading.Lock()

    def __len__(self) -> int:
        return len(self.index)

    def _reader(self, file_index: int) -> HDF5Reader:
        with self._reader_lock:
            if file_index not in self._readers:
                self._readers[file_index] = HDF5Reader(
                    self.paths[file_index],
                    fields=["state", "mask", "omega", "force", "time"],
                )
        return self._readers[file_index]

    def _load(self, index: int) -> tuple[TensorDict, dict]:
        file_index, step = self.index[index]
        reader = self._reader(file_index)
        current, _ = reader[step]
        following, _ = reader[step + 1]
        mask = current["mask"].float()
        state = (current["state"].float() - self.state_mean) / self.state_std
        next_state = (following["state"].float() - self.state_mean) / self.state_std
        state *= mask
        next_state *= mask
        height, width = mask.shape[-2:]
        omega_now = current["omega"].float().reshape(1, 1, 1).expand(1, height, width)
        omega_next = following["omega"].float().reshape(1, 1, 1).expand(1, height, width)
        inputs = torch.cat([state, mask, omega_now, omega_next], dim=0)
        force = (following["force"].float()[2:4] - self.force_mean) / self.force_std
        sample = TensorDict({
            "x": inputs,
            "delta": next_state - state,
            "force": force,
            "mask": mask,
            "time": current["time"].float(),
        }, batch_size=[])
        metadata = {
            "case": self.paths[file_index].stem,
            "step": step,
            "split": self.split,
        }
        return sample, metadata

    def close(self) -> None:
        super().close()
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()

    def __del__(self) -> None:
        self.close()


class TandemRolloutDataset(TandemWindowDataset):
    """PhysicsNeMo dataset that returns contiguous autoregressive windows."""

    def __init__(self, root: str | Path, split: str, rollout_steps: int,
                 stride: int = 1, num_workers: int = 2) -> None:
        if rollout_steps < 1:
            raise ValueError(f"rollout_steps must be positive, got {rollout_steps}")
        super().__init__(root, split, stride=1, num_workers=num_workers)
        self.rollout_steps = int(rollout_steps)
        self.index = []
        for file_index, path in enumerate(self.paths):
            with h5py.File(path, "r") as handle:
                count = len(handle["state"])
            self.index.extend(
                (file_index, step)
                for step in range(0, count - self.rollout_steps, stride)
            )

    def _load(self, index: int) -> tuple[TensorDict, dict]:
        file_index, step = self.index[index]
        reader = self._reader(file_index)
        frames = [reader[offset][0] for offset in range(step, step + self.rollout_steps + 1)]
        mask = frames[0]["mask"].float()

        states = torch.stack([frame["state"].float() for frame in frames])
        normalized = (states - self.state_mean[None]) / self.state_std[None]
        normalized *= mask[None]
        omega = torch.stack([frame["omega"].float().reshape(1) for frame in frames])
        forces = torch.stack([frame["force"].float()[2:4] for frame in frames[1:]])
        forces = (forces - self.force_mean[None]) / self.force_std[None]

        sample = TensorDict({
            "state": normalized[0],
            "target_state": normalized[1:],
            "omega": omega,
            "target_force": forces,
            "mask": mask,
            "time": frames[0]["time"].float(),
        }, batch_size=[])
        metadata = {
            "case": self.paths[file_index].stem,
            "step": step,
            "rollout_steps": self.rollout_steps,
            "split": self.split,
        }
        return sample, metadata
