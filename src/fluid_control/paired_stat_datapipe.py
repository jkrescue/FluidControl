"""Train-only matched-start pair samples backed by PhysicsNeMo HDF5Reader."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import torch
from physicsnemo.datapipes import DatasetBase
from physicsnemo.datapipes.readers.hdf5 import HDF5Reader
from tensordict import TensorDict

HORIZONS = (20, 50, 100)
STAT_NAMES = ("mean_total_cd", "mean_rear_cl", "rear_cl_fluctuation_rms")
FORCE_CHANNELS = ("front_cd", "front_cl", "rear_cd", "rear_cl")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def physical_pair_targets(
    action_force: torch.Tensor, zero_force: torch.Tensor
) -> torch.Tensor:
    """Return H20/H50/H100 action-minus-zero force statistics."""
    rows = []
    for horizon in HORIZONS:
        action = action_force[1 : horizon + 1]
        zero = zero_force[1 : horizon + 1]
        action_rear = action[:, 3]
        zero_rear = zero[:, 3]
        rows.append(
            torch.stack(
                (
                    (action[:, 0] + action[:, 2]).mean()
                    - (zero[:, 0] + zero[:, 2]).mean(),
                    action_rear.mean() - zero_rear.mean(),
                    (action_rear - action_rear.mean()).square().mean().sqrt()
                    - (zero_rear - zero_rear.mean()).square().mean().sqrt(),
                )
            )
        )
    return torch.stack(rows)


class MatchedPairStatDataset(DatasetBase):
    """Sixteen exact matched-start action/zero H100 pairs from train20.

    This adapter deliberately composes the official PhysicsNeMo ``DatasetBase``
    and ``HDF5Reader`` APIs.  It does not copy or mutate curated CFD data.
    """

    def __init__(
        self,
        root: str | Path,
        pair_manifest: str | Path,
        *,
        num_workers: int = 2,
        verify_hdf_sha256: bool = True,
    ) -> None:
        super().__init__(num_workers=num_workers)
        self.root = Path(root).resolve()
        self.manifest_path = Path(pair_manifest).resolve()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self._validate_manifest(verify_hdf_sha256)

        normalization_path = self.root / "normalization.json"
        normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
        if _sha256(normalization_path) != self.manifest["normalization_sha256"]:
            raise ValueError("normalization SHA differs from paired manifest")
        if tuple(normalization["all_force_channels"]) != FORCE_CHANNELS:
            raise ValueError("force channel order differs from paired contract")
        force_norm = {
            "channels": normalization["all_force_channels"],
            "mean": normalization["all_force_mean"],
            "std": normalization["all_force_std"],
        }
        if _json_sha(force_norm) != self.manifest["force_normalization_sha256"]:
            raise ValueError("force normalization payload differs")

        self.state_mean = torch.tensor(normalization["state_mean"], dtype=torch.float32)[:, None, None]
        self.state_std = torch.tensor(normalization["state_std"], dtype=torch.float32)[:, None, None]
        self.force_mean = torch.tensor(normalization["all_force_mean"], dtype=torch.float32)
        self.force_std = torch.tensor(normalization["all_force_std"], dtype=torch.float32)
        self.action_scale = float(self.manifest["max_abs_omega"])
        self.pairs = self.manifest["pairs"]
        self._readers: dict[Path, HDF5Reader] = {}
        self._reader_lock = threading.Lock()

    def _validate_manifest(self, verify_hdf_sha256: bool) -> None:
        required = {
            "status": "TRAIN20_MATCHED_PAIR_DATAPIPE_READY",
            "split": "train",
            "start": 0,
            "sequence_length": 101,
            "dt": 0.1,
            "horizons": list(HORIZONS),
            "force_channels": list(FORCE_CHANNELS),
            "pair_count": 16,
            "validation_or_frozen_accessed": False,
        }
        for key, expected in required.items():
            if self.manifest.get(key) != expected:
                raise ValueError(f"paired manifest {key} differs: {self.manifest.get(key)!r}")
        if float(self.manifest.get("max_abs_omega", -1)) != 0.75:
            raise ValueError("paired action normalization must be 0.75")
        if any((self.root / name).exists() for name in ("frozen_test", "test")):
            raise ValueError("paired adapter root must not expose frozen/test data")
        pairs = self.manifest.get("pairs", [])
        identities = set()
        expected_ids = {
            (f"b{phase:02d}", action)
            for phase in (0, 2, 4, 6)
            for action in ("m075", "m0375", "p0375", "p075")
        }
        for pair in pairs:
            identity = (pair.get("phase"), pair.get("action"))
            if identity in identities:
                raise ValueError(f"duplicate paired identity: {identity}")
            identities.add(identity)
            if pair.get("split") != "train" or pair.get("start") != 0:
                raise ValueError("only train/start0 pairs are allowed")
            action_path = self._safe_train_path(pair["action_file"])
            zero_path = self._safe_train_path(pair["zero_file"])
            if pair.get("initial_sha256") != pair.get("zero_initial_sha256"):
                raise ValueError(f"non-identical matched start for {identity}")
            if verify_hdf_sha256:
                if _sha256(action_path) != pair["action_hdf_sha256"]:
                    raise ValueError(f"action HDF SHA differs: {action_path.name}")
                if _sha256(zero_path) != pair["zero_hdf_sha256"]:
                    raise ValueError(f"zero HDF SHA differs: {zero_path.name}")
        if identities != expected_ids:
            raise ValueError("paired manifest is not the exact 4-phase x 4-action set")

    def _safe_train_path(self, filename: str) -> Path:
        if Path(filename).name != filename or not filename.endswith(".h5"):
            raise ValueError(f"unsafe paired filename: {filename}")
        path = (self.root / "train" / filename).resolve()
        if path.parent != (self.root / "train").resolve() or not path.is_file():
            raise ValueError(f"paired file is not a train HDF: {filename}")
        return path

    def __len__(self) -> int:
        return len(self.pairs)

    def _reader(self, path: Path) -> HDF5Reader:
        with self._reader_lock:
            if path not in self._readers:
                self._readers[path] = HDF5Reader(
                    path, fields=["state", "mask", "omega", "force", "time"]
                )
        return self._readers[path]

    def _sequence(self, filename: str) -> dict[str, torch.Tensor]:
        reader = self._reader(self._safe_train_path(filename))
        frames = [reader[index][0] for index in range(101)]
        return {
            key: torch.stack([frame[key].float() for frame in frames])
            for key in ("state", "mask", "omega", "force", "time")
        }

    def _load(self, index: int) -> tuple[TensorDict, dict]:
        pair = self.pairs[index]
        action = self._sequence(pair["action_file"])
        zero = self._sequence(pair["zero_file"])
        if not torch.equal(action["mask"], zero["mask"]):
            raise ValueError(f"mask mismatch for pair {pair['phase']}/{pair['action']}")
        if not torch.equal(action["state"][0], zero["state"][0]):
            raise ValueError(f"initial state mismatch for pair {pair['phase']}/{pair['action']}")
        if not torch.equal(action["force"][0], zero["force"][0]):
            raise ValueError(f"initial force mismatch for pair {pair['phase']}/{pair['action']}")
        if not torch.allclose(action["time"], zero["time"], rtol=0, atol=2e-5):
            raise ValueError(f"time grid mismatch for pair {pair['phase']}/{pair['action']}")
        expected_time = action["time"][0] + torch.arange(101, dtype=torch.float32)[:, None] * 0.1
        if not torch.allclose(action["time"], expected_time, rtol=0, atol=2e-5):
            raise ValueError(f"non-canonical dt for pair {pair['phase']}/{pair['action']}")

        targets = torch.tensor(
            [[pair["targets"][str(h)][name] for name in STAT_NAMES] for h in HORIZONS],
            dtype=torch.float32,
        )
        recomputed_targets = physical_pair_targets(action["force"], zero["force"])
        if not torch.allclose(targets, recomputed_targets, rtol=2e-5, atol=2e-6):
            raise ValueError(
                f"paired targets differ from force sequence for "
                f"{pair['phase']}/{pair['action']}"
            )
        action_state = (action["state"] - self.state_mean[None]) / self.state_std[None]
        zero_state = (zero["state"] - self.state_mean[None]) / self.state_std[None]
        action_state *= action["mask"]
        zero_state *= zero["mask"]
        sample = TensorDict(
            {
                "action_state": action_state,
                "zero_state": zero_state,
                "action_omega": action["omega"] / self.action_scale,
                "zero_omega": zero["omega"] / self.action_scale,
                "action_force": (action["force"] - self.force_mean[None]) / self.force_std[None],
                "zero_force": (zero["force"] - self.force_mean[None]) / self.force_std[None],
                "mask": action["mask"][0],
                "paired_targets": targets,
            },
            batch_size=[],
        )
        if not all(torch.isfinite(value).all() for value in sample.values()):
            raise ValueError(f"non-finite paired sample {pair['phase']}/{pair['action']}")
        metadata = {
            "phase": pair["phase"],
            "action": pair["action"],
            "case": Path(pair["action_file"]).stem,
            "zero_case": Path(pair["zero_file"]).stem,
            "split": "train",
            "start": 0,
            "horizons": HORIZONS,
        }
        return sample, metadata

    def close(self) -> None:
        super().close()
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()

    def __del__(self) -> None:
        self.close()
