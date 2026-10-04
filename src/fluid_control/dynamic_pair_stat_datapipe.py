"""Train-only dynamic matched-pair adapter using official PhysicsNeMo readers."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import torch
from physicsnemo.datapipes import DatasetBase
from physicsnemo.datapipes.readers.hdf5 import HDF5Reader
from tensordict import TensorDict

from .paired_stat_datapipe import (
    FORCE_CHANNELS,
    HORIZONS,
    STAT_NAMES,
    physical_pair_targets,
)

SOURCE_KEYS = {"U", "U_0", "p", "phi", "phi_0"}
STATE_ATOL = 3e-7


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class DynamicMatchedPairStatDataset(DatasetBase):
    """Eight existing train-only dynamic-action/zero H100 pairs.

    Action and zero trajectories live in separate curated roots.  The project
    adapter composes official ``DatasetBase`` and ``HDF5Reader`` APIs without
    changing either curated release.
    """

    def __init__(
        self,
        action_root: str | Path,
        zero_root: str | Path,
        pair_manifest: str | Path,
        *,
        num_workers: int = 2,
        verify_hdf_sha256: bool = True,
    ) -> None:
        super().__init__(num_workers=num_workers)
        self._readers: dict[Path, HDF5Reader] = {}
        self._reader_lock = threading.Lock()
        self.action_root = Path(action_root).resolve()
        self.zero_root = Path(zero_root).resolve()
        self.manifest_path = Path(pair_manifest).resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        self._validate_manifest(verify_hdf_sha256)

        action_norm_path = self.action_root / "normalization.json"
        zero_norm_path = self.zero_root / "normalization.json"
        action_norm_bytes = action_norm_path.read_bytes()
        if action_norm_bytes != zero_norm_path.read_bytes():
            raise ValueError("action and zero normalization files differ")
        if hashlib.sha256(action_norm_bytes).hexdigest() != self.manifest["normalization_sha256"]:
            raise ValueError("normalization SHA differs from dynamic pair manifest")
        normalization = json.loads(action_norm_bytes)
        if tuple(normalization["all_force_channels"]) != FORCE_CHANNELS:
            raise ValueError("force channel order differs from dynamic pair contract")
        self.state_mean = torch.tensor(normalization["state_mean"], dtype=torch.float32)[:, None, None]
        self.state_std = torch.tensor(normalization["state_std"], dtype=torch.float32)[:, None, None]
        self.force_mean = torch.tensor(normalization["all_force_mean"], dtype=torch.float32)
        self.force_std = torch.tensor(normalization["all_force_std"], dtype=torch.float32)
        self.action_scale = 0.75
        self.pairs = self.manifest["pairs"]

    def _safe_path(self, root: Path, relative: str) -> Path:
        value = Path(relative)
        if value.parts[:1] != ("train",) or len(value.parts) != 2 or value.suffix != ".h5":
            raise ValueError(f"unsafe/non-train dynamic pair path: {relative}")
        path = (root / value).resolve()
        if path.parent != (root / "train").resolve() or not path.is_file():
            raise ValueError(f"dynamic pair file missing: {relative}")
        return path

    def _validate_manifest(self, verify_hdf_sha256: bool) -> None:
        required = {
            "status": "FC_P003_DYNAMIC8_PAIR_CANDIDATE_QC_PASS",
            "scope": "train-only existing-data candidate; not approved for training",
            "split": "train",
            "pair_count": 8,
            "sequence_length": 101,
            "horizon": 100,
            "horizons": list(HORIZONS),
            "force_channels": list(FORCE_CHANNELS),
            "state0_tolerance": {"rtol": 0.0, "atol": STATE_ATOL},
            "validation_or_frozen_accessed": False,
            "unique_initial_restart_count": 4,
            "profiles_per_phase": ["multisine", "prbs"],
        }
        for key, expected in required.items():
            if self.manifest.get(key) != expected:
                raise ValueError(f"dynamic pair manifest {key} differs")
        expected_ids = {(f"b{phase:02d}", profile) for phase in (0, 2, 4, 6) for profile in ("multisine", "prbs")}
        identities = set()
        for pair in self.manifest.get("pairs", []):
            identity = (pair.get("phase"), pair.get("profile"))
            if identity in identities:
                raise ValueError(f"duplicate dynamic pair identity: {identity}")
            identities.add(identity)
            if pair.get("split") != "train" or pair.get("start") != 0 or pair.get("horizon") != 100:
                raise ValueError("dynamic pairs must be train/start0/H100")
            action_source = pair.get("action_source_state_sha256", {})
            zero_source = pair.get("zero_source_state_sha256", {})
            if set(action_source) != SOURCE_KEYS or action_source != zero_source:
                raise ValueError(f"source restart SHA differs for {identity}")
            action_path = self._safe_path(self.action_root, pair["action_file"])
            zero_path = self._safe_path(self.zero_root, pair["zero_file"])
            if verify_hdf_sha256 and (
                _sha256(action_path) != pair["action_hdf_sha256"]
                or _sha256(zero_path) != pair["zero_hdf_sha256"]
            ):
                raise ValueError(f"dynamic pair HDF SHA differs for {identity}")
        if identities != expected_ids:
            raise ValueError("dynamic manifest is not exact 4-phase x 2-profile set")

    def __len__(self) -> int:
        return len(self.pairs)

    def _reader(self, path: Path) -> HDF5Reader:
        with self._reader_lock:
            if path not in self._readers:
                self._readers[path] = HDF5Reader(path, fields=["state", "mask", "omega", "force", "time"])
        return self._readers[path]

    def _sequence(self, root: Path, relative: str) -> dict[str, torch.Tensor]:
        reader = self._reader(self._safe_path(root, relative))
        frames = [reader[index][0] for index in range(101)]
        return {key: torch.stack([frame[key].float() for frame in frames]) for key in ("state", "mask", "omega", "force", "time")}

    def _load(self, index: int):
        pair = self.pairs[index]
        action = self._sequence(self.action_root, pair["action_file"])
        zero = self._sequence(self.zero_root, pair["zero_file"])
        identity = f"{pair['phase']}/{pair['profile']}"
        if not torch.allclose(action["state"][0], zero["state"][0], rtol=0, atol=STATE_ATOL):
            raise ValueError(f"state0 tolerance failed for {identity}")
        if not torch.equal(action["mask"], zero["mask"]):
            raise ValueError(f"H100 mask mismatch for {identity}")
        if not torch.equal(action["force"][0], zero["force"][0]):
            raise ValueError(f"initial force mismatch for {identity}")
        if not torch.allclose(action["time"], zero["time"], rtol=0, atol=2e-5):
            raise ValueError(f"H100 time mismatch for {identity}")
        if not torch.allclose(zero["omega"], torch.zeros_like(zero["omega"]), rtol=0, atol=1e-7):
            raise ValueError(f"zero action mismatch for {identity}")
        targets = torch.tensor([[pair["targets"][str(h)][name] for name in STAT_NAMES] for h in HORIZONS], dtype=torch.float32)
        recomputed = physical_pair_targets(action["force"], zero["force"])
        if not torch.allclose(targets, recomputed, rtol=2e-5, atol=2e-6):
            raise ValueError(f"dynamic paired targets differ for {identity}")
        action_state = (action["state"] - self.state_mean[None]) / self.state_std[None]
        zero_state = (zero["state"] - self.state_mean[None]) / self.state_std[None]
        action_state *= action["mask"]
        zero_state *= zero["mask"]
        sample = TensorDict({
            "action_state": action_state, "zero_state": zero_state,
            "action_omega": action["omega"] / self.action_scale,
            "zero_omega": zero["omega"] / self.action_scale,
            "action_force": (action["force"] - self.force_mean[None]) / self.force_std[None],
            "zero_force": (zero["force"] - self.force_mean[None]) / self.force_std[None],
            "mask": action["mask"][0], "paired_targets": targets,
        }, batch_size=[])
        if not all(torch.isfinite(value).all() for value in sample.values()):
            raise ValueError(f"non-finite dynamic pair {identity}")
        return sample, {
            "pair_id": f"{pair['phase']}:{pair['profile']}",
            "phase": pair["phase"],
            "profile": pair["profile"],
            "split": "train",
            "start": 0,
            "horizon": 100,
        }

    def close(self) -> None:
        super().close()
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()

    def __del__(self) -> None:
        self.close()
