"""Fixed-budget A/B/C data schedule for a controlled-aerodynamic continuation.

This project adapter does not implement a new model, reader, or loss.  It
selects a predeclared 256-window prefix from the already-audited P026 sampler
Arm B replaces two positions and arm C replaces four positions in every
eight-window optimizer update with mechanically selected windows from one
reviewed b00 trajectory. Arm C changes schedule density only.
The wrapped datasets remain the official PhysicsNeMo HDF5Reader-backed
datasets used by the parent trainer.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from collections.abc import Mapping
from typing import Any, Literal, Sequence

import torch
from physicsnemo.datapipes import DatasetBase


PARENT_ORDER_SHA256 = "177ebd9523cde918eb0c1fb8026286dac7e95f3d228ae0e349757a2a9a288f9f"
PARENT_PREFIX_256_SHA256 = "06c922e8f1476af52705fcc88521bc039d3fa6bd7a9c00b130179af712e36691"
PARENT_RUN_LOG_SHA256 = "9ce79ec016d57237b0945190943b4aa0a3f766a28c7e806ffe51dede332b690a"
PARENT_RESOURCE_WATCH_SHA256 = "9fe5533e0e15a5e05100a36782520e85deed7cdfb1f5a20962bd5447edc666ca"
PARENT_WINDOWS = 1368
TRAINING_WINDOWS = 256
ACCUMULATION_WINDOWS = 8
OPTIMIZER_STEPS = 32
B00_WINDOWS = 64
C00_WINDOWS = 128
B00_VALID_STARTS = 701
ROLLOUT_STEPS = 100


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    return value


def evenly_spaced_indices(length: int, count: int) -> tuple[int, ...]:
    """Return inclusive, integer-only, mechanically even indices."""
    length = _integer("length", length)
    count = _integer("count", count)
    if length < 1 or count < 1 or count > length:
        raise ValueError(f"invalid evenly-spaced request: length={length}, count={count}")
    if count == 1:
        return (0,)
    return tuple(index * (length - 1) // (count - 1) for index in range(count))


@dataclass(frozen=True)
class ScheduleRow:
    consumed_position: int
    update: int
    within_update: int
    original_global_index: int
    source: Literal["original44", "controlled_b00"]
    b00_start: int | None


def compile_schedule(
    parent_order: Sequence[int], arm: Literal["A", "B", "C"]
) -> tuple[ScheduleRow, ...]:
    """Compile the exact 32-update schedule; no sampling occurs here."""
    if arm not in ("A", "B", "C"):
        raise ValueError("arm must be exactly 'A', 'B', or 'C'")
    order = tuple(_integer("parent_order entry", value) for value in parent_order)
    if len(order) != PARENT_WINDOWS or len(set(order)) != PARENT_WINDOWS:
        raise ValueError("parent order must be a permutation of 1368 entries")
    if set(order) != set(range(PARENT_WINDOWS)):
        raise ValueError("parent order must contain exactly global indices 0..1367")
    if canonical_sha256(order) != PARENT_ORDER_SHA256:
        raise ValueError("parent sampler order SHA differs from actual P026 K1")
    prefix = order[:TRAINING_WINDOWS]
    if canonical_sha256(prefix) != PARENT_PREFIX_256_SHA256:
        raise ValueError("256-window parent prefix SHA differs")

    replacement_stride = {"A": None, "B": 4, "C": 2}[arm]
    replacement_positions = (
        set() if replacement_stride is None
        else set(range(0, TRAINING_WINDOWS, replacement_stride))
    )
    controlled_windows = {"A": 0, "B": B00_WINDOWS, "C": C00_WINDOWS}[arm]
    starts = iter(evenly_spaced_indices(B00_VALID_STARTS, controlled_windows)) if controlled_windows else iter(())
    rows = []
    for position, original_index in enumerate(prefix):
        replace = position in replacement_positions
        rows.append(
            ScheduleRow(
                consumed_position=position,
                update=position // ACCUMULATION_WINDOWS,
                within_update=position % ACCUMULATION_WINDOWS,
                original_global_index=original_index,
                source="controlled_b00" if replace else "original44",
                b00_start=next(starts) if replace else None,
            )
        )
    if arm in ("B", "C") and next(starts, None) is not None:
        raise AssertionError("not all b00 starts were consumed")
    return tuple(rows)


def schedule_sha256(rows: Sequence[ScheduleRow]) -> str:
    return canonical_sha256([asdict(row) for row in rows])


def scheduled_training_identity(metadata: Any) -> dict[str, Any]:
    """Normalize one official collated row, allowing reviewed b00 index 3."""
    if isinstance(metadata, list) and len(metadata) == 1 and isinstance(metadata[0], dict):
        row = metadata[0]
    elif isinstance(metadata, Mapping):
        lengths = {
            len(value)
            for value in metadata.values()
            if isinstance(value, (list, tuple))
        }
        if lengths != {1}:
            raise ValueError("expected one collated metadata row")
        row = {
            key: value[0] if isinstance(value, (list, tuple)) else value
            for key, value in metadata.items()
        }
    else:
        raise ValueError("expected one-item metadata")
    result = {
        "case": str(row.get("case", "")),
        "start": int(row.get("start", row.get("step", -1))),
        "dataset_index": int(row.get("dataset_index", -1)),
        "split": str(row.get("split", "")),
        "rollout_steps": int(row.get("rollout_steps", -1)),
    }
    if (
        not result["case"]
        or result["start"] < 0
        or result["dataset_index"] not in (0, 1, 2, 3)
        or result["split"] != "train"
        or result["rollout_steps"] != ROLLOUT_STEPS
    ):
        raise ValueError("scheduled training metadata identity is incomplete")
    return result


def _same_tensor(name: str, left: Any, right: Any) -> None:
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        raise TypeError(f"{name} must be tensors")
    if not torch.equal(left, right):
        raise ValueError(f"{name} differs from parent train-only normalization")


class ScheduledABCDataset(DatasetBase):
    """Resolve a fixed schedule through existing official datasets.

    The caller must use ``shuffle=False``: the schedule itself is the audited
    order.  Arm A never touches ``b00_dataset``.  Arm B requires a single-file
    801-frame dataset whose H100 index is exactly starts 0..700.
    """

    def __init__(
        self,
        original_dataset: Any,
        reference_dataset: Any,
        parent_order: Sequence[int],
        arm: Literal["A", "B", "C"],
        *,
        b00_dataset: Any | None = None,
        num_workers: int = 0,
    ) -> None:
        super().__init__(num_workers=num_workers)
        if len(original_dataset) != PARENT_WINDOWS:
            raise ValueError("original dataset must retain all 1368 parent windows")
        self.original_dataset = original_dataset
        self.reference_dataset = reference_dataset
        self.b00_dataset = b00_dataset
        self.arm = arm
        self.schedule = compile_schedule(parent_order, arm)
        self._datasets = list(getattr(original_dataset, "_datasets", ()))
        if len(self._datasets) != 3:
            raise ValueError("original dataset must retain the three audited families")
        if arm in ("B", "C"):
            self._validate_b00()
            self._datasets.append(b00_dataset)
            self._b00_local = {start: start for start in range(B00_VALID_STARTS)}
        elif b00_dataset is not None:
            raise ValueError("arm A must not expose controlled b00 data")

    def _validate_b00(self) -> None:
        data = self.b00_dataset
        if data is None:
            raise ValueError("controlled-data arms require b00_dataset")
        if getattr(data, "rollout_steps", None) != ROLLOUT_STEPS:
            raise ValueError("b00 rollout_steps must be exactly 100")
        if getattr(data, "force_indices", None) != (0, 1, 2, 3):
            raise ValueError("b00 must expose all four force channels")
        if getattr(data, "index", None) != [
            (0, start) for start in range(B00_VALID_STARTS)
        ]:
            raise ValueError("b00 must be one 801-frame trajectory with H100 starts 0..700")
        if float(data.action_scale) != float(self.reference_dataset.action_scale):
            raise ValueError("b00 action normalization differs")
        for name in ("state_mean", "state_std", "force_mean", "force_std"):
            _same_tensor(name, getattr(data, name), getattr(self.reference_dataset, name))

    def __len__(self) -> int:
        return TRAINING_WINDOWS

    def _load(self, index: int):
        index = _integer("index", index)
        if index < 0 or index >= len(self.schedule):
            raise IndexError(index)
        row = self.schedule[index]
        if row.source == "original44":
            sample, metadata = self.original_dataset[row.original_global_index]
        else:
            sample, metadata = self.b00_dataset[self._b00_local[row.b00_start]]
            metadata = dict(metadata)
            metadata["dataset_index"] = 3
        metadata = dict(metadata)
        metadata.update(
            ab_arm=self.arm,
            ab_consumed_position=row.consumed_position,
            ab_update=row.update,
            ab_within_update=row.within_update,
            ab_source=row.source,
            ab_counterfactual_original_global_index=row.original_global_index,
            ab_b00_start=-1 if row.b00_start is None else row.b00_start,
        )
        return sample, metadata

    def close(self) -> None:
        super().close()
        self.original_dataset.close()
        if self.b00_dataset is not None:
            self.b00_dataset.close()


def protocol_summary(parent_order: Sequence[int]) -> dict[str, Any]:
    a = compile_schedule(parent_order, "A")
    b = compile_schedule(parent_order, "B")
    c = compile_schedule(parent_order, "C")
    return {
        "status": "FC_P064_CONTROLLED_AERO_AB_CPU_PREPARATION_NOT_APPROVED",
        "parent_sampler_order_sha256": PARENT_ORDER_SHA256,
        "parent_prefix_256_sha256": PARENT_PREFIX_256_SHA256,
        "parent_run_log_sha256": PARENT_RUN_LOG_SHA256,
        "parent_resource_watch_sha256": PARENT_RESOURCE_WATCH_SHA256,
        "training_windows_per_arm": TRAINING_WINDOWS,
        "optimizer_steps_per_arm": OPTIMIZER_STEPS,
        "accumulation_windows": ACCUMULATION_WINDOWS,
        "b00_windows_arm_a": 0,
        "b00_windows_arm_b": B00_WINDOWS,
        "b00_weight_arm_b": 0.25,
        "b00_windows_arm_c": C00_WINDOWS,
        "b00_weight_arm_c": 0.5,
        "replacement_within_each_update_arm_b": [0, 4],
        "replacement_within_each_update_arm_c": [0, 2, 4, 6],
        "arm_a_original_family_windows": {"base": 140, "train8": 64, "train16": 52},
        "arm_b_original_family_windows": {"base": 109, "train8": 45, "train16": 38},
        "arm_a_schedule_sha256": schedule_sha256(a),
        "arm_b_schedule_sha256": schedule_sha256(b),
        "arm_c_schedule_sha256": schedule_sha256(c),
        "selection_performed": False,
        "gpu_execution_authorized": False,
    }
