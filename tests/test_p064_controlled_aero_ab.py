from __future__ import annotations

import json
from pathlib import Path
import sys
import types

import pytest
import torch

try:
    from physicsnemo.datapipes import DatasetBase, MultiDataset
    USING_OFFICIAL_PHYSICSNEMO = True
except ModuleNotFoundError:
    class DatasetBase:
        """Dependency mock only; a separate pinned-env run is mandatory."""

        def __init__(self, num_workers=0):
            self.num_workers = num_workers

        def __getitem__(self, index):
            return self._load(index)

        def close(self):
            pass

    physicsnemo = types.ModuleType("physicsnemo")
    datapipes = types.ModuleType("physicsnemo.datapipes")
    datapipes.DatasetBase = DatasetBase
    physicsnemo.datapipes = datapipes
    sys.modules.setdefault("physicsnemo", physicsnemo)
    sys.modules.setdefault("physicsnemo.datapipes", datapipes)
    MultiDataset = None
    USING_OFFICIAL_PHYSICSNEMO = False

from fluid_control.p064_controlled_aero_ab import (
    B00_WINDOWS,
    PARENT_ORDER_SHA256,
    PARENT_PREFIX_256_SHA256,
    ScheduledABDataset,
    canonical_sha256,
    compile_schedule,
    evenly_spaced_indices,
    protocol_summary,
)


ROOT = Path(__file__).resolve().parents[1]


def actual_parent_order() -> list[int]:
    path = Path(__file__).with_name("p064_parent_order.json")
    return json.loads(path.read_text())


class ToyOriginal:
    def __init__(self):
        self._datasets = [object(), object(), object()]

    def __len__(self):
        return 1368

    def __getitem__(self, index):
        return {"value": index}, {"case": "original", "dataset_index": index % 3}

    def close(self):
        pass


class ToyReference:
    action_scale = 0.75
    state_mean = torch.tensor([1.0, 2.0, 3.0])[:, None, None]
    state_std = torch.tensor([4.0, 5.0, 6.0])[:, None, None]
    force_mean = torch.tensor([1.0, 2.0, 3.0, 4.0])
    force_std = torch.tensor([5.0, 6.0, 7.0, 8.0])


class ToyB00(ToyReference):
    rollout_steps = 100
    force_indices = (0, 1, 2, 3)
    index = [(0, start) for start in range(701)]

    def __getitem__(self, index):
        return {"value": index}, {"case": "controlled_b00", "step": index}

    def close(self):
        pass


def test_parent_order_fixture_is_actual_p026_prefix():
    order = actual_parent_order()
    assert len(order) == 1368
    assert canonical_sha256(order) == PARENT_ORDER_SHA256
    assert canonical_sha256(order[:256]) == PARENT_PREFIX_256_SHA256


def test_fixed_budget_and_two_replacements_per_update():
    order = actual_parent_order()
    a = compile_schedule(order, "A")
    b = compile_schedule(order, "B")
    assert len(a) == len(b) == 256
    assert all(row.source == "original44" for row in a)
    replacements = [row for row in b if row.source == "controlled_b00"]
    assert len(replacements) == B00_WINDOWS == 64
    for update in range(32):
        rows = b[update * 8 : (update + 1) * 8]
        assert [row.within_update for row in rows if row.b00_start is not None] == [0, 4]
    assert [row.original_global_index for row in a] == [
        row.original_global_index for row in b
    ]


def test_b00_starts_are_predeclared_even_and_cover_endpoints():
    starts = [
        row.b00_start
        for row in compile_schedule(actual_parent_order(), "B")
        if row.b00_start is not None
    ]
    assert starts == list(evenly_spaced_indices(701, 64))
    assert starts[0] == 0 and starts[-1] == 700
    assert set(b - a for a, b in zip(starts, starts[1:])) == {11, 12}


def test_adapter_routes_a_and_b_without_changing_counterfactual_order():
    order = actual_parent_order()
    a = ScheduledABDataset(ToyOriginal(), ToyReference(), order, "A")
    b = ScheduledABDataset(
        ToyOriginal(), ToyReference(), order, "B", b00_dataset=ToyB00()
    )
    sample_a, meta_a = a._load(0)
    sample_b, meta_b = b._load(0)
    assert sample_a["value"] == order[0]
    assert meta_a["ab_source"] == "original44"
    assert sample_b["value"] == 0
    assert meta_b["dataset_index"] == 3
    assert meta_b["ab_counterfactual_original_global_index"] == order[0]
    sample_b1, meta_b1 = b._load(1)
    assert sample_b1["value"] == order[1]
    assert meta_b1["ab_source"] == "original44"


def test_adapter_rejects_normalization_or_window_contract_changes():
    order = actual_parent_order()
    bad = ToyB00()
    bad.action_scale = 1.0
    with pytest.raises(ValueError, match="action normalization"):
        ScheduledABDataset(ToyOriginal(), ToyReference(), order, "B", b00_dataset=bad)
    bad = ToyB00()
    bad.index = bad.index[:-1]
    with pytest.raises(ValueError, match="801-frame"):
        ScheduledABDataset(ToyOriginal(), ToyReference(), order, "B", b00_dataset=bad)


def test_fail_closed_identity_and_integer_validation():
    order = actual_parent_order()
    with pytest.raises(ValueError, match="sampler order SHA"):
        compile_schedule(order[1:] + order[:1], "A")
    with pytest.raises(TypeError, match="integer"):
        evenly_spaced_indices(True, 2)
    with pytest.raises(ValueError, match="arm"):
        compile_schedule(order, "C")


def test_protocol_is_preparation_only_and_equal_budget():
    value = protocol_summary(actual_parent_order())
    assert value["training_windows_per_arm"] == 256
    assert value["optimizer_steps_per_arm"] == 32
    assert value["b00_windows_arm_a"] == 0
    assert value["b00_windows_arm_b"] == 64
    assert value["b00_weight_arm_b"] == 0.25
    assert value["arm_a_original_family_windows"] == {
        "base": 140,
        "train8": 64,
        "train16": 52,
    }
    assert value["arm_b_original_family_windows"] == {
        "base": 109,
        "train8": 45,
        "train16": 38,
    }
    assert value["selection_performed"] is False
    assert value["gpu_execution_authorized"] is False


@pytest.mark.skipif(not USING_OFFICIAL_PHYSICSNEMO, reason="pinned PhysicsNeMo absent")
def test_actual_physicsnemo_multidataset_getitem_contract():
    class Child(DatasetBase):
        def __init__(self, family, count):
            super().__init__(num_workers=0)
            self.family = family
            self.count = count

        def __len__(self):
            return self.count

        def _load(self, index):
            return {"value": index}, {
                "case": self.family,
                "start": index,
                "split": "train",
            }

    counts = (720, 408, 240)
    children = [Child(f"family{i}", count) for i, count in enumerate(counts)]
    original = MultiDataset(*children, output_strict=True)
    adapter = ScheduledABDataset(
        original, ToyReference(), actual_parent_order(), "A", num_workers=0
    )
    sample, metadata = adapter._load(0)
    assert sample["value"] == actual_parent_order()[0]
    assert metadata["dataset_index"] == 0
    assert metadata["ab_source"] == "original44"
    assert not hasattr(original, "_load")
