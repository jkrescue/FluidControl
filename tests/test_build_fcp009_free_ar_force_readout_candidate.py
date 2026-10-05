from dataclasses import replace

import numpy as np
import pytest

from build_fcp008_force_readout_candidate import Trajectory
from build_fcp009_free_ar_force_readout_candidate import (
    FAMILY_WINDOWS,
    TOTAL_ROWS,
    canonical_force_targets,
    cross_domain_phase_oof,
    expand_matched_h1_cache,
    free_ar_batch,
    phase_oof_alpha_zero,
    regular_window_records,
    window_inventory_payload,
)


def records():
    rows = []
    for phase in ("b00", "b02", "b04", "b06"):
        for action in range(5):
            rows.append(Trajectory("base20", f"base_{phase}_{action}", None, "x", 801, phase))
        for profile in range(2):
            rows.append(Trajectory("train8", f"dynamic_{phase}_{profile}", None, "x", 201, phase))
    for phase in ("b00", "b02"):
        for episode in range(8):
            rows.append(Trajectory("train16", f"ppo_{phase}_{episode}", None, "x", 129, phase))
    return rows


def test_exact_regular_window_and_physical_endpoint_disclosure():
    windows = regular_window_records(records())
    assert len(windows) == 1368
    assert {family: sum(row.family == family for row in windows) for family in FAMILY_WINDOWS} == FAMILY_WINDOWS
    payload = window_inventory_payload(records())
    assert payload["window_step_row_count"] == TOTAL_ROWS == 136800
    assert payload["unique_cfd_endpoint_count"] == 19648
    assert payload["rows_are_independent_physical_samples"] is False
    assert payload["family"]["base20"]["endpoint_multiplicity_max"] == 5
    assert payload["family"]["train8"]["endpoint_multiplicity_max"] == 50
    assert payload["family"]["train16"]["endpoint_multiplicity_max"] == 15


def test_window_inventory_rejects_changed_length():
    changed = records()
    changed[0] = replace(changed[0], frames=800)
    with pytest.raises(ValueError, match="inventory differs"):
        regular_window_records(changed)


def test_alpha_zero_oof_is_fixed_not_selected():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(24, 128))
    y = rng.normal(size=(24, 4))
    phases = np.repeat(np.array(["b00", "b02", "b04", "b06"]), 6)
    result = phase_oof_alpha_zero(x, y, phases)
    assert result["alpha"] == 0.0
    assert result["selection_performed"] is False
    assert set(result["fold_fits"]) == set(phases)


def test_cross_domain_oof_has_all_four_directions_and_fixed_horizons():
    rng = np.random.default_rng(11)
    rows = 4 * 100
    free_ar = rng.normal(size=(rows, 128))
    h1 = free_ar + 0.1 * rng.normal(size=free_ar.shape)
    targets = rng.normal(size=(rows, 4))
    phases = np.repeat(np.array(["b00", "b02", "b04", "b06"]), 100)
    steps = np.tile(np.arange(1, 101), 4)
    result = cross_domain_phase_oof(free_ar, h1, targets, phases, steps)
    assert result["alpha"] == 0.0
    assert result["selection_performed"] is False
    assert set(result["reports"]) == {
        "fit_free_ar_predict_free_ar",
        "fit_free_ar_predict_h1",
        "fit_h1_predict_free_ar",
        "fit_h1_predict_h1",
    }
    for report in result["reports"].values():
        assert set(report) == {"normalized", "physical"}
        assert set(report["normalized"]["relative_horizons"]) == {"H1", "H10", "H50", "H100"}


def test_matched_h1_expansion_preserves_targets_and_duplicate_weighting():
    cache = {
        "features": np.arange(4 * 128, dtype=np.float32).reshape(4, 128),
        "targets_normalized": np.arange(16, dtype=np.float32).reshape(4, 4),
        "phases": np.array(["b00", "b00", "b02", "b02"]),
        "families": np.array(["base20"] * 4),
        "trajectory_names": np.array(["a", "a", "b", "b"]),
    }
    names = np.array(["a", "a", "a", "b"])
    starts = np.array([0, 0, 1, 0])
    steps = np.array([1, 2, 1, 2])
    indices = np.array([0, 1, 1, 3])
    result = expand_matched_h1_cache(
        cache,
        names,
        starts,
        steps,
        cache["targets_normalized"][indices],
        cache["phases"][indices],
        cache["families"][indices],
    )
    np.testing.assert_array_equal(result["indices"], indices)
    assert result["unique_source_rows"] == 3
    np.testing.assert_array_equal(result["features"], cache["features"][indices])


def test_matched_h1_rejects_target_mismatch():
    cache = {
        "features": np.zeros((19648, 128), np.float32),
        "targets_normalized": np.zeros((19648, 4), np.float32),
        "phases": np.array(["b00"] * 19648),
        "families": np.array(["base20"] * 19648),
        "trajectory_names": np.array(["a"] * 19648),
    }
    with pytest.raises(ValueError, match="targets differ"):
        expand_matched_h1_cache(
            cache,
            np.array(["a"]),
            np.array([0]),
            np.array([1]),
            np.ones((1, 4), np.float32),
            np.array(["b00"]),
            np.array(["base20"]),
        )


def test_canonical_force_target_uses_float64_then_float32(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "case.h5"
    force = np.arange(101 * 4, dtype=np.float32).reshape(101, 4) / 17
    with h5py.File(path, "w") as hdf:
        hdf["force"] = force
    record = Trajectory("train8", "case", path, "x", 101, "b00")
    stats = {"all_force_mean": [0.1, 0.2, 0.3, 0.4], "all_force_std": [1.1, 1.2, 1.3, 1.4]}
    actual, physical = canonical_force_targets(record, 0, stats)
    expected = ((force[1:].astype(np.float64) - stats["all_force_mean"]) / stats["all_force_std"]).astype(np.float32)
    np.testing.assert_array_equal(physical, force[1:])
    np.testing.assert_array_equal(actual, expected)


torch = pytest.importorskip("torch")


class HookableLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 7)

    def forward(self, hidden):
        return self.linear(hidden)


class FakeFNO(torch.nn.Module):
    def __init__(self, height=2, width=3):
        super().__init__()
        self.height = height
        self.width = width
        self.final = HookableLayer()
        self.steps = 0

    def forward(self, inputs):
        batch = len(inputs)
        base = inputs[:, :1].permute(0, 2, 3, 1).reshape(-1, 1)
        hidden = base.repeat(1, 128)
        raw = self.final(hidden).reshape(batch, self.height, self.width, 7).permute(0, 3, 1, 2)
        self.steps += 1
        return raw


def fake_batch(batch=2, height=2, width=3):
    return {
        "state": torch.zeros(batch, 3, height, width),
        "target_state": torch.zeros(batch, 100, 3, height, width),
        "target_force": torch.arange(batch * 100 * 4, dtype=torch.float32).reshape(batch, 100, 4),
        "omega": torch.zeros(batch, 101, 1),
        "mask": torch.ones(batch, 1, height, width),
    }


def test_free_ar_real_shape_alignment_and_100_step_recurrence():
    model = FakeFNO()
    result = free_ar_batch(model, model.final, fake_batch(), torch.device("cpu"))
    assert result["features"].shape == (2, 100, 128)
    assert result["targets"].shape == (2, 100, 4)
    assert result["native"].shape == (2, 100, 4)
    assert len(result["state_sha256"]) == 64
    assert model.steps == 100
    np.testing.assert_array_equal(result["targets"][:, -1], fake_batch()["target_force"].numpy()[:, -1])


def test_free_ar_collect_false_keeps_native_and_state():
    model = FakeFNO()
    result = free_ar_batch(model, model.final, fake_batch(batch=1), torch.device("cpu"), collect_features=False)
    assert result["features"] is None
    assert result["native"].shape == (1, 100, 4)


def test_free_ar_rejects_wrong_target_length():
    model = FakeFNO()
    batch = fake_batch(batch=1)
    batch["target_force"] = batch["target_force"][:, :-1]
    with pytest.raises(ValueError, match="shape differs"):
        free_ar_batch(model, model.final, batch, torch.device("cpu"))
