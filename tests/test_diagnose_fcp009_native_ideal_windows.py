import numpy as np
import pytest

from diagnose_fcp009_native_ideal_windows import (
    actual_force_head,
    physical_window_comparison,
    validate_runtime_batch_context,
)


def test_native_ideal_comparison_separates_bias_and_centered_rms():
    truth = np.zeros((100, 4))
    truth[:, 3] = np.linspace(-1.0, 1.0, 100)
    ideal = truth.copy()
    ideal[:, 3] += 0.25
    native = truth.copy()
    native[:, 3] *= 1.5
    result = physical_window_comparison(
        truth,
        ideal,
        native,
        np.zeros(4),
        np.ones(4),
    )
    assert result["ideal_absolute_statistic_error"]["mean_rear_cl"] == pytest.approx(0.25)
    assert result["ideal_absolute_statistic_error"]["rear_cl_fluctuation_rms"] == pytest.approx(0.0)
    expected = 0.5 * result["truth_window"]["rear_cl_fluctuation_rms"]
    assert result["native_absolute_statistic_error"]["rear_cl_fluctuation_rms"] == pytest.approx(expected)


def test_physical_total_cd_uses_both_unequal_scales():
    truth = np.zeros((100, 4))
    ideal = truth.copy()
    ideal[:, 0] = 1.0
    ideal[:, 2] = -1.0
    result = physical_window_comparison(
        truth,
        ideal,
        truth,
        np.array([1.0, 0.0, 2.0, 0.0]),
        np.array([2.0, 1.0, 5.0, 1.0]),
    )
    assert result["ideal_absolute_statistic_error"]["mean_total_cd"] == pytest.approx(3.0)
    assert result["native_absolute_statistic_error"]["mean_total_cd"] == pytest.approx(0.0)


def test_comparison_rejects_nonfinite_or_wrong_length():
    good = np.zeros((100, 4))
    with pytest.raises(ValueError, match="arrays differ"):
        physical_window_comparison(good[:-1], good[:-1], good[:-1], np.zeros(4), np.ones(4))
    bad = good.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="arrays differ"):
        physical_window_comparison(good, bad, good, np.zeros(4), np.ones(4))


def test_actual_force_head_uses_only_saved_rows_three_through_six():
    weight = np.arange(7 * 128, dtype=np.float32).reshape(7, 128)
    bias = np.arange(7, dtype=np.float32)
    actual = actual_force_head(weight, bias)
    assert actual.shape == (129, 4)
    np.testing.assert_array_equal(actual[:128], weight[3:7].T)
    np.testing.assert_array_equal(actual[128], bias[3:7])


def test_runtime_batch_context_checks_all_four_neighbors():
    cache = {
        "case_names": np.repeat(np.array(["a", "b", "c", "d"]), 100),
        "window_starts": np.repeat(np.array([0, 2, 4, 6]), 100),
        "families": np.repeat(np.array(["train8"] * 4), 100),
    }
    selected = {
        "family": "train8",
        "global_window_index": 2,
        "local_dataset_index": 2,
    }
    metadata = [
        {"case": case, "step": step, "rollout_steps": 100, "split": "train"}
        for case, step in zip(("a", "b", "c", "d"), (0, 2, 4, 6), strict=True)
    ]
    validate_runtime_batch_context(selected, metadata, cache)
    metadata[1] = {**metadata[1], "step": 99}
    with pytest.raises(ValueError, match="batch4 cache identity differs"):
        validate_runtime_batch_context(selected, metadata, cache)
