import numpy as np
import pytest

from analyze_fcp009_window_statistics import (
    FORMAL_RECEIPT_SHA,
    STEPS_PER_WINDOW,
    aggregate_rows,
    force_window_statistics,
    validate_formal_receipt,
    window_error_rows,
)


def test_constant_rear_cl_bias_changes_mean_not_centered_rms():
    time = np.linspace(0.0, 4.0 * np.pi, STEPS_PER_WINDOW)
    truth = np.zeros((STEPS_PER_WINDOW, 4))
    truth[:, 3] = np.sin(time)
    prediction = truth.copy()
    prediction[:, 3] += 0.25
    truth_stats = force_window_statistics(truth)
    pred_stats = force_window_statistics(prediction)
    assert pred_stats["mean_rear_cl"] - truth_stats["mean_rear_cl"] == pytest.approx(0.25)
    assert pred_stats["rear_cl_fluctuation_rms"] == pytest.approx(
        truth_stats["rear_cl_fluctuation_rms"]
    )


def test_known_scaling_changes_centered_rms_by_known_amount():
    truth = np.zeros((STEPS_PER_WINDOW, 4))
    truth[:, 3] = np.arange(STEPS_PER_WINDOW, dtype=float)
    scaled = truth.copy()
    scaled[:, 3] *= 1.5
    a = force_window_statistics(truth)["rear_cl_fluctuation_rms"]
    b = force_window_statistics(scaled)["rear_cl_fluctuation_rms"]
    assert b - a == pytest.approx(0.5 * a)


def test_physical_total_cd_uses_unequal_channel_scales():
    target = np.zeros((STEPS_PER_WINDOW, 4))
    prediction = np.zeros_like(target)
    prediction[:, 0] = 1.0
    prediction[:, 2] = -1.0
    windows = [
        {
            "families": "base20",
            "case_names": "case",
            "phases": "b00",
            "window_starts": 0,
            "window_index": 0,
        }
    ]
    # Normalized front/rear errors cancel, but physical total-Cd error is 2 - 5 = -3.
    rows = window_error_rows(
        prediction,
        target,
        np.array([10.0, 0.0, 20.0, 0.0]),
        np.array([2.0, 1.0, 5.0, 1.0]),
        windows,
    )
    assert rows[0]["signed_error"]["mean_total_cd"] == pytest.approx(-3.0)
    assert rows[0]["absolute_error"]["mean_total_cd"] == pytest.approx(3.0)


def test_aggregate_records_nonindependent_window_counts():
    row = {
        "families": "base20",
        "case_names": "case",
        "phases": "b00",
        "signed_error": {
            "mean_total_cd": 1.0,
            "mean_rear_cl": -2.0,
            "rear_cl_fluctuation_rms": 3.0,
        },
    }
    result = aggregate_rows([row, dict(row)])
    pooled = result["pooled"]["all"]
    assert pooled["window_count"] == 2
    assert pooled["unique_case_count"] == 1
    assert pooled["windows_are_not_independent_physical_samples"] is True
    assert pooled["metrics"]["mean_rear_cl"]["mean_absolute_error"] == pytest.approx(2.0)


def test_formal_receipt_requires_real_no_ppo_schema():
    receipt = {
        "status": "FC_P009_POSTEVAL_COMPLETE",
        "ppo_auto_launched": False,
        "frozen_test_accessed": False,
    }
    validate_formal_receipt(receipt, FORMAL_RECEIPT_SHA)
    for bad in (
        {key: value for key, value in receipt.items() if key != "ppo_auto_launched"},
        {**receipt, "ppo_auto_launched": True},
        {**receipt, "ppo_authorized": True},
    ):
        with pytest.raises(ValueError, match="formal receipt differs"):
            validate_formal_receipt(bad, FORMAL_RECEIPT_SHA)
