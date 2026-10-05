import numpy as np
import pytest

from analyze_fcp009_joint_readout import (
    fit_joint_coefficients,
    joint_oof_predictions,
    metric_block,
)
from build_fcp008_force_readout_candidate import predict


def test_joint_fit_is_exact_fixed_half_half_design():
    rng = np.random.default_rng(17)
    free_ar = rng.normal(size=(180, 128))
    h1 = rng.normal(size=(180, 128))
    truth = rng.normal(size=(180, 4))
    mask = np.arange(180) < 160
    actual, fit = fit_joint_coefficients(free_ar, h1, truth, mask)
    design = np.concatenate((free_ar[mask], h1[mask]))
    target = np.concatenate((truth[mask], truth[mask]))
    augmented = np.column_stack((design, np.ones(len(design))))
    expected = np.linalg.lstsq(augmented, target, rcond=1e-10)[0]
    np.testing.assert_allclose(predict(design, actual), augmented @ expected, rtol=1e-10, atol=1e-10)
    assert fit["domain_mix"] == {"free_ar": 0.5, "matched_weight_h1": 0.5}
    assert fit["joint_design_rows"] == 320
    assert fit["per_domain_copy_weight"] == pytest.approx(0.5 / 160)


def test_metric_block_uses_correlated_total_cd_residual():
    error = np.array([[1.0, 0.0, -1.0, 2.0], [2.0, 0.0, 3.0, -2.0]])
    result = metric_block(error, np.array([[2.0, 1.0, 3.0, 1.0]]), np.array([True, True]))
    assert result["physical"]["rear_cd"]["mae"] == pytest.approx(6.0)
    # Physical total Cd is (front normalized residual * front std) plus
    # (rear normalized residual * rear std): [-1, 13], MAE=7.
    assert result["physical"]["total_cd"]["mae"] == pytest.approx(7.0)
    assert set(result["normalized"]) == {"rear_cd", "rear_cl"}
    assert "total_cd" not in result["normalized"]


def test_joint_fit_rejects_row_target_mismatch_and_nonfinite():
    x = np.zeros((10, 128))
    y = np.zeros((10, 4))
    with pytest.raises(ValueError, match="row/target contract"):
        fit_joint_coefficients(x, x[:-1], y, np.ones(10, dtype=bool))
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match="row/target contract"):
        fit_joint_coefficients(x, x, y, np.ones(10, dtype=bool))


def test_joint_oof_uses_complete_held_phase_folds():
    rng = np.random.default_rng(23)
    phases = np.repeat(np.array(["b00", "b02", "b04", "b06"]), 40)
    free_ar = rng.normal(size=(160, 128))
    h1 = rng.normal(size=(160, 128))
    targets = rng.normal(size=(160, 4))
    predictions, fits = joint_oof_predictions(free_ar, h1, targets, phases)
    assert set(fits) == {"b00", "b02", "b04", "b06"}
    assert all(fit["original_train_rows"] == 120 for fit in fits.values())
    assert all(np.isfinite(value).all() for value in predictions.values())
    with pytest.raises(ValueError, match="phase fold contract"):
        joint_oof_predictions(free_ar, h1, targets, np.array(["b00"] * 160))
