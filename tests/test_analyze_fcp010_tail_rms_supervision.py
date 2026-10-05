import numpy as np
import pytest
import torch

from analyze_fcp010_tail_rms_supervision import (
    joint_standardization,
    optimize_rear_cl,
    report,
    replace_rear_cl,
    tail_rms,
)


def test_tail_rms_is_bias_invariant_and_scales_amplitude():
    values = torch.linspace(-2.0, 2.0, 100, dtype=torch.float64).reshape(1, 100)
    baseline = tail_rms(values)
    torch.testing.assert_close(tail_rms(values + 7.0), baseline)
    torch.testing.assert_close(tail_rms(values * 1.5), baseline * 1.5)


def test_joint_standardization_uses_only_masked_rows():
    ar = np.arange(24, dtype=float).reshape(6, 4)
    h1 = ar + 10.0
    mask = np.array([True, True, False, False, False, False])
    mean, std, active = joint_standardization(ar, h1, mask)
    expected = np.concatenate((ar[:2], h1[:2]))
    np.testing.assert_allclose(mean, expected.mean(axis=0))
    np.testing.assert_allclose(std, expected.std(axis=0))
    assert active.all()


def test_optimizer_uses_train_phases_and_reduces_known_amplitude_error():
    rng = np.random.default_rng(4)
    phases = np.repeat(np.array(["b00", "b02", "b04", "b06"]), 100)
    feature = rng.normal(size=(400, 128))
    h1 = feature.copy()
    target = np.zeros((400, 4))
    target[:, 3] = 2.0 * feature[:, 0]
    init = np.zeros(129)
    init[0] = 0.5
    coefficient, info = optimize_rear_cl(
        feature, h1, target, phases, 1.0, init, held_phase="b06"
    )
    before = np.mean(np.square(feature[phases != "b06"] @ init[:128] - target[phases != "b06", 3]))
    after = np.mean(np.square(feature[phases != "b06"] @ coefficient[:128] + coefficient[128] - target[phases != "b06", 3]))
    assert after < before
    assert info["train_window_count"] == 3
    assert info["train_row_count"] == 300
    assert info["finite"] is True


def test_optimizer_rejects_nonpositive_physical_scale():
    x = np.ones((400, 128))
    y = np.zeros((400, 4))
    phases = np.repeat(np.array(["b00", "b02", "b04", "b06"]), 100)
    with pytest.raises(ValueError, match="input differs"):
        optimize_rear_cl(x, x, y, phases, 0.0, np.zeros(129), "b06")

    rng = np.random.default_rng(12)
    varying_features = rng.normal(size=(400, 128))
    with pytest.raises(ValueError, match="RMS scale is nonfinite or nonpositive"):
        optimize_rear_cl(
            varying_features,
            varying_features,
            y,
            phases,
            1.0,
            np.zeros(129),
            "b06",
        )


def test_rear_cl_replacement_preserves_other_force_rows_exactly():
    rng = np.random.default_rng(22)
    baseline = rng.normal(size=(129, 4))
    rear = rng.normal(size=129)
    changed = replace_rear_cl(baseline, rear)
    np.testing.assert_array_equal(changed[:, :3], baseline[:, :3])
    np.testing.assert_array_equal(changed[:, 3], rear)
    features = rng.normal(size=(100, 128))
    design = np.column_stack((features, np.ones(100)))
    before = design @ baseline
    after = design @ changed
    np.testing.assert_array_equal(after[:, :3], before[:, :3])
    assert not np.array_equal(after[:, 3], before[:, 3])


def test_report_requires_full_head_and_preserves_nonzero_cd_error():
    rng = np.random.default_rng(31)
    features = rng.normal(size=(100, 128))
    targets = np.zeros((100, 4))
    coefficients = np.zeros((129, 4))
    coefficients[128, 0] = 1.0
    coefficients[128, 2] = 2.0
    cache = {
        "families": np.repeat("base20", 100),
        "phases": np.repeat("b00", 100),
        "case_names": np.repeat("matched_start_acquisition_train_b00_zero", 100),
    }
    windows = [
        {
            "families": "base20",
            "phases": "b00",
            "case_names": "matched_start_acquisition_train_b00_zero",
            "window_starts": 0,
            "window_index": 0,
        }
    ]
    result = report(
        cache,
        features,
        targets,
        coefficients,
        np.ones(100, dtype=bool),
        np.zeros(4),
        np.ones(4),
        windows,
    )
    total_cd = result["tail_window"]["pooled"]["all"]["metrics"]["mean_total_cd"]
    assert total_cd["mean_absolute_error"] == pytest.approx(3.0)
    with pytest.raises(ValueError, match="coefficient shape differs"):
        report(
            cache,
            features,
            targets,
            coefficients[:, 3],
            np.ones(100, dtype=bool),
            np.zeros(4),
            np.ones(4),
            windows,
        )
