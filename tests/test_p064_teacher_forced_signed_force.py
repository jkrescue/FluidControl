import ast
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


SOURCE = Path(__file__).parents[1] / "scripts/diagnose_fno_force_window_p064_teacher_forced.py"
spec = importlib.util.spec_from_file_location("p064_teacher_force", SOURCE)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def fixture():
    states = np.zeros((101, 3, 1, 1), dtype=np.float32)
    states[:, 0, 0, 0] = np.arange(101, dtype=np.float32) * 100.0
    mask = np.ones((101, 1, 1, 1), dtype=np.float32)
    omega = np.linspace(-0.5, 0.5, 101, dtype=np.float32)
    truth = np.stack([np.arange(101) + channel for channel in range(4)], axis=1)
    times = 20.0 + 0.1 * np.arange(101)
    common = {
        "action_scale": 0.75,
        "state_mean": torch.zeros((1, 3, 1, 1)),
        "state_std": torch.ones((1, 3, 1, 1)),
        "force_mean": torch.zeros(4),
        "force_std": torch.ones(4),
        "network": object(),
        "history_k": 1,
    }
    return states, mask, omega, truth, times, common


def test_each_prediction_resets_from_true_current_state_without_future_truth():
    states, mask, omega, truth, times, common = fixture()

    def reset_history(q, now, *, k):
        assert k == 1
        return {"q": q.clone(), "now": now.clone()}

    def step(_network, history, spatial_mask, following):
        step.calls += 1
        raw = torch.zeros((1, 7, 1, 1))
        raw[:, 3:] = history["q"][:, :1] + following.reshape(1, 1, 1, 1)
        return raw, torch.full_like(history["q"], -999), "unused"
    step.calls = 0

    rows = module.teacher_forced_h1_predictions(
        states, mask, omega, truth, times,
        reset_history=reset_history, history_dual_step=step, **common,
    )
    assert len(rows) == 100
    assert step.calls == 100
    # The first prediction uses q_0=0, not the deliberately large q_1=100.
    expected0 = float(torch.tensor([omega[1]], dtype=torch.float32).div(0.75)[0])
    assert rows[0]["predicted_force_s_plus_1"] == pytest.approx([expected0] * 4)
    assert rows[0]["true_force_s_plus_1"] == [1.0, 2.0, 3.0, 4.0]
    assert rows[0]["input_time"] == 20.0
    assert rows[-1]["target_time"] == 30.0
    assert rows[-1]["omega_s_plus_1"] == pytest.approx(float(omega[100]))


def test_signed_summary_preserves_bias_and_amplitude():
    rows = [
        {"true_force_s_plus_1": [1, -2, 3, -4], "predicted_force_s_plus_1": [2, -1, 1, -5]},
        {"true_force_s_plus_1": [-1, 2, -3, 4], "predicted_force_s_plus_1": [0, 1, -1, 3]},
    ]
    summary = module.summarize_signed_force_rows(rows)
    assert summary["front_cd"]["signed_error_mean"] == 1.0
    assert summary["rear_cl"]["mean_absolute_error"] == 1.0
    assert summary["rear_cl"]["truth_centered_rms"] == 4.0
    assert summary["rear_cl"]["predicted_centered_rms"] == 4.0


def test_history_profile_is_required():
    states, mask, omega, truth, times, common = fixture()
    common["history_k"] = None
    with pytest.raises(ValueError, match="explicit history profile"):
        module.teacher_forced_h1_predictions(
            states, mask, omega, truth, times,
            reset_history=lambda *a, **k: None,
            history_dual_step=lambda *a, **k: None,
            **common,
        )


def test_nonfinite_true_input_is_rejected_before_forward():
    states, mask, omega, truth, times, common = fixture()
    states[3, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite aligned 101-frame"):
        module.teacher_forced_h1_predictions(
            states, mask, omega, truth, times,
            reset_history=lambda *a, **k: None,
            history_dual_step=lambda *a, **k: None,
            **common,
        )


def test_production_cli_is_explicit_and_default_ar_is_preserved():
    source = ast.unparse(ast.parse(SOURCE.read_text()))
    assert "choices=('autoregressive', 'teacher_forced_h1')" in source
    assert "default='autoregressive'" in source
    assert "allocator_target_bytes = 6 * 1024 ** 3" in source
    assert "torch.set_float32_matmul_precision('high')" in source
    assert "'total_submodel_forwards': 2 * len(all_rows)" in source
    assert "reference_cases[path.stem] != sha256(path)" in source
    assert "q_(s+1)" in SOURCE.read_text()
