import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "diagnose_fcp003c_fixed_feature_force_readout.py"
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location("fixed_readout", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_float64_affine_fit_recovers_exact_four_channel_map():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(400, 128))
    beta = rng.normal(size=(129, 4))
    x = x.astype(np.float32)
    y = x.astype(np.float64) @ beta[:128] + beta[128]
    fitted, diagnostic = module.fit_affine_readout(x, y)
    assert fitted.dtype == np.float64
    assert diagnostic["rcond"] == 1e-10
    assert diagnostic["rank"] == 129
    assert len(diagnostic["singular_values"]) == 129
    assert diagnostic["retained_singular_value_count"] == 129
    assert diagnostic["condition_number_retained"] >= 1.0
    np.testing.assert_allclose(module.predict_affine(x, fitted), y, rtol=0, atol=2e-12)


def _panel():
    rng = np.random.default_rng(9)
    phases = ("b00", "b02", "b04", "b06")
    panel = {
        "action_features": {}, "zero_features": {},
        "action_targets_normalized": {}, "zero_targets_normalized": {},
        "force_std": np.ones((1, 4)),
    }
    beta = rng.normal(size=(129, 4))
    for phase in phases:
        zero_x = rng.normal(size=(100, 128))
        panel["zero_features"][phase] = zero_x
        panel["zero_targets_normalized"][phase] = module.predict_affine(zero_x, beta)
        for profile in ("multisine", "prbs"):
            key = f"{phase}:{profile}"
            action_x = rng.normal(size=(100, 128))
            panel["action_features"][key] = action_x
            panel["action_targets_normalized"][key] = module.predict_affine(action_x, beta)
    return panel, beta


def test_symmetric_weighting_repeats_each_unique_zero_twice():
    panel, _ = _panel()
    ids = sorted(panel["action_features"])
    x, y = module.weighted_fit_arrays(panel, ids)
    assert x.shape == (1600, 128)
    assert y.shape == (1600, 4)
    for phase in ("b00", "b02", "b04", "b06"):
        zero = panel["zero_features"][phase]
        matches = sum(np.array_equal(row, candidate) for row in x for candidate in zero)
        assert matches == 200


def test_metrics_keep_action_unique_zero_and_delta_separate():
    panel, beta = _panel()
    metrics = module.evaluate_panel(panel, sorted(panel["action_features"]), beta)
    assert metrics["endpoint_counts"] == {"action": 800, "unique_zero": 400, "delta": 800}
    assert set(metrics["per_pair"]) == set(panel["action_features"])
    assert set(metrics["unique_zero_by_phase"]) == {"b00", "b02", "b04", "b06"}
    for group in ("action", "unique_zero", "delta"):
        for channel in module.CHANNELS:
            assert metrics[group][channel]["mae"] < 1e-12


def test_prefix_and_late_windows_are_disjoint():
    prefix = set(module.PANELS["prefix_targets_1_100"])
    late = set(module.PANELS["late_targets_101_200"])
    assert prefix == set(range(1, 101))
    assert late == set(range(101, 201))
    assert not prefix & late


def test_true_state_steps_are_local_to_each_window():
    """Late target 101 is local step 0, not an invalid global step 100."""
    endpoints = list(module.PANELS["late_targets_101_200"])
    calls = []
    for begin in range(0, len(endpoints), 10):
        local = endpoints[begin : begin + 10]
        calls.extend(module.local_step_indices(begin, len(local)))
    assert calls == list(range(100))
    assert endpoints == list(range(101, 201))
    assert list(zip(calls, endpoints, strict=True))[0] == (0, 101)
    assert list(zip(calls, endpoints, strict=True))[-1] == (99, 200)


def test_extract_branch_uses_real_shapes_and_global_physical_targets():
    torch = pytest.importorskip("torch")

    class Final(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(128, 7)

        def forward(self, value):
            return self.linear(value)

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.final = Final()

        def forward(self, value):
            batch, _, height, width = value.shape
            hidden = torch.arange(
                batch * height * width * 128, dtype=value.dtype, device=value.device
            ).reshape(batch * height * width, 128)
            return self.final(hidden).reshape(batch, height, width, 7).permute(0, 3, 1, 2)

    model = FakeModel().eval()
    state = torch.zeros((1, 101, 3, 2, 2))
    omega = torch.zeros((1, 101, 1))
    mask = torch.ones((1, 1, 2, 2))
    force = np.arange(201 * 4, dtype=np.float32).reshape(201, 4)
    result = module.extract_branch_features(
        model, model.final, state, omega, mask, force, list(range(101, 201)), 10
    )
    assert result["features"].shape == (100, 128)
    np.testing.assert_array_equal(result["targets_physical"][0], force[101])
    np.testing.assert_array_equal(result["targets_physical"][-1], force[200])
    assert result["first_input"].shape[0] == 10
    assert result["first_state_output"].shape[0] == 10


@pytest.mark.parametrize(
    "features,targets",
    [
        (np.zeros((129, 128)), np.zeros((129, 4))),
        (np.zeros((200, 127)), np.zeros((200, 4))),
        (np.zeros((200, 128)), np.zeros((200, 3))),
        (np.full((200, 128), np.nan), np.zeros((200, 4))),
    ],
)
def test_lstsq_rejects_invalid_contract(features, targets):
    with pytest.raises(ValueError):
        module.fit_affine_readout(features, targets)


def test_summarize_rejects_nonfinite_and_reports_signed_bias():
    error = np.tile(np.array([[1.0, -2.0, 3.0, -4.0]]), (3, 1))
    summary = module.summarize_error(error)
    assert summary["rear_cl"] == {"count": 3, "mae": 4.0, "rmse": 4.0, "bias": -4.0}
    with pytest.raises(ValueError):
        module.summarize_error(np.full((1, 4), np.inf))


def test_diagnostic_precision_is_explicit_and_restorable():
    torch = pytest.importorskip("torch")
    saved_matmul = torch.backends.cuda.matmul.allow_tf32
    saved_cudnn = torch.backends.cudnn.allow_tf32
    saved_precision = torch.get_float32_matmul_precision()
    try:
        before, effective = module.configure_fixed_highest_fp32(torch)
        assert set(before) == {
            "NVIDIA_TF32_OVERRIDE",
            "cuda_matmul_allow_tf32",
            "cudnn_allow_tf32",
            "float32_matmul_precision",
        }
        assert effective["cuda_matmul_allow_tf32"] is False
        assert effective["cudnn_allow_tf32"] is False
        assert effective["float32_matmul_precision"] == "highest"
    finally:
        torch.backends.cuda.matmul.allow_tf32 = saved_matmul
        torch.backends.cudnn.allow_tf32 = saved_cudnn
        torch.set_float32_matmul_precision(saved_precision)
