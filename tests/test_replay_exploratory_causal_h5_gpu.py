import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SOURCE = Path(__file__).parents[1] / "scripts/replay_exploratory_causal_h5_gpu.py"
SPEC = importlib.util.spec_from_file_location("h5_gpu_replay", SOURCE)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def decision(offset=0.0, selected=2):
    components = {
        "drag_screen": 1.0 + offset,
        "drag_gate_violation": 2.0,
        "rear_cl_fluctuation_gate_violation": 0.0,
        "rear_cl_mean_bias_gate_violation": 0.0,
        "actuation": .1,
        "rate": .2,
    }
    return {
        "predicted_forces_h5": (np.arange(100).reshape(5, 5, 4) / 100 + offset).tolist(),
        "h5_cost": [5.0, 4.0, 3.0, 2.0, 1.0 + offset],
        "candidate_reports": [
            {"stages": [{"components": components} for _ in range(5)]}
            for _ in range(5)],
        "selected_index": selected,
        "selected_action": float(selected),
    }


def test_comparison_reports_raw_force_component_rank_and_selection():
    row = M.compare_decision(decision(), decision(1e-4))
    assert np.allclose(row["force_difference"]["values"], 1e-4, rtol=0.0, atol=1e-15)
    assert np.isclose(row["force_difference"]["max_abs"], 1e-4)
    assert np.isclose(row["component_difference"]["max_abs"], 1e-4)
    assert row["cpu_rank"] == row["gpu_rank"] == [4, 3, 2, 1, 0]
    assert row["selection_same"]


def test_selection_mismatch_is_explicit_not_admission():
    row = M.compare_decision(decision(selected=2), decision(selected=3))
    assert not row["selection_same"]
    assert M.COMPLETE.endswith("NOT_ADMISSION")


def test_actual_endpoint_force_order():
    row = {"actual_endpoint_forces": {"mpc": {
        "front_cd": 1.0, "front_cl": 2.0, "rear_cd": 3.0, "rear_cl": 4.0}}}
    assert M.actual_force(row) == [1.0, 2.0, 3.0, 4.0]


def test_precision_and_no_cfd_or_optimizer_contract_in_source():
    text = SOURCE.read_text()
    load = text.index("flow, aero, identity = load_bound_k1(")
    override = text.index("precision = override_inference_precision(torch)")
    assert load < override
    assert "cuda.mem_get_info()[0] >=" not in text
    assert '"optimizer_used": False' in text
    assert '"cfd_executed": False' in text
    assert "rollout_five_held_horizon" in text
    assert "append_actual_endpoint" in text


def test_post_load_precision_override_is_exact_and_recorded():
    fake = SimpleNamespace(precision="high", backends=SimpleNamespace(
        cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
        cudnn=SimpleNamespace(allow_tf32=True)))
    fake.get_float32_matmul_precision = lambda: fake.precision
    fake.set_float32_matmul_precision = lambda value: setattr(fake, "precision", value)
    record = M.override_inference_precision(fake)
    assert record["before_override"] == {
        "float32_matmul_precision": "high",
        "cuda_matmul_allow_tf32": True, "cudnn_allow_tf32": True}
    assert record["effective"] == {
        "float32_matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False}


def test_precision_override_rejects_wrong_historical_load_state():
    fake = SimpleNamespace(precision="highest", backends=SimpleNamespace(
        cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
        cudnn=SimpleNamespace(allow_tf32=False)))
    fake.get_float32_matmul_precision = lambda: fake.precision
    fake.set_float32_matmul_precision = lambda value: setattr(fake, "precision", value)
    import pytest
    with pytest.raises(ValueError, match="historical load precision"):
        M.override_inference_precision(fake)
