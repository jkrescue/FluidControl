import hashlib
import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/serve_live_research_dashboard.py"
spec = importlib.util.spec_from_file_location("dashboard_readout", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(root):
    base = root / module.FIXED_READOUT
    (base / "result_bundle").mkdir(parents=True)
    result = {
        "status": "FCP003C_FIXED_FEATURE_FORCE_READOUT_DIAGNOSTIC_COMPLETE",
        "model_sha256": module.C_FINAL_SHA,
        "optimizer_steps": 0, "candidate_saved": False,
        "validation_accessed": False, "frozen_test_accessed": False, "ppo_executed": False,
        "numerical_protocol": {"precision_effective": {
            "float32_matmul_precision": "highest", "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False}},
        "model_tensor_state_sha256_before": "test-only",
        "model_tensor_state_sha256_after": "test-only",
        "field_repeat_bitwise_identical": True,
    }
    for kind in ("original", "fitted"):
        result[kind + "_readout_metrics_physical"] = {
            panel: {"action": {channel: {"count": 800, "mae": 0.2}
                               for channel in ("rear_cd", "rear_cl")}}
            for panel in ("prefix_targets_1_100", "late_targets_101_200")}
    cache = base / "result_bundle/fixed_features.npz"
    cache.write_bytes(b"unit-test cache, not experimental evidence")
    receipt = {k: result[k] for k in ("optimizer_steps", "candidate_saved", "validation_accessed", "frozen_test_accessed", "ppo_executed")}
    receipt["status"] = "FCP003C_FIXED_FEATURE_FORCE_READOUT_V2_EXECUTION_COMPLETE_NOT_ADMISSION"
    receipt["cache_sha256"] = hashlib.sha256(cache.read_bytes()).hexdigest()
    save(base, result, receipt)
    return base, result, receipt


def save(base, result, receipt):
    path = base / "result_bundle/result.json"
    path.write_text(json.dumps(result))
    receipt["result_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (base / "completion_receipt.json").write_text(json.dumps(receipt))


def test_missing_not_ready(tmp_path):
    assert module._fixed_feature_readout(tmp_path) == {"ready": False}


def test_bound_diagnostic_is_not_admission(tmp_path):
    fixture(tmp_path)
    data = module._fixed_feature_readout(tmp_path)
    assert data["ready"] and data["admission"] is False
    assert len(data["rows"]) == 2


def test_changed_cache_hidden(tmp_path):
    base, _, _ = fixture(tmp_path)
    (base / "result_bundle/fixed_features.npz").write_bytes(b"changed")
    assert not module._fixed_feature_readout(tmp_path)["ready"]


def test_wrong_precision_hidden(tmp_path):
    base, result, receipt = fixture(tmp_path)
    result["numerical_protocol"]["precision_effective"]["cuda_matmul_allow_tf32"] = True
    save(base, result, receipt)
    assert not module._fixed_feature_readout(tmp_path)["ready"]


def test_nonfinite_metric_hidden(tmp_path):
    base, result, receipt = fixture(tmp_path)
    result["fitted_readout_metrics_physical"]["late_targets_101_200"]["action"]["rear_cd"]["mae"] = float("nan")
    save(base, result, receipt)
    assert not module._fixed_feature_readout(tmp_path)["ready"]


def test_candidate_flag_hidden(tmp_path):
    base, result, receipt = fixture(tmp_path)
    result["candidate_saved"] = True
    save(base, result, receipt)
    assert not module._fixed_feature_readout(tmp_path)["ready"]
