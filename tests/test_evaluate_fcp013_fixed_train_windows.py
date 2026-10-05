"""Software-contract fixtures only; these values are not CFD evidence."""
import copy

import pytest

from evaluate_fcp013_fixed_train_windows import check_diagnostic_comparison, check_training_result


def terminal():
    return {
        "status": "FC_P013_INDEPENDENT_FORCE_FNO_TRAINING_COMPLETE_NOT_ADMISSION",
        "optimizer_steps": 1368, "dual_fresh_reload_verified": True,
        "dual_fresh_reload_manifest_sha256": "fixture", "dual_model_manifest_sha256": "fixture",
        "validation_accessed": False, "frozen_test_accessed": False,
        "ppo_executed": False, "selection_performed": False,
        "flow_tensor_sha256_before": "unchanged", "flow_tensor_sha256_after": "unchanged",
        "records": [{"step": i, "optimizer_steps": 1, "total": 0.1, "h1_balanced": 0.1,
                     "ar_balanced": 0.1, "preclip_gradient_norm": 0.1,
                     "identity": {"split": "train", "rollout_steps": 100}} for i in range(1, 1369)],
    }


def test_terminal_requires_full_actual_record_sequence():
    result = terminal()
    check_training_result(result, "fixture")
    result["records"].pop()
    with pytest.raises(ValueError):
        check_training_result(result, "fixture")


@pytest.mark.parametrize("key,value", [("ppo_executed", True), ("selection_performed", True),
                                       ("flow_tensor_sha256_after", "changed"),
                                       ("dual_fresh_reload_verified", False)])
def test_terminal_rejects_contract_violation(key, value):
    result = terminal()
    result[key] = value
    with pytest.raises(ValueError):
        check_training_result(result, "fixture")


@pytest.mark.parametrize("key,value", [("step", 2), ("total", float("nan")), ("optimizer_steps", 2)])
def test_terminal_rejects_bad_update(key, value):
    result = terminal()
    result["records"][0][key] = value
    with pytest.raises(ValueError):
        check_training_result(result, "fixture")


def panel():
    vector = {"free_ar_field_relative_l2_uvp": [0.1] * 3,
              "true_state_h1_field_relative_l2_uvp": [0.1] * 3,
              "initial_endpoint_h1_force_absolute_error": [0.1] * 4,
              "true_state_h1_force_mae": [0.1] * 4, "free_ar_force_mae": [0.1] * 4}
    scalar = {key: 0.1 for key in (
        "tail62_mean_rear_cl_absolute_error", "tail62_rear_cl_rms_absolute_error",
        "true_state_h1_tail62_mean_rear_cl_absolute_error", "true_state_h1_tail62_rear_cl_rms_absolute_error")}
    return {"windows": [{"global_index": i, "family": "fixture", "identity": {"case": str(i)},
                         **copy.deepcopy(vector), **scalar} for i in range(6)]}


def test_physical_metrics_and_identity_completeness():
    a, b = panel(), panel()
    check_diagnostic_comparison(a, b)
    b["windows"][0]["identity"] = {"case": "wrong"}
    with pytest.raises(ValueError):
        check_diagnostic_comparison(a, b)


@pytest.mark.parametrize("key,value", [("free_ar_force_mae", [0.1] * 3),
                                       ("tail62_rear_cl_rms_absolute_error", float("inf"))])
def test_missing_or_invalid_physical_metric_rejected(key, value):
    a, b = panel(), panel()
    b["windows"][0][key] = value
    with pytest.raises(ValueError):
        check_diagnostic_comparison(a, b)
