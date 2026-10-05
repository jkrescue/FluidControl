"""CPU software fixtures only; no training, CFD or admission evidence."""
import pytest
import torch
import copy
import json

import audit_fcp015_candidate as audit


def optimizer_state():
    return {"epoch": 1, "optimizer_state_dict": {
        "param_groups": [{"lr": 1e-5, "weight_decay": 1e-4,
                          "betas": (.9, .999), "eps": 1e-8,
                          "params": list(range(28))}],
        "state": {i: {"step": torch.tensor(171.), "exp_avg": torch.zeros(2),
                      "exp_avg_sq": torch.ones(2)} for i in range(28)}}}


def test_actual_p015_optimizer():
    audit.validate_optimizer(optimizer_state())


@pytest.mark.parametrize("bad", [1368, 170, 172, True, float("nan")])
def test_rejects_non_p015_actual_steps(bad):
    value = optimizer_state()
    value["optimizer_state_dict"]["state"][0]["step"] = bad
    with pytest.raises(ValueError, match="actual AdamW step"):
        audit.validate_optimizer(value)


@pytest.mark.parametrize("key,bad", [("lr", 1e-4), ("weight_decay", 0),
    ("betas", (.5, .9)), ("eps", 1e-4), ("maximize", True), ("amsgrad", True)])
def test_rejects_changed_adamw(key, bad):
    value = optimizer_state()
    value["optimizer_state_dict"]["param_groups"][0][key] = bad
    with pytest.raises(ValueError):
        audit.validate_optimizer(value)


@pytest.mark.parametrize("key,bad", [("exp_avg", torch.tensor([float("nan")]*2)),
    ("exp_avg_sq", torch.tensor([float("inf")]*2)), ("exp_avg_sq", torch.ones(3)),
    ("exp_avg_sq", -torch.ones(2)), ("exp_avg", None)])
def test_rejects_bad_moments(key, bad):
    value = optimizer_state()
    value["optimizer_state_dict"]["state"][7][key] = bad
    with pytest.raises(ValueError, match="moments"):
        audit.validate_optimizer(value)


def test_rejects_missing_parameter_state():
    value = optimizer_state()
    del value["optimizer_state_dict"]["state"][7]
    with pytest.raises(ValueError, match="identity/count"):
        audit.validate_optimizer(value)


def test_nested_nonfinite():
    audit.require_finite({"ok": [0, True, 1.0, "train"]})
    with pytest.raises(ValueError, match="nonfinite"):
        audit.require_finite({"panel": [0, {"loss": float("nan")} ]})


def records():
    row = {"identity": {"case": "synthetic_train_case", "start": 0,
                        "dataset_index": 0, "split": "train", "rollout_steps": 100},
           "chunk_size": 10, "chunks": 10,
           "h1_balanced": 1., "ar_balanced": 1., "total": 1.,
           "h1_channel_mse": [1.] * 4, "ar_channel_mse": [1.] * 4}
    gradient = {"official_frozen_parameter_names": list(audit.OFFICIAL_FROZEN_PARAMETER_NAMES),
        "official_frozen_parameter_shapes": {"spec_encoder.lift_network.0.conv.bias": [24],
                                             "spec_encoder.lift_network.2.conv.bias": [48]},
        "trainable_parameter_tensor_count": 28, "trainable_gradient_all_finite": True}
    return [{"update": n, "consumed_windows": n * 8, "windows": 8, "optimizer_steps": 1,
             "preclip_mean_gradient_norm": 2., "gradient_audit": copy.deepcopy(gradient),
             "records": [copy.deepcopy(row) for _ in range(8)],
             "mean_objective": {"h1_balanced": 1., "ar_balanced": 1., "total": 1.}}
            for n in range(1, 172)]


def test_group_flattening_preserves_consumption_order():
    groups = records()
    for n, row in enumerate(r for group in groups for r in group["records"]):
        row["identity"]["start"] = n
    flattened = audit.validate_records(groups)
    assert [r["identity"]["start"] for r in flattened] == list(range(1368))


@pytest.mark.parametrize("key,bad", [("update", 2), ("consumed_windows", 7),
    ("windows", 1), ("optimizer_steps", 8), ("preclip_mean_gradient_norm", float("nan"))])
def test_update_boundary_rejected(key, bad):
    groups = records()
    groups[0][key] = bad
    with pytest.raises(ValueError):
        audit.validate_records(groups)


def test_eight_records_and_correct_mean_required():
    groups = records()
    groups[0]["records"].pop()
    with pytest.raises(ValueError, match="eight"):
        audit.validate_records(groups)
    groups = records()
    groups[0]["mean_objective"]["total"] = 8.
    with pytest.raises(ValueError, match="mean objective"):
        audit.validate_records(groups)


def test_chunk_normalization_and_train_only():
    groups = records()
    groups[0]["records"][0]["h1_balanced"] = 10.
    with pytest.raises(ValueError, match="normalization"):
        audit.validate_records(groups)
    groups = records()
    groups[0]["records"][0]["identity"]["split"] = "validation"
    with pytest.raises(ValueError, match="train window"):
        audit.validate_records(groups)


def panels():
    data = {"force_forward_batch_sizes": [20] * 10,
            "force_predictor_training_mode": True, "autograd_enabled": False,
            "domains": {domain: {"rear_cl_physical_residual": [0.] * 100,
                "h100": audit.diagnostic.residual_statistics([0.] * 100),
                "tail62": audit.diagnostic.residual_statistics([0.] * 62)} for domain in ("h1", "ar")}}
    rows = [{"global_index": index, "family": family,
             "identity": {"case": case, "start": start, "split": "train", "rollout_steps": 100},
             "panel": copy.deepcopy(data)} for index, family, case, start in audit.diagnostic.WINDOWS]
    return [{"consumed_windows": consumed, "optimizer_steps": consumed // 8,
             "no_grad": True, "model_selection": False,
             "tensor_sha256_before": ["flow", aero], "tensor_sha256_after": ["flow", aero],
             "rows": copy.deepcopy(rows)}
            for consumed, aero in [(0, "initial"), (456, "a"), (912, "b"), (1368, "terminal")]]


def test_fixed_panels():
    audit.validate_panels(panels(), "flow", "initial", "terminal")


@pytest.mark.parametrize("failure", ["boundary", "mutation", "selection", "residual", "window", "batch"])
def test_bad_panel_rejected(failure):
    p = panels()
    if failure == "boundary": p[1]["consumed_windows"] = 457
    if failure == "mutation": p[2]["tensor_sha256_after"] = ["flow", "changed"]
    if failure == "selection": p[1]["model_selection"] = True
    if failure == "residual": p[0]["rows"][0]["panel"]["domains"]["h1"]["h100"]["mse"] = 1.
    if failure == "window": p[0]["rows"][0]["identity"]["start"] += 1
    if failure == "batch": p[0]["rows"][0]["panel"]["force_forward_batch_sizes"] = [10] * 10
    with pytest.raises(ValueError):
        audit.validate_panels(p, "flow", "initial", "terminal")


def test_actual_p015_resource_schema():
    row = {"timestamp": 123, "mem_available_kib": 30 * 1024**2, "mem_free_kib": 21 * 1024**2}
    assert audit.validate_resource_watch(json.dumps(row))["min_mem_free_gib"] == 21
    row["mem_free_kib"] = 19 * 1024**2
    with pytest.raises(ValueError, match="floor"):
        audit.validate_resource_watch(json.dumps(row))
    with pytest.raises(ValueError):
        audit.validate_resource_watch("stopped container")


def training_result():
    return {**audit.EXPERIMENT,
        "status": "FC_P015_WINDOW_ACCUMULATION_TRAINING_COMPLETE_NOT_ADMISSION",
        "dual_model_manifest_sha256": "manifest",
        "official_pair_fresh_reload_verified": True,
        "dual_adapter_fresh_reload_verified": False,
        "dual_adapter_reload_status": "SEPARATE_P015_PROFILE_VERIFICATION_REQUIRED",
        "input_sha256": audit.INPUT_SHA, "config_sha256": audit.CONFIG_SHA,
        "sampler_order_sha256": audit.ORDER_SHA, "precision": audit.PRECISION_PROTOCOL,
        "p014_diagnostic_sha256": audit.P014_SHA, "source_sha256": audit.diagnostic.SOURCE_SHA,
        "selection_performed": False, "validation_accessed": False,
        "frozen_test_accessed": False, "ppo_executed": False,
        "flow_tensor_sha256_before": "flow", "aerodynamic_tensor_sha256_before": "initial",
        "aerodynamic_tensor_sha256_after": "terminal", "records": records(), "fixed_train_panels": panels()}


def test_p015_result_is_separate_from_p013():
    assert len(audit.validate_training_result(training_result(), "manifest")) == 1368


@pytest.mark.parametrize("key,bad", [("training_experiment", "FC-P013"),
    ("optimizer_steps", 1368), ("accumulation_windows", 1),
    ("official_pair_fresh_reload_verified", False), ("dual_adapter_fresh_reload_verified", True),
    ("validation_accessed", True), ("frozen_test_accessed", True), ("ppo_executed", True),
    ("sampler_order_sha256", "changed"), ("selection_performed", 0)])
def test_result_contract_failures(key, bad):
    result = training_result()
    result[key] = bad
    with pytest.raises(ValueError, match="result contract"):
        audit.validate_training_result(result, "manifest")


def execution_fixture(tmp_path):
    def write(name, value):
        (tmp_path / name).write_text(json.dumps(value))
    (tmp_path / "immutable_launcher.sh").write_text("software fixture")
    launcher_sha = audit.sha256(tmp_path / "immutable_launcher.sh")
    write("execution_approval.json", {
        "status": "FC_P015_APPROVED_FIXED_ACCUMULATION_TRAINING_NOT_ADMISSION",
        "source_sha256": {audit.TRAINER_KEY: "1" * 64}, "launcher_sha256": launcher_sha})
    approval_sha = audit.sha256(tmp_path / "execution_approval.json")
    write("running_execution_evidence.json", {
        "status": "FC_P015_RUNNING_EXECUTION_OBSERVED", "image_id": audit.IMAGE,
        "approval_sha256": approval_sha, "attempt": 1,
        "retrospective_launch_approval": False, "validation_or_frozen_mounted": False,
        "training_complete": False, "source_sha256": {audit.TRAINER_KEY: "1" * 64},
        "executed_launcher_sha256": launcher_sha, "container_id": "fixture-container",
        "unit": {"ActiveState": "active", "SubState": "running", "MainPID": "123",
                 "InvocationID": "fixture-invocation"}})
    write("run.log", {"event": "gpu_guard_complete", "exit_code": 0,
        "min_required_mem_available_gib": 20, "min_observed_mem_available_gib": 30, "memory_samples": 10})
    write("resource_watch.jsonl", {"timestamp": 123, "mem_available_kib": 30 * 1024**2,
                                   "mem_free_kib": 21 * 1024**2})
    return approval_sha, audit.sha256(tmp_path / "running_execution_evidence.json")


def test_pinned_execution_and_actual_guard(tmp_path):
    approval, observation = execution_fixture(tmp_path)
    assert audit.validate_execution(tmp_path, approval, observation)["guard"]["exit_code"] == 0
    (tmp_path / "resource_or_deadline_violation").touch()
    with pytest.raises(ValueError, match="watchdog"):
        audit.validate_execution(tmp_path, approval, observation)


@pytest.mark.parametrize("name", ["execution_approval.json", "running_execution_evidence.json", "immutable_launcher.sh"])
def test_execution_bytes_must_match_external_pins(tmp_path, name):
    approval, observation = execution_fixture(tmp_path)
    with (tmp_path / name).open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="SHA"):
        audit.validate_execution(tmp_path, approval, observation)


def test_real_nonzero_guard_rejected_even_with_approved_observation(tmp_path):
    approval, observation = execution_fixture(tmp_path)
    p = tmp_path / "run.log"
    data = json.loads(p.read_text())
    data["exit_code"] = 75
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="guard"):
        audit.validate_execution(tmp_path, approval, observation)
