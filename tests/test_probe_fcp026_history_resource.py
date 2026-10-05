import inspect
import json
from pathlib import Path

import pytest
import torch
import probe_fcp026_history_resource as p

SCRIPTS = Path(p.__file__).resolve().parent


def test_limits_and_deadline():
    p.limits(dict(MemFree=30, MemAvailable=50), 0, True)
    for memory, elapsed, startup in [
        (dict(MemFree=29, MemAvailable=80), 0, True),
        (dict(MemFree=40, MemAvailable=49), 0, True),
        (dict(MemFree=19, MemAvailable=80), 1, False),
        (dict(MemFree=40, MemAvailable=19), 1, False),
        (dict(MemFree=40, MemAvailable=80), 901, False),
    ]:
        with pytest.raises(RuntimeError):
            p.limits(memory, elapsed, startup)


class Model:
    def __init__(self):
        self.names = {name: torch.nn.Parameter(torch.ones(1)) for name in p.FROZEN}
        self.names["lift"] = torch.nn.Parameter(torch.ones(24, 20, 1, 1))
        self.names.update(
            {f"other{i}": torch.nn.Parameter(torch.ones(1)) for i in range(27)}
        )

    def named_parameters(self):
        return self.names.items()

    def parameters(self):
        return self.names.values()


def test_exact_scope_finite_newcolumns_and_frozen_bias():
    m = Model()
    p.trainable_scope(m)
    for param in m.parameters():
        if param.requires_grad:
            param.grad = torch.ones_like(param)
    report = p.gradient_report(m, "lift", 4)
    assert report["trainable_tensors"] == 28 and report["new_history_scalars"] == 288
    assert report["new_history_gradient_norm"] > 0
    m.names[p.FROZEN[0]].grad = torch.ones(1)
    with pytest.raises(RuntimeError, match="bias"):
        p.gradient_report(m, "lift", 4)
    m.names[p.FROZEN[0]].grad = None
    m.names["lift"].grad.fill_(float("nan"))
    with pytest.raises(RuntimeError, match="finite"):
        p.gradient_report(m, "lift", 4)


def test_zero_new_history_gradient_rejected():
    m = Model()
    p.trainable_scope(m)
    for param in m.parameters():
        if param.requires_grad:
            param.grad = torch.zeros_like(param)
    with pytest.raises(RuntimeError, match="288"):
        p.gradient_report(m, "lift", 4)


def test_parent_metadata_roles_are_not_interchangeable():
    root = Path(
        "/workspace/fluid_control/artifacts/fcp018_reduced_rate_training_20261005/candidate"
    )
    manifest = json.loads((root / "dual_model_manifest.json").read_text())
    keys = (
        "training_experiment",
        "accumulation_windows",
        "training_windows",
        "optimizer_steps",
        "actual_learning_rate",
        "training_protocol_sha256",
        "training_protocol_file",
        "flow_parent_model_sha256",
        "flow_parent_state_sha256",
        "aerodynamic_initial_model_sha256",
        "aerodynamic_initial_state_sha256",
    )
    metadata = {key: manifest[key] for key in keys}
    metadata.update(
        status=manifest["aerodynamic"]["metadata_kind"],
        selection_performed=False,
        validation_accessed=False,
        frozen_test_accessed=False,
        ppo_executed=False,
    )
    p.validate_metadata("aerodynamic", 1, metadata, manifest)
    with pytest.raises(ValueError):
        p.validate_metadata("flow", 0, metadata, manifest)
    metadata["actual_learning_rate"] = 1e-5
    with pytest.raises(ValueError):
        p.validate_metadata("aerodynamic", 1, metadata, manifest)
    flow = dict(
        status=manifest["flow"]["metadata_kind"],
        candidate_checkpoint_epoch=0,
        parent_checkpoint_epoch=2,
        calibration_generation=1,
        alpha=0.0,
        domain_mix=dict(free_ar=0.5, matched_weight_h1=0.5),
        calibration_fit_performed=True,
        optimizer_training_performed=False,
        validation_accessed=False,
        frozen_test_accessed=False,
        ppo_executed=False,
    )
    p.validate_metadata("flow", 0, flow, manifest)
    flow["optimizer_training_performed"] = True
    with pytest.raises(ValueError):
        p.validate_metadata("flow", 0, flow, manifest)


def test_dependencies_and_no_optimizer():
    stage = SCRIPTS
    assert p.sha(stage / "p026_state_history.py") == p.HISTORY_SHA
    assert p.sha(stage / "p026_history_objective.py") == p.OBJECTIVE_SHA
    text = inspect.getsource(p.execute)
    assert "torch.optim" not in text and "save_checkpoint" not in text
    assert "rtol=1e-5, atol=1e-6" in text
    assert "range(0,100" not in text  # objective owns time chunks, not a second loop


def test_launcher_only_actual_dependencies_and_both_memory_guards():
    text = (SCRIPTS / "run_fcp026_history_resource_spark.sh").read_text()
    assert (
        "raw_source_view" not in text
        and "causal_audit" not in text
        and "prior_result" not in text
    )
    assert "--memory 12g" in text and "--allocator-fraction .06" in text
    assert "available<20*1024*1024 || free<20*1024*1024" in text
    assert "O_NOFOLLOW" in text and "POSIX_FADV_DONTNEED" in text
    assert "fc.p026.approval" in text and "is_relative_to" in text
    assert "gpu_pids=$(timeout 10 nvidia-smi" in text
    assert 'if ! kill -0 "$watch_pid"' in text
    assert 'wait "$watch_pid" ||' in text
    assert "runtime_container.json" in text
    parent_loop = text.split('while kill -0 "$run_pid"', 1)[1].split("done", 1)[0]
    assert "owned_stop || true" in parent_loop
    assert 'kill -TERM "$run_pid"' not in parent_loop
    assert 'wait "$run_pid"' not in parent_loop
    assert "exit 70" not in parent_loop
    assert "timeout -k 20 1040 docker run" in text


def test_exact_parent_parameter_schema():
    parent = Model()
    parent.names["lift"] = torch.nn.Parameter(torch.ones(24, 8, 1, 1))
    arm = Model()
    p.trainable_scope(arm)
    p.validate_arm_schema(arm, parent, 4, "lift")
    arm.names["renamed"] = arm.names.pop("other0")
    with pytest.raises(ValueError, match="names"):
        p.validate_arm_schema(arm, parent, 4, "lift")
