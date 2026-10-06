import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SCRIPT = Path(__file__).parents[1] / "src/fluid_control/dual_fno.py"
SPEC = importlib.util.spec_from_file_location("dual_fno_p064_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _checkpoint(root, name, epoch):
    directory = root / name
    directory.mkdir()
    model = directory / f"FNO.0.{epoch}.mdlus"
    state = directory / f"checkpoint.0.{epoch}.pt"
    model.write_bytes(f"{name}-model".encode())
    state.write_bytes(f"{name}-state".encode())
    return directory, model, state


def _row(role, directory, model, state, epoch, kind):
    return {
        "role": role,
        "checkpoint_relative_directory": directory.name,
        "model_file": model.name,
        "state_file": state.name,
        "checkpoint_epoch": epoch,
        "model_sha256": MODULE.sha256(model),
        "state_sha256": MODULE.sha256(state),
        "metadata_kind": kind,
        "frozen": role == "flow",
    }


def _protocol(arm):
    return {
        "training_experiment": "FC-P064",
        "arm": arm,
        "parent_experiment": "FC-P026-K1",
        "history_input": MODULE._p026_history_input(1),
        "training_windows": 256,
        "accumulation_windows": 8,
        "optimizer_steps": 32,
        "learning_rate": MODULE.P026_LEARNING_RATE,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
        "seed": 20261003,
        "chunk_size": 10,
        "rollout_steps": 100,
        "parent_sampler_order_sha256": MODULE.P026_ORDER_SHA256,
        "schedule_sha256": MODULE.P064_SCHEDULE_SHA256[arm],
        "diagnostic_counts": [0, 256],
        "objective": "equal_H1_AR_half_equal_four_half_rearCl_normalized_MSE",
        "history_state_module_sha256": MODULE.P026_HISTORY_STATE_SHA256,
        "history_objective_sha256": MODULE.P026_HISTORY_OBJECTIVE_SHA256,
        "history_inference_module_sha256": MODULE.P026_HISTORY_INFERENCE_SHA256,
        "action_semantics": "stored_prescribed_action_samples_not_exact_nominal_time_commands",
        "controlled_b00_action_semantics": (
            "actual_closed_loop_applied_endpoint_omega_samples"
            if arm == "B"
            else "not_applicable_no_b00_windows"
        ),
        "b00_windows": 0 if arm == "A" else 64,
        "b00_weight": 0.0 if arm == "A" else 0.25,
        "replacement_within_each_update": [] if arm == "A" else [0, 4],
        "allocator_fraction": 0.06,
        "wall_seconds": 3600,
        "validation_accessed": False,
        "frozen_test_accessed": False,
        "selection_performed": False,
    }


def fixture(tmp_path, monkeypatch, arm):
    flow_dir, flow_model, flow_state = _checkpoint(tmp_path, "flow", 0)
    aero_dir, aero_model, aero_state = _checkpoint(tmp_path, "aero", 1)
    monkeypatch.setattr(MODULE, "FLOW_MODEL_SHA256", MODULE.sha256(flow_model))
    monkeypatch.setattr(MODULE, "FLOW_STATE_SHA256", MODULE.sha256(flow_state))
    protocol = _protocol(arm)
    protocol_path = tmp_path / "training_protocol.json"
    protocol_path.write_text(json.dumps(protocol, sort_keys=True, separators=(",", ":")))
    payload = {
        "schema_version": 1,
        "status": MODULE.P064_MANIFEST_STATUS[arm],
        "kind": MODULE.P064_SYSTEM_KIND[arm],
        "config_sha256": MODULE.CONFIG_SHA256,
        "normalization_sha256": MODULE.NORMALIZATION_SHA256,
        "precision_protocol": MODULE.PRECISION_PROTOCOL,
        "architecture": copy.deepcopy(MODULE.ARCHITECTURE),
        "flow_architecture": copy.deepcopy(MODULE.ARCHITECTURE),
        "aerodynamic_architecture": copy.deepcopy(MODULE.ARCHITECTURE),
        "flow_parent_model_sha256": MODULE.FLOW_MODEL_SHA256,
        "flow_parent_state_sha256": MODULE.FLOW_STATE_SHA256,
        "aerodynamic_initial_model_sha256": MODULE.P028_AERO_PARENT_MODEL_SHA256,
        "aerodynamic_initial_state_sha256": MODULE.P028_AERO_PARENT_STATE_SHA256,
        "aerodynamic_parent_model_sha256": MODULE.P028_AERO_PARENT_MODEL_SHA256,
        "aerodynamic_parent_state_sha256": MODULE.P028_AERO_PARENT_STATE_SHA256,
        "training_experiment": "FC-P064",
        "arm": arm,
        "accumulation_windows": 8,
        "training_windows": 256,
        "optimizer_steps": 32,
        "actual_learning_rate": MODULE.P026_LEARNING_RATE,
        "training_protocol_file": protocol_path.name,
        "training_protocol_sha256": MODULE.sha256(protocol_path),
        "training_semantics": protocol,
        "history_input": MODULE._p026_history_input(1),
        "parent_history_inventory": MODULE._p026_inventory(),
        "history_state_module_sha256": MODULE.P026_HISTORY_STATE_SHA256,
        "history_inference_module_sha256": MODULE.P026_HISTORY_INFERENCE_SHA256,
        "flow": _row("flow", flow_dir, flow_model, flow_state, 0, MODULE.FLOW_KIND),
        "aerodynamic": _row(
            "aerodynamic", aero_dir, aero_model, aero_state, 1, MODULE.P064_AERO_KIND[arm]
        ),
    }
    path = tmp_path / "dual_model_manifest.json"
    path.write_text(json.dumps(payload))
    return path, payload


def _cfg():
    return SimpleNamespace(
        model=SimpleNamespace(
            **{k: v for k, v in MODULE.ARCHITECTURE.items() if k != "force_channels"}
        )
    )


@pytest.mark.parametrize("arm", ("A", "B"))
def test_p064_manifest_and_injected_official_loader_contract(tmp_path, monkeypatch, arm):
    path, payload = fixture(tmp_path, monkeypatch, arm)
    identity = MODULE.validate_dual_fno_manifest(path)
    assert identity.payload["training_windows"] == 256
    assert identity.payload["optimizer_steps"] == 32

    class Toy(torch.nn.Module):
        def forward(self, value):
            return value[:, :1].expand(-1, 7, -1, -1)

    def load_checkpoint(directory, *, models, metadata_dict, device):
        del models, device
        if directory.name == "flow":
            return 0
        metadata_dict.update(
            status=MODULE.P064_AERO_KIND[arm],
            checkpoint_epoch=1,
            training_experiment="FC-P064",
            arm=arm,
            history_profile="p026_k1",
            history_k=1,
            model_in_channels=6,
            training_protocol_sha256=payload["training_protocol_sha256"],
            training_protocol_file="training_protocol.json",
            history_state_module_sha256=MODULE.P026_HISTORY_STATE_SHA256,
            history_inference_module_sha256=MODULE.P026_HISTORY_INFERENCE_SHA256,
            training_windows=256,
            optimizer_steps=32,
            accumulation_windows=8,
            actual_learning_rate=MODULE.P026_LEARNING_RATE,
            parent_sampler_order_sha256=MODULE.P026_ORDER_SHA256,
            schedule_sha256=MODULE.P064_SCHEDULE_SHA256[arm],
            parent_history_inventory=MODULE._p026_inventory(),
            flow_parent_model_sha256=MODULE.FLOW_MODEL_SHA256,
            flow_parent_state_sha256=MODULE.FLOW_STATE_SHA256,
            aerodynamic_parent_model_sha256=MODULE.P028_AERO_PARENT_MODEL_SHA256,
            aerodynamic_parent_state_sha256=MODULE.P028_AERO_PARENT_STATE_SHA256,
            selection_performed=False,
            validation_accessed=False,
            frozen_test_accessed=False,
            ppo_executed=False,
        )
        return 1

    import fluid_control.calibrated_checkpoint as calibrated

    monkeypatch.setattr(calibrated, "validate_calibrated_epoch_zero", lambda *a, **k: {})
    monkeypatch.setattr(MODULE, "validate_runtime_precision", lambda: MODULE.PRECISION_PROTOCOL)
    adapter, loaded_identity = MODULE.load_dual_fno(
        path,
        _cfg(),
        torch.device("cpu"),
        build_model=lambda cfg: Toy(),
        load_checkpoint=load_checkpoint,
    )
    assert loaded_identity.payload["arm"] == arm
    assert adapter(torch.zeros(1, 6, 2, 2)).shape == (1, 7, 2, 2)


@pytest.mark.parametrize(
    "mutation,message",
    (
        (lambda p: p.update(training_windows=1368), "fixed identity"),
        (lambda p: p.update(optimizer_steps=171), "fixed identity"),
        (lambda p: p.update(kind=MODULE.P064_SYSTEM_KIND["B"]), "fixed identity"),
        (lambda p: p["training_semantics"].update(schedule_sha256="0" * 64), "protocol"),
    ),
)
def test_p064_rejects_cross_arm_or_old_budget(tmp_path, monkeypatch, mutation, message):
    path, payload = fixture(tmp_path, monkeypatch, "A")
    mutation(payload)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=message):
        MODULE.validate_dual_fno_manifest(path)


def test_p064_engineering_fixture_requires_explicit_opt_in(tmp_path, monkeypatch):
    path, payload = fixture(tmp_path, monkeypatch, "A")
    payload["engineering_fixture_not_candidate"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="not a scientific candidate"):
        MODULE.validate_dual_fno_manifest(path)
    assert MODULE.validate_dual_fno_manifest(
        path, allow_engineering_fixture=True
    ).payload["engineering_fixture_not_candidate"] is True
