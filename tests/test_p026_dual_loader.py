import importlib.util
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SCRIPT = Path(__file__).parents[1] / "src/fluid_control/dual_fno.py"
SPEC = importlib.util.spec_from_file_location("dual_fno_p026_test", SCRIPT)
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


def _protocol(k):
    value = {
        "training_experiment": "FC-P026",
        "history_input": MODULE._p026_history_input(k),
        "training_windows": 1368,
        "accumulation_windows": 8,
        "optimizer_steps": 171,
        "learning_rate": MODULE.P026_LEARNING_RATE,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
        "seed": 20261003,
        "chunk_size": 10,
        "rollout_steps": 100,
        "sampler_order_sha256": MODULE.P026_ORDER_SHA256,
        "diagnostic_counts": [0, 456, 912, 1368],
        "objective": "equal_H1_AR_half_equal_four_half_rearCl_normalized_MSE",
        "history_state_module_sha256": MODULE.P026_HISTORY_STATE_SHA256,
        "history_objective_sha256": MODULE.P026_HISTORY_OBJECTIVE_SHA256,
        "history_inference_module_sha256": MODULE.P026_HISTORY_INFERENCE_SHA256,
        "action_semantics": "stored_prescribed_action_samples_not_exact_nominal_time_commands",
        "inventory": MODULE._p026_inventory(),
        "allocator_fraction": 0.06,
        "wall_seconds": 14400,
        "validation_accessed": False,
        "frozen_test_accessed": False,
        "selection_performed": False,
    }
    return value


def p026_fixture(tmp_path, monkeypatch, k=4):
    flow_dir, flow_model, flow_state = _checkpoint(tmp_path, "flow", 0)
    aero_dir, aero_model, aero_state = _checkpoint(tmp_path, "aero", 1)
    monkeypatch.setattr(MODULE, "FLOW_MODEL_SHA256", MODULE.sha256(flow_model))
    monkeypatch.setattr(MODULE, "FLOW_STATE_SHA256", MODULE.sha256(flow_state))
    protocol = _protocol(k)
    protocol_path = tmp_path / "training_protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    kind = MODULE.P026_K1_SYSTEM_KIND if k == 1 else MODULE.P026_K4_SYSTEM_KIND
    status = (
        MODULE.P026_K1_MANIFEST_STATUS if k == 1 else MODULE.P026_K4_MANIFEST_STATUS
    )
    aero_kind = MODULE.P026_K1_AERO_KIND if k == 1 else MODULE.P026_K4_AERO_KIND
    aerodynamic_architecture = dict(MODULE.ARCHITECTURE)
    aerodynamic_architecture["in_channels"] = 6 if k == 1 else 18
    payload = {
        "schema_version": 1,
        "status": status,
        "kind": kind,
        "config_sha256": MODULE.CONFIG_SHA256,
        "normalization_sha256": MODULE.NORMALIZATION_SHA256,
        "precision_protocol": MODULE.PRECISION_PROTOCOL,
        "architecture": copy.deepcopy(MODULE.ARCHITECTURE),
        "flow_architecture": copy.deepcopy(MODULE.ARCHITECTURE),
        "aerodynamic_architecture": aerodynamic_architecture,
        "flow_parent_model_sha256": MODULE.FLOW_MODEL_SHA256,
        "flow_parent_state_sha256": MODULE.FLOW_STATE_SHA256,
        "aerodynamic_initial_model_sha256": MODULE.P026_AERO_INITIAL_MODEL_SHA256,
        "aerodynamic_initial_state_sha256": MODULE.P026_AERO_INITIAL_STATE_SHA256,
        "training_experiment": "FC-P026",
        "accumulation_windows": 8,
        "training_windows": 1368,
        "optimizer_steps": 171,
        "actual_learning_rate": MODULE.P026_LEARNING_RATE,
        "training_protocol_file": protocol_path.name,
        "training_protocol_sha256": MODULE.sha256(protocol_path),
        "training_semantics": protocol,
        "history_input": MODULE._p026_history_input(k),
        "history_inventory": MODULE._p026_inventory(),
        "history_state_module_sha256": MODULE.P026_HISTORY_STATE_SHA256,
        "history_inference_module_sha256": MODULE.P026_HISTORY_INFERENCE_SHA256,
        "flow": _row(
            "flow", flow_dir, flow_model, flow_state, 0, MODULE.FLOW_KIND
        ),
        "aerodynamic": _row(
            "aerodynamic", aero_dir, aero_model, aero_state, 1, aero_kind
        ),
    }
    path = tmp_path / "dual_model_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def _cfg():
    return SimpleNamespace(
        model=SimpleNamespace(
            **{
                key: value
                for key, value in MODULE.ARCHITECTURE.items()
                if key != "force_channels"
            }
        )
    )


@pytest.mark.parametrize("k", (1, 4))
def test_p026_role_architecture_injected_loader_contract(tmp_path, monkeypatch, k):
    path, payload = p026_fixture(tmp_path, monkeypatch, k)
    identity = MODULE.validate_dual_fno_manifest(path)
    assert identity.payload["history_input"]["history_length"] == k

    class Toy(torch.nn.Module):
        def __init__(self, in_channels):
            super().__init__()
            self.in_channels = in_channels

        def forward(self, value):
            assert value.shape[1] == self.in_channels
            return value[:, :1].expand(-1, 7, -1, -1)

    built = []

    def build_model(cfg):
        model = Toy(int(cfg.model.in_channels))
        built.append(model)
        return model

    contract = MODULE._experiment_contract(payload["kind"])

    def load_checkpoint(directory, *, models, metadata_dict, device):
        del models, device
        if directory.name == "flow":
            return 0
        metadata_dict.update(
            {
                "status": contract["aero_kind"],
                "checkpoint_epoch": 1,
                "flow_parent_model_sha256": MODULE.FLOW_MODEL_SHA256,
                "flow_parent_state_sha256": MODULE.FLOW_STATE_SHA256,
                "aerodynamic_initial_model_sha256": MODULE.P026_AERO_INITIAL_MODEL_SHA256,
                "aerodynamic_initial_state_sha256": MODULE.P026_AERO_INITIAL_STATE_SHA256,
                "optimizer_steps": 171,
                **contract["extra"],
                "history_profile": f"p026_k{k}",
                "history_k": k,
                "model_in_channels": 6 if k == 1 else 18,
                "training_protocol_sha256": payload["training_protocol_sha256"],
                "history_state_module_sha256": MODULE.P026_HISTORY_STATE_SHA256,
                "history_inference_module_sha256": MODULE.P026_HISTORY_INFERENCE_SHA256,
                "sampler_order_sha256": MODULE.P026_ORDER_SHA256,
                "history_inventory": MODULE._p026_inventory(),
                "selection_performed": False,
                "validation_accessed": False,
                "frozen_test_accessed": False,
                "ppo_executed": False,
            }
        )
        return 1

    import fluid_control.calibrated_checkpoint as calibrated

    monkeypatch.setattr(calibrated, "validate_calibrated_epoch_zero", lambda *a, **k: {})
    monkeypatch.setattr(MODULE, "validate_runtime_precision", lambda: MODULE.PRECISION_PROTOCOL)
    adapter, _ = MODULE.load_dual_fno(
        path,
        _cfg(),
        torch.device("cpu"),
        build_model=build_model,
        load_checkpoint=load_checkpoint,
    )
    assert [item.in_channels for item in built] == [6, 6 if k == 1 else 18]
    flow = torch.ones(2, 6, 3, 4)
    if k == 1:
        assert torch.equal(adapter(flow), torch.ones(2, 7, 3, 4))
    else:
        with pytest.raises(ValueError, match="requires an explicit"):
            adapter(flow)
        aero = torch.zeros(2, 18, 3, 4)
        aero[:, 9:12] = flow[:, :3]
        aero[:, 12:13] = flow[:, 3:4]
        aero[:, 16:17] = flow[:, 4:5]
        aero[:, 17:18] = flow[:, 5:6]
        assert adapter(flow, aero).shape == (2, 7, 3, 4)


@pytest.mark.parametrize(
    "mutation, message",
    (
        (lambda p: p.update(status="FC_P026_HISTORY_ENGINEERING_FIXTURE_NOT_CANDIDATE"), "fixed identity"),
        (lambda p: p.update(kind="FC_P026_HISTORY_ENGINEERING_FIXTURE_NOT_CANDIDATE"), "not supported"),
        (lambda p: p["flow_architecture"].update(in_channels=18), "role architecture"),
        (lambda p: p["aerodynamic_architecture"].update(in_channels=6), "role architecture"),
        (lambda p: p["history_input"].update(history_length=1), "history or role"),
        (lambda p: p["aerodynamic"].update(metadata_kind=MODULE.P026_K1_AERO_KIND), "epoch/kind"),
    ),
)
def test_p026_rejects_fixture_wrong_role_history_or_kind(
    tmp_path, monkeypatch, mutation, message
):
    path, payload = p026_fixture(tmp_path, monkeypatch, 4)
    mutation(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        MODULE.validate_dual_fno_manifest(path)


def test_p026_rejects_metadata_from_other_history_arm(tmp_path, monkeypatch):
    path, payload = p026_fixture(tmp_path, monkeypatch, 4)

    class Toy(torch.nn.Module):
        def forward(self, value):
            return value[:, :1].expand(-1, 7, -1, -1)

    def load_checkpoint(directory, *, models, metadata_dict, device):
        del models, device
        if directory.name == "flow":
            return 0
        contract = MODULE._experiment_contract(payload["kind"])
        metadata_dict.update(
            status=contract["aero_kind"],
            checkpoint_epoch=1,
            flow_parent_model_sha256=MODULE.FLOW_MODEL_SHA256,
            flow_parent_state_sha256=MODULE.FLOW_STATE_SHA256,
            aerodynamic_initial_model_sha256=MODULE.P026_AERO_INITIAL_MODEL_SHA256,
            aerodynamic_initial_state_sha256=MODULE.P026_AERO_INITIAL_STATE_SHA256,
            optimizer_steps=171,
            training_experiment="FC-P026",
            accumulation_windows=8,
            training_windows=1368,
            actual_learning_rate=MODULE.P026_LEARNING_RATE,
            training_protocol_file="training_protocol.json",
            history_profile="p026_k1",
            history_k=1,
            model_in_channels=6,
            training_protocol_sha256=payload["training_protocol_sha256"],
            history_state_module_sha256=MODULE.P026_HISTORY_STATE_SHA256,
            history_inference_module_sha256=MODULE.P026_HISTORY_INFERENCE_SHA256,
            sampler_order_sha256=MODULE.P026_ORDER_SHA256,
            history_inventory=MODULE._p026_inventory(),
            selection_performed=False,
            validation_accessed=False,
            frozen_test_accessed=False,
            ppo_executed=False,
        )
        return 1

    import fluid_control.calibrated_checkpoint as calibrated

    monkeypatch.setattr(calibrated, "validate_calibrated_epoch_zero", lambda *a, **k: {})
    monkeypatch.setattr(MODULE, "validate_runtime_precision", lambda: MODULE.PRECISION_PROTOCOL)
    with pytest.raises(ValueError, match="metadata differs"):
        MODULE.load_dual_fno(
            path,
            _cfg(),
            torch.device("cpu"),
            build_model=lambda cfg: Toy(),
            load_checkpoint=load_checkpoint,
        )


def test_p026_rejects_imported_history_module_with_different_bytes(
    tmp_path, monkeypatch
):
    mismatched = tmp_path / "p026_history_inference.py"
    mismatched.write_text("# wrong project glue bytes\n", encoding="utf-8")
    fake = SimpleNamespace(
        __file__=str(mismatched), make_history_dual_fno_adapter=lambda *a, **k: None
    )
    monkeypatch.setitem(sys.modules, "p026_history_inference", fake)
    with pytest.raises(ValueError, match="history module source SHA differs"):
        MODULE._p026_history_adapter_factory()


def test_legacy_adapter_path_remains_single_input_and_byte_equivalent():
    class Toy(torch.nn.Module):
        def forward(self, value):
            return torch.cat((value[:, :3], value[:, :4] * 2), 1)

    x = torch.arange(2 * 6 * 2 * 3, dtype=torch.float32).reshape(2, 6, 2, 3)
    adapter = MODULE.make_dual_fno_adapter(Toy(), Toy())
    assert torch.equal(adapter(x), Toy()(x))
