import importlib.util
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).parents[1] / "src/fluid_control/dual_fno.py"
SPEC = importlib.util.spec_from_file_location("dual_fno_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def checkpoint(root, name, epoch):
    directory = root / name
    directory.mkdir()
    model = directory / f"FNO.0.{epoch}.mdlus"
    state = directory / f"checkpoint.0.{epoch}.pt"
    model.write_bytes(f"{name}-model".encode())
    state.write_bytes(f"{name}-state".encode())
    return directory, model, state


def manifest_fixture(tmp_path, monkeypatch):
    _, flow_model, flow_state = checkpoint(tmp_path, "flow", 0)
    _, aero_model, aero_state = checkpoint(tmp_path, "aero", 1)
    flow_model_sha = MODULE.sha256(flow_model)
    flow_state_sha = MODULE.sha256(flow_state)
    monkeypatch.setattr(MODULE, "FLOW_MODEL_SHA256", flow_model_sha)
    monkeypatch.setattr(MODULE, "FLOW_STATE_SHA256", flow_state_sha)
    def row(role, directory, model, state, epoch, kind):
        return {
            "role": role,
            "checkpoint_relative_directory": directory,
            "model_file": model.name,
            "state_file": state.name,
            "checkpoint_epoch": epoch,
            "model_sha256": MODULE.sha256(model),
            "state_sha256": MODULE.sha256(state),
            "metadata_kind": kind,
            "frozen": role == "flow",
        }
    payload = {
        "schema_version": 1,
        "status": MODULE.MANIFEST_STATUS,
        "kind": MODULE.SYSTEM_KIND,
        "config_sha256": MODULE.CONFIG_SHA256,
        "normalization_sha256": MODULE.NORMALIZATION_SHA256,
        "precision_protocol": MODULE.PRECISION_PROTOCOL,
        "architecture": MODULE.ARCHITECTURE,
        "flow_parent_model_sha256": flow_model_sha,
        "flow_parent_state_sha256": flow_state_sha,
        "aerodynamic_initial_model_sha256": flow_model_sha,
        "aerodynamic_initial_state_sha256": flow_state_sha,
        "flow": row("flow", "flow", flow_model, flow_state, 0, MODULE.FLOW_KIND),
        "aerodynamic": row(
            "aerodynamic", "aero", aero_model, aero_state, 1, MODULE.AERO_KIND
        ),
    }
    path = tmp_path / "dual_fno_manifest.json"
    path.write_text(json.dumps(payload))
    return path, payload


def test_dual_adapter_identical_models_matches_single_raw_and_state_semantics():
    class Toy(torch.nn.Module):
        def forward(self, value):
            return torch.cat((value[:, :3] * 0.25, value[:, :4] - 0.1), dim=1)

    model = Toy()
    adapter = MODULE.make_dual_fno_adapter(model, Toy())
    inputs = torch.arange(2 * 6 * 3 * 4, dtype=torch.float32).reshape(2, 6, 3, 4)
    mask = torch.ones(2, 1, 3, 4)
    old_raw = model(inputs)
    new_raw = adapter(inputs)
    assert torch.equal(new_raw, old_raw)
    old_state = (inputs[:, :3] + old_raw[:, :3]) * mask
    new_state = (inputs[:, :3] + new_raw[:, :3]) * mask
    old_force = (old_raw[:, 3:] * mask).sum((-2, -1)) / mask.sum((-2, -1))
    new_force = (new_raw[:, 3:] * mask).sum((-2, -1)) / mask.sum((-2, -1))
    assert torch.equal(new_state, old_state)
    assert torch.equal(new_force, old_force)


def test_changing_aerodynamic_model_cannot_change_state_trajectory():
    class Constant(torch.nn.Module):
        def __init__(self, values):
            super().__init__()
            self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

        def forward(self, value):
            return self.values[None, :, None, None].expand(
                value.shape[0], -1, value.shape[2], value.shape[3]
            )

    flow = Constant(range(7))
    first = MODULE.make_dual_fno_adapter(flow, Constant(range(7)))
    second = MODULE.make_dual_fno_adapter(flow, Constant(range(10, 17)))
    state1 = state2 = torch.zeros(1, 3, 2, 2)
    mask = torch.ones(1, 1, 2, 2)
    for _ in range(100):
        actions = torch.zeros(1, 3, 2, 2)
        raw1, raw2 = first(torch.cat((state1, actions), 1)), second(
            torch.cat((state2, actions), 1)
        )
        state1 = (state1 + raw1[:, :3]) * mask
        state2 = (state2 + raw2[:, :3]) * mask
    assert torch.equal(state1, state2)
    assert not torch.equal(raw1[:, 3:], raw2[:, 3:])


def test_manifest_accepts_exact_independent_pair(tmp_path, monkeypatch):
    path, _ = manifest_fixture(tmp_path, monkeypatch)
    identity = MODULE.validate_dual_fno_manifest(
        path, expected_sha256=MODULE.sha256(path)
    )
    assert identity.flow.epoch == 0
    assert identity.aerodynamic.epoch == 1
    assert identity.flow.directory != identity.aerodynamic.directory


def test_runtime_files_are_bound_to_manifest(tmp_path, monkeypatch):
    path, payload = manifest_fixture(tmp_path, monkeypatch)
    config = tmp_path / "resolved_config.yaml"
    normalization = tmp_path / "normalization.json"
    config.write_text("config")
    normalization.write_text("normalization")
    payload["config_sha256"] = MODULE.sha256(config)
    payload["normalization_sha256"] = MODULE.sha256(normalization)
    monkeypatch.setattr(MODULE, "CONFIG_SHA256", payload["config_sha256"])
    monkeypatch.setattr(MODULE, "NORMALIZATION_SHA256", payload["normalization_sha256"])
    path.write_text(json.dumps(payload))
    identity = MODULE.validate_dual_fno_manifest(path)
    MODULE.validate_dual_runtime_files(
        identity, config_path=config, normalization_path=normalization
    )
    normalization.write_text("tampered")
    with pytest.raises(ValueError, match="normalization SHA differs"):
        MODULE.validate_dual_runtime_files(
            identity, config_path=config, normalization_path=normalization
        )


def test_official_loader_constructs_two_models_and_checks_aero_metadata(
    tmp_path, monkeypatch
):
    path, _ = manifest_fixture(tmp_path, monkeypatch)

    class Toy(torch.nn.Module):
        def forward(self, value):
            return value[:, :1].expand(-1, 7, -1, -1)

    built = []

    def build_model(_cfg):
        model = Toy()
        built.append(model)
        return model

    def load_checkpoint(directory, *, models, metadata_dict, device):
        del models, device
        if directory.name == "flow":
            return 0
        metadata_dict.update(
            {
                "status": MODULE.AERO_KIND,
                "checkpoint_epoch": 1,
                "flow_parent_model_sha256": MODULE.FLOW_MODEL_SHA256,
                "flow_parent_state_sha256": MODULE.FLOW_STATE_SHA256,
                "aerodynamic_initial_model_sha256": MODULE.FLOW_MODEL_SHA256,
                "aerodynamic_initial_state_sha256": MODULE.FLOW_STATE_SHA256,
                "optimizer_steps": 1368,
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
    model = MODULE.ARCHITECTURE
    cfg = SimpleNamespace(model=SimpleNamespace(**{k: v for k, v in model.items() if k != "force_channels"}))
    adapter, identity = MODULE.load_dual_fno(
        path,
        cfg,
        torch.device("cpu"),
        build_model=build_model,
        load_checkpoint=load_checkpoint,
    )
    assert len(built) == 2 and built[0] is not built[1]
    assert adapter.flow_model is built[0]
    assert adapter.aerodynamic_model is built[1]
    assert identity.aerodynamic.epoch == 1


@pytest.mark.parametrize("failure", ("missing", "swapped", "escape", "extra_epoch", "wrong_sha"))
def test_manifest_rejects_invalid_checkpoint_contract(tmp_path, monkeypatch, failure):
    path, payload = manifest_fixture(tmp_path, monkeypatch)
    if failure == "missing":
        del payload["aerodynamic"]["state_file"]
    elif failure == "swapped":
        payload["aerodynamic"]["role"] = "flow"
    elif failure == "escape":
        payload["aerodynamic"]["checkpoint_relative_directory"] = "../outside"
    elif failure == "extra_epoch":
        (tmp_path / "aero/FNO.0.2.mdlus").write_bytes(b"extra")
    else:
        payload["aerodynamic"]["model_sha256"] = "0" * 64
    path.write_text(json.dumps(payload))
    with pytest.raises((ValueError, FileNotFoundError)):
        MODULE.validate_dual_fno_manifest(path)


def test_combine_rejects_malformed_shapes():
    with pytest.raises(ValueError, match="shapes differ"):
        MODULE.combine_dual_raw(torch.zeros(1, 7, 2, 2), torch.zeros(1, 6, 2, 2))


def test_single_model_selection_remains_default_and_dual_is_mutually_exclusive():
    assert MODULE.dual_fno_requested(None, None) is False
    assert MODULE.dual_fno_requested(Path("manifest.json"), "a" * 64) is True
    with pytest.raises(ValueError, match="without a manifest"):
        MODULE.dual_fno_requested(None, "a" * 64)
    with pytest.raises(ValueError, match="expected manifest SHA"):
        MODULE.dual_fno_requested(Path("manifest.json"), None)
    with pytest.raises(ValueError, match="single-model"):
        MODULE.dual_fno_requested(
            Path("manifest.json"),
            "a" * 64,
            single_model_calibrated_arguments=(True, None),
        )
