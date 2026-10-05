"""Cross-module contract between the FC-P013 trainer and dual evaluator."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_trainer_manifest_validates_and_loads_through_dual_adapter(tmp_path, monkeypatch):
    trainer = load_module(
        "fcp013_trainer_contract",
        ROOT / "scripts/train_fcp013_independent_force_fno.py",
    )
    dual = load_module(
        "fcp013_dual_contract", ROOT / "src/fluid_control/dual_fno.py"
    )
    flow = tmp_path / "flow"
    aerodynamic = tmp_path / "aerodynamic"
    flow.mkdir()
    aerodynamic.mkdir()
    flow_model = flow / "FNO.0.0.mdlus"
    flow_state = flow / "checkpoint.0.0.pt"
    aero_model = aerodynamic / "FNO.0.1.mdlus"
    aero_state = aerodynamic / "checkpoint.0.1.pt"
    flow_model.write_bytes(b"fake-flow-model")
    flow_state.write_bytes(b"fake-flow-state")
    aero_model.write_bytes(b"fake-aerodynamic-model")
    aero_state.write_bytes(b"fake-aerodynamic-state")
    parent_model_sha = trainer.sha256(flow_model)
    parent_state_sha = trainer.sha256(flow_state)
    for module in (trainer, dual):
        monkeypatch.setattr(module, "PARENT_MODEL_SHA" if module is trainer else "FLOW_MODEL_SHA256", parent_model_sha)
        monkeypatch.setattr(module, "PARENT_STATE_SHA" if module is trainer else "FLOW_STATE_SHA256", parent_state_sha)
    manifest = trainer.build_dual_manifest(tmp_path, aero_model, aero_state)
    identity = dual.validate_dual_fno_manifest(manifest)
    assert identity.flow.model == flow_model
    assert identity.aerodynamic.model == aero_model

    class Toy(torch.nn.Module):
        def forward(self, value):
            return value[:, :1].expand(-1, 7, -1, -1)

    built = []

    def build_model(_cfg):
        result = Toy()
        built.append(result)
        return result

    def load_checkpoint(directory, *, models, metadata_dict, device):
        del models, device
        if directory == flow:
            return 0
        metadata_dict.update(
            {
                "status": dual.AERO_KIND,
                "checkpoint_epoch": 1,
                "optimizer_steps": 1368,
                "flow_parent_model_sha256": parent_model_sha,
                "flow_parent_state_sha256": parent_state_sha,
                "aerodynamic_initial_model_sha256": parent_model_sha,
                "aerodynamic_initial_state_sha256": parent_state_sha,
                "selection_performed": False,
                "validation_accessed": False,
                "frozen_test_accessed": False,
                "ppo_executed": False,
            }
        )
        return 1

    import fluid_control.calibrated_checkpoint as calibrated

    monkeypatch.setattr(calibrated, "validate_calibrated_epoch_zero", lambda *a, **k: {})
    monkeypatch.setattr(dual, "validate_runtime_precision", lambda: dual.PRECISION_PROTOCOL)
    cfg = SimpleNamespace(
        model=SimpleNamespace(
            **{
                key: value
                for key, value in dual.ARCHITECTURE.items()
                if key != "force_channels"
            }
        )
    )
    adapter, loaded_identity = dual.load_dual_fno(
        manifest,
        cfg,
        torch.device("cpu"),
        build_model=build_model,
        load_checkpoint=load_checkpoint,
    )
    assert len(built) == 2 and built[0] is not built[1]
    assert adapter.flow_model is built[0]
    assert adapter.aerodynamic_model is built[1]
    assert loaded_identity.manifest_sha256 == trainer.sha256(manifest)
