import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
DUAL = ROOT / "src/fluid_control/dual_fno_absolute64.py"
WORKER = ROOT / "scripts/evaluate_p064_absolute64_development_h1_h5_r2.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_absolute64_contract_is_explicit_epoch2_and_64_updates():
    dual = load(DUAL, "absolute64_dual")
    contract = dual._experiment_contract(
        "FC_P064_ABSOLUTE64_ARM_B_CONTROLLED_AERO_FORCE_FNO"
    )
    assert contract["status"] == "FC_P064_ABSOLUTE64_ARM_B_DUAL_FNO_MANIFEST_VERIFIED"
    assert contract["aero_kind"] == "FC_P064_ABSOLUTE64_ARM_B_CONTROLLED_AERO_CHECKPOINT"
    assert contract["aero_epoch"] == 2
    assert contract["optimizer_steps"] == 64
    assert contract["flow_frozen"] is True
    assert contract["p064_arm"] == "B"
    assert contract["extra"]["training_experiment"] == "FC-P064-ABSOLUTE64"
    assert contract["extra"]["training_windows"] == 512
    assert contract["extra"]["optimizer_steps"] == 64
    assert contract["extra"]["arm"] == "B"


def test_old_b_contract_is_unchanged():
    dual = load(DUAL, "absolute64_dual_old")
    contract = dual._experiment_contract(dual.P064_SYSTEM_KIND["B"])
    assert contract["aero_epoch"] == 1
    assert contract["optimizer_steps"] == 32
    assert contract["extra"]["training_experiment"] == "FC-P064"
    assert "absolute64" not in contract


def test_worker_changes_identity_not_fixed_numerics():
    text = WORKER.read_text()
    assert "B_ABSOLUTE64" in text
    assert "FC_P064_ABSOLUTE64_ARM_B_CONTROLLED_AERO_FORCE_FNO" in text
    assert "'starts':[0,100,200,300,400,500,600,700]" in text
    assert "'branches':['b01','b03']" in text
    assert "'horizon':5" in text and "'endpoints':80" in text
    assert "'allocator_bytes':16*2**30" in text
    assert "cap = 16*2**30" in text
    assert "'memory_bytes':24*2**30" in text
    assert "'optimizer_steps':0" in text
    assert "precision = base.override_inference_precision(torch)" in text
    assert "base.inference_precision(torch) == precision['effective']" in text


def test_mock_official_load_accepts_epoch2_absolute64_metadata(monkeypatch, tmp_path):
    dual = load(DUAL, "absolute64_dual_load")
    flow_dir, aero_dir = tmp_path / "flow", tmp_path / "aerodynamic"
    flow_dir.mkdir(); aero_dir.mkdir()
    identity = dual.DualFNOIdentity(
        tmp_path / "dual_model_manifest.json", "manifest",
        dual.CheckpointIdentity("flow", flow_dir, flow_dir / "model", flow_dir / "state",
                                0, dual.FLOW_MODEL_SHA256, dual.FLOW_STATE_SHA256, dual.FLOW_KIND),
        dual.CheckpointIdentity("aerodynamic", aero_dir, aero_dir / "model", aero_dir / "state",
                                2, "model2", "state2",
                                "FC_P064_ABSOLUTE64_ARM_B_CONTROLLED_AERO_CHECKPOINT"),
        {
            "kind": "FC_P064_ABSOLUTE64_ARM_B_CONTROLLED_AERO_FORCE_FNO",
            "flow_architecture": dual.ARCHITECTURE,
            "aerodynamic_architecture": dual.ARCHITECTURE,
            "training_protocol_sha256": "protocol",
        },
    )
    monkeypatch.setattr(dual, "validate_dual_fno_manifest", lambda *a, **k: identity)
    monkeypatch.setattr(dual, "validate_runtime_precision", lambda: dual.PRECISION_PROTOCOL)
    monkeypatch.setattr(dual, "_runtime_architecture", lambda cfg: dual.ARCHITECTURE)
    monkeypatch.setattr(dual, "_p026_history_adapter_factory",
                        lambda: (lambda flow, aero, k: SimpleNamespace(flow_model=flow,
                                                                        aerodynamic_model=aero, k=k)))
    if "fluid_control" not in sys.modules:
        package = ModuleType("fluid_control"); package.__path__ = []
        sys.modules["fluid_control"] = package
    calibrated = ModuleType("fluid_control.calibrated_checkpoint")
    calibrated.validate_calibrated_epoch_zero = lambda *a, **k: None
    sys.modules["fluid_control.calibrated_checkpoint"] = calibrated

    class Model:
        def to(self, device):
            return self

    aero_metadata = {
        "status": "FC_P064_ABSOLUTE64_ARM_B_CONTROLLED_AERO_CHECKPOINT",
        "checkpoint_epoch": 2,
        "training_experiment": "FC-P064-ABSOLUTE64",
        "arm": "B",
        "history_profile": "p026_k1",
        "history_k": 1,
        "model_in_channels": 6,
        "training_protocol_sha256": "protocol",
        "training_protocol_file": "training_protocol.json",
        "history_state_module_sha256": dual.P026_HISTORY_STATE_SHA256,
        "history_inference_module_sha256": dual.P026_HISTORY_INFERENCE_SHA256,
        "training_windows": 512,
        "optimizer_steps": 64,
        "accumulation_windows": 8,
        "actual_learning_rate": dual.P026_LEARNING_RATE,
        "parent_sampler_order_sha256": dual.P026_ORDER_SHA256,
        "schedule_sha256": dual.P064_SCHEDULE_SHA256["B"],
        "parent_history_inventory": dual._p026_inventory(),
        "flow_parent_model_sha256": dual.FLOW_MODEL_SHA256,
        "flow_parent_state_sha256": dual.FLOW_STATE_SHA256,
        "aerodynamic_parent_model_sha256": dual.P028_AERO_PARENT_MODEL_SHA256,
        "aerodynamic_parent_state_sha256": dual.P028_AERO_PARENT_STATE_SHA256,
        "selection_performed": False,
        "validation_accessed": False,
        "frozen_test_accessed": False,
        "ppo_executed": False,
    }

    def official_load(directory, models, metadata_dict, device):
        if Path(directory) == flow_dir:
            return 0
        metadata_dict.update(aero_metadata)
        return 2

    adapter, loaded = dual.load_dual_fno(
        tmp_path / "dual_model_manifest.json", SimpleNamespace(model=SimpleNamespace(in_channels=6)), "cpu",
        build_model=lambda cfg: Model(), load_checkpoint=official_load,
        expected_manifest_sha256="manifest",
    )
    assert loaded is identity
    assert adapter.k == 1
