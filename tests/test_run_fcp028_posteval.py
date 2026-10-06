import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_fcp028_posteval.py"
BASE_SCRIPT = Path(
    "/workspace/fluid_control/scripts/run_fcp026_posteval.py"
)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MODULE = load(SCRIPT, "run_fcp028_posteval_test")
BASE = load(BASE_SCRIPT, "run_fcp026_posteval_test_base")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(json.dumps(value), encoding="utf-8")
    return BASE.sha(path)


def fixture(tmp_path):
    candidate = tmp_path / "candidate"
    protocol = {
        "experiment": "FC-P028",
        "optimized_role": "flow",
        "fixed_role": "aerodynamic",
        "horizon": 10,
        "training_windows": 1368,
        "optimizer_steps": 171,
        "accumulation_windows": 8,
        "learning_rate": 1e-5,
        "objective": "ten_equal_masked_normalized_state_MSE",
        "force_loss": False,
        "terminal_selection": False,
        "future_truth_inputs": False,
        "sampler_order_sha256": "1" * 64,
    }
    protocol_sha = write(candidate / "training_protocol.json", protocol)
    result = {
        "status": MODULE.TRAINING_COMPLETE,
        "mode": "train",
        "training_windows": 1368,
        "optimizer_steps": 171,
        "training_protocol_sha256": protocol_sha,
        "scientific_admission": False,
        "validation_accessed": False,
        "frozen_test_accessed": False,
        "official_fresh_reload_verified": True,
        "flow_terminal_tensor_sha256": "6" * 64,
        "frozen_aerodynamic_tensor_sha256": "7" * 64,
        "sampler_order_sha256": "1" * 64,
        "records": [
            {"update": index, "consumed_windows": index * 8}
            for index in range(1, 172)
        ],
    }
    result_sha = write(candidate / "result.json", result)
    role_hashes = {}
    for role in ("flow", "aerodynamic"):
        role_hashes[(role, "model")] = write(
            candidate / role / "FNO.0.1.mdlus", f"{role}-model".encode()
        )
        role_hashes[(role, "state")] = write(
            candidate / role / "checkpoint.0.1.pt", f"{role}-state".encode()
        )
    parent = "2" * 64
    manifest = {
        "status": MODULE.P028_STATUS,
        "kind": MODULE.P028_KIND,
        "training_experiment": "FC-P028",
        "training_protocol_sha256": protocol_sha,
        "training_semantics": protocol,
        "parent_manifest_sha256": parent,
        "scientific_admission": False,
        "history_input": {
            "profile": "p026_k1",
            "history_length": 1,
            "flow_input_channels": 6,
            "aerodynamic_input_channels": 6,
            "future_state_inputs": False,
            "future_force_inputs": False,
        },
        "flow": {
            "role": "flow",
            "frozen": False,
            "checkpoint_epoch": 1,
            "metadata_kind": MODULE.P028_FLOW_KIND,
            "model_sha256": role_hashes[("flow", "model")],
            "state_sha256": role_hashes[("flow", "state")],
        },
        "aerodynamic": {
            "role": "aerodynamic",
            "frozen": True,
            "checkpoint_epoch": 1,
            "metadata_kind": MODULE.P026_AERO_KIND,
            "model_sha256": role_hashes[("aerodynamic", "model")],
            "state_sha256": role_hashes[("aerodynamic", "state")],
        },
    }
    manifest_sha = write(candidate / "dual_model_manifest.json", manifest)
    approval = {
        "parent_manifest_sha256": parent,
        "candidate_sha256": {
            "result.json": result_sha,
            "training_protocol.json": protocol_sha,
            "dual_model_manifest.json": manifest_sha,
            "flow/FNO.0.1.mdlus": role_hashes[("flow", "model")],
            "flow/checkpoint.0.1.pt": role_hashes[("flow", "state")],
            "aerodynamic/FNO.0.1.mdlus": role_hashes[("aerodynamic", "model")],
            "aerodynamic/checkpoint.0.1.pt": role_hashes[("aerodynamic", "state")],
        },
    }
    return candidate, approval, manifest


def test_p028_candidate_contract_is_trained_flow_frozen_aero(tmp_path):
    candidate, approval, _ = fixture(tmp_path)
    manifest = MODULE.candidate_contract(BASE, candidate, approval)
    assert manifest["flow"]["frozen"] is False
    assert manifest["aerodynamic"]["frozen"] is True


@pytest.mark.parametrize(
    "target,key,value,message",
    (
        ("flow", "frozen", True, "contract differs"),
        ("aerodynamic", "frozen", False, "contract differs"),
        ("flow", "checkpoint_epoch", 0, "contract differs"),
        ("root", "kind", "FC_P026_K1_HISTORY_FORCE_FNO", "contract differs"),
    ),
)
def test_p028_candidate_contract_rejects_old_or_swapped_scope(
    tmp_path, target, key, value, message
):
    candidate, approval, manifest = fixture(tmp_path)
    (manifest if target == "root" else manifest[target])[key] = value
    approval["candidate_sha256"]["dual_model_manifest.json"] = write(
        candidate / "dual_model_manifest.json", manifest
    )
    with pytest.raises(ValueError, match=message):
        MODULE.candidate_contract(BASE, candidate, approval)


def test_commands_only_change_identity_not_numerical_cli():
    manifest = "a" * 64
    model = "b" * 64
    old = BASE.commands(1, model, manifest)
    new = MODULE.commands(BASE, manifest, model)
    assert [row[:2] for row in new] == [row[:2] for row in old]
    for (name_old, _, command_old), (name_new, _, command_new) in zip(old, new):
        assert name_old == name_new
        if name_old == "validation_diagnostic":
            old_index = command_old.index("--candidate-kind") + 1
            new_index = command_new.index("--candidate-kind") + 1
            assert command_old[old_index] == "FC_P026_K1_HISTORY_FORCE_FNO"
            assert command_new[new_index] == MODULE.P028_KIND
            command_new = list(command_new)
            command_new[new_index] = command_old[old_index]
        assert command_new == command_old


def test_terminal_proofs_are_sha_bound_and_nonadmission(tmp_path):
    candidate, approval, _ = fixture(tmp_path)
    approval.update(
        training_unit="fluid-control-fcp028-training-20261006.service",
        training_invocation="8" * 32,
        training_config_sha256="9" * 64,
        source_sha256={"src/fluid_control/dual_fno.py": "a" * 64},
    )
    common = {
        "candidate_sha256": {
            "candidate/" + name: digest
            for name, digest in approval["candidate_sha256"].items()
        },
        "training_protocol_sha256": approval["candidate_sha256"]["training_protocol.json"],
        "dual_manifest_sha256": approval["candidate_sha256"]["dual_model_manifest.json"],
        "candidate_result_sha256": approval["candidate_sha256"]["result.json"],
        "scientific_admission": False,
        "ppo_authorized": False,
        "tensor_sha256": {"flow": "6" * 64, "aerodynamic": "7" * 64},
    }
    audit = {
        **common,
        "status": "FC_P028_CANDIDATE_INTEGRITY_VERIFIED_NOT_ADMISSION",
        "actual_optimizer_steps": 171,
        "actual_training_windows": 1368,
        "accumulation_windows": 8,
        "training_unit": approval["training_unit"],
        "training_invocation": approval["training_invocation"],
        "flow_trained": True,
        "aerodynamic_frozen": True,
        "dual_adapter_fresh_reload_verified": False,
        "terminal_evidence": {
            "InvocationID": approval["training_invocation"],
            "ActiveState": "active",
            "SubState": "exited",
            "Result": "success",
            "ExecMainCode": "1",
            "ExecMainStatus": "0",
            "MainPID": "0",
            "LoadState": "loaded",
        },
        "role_loader_sha256": "a" * 64,
    }
    audit_relative = "proof/audit.json"
    audit_sha = write(tmp_path / audit_relative, audit)
    approval["independent_terminal_audit"] = {
        "path": audit_relative,
        "sha256": audit_sha,
        "reviewed_by_lead": True,
    }
    reload = {
        **common,
        "status": "FC_P028_OFFICIAL_CPU_DUAL_RELOAD_VERIFIED_NOT_ADMISSION",
        "candidate_audit_sha256": audit_sha,
        "config_sha256": approval["training_config_sha256"],
        "required_official_image_id": BASE.IMAGE,
        "device": "cpu",
        "official_dual_reload_verified": True,
        "forward_performed": False,
        "optimizer_created": False,
        "model_saved": False,
        "gpu_used": False,
        "runtime_source_sha256": {"src/fluid_control/dual_fno.py": "a" * 64},
    }
    reload_relative = "proof/reload.json"
    approval["official_dual_reload"] = {
        "path": reload_relative,
        "sha256": write(tmp_path / reload_relative, reload),
        "reviewed_by_lead": True,
    }
    MODULE.validate_terminal_proofs(BASE, tmp_path, candidate, approval)
    approval["official_dual_reload"]["reviewed_by_lead"] = False
    with pytest.raises(ValueError, match="unreviewed"):
        MODULE.validate_terminal_proofs(BASE, tmp_path, candidate, approval)


def test_numerical_runner_is_external_fixed_dependency():
    source = SCRIPT.read_text(encoding="utf-8")
    assert MODULE.NUMERICAL_RUNNER_SHA256 == BASE.sha(BASE_SCRIPT)
    assert 'args.source / "scripts/run_fcp026_posteval.py"' not in source
    assert "external numerical runner bytes differ" in source


def test_numerical_runner_is_hashed_before_import(tmp_path):
    marker = tmp_path / "executed"
    malicious = tmp_path / "runner.py"
    malicious.write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n")
    with pytest.raises(ValueError, match="before import"):
        MODULE._base_runner(malicious)
    assert not marker.exists()


def test_execution_path_keeps_nonadmission_receipt_and_original_runner():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "FC_P028_FORMAL_PREFLIGHT_ONLY" in source
    assert "FC_P028_ORIGINAL_FORMAL_COMPLETE_NOT_ADMISSION" in source
    assert "base.run_container" in source
    assert "scientific_admission\": False" in source
