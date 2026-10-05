"""CPU software fixtures only: no trained P013 candidate or scientific PASS."""
import importlib.util
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_module(name):
    path = Path(__file__).parent / (name + ".py")
    spec = importlib.util.spec_from_file_location("p013_fixture_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


test_module.__test__ = False
LAUNCH = test_module("test_candidate_canonical_ppo_launcher")
EXPORT = test_module("test_candidate_openfoam_readiness_adapter")
READY = test_module("test_candidate_ppo_readiness")
KIND = "fcp013_independent_force_dual_fno"


def binding():
    return {
        "status": "DUAL_CONTROL_IDENTITY_VERIFIED_NOT_CONTROL_SUCCESS",
        "checkpoint_sha256": "a" * 64, "checkpoint_state_sha256": "b" * 64,
        "training_config_sha256": "c" * 64, "normalization_sha256": "d" * 64,
        "dual_manifest_sha256": "e" * 64, "flow_model_sha256": "f" * 64,
        "flow_state_sha256": "1" * 64, "posteval_receipt_sha256": "2" * 64,
        "canonical_endpoint_window_dynamic_gates_still_required": True,
        "policy_trained": False, "real_cfd_control_validated": False,
    }


def readiness():
    result = LAUNCH.readiness()
    result["candidate_identity"].update({
        "candidate_kind": KIND, "checkpoint_epoch": 1,
        "checkpoint_relative_directory": "candidate/aerodynamic",
        "checkpoint_model_file": "FNO.0.1.mdlus",
        "checkpoint_state_file": "checkpoint.0.1.pt",
        "resolved_config_path": "artifacts/fcp011_resource_probe_runtime_20261005/resolved_config.yaml",
        "dual_manifest_path": "artifacts/candidate/candidate/dual_model_manifest.json",
        "dual_posteval_receipt_path": "artifacts/candidate/posteval_fc_p013/receipt.json",
        "dual_control_binding": binding(),
        "endpoint_evaluation_config_sha256": hashlib.sha256(b"{}\n").hexdigest(),
    })
    return result


@pytest.mark.parametrize("mode", ["dry-run", "execute"])
def test_dual_command_has_all_identities_and_vecnormalize(tmp_path, mode):
    args = LAUNCH.make_args(tmp_path)
    ready = readiness()
    command = LAUNCH.MODULE.build_trainer_command(args, args.output, mode, ready)
    identity = ready["candidate_identity"]
    expected = {
        "--config": args.endpoint_evaluation_config,
        "--checkpoint-dir": args.candidate_root / "candidate/aerodynamic",
        "--dual-fno-manifest": args.repo / identity["dual_manifest_path"],
        "--expected-dual-fno-manifest-sha256": "e" * 64,
        "--dual-training-config": args.repo / identity["resolved_config_path"],
        "--dual-posteval-receipt": args.repo / identity["dual_posteval_receipt_path"],
        "--expected-dual-posteval-receipt-sha256": "2" * 64,
        "--vecnormalize-output": args.output / "vecnormalize.pkl",
    }
    for key, value in expected.items():
        assert command.count(key) == 1
        assert command[command.index(key) + 1] == str(value)
    assert "--allow-calibrated-epoch-zero" not in command
    assert command[command.index("--config") + 1] != command[command.index("--dual-training-config") + 1]
    assert command[-1] == "--" + mode


def test_command_contract_binds_distinct_endpoint_config(tmp_path):
    args = LAUNCH.make_args(tmp_path)
    ready = readiness()
    contract = LAUNCH.MODULE.command_contract(args, ready)
    assert contract["endpoint_evaluation_config_sha256"] == LAUNCH.MODULE.sha256(args.endpoint_evaluation_config)
    assert contract["resolved_config_sha256"] == "c" * 64
    args.endpoint_evaluation_config.write_text("changed evaluator config")
    with pytest.raises(ValueError, match="endpoint evaluation config"):
        LAUNCH.MODULE.command_contract(args, ready)


@pytest.mark.parametrize("key", ["dual_control_binding", "dual_manifest_path", "dual_posteval_receipt_path"])
def test_contract_rejects_missing_dual_identity(tmp_path, key):
    args = LAUNCH.make_args(tmp_path)
    ready = readiness()
    ready["candidate_identity"].pop(key)
    with pytest.raises(ValueError, match="incomplete"):
        LAUNCH.MODULE.command_contract(args, ready)


def test_dry_run_rejects_lowlevel_single_model_result(tmp_path, monkeypatch):
    args = LAUNCH.make_args(tmp_path)
    def run(command, **kwargs):
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps({"status": "FULL40_CANONICAL_PPO_EXECUTION_READY",
                                      "checkpoint_sha256": "a" * 64}))
    monkeypatch.setattr(LAUNCH.MODULE.subprocess, "run", run)
    with pytest.raises(ValueError, match="dual binding"):
        LAUNCH.MODULE.dry_run(args, readiness())


def test_execute_rejects_final_audit_from_another_dual_system(tmp_path, monkeypatch):
    args = LAUNCH.make_args(tmp_path)
    ready = readiness()
    contract = LAUNCH.MODULE.command_contract(args, ready)
    args.approved_preflight = tmp_path / "approved.json"
    args.approved_preflight.write_text(json.dumps({
        "status": "CANDIDATE_CANONICAL_PPO_DRY_RUN_READY", "candidate_readiness": ready,
        "command_contract": contract, "training_executed": False,
        "canonical_preflight": {"status": "FULL40_CANONICAL_PPO_EXECUTION_READY",
                                "checkpoint_sha256": "a" * 64, "dual_control_binding": binding()},
    }))
    def run(command, **kwargs):
        output = Path(command[command.index("--output") + 1])
        output.mkdir(parents=True)
        changed = binding()
        changed["flow_model_sha256"] = "0" * 64
        (output / "audit.json").write_text(json.dumps({
            "status": "FULL40_CANONICAL_PPO_SURROGATE_RUN_COMPLETE",
            "physicsnemo_checkpoint_sha256": "a" * 64, "dual_control_binding": changed,
            "vecnormalize_contract": EXPORT.MODULE.IDENTITY_VEC,
            "precision_protocol": contract["precision_protocol"], "iterations": [{}],
        }))
    monkeypatch.setattr(LAUNCH.MODULE.subprocess, "run", run)
    with pytest.raises(ValueError, match="audit contract"):
        LAUNCH.MODULE.execute(args, ready)


def export_fixture(tmp_path):
    paths = EXPORT.fixture(tmp_path)
    approved = EXPORT.MODULE.load(paths["approved_preflight_path"])
    approved["command_contract"].update(candidate_kind=KIND, dual_control_binding=binding(),
                                         full40_normalization_sha256="d" * 64)
    approved["candidate_readiness"]["candidate_identity"].update(
        candidate_kind=KIND, dual_control_binding=binding(), normalization_sha256="d" * 64)
    approved["canonical_preflight"]["dual_control_binding"] = binding()
    audit = EXPORT.MODULE.load(paths["canonical_audit_path"])
    audit["dual_control_binding"] = binding()
    EXPORT.dump(paths["approved_preflight_path"], approved)
    EXPORT.dump(paths["canonical_audit_path"], audit)
    repair_binding(paths)
    return paths


def repair_binding(paths):
    approved = EXPORT.MODULE.load(paths["approved_preflight_path"])
    receipt = EXPORT.MODULE.load(paths["binding_receipt_path"])
    receipt.update(candidate_readiness_sha256=EXPORT.MODULE.sha256(paths["approved_preflight_path"]),
                   canonical_audit_sha256=EXPORT.MODULE.sha256(paths["canonical_audit_path"]),
                   command_contract=approved["command_contract"])
    EXPORT.dump(paths["binding_receipt_path"], receipt)


def test_export_preserves_binding_and_translates_only_policy(tmp_path):
    paths = export_fixture(tmp_path)
    before = EXPORT.MODULE.load(paths["canonical_audit_path"])
    ready, audit = EXPORT.MODULE.adapt(**paths)
    assert ready["dual_control_binding"] == audit["dual_control_binding"] == binding()
    assert ready["candidate_openfoam_compatibility"]["dual_control_binding"] == binding()
    assert audit["iterations"][-1]["checkpoint"] == str(paths["policy_path"].resolve())
    audit.pop("candidate_openfoam_compatibility")
    audit["iterations"][-1]["checkpoint"] = before["iterations"][-1]["checkpoint"]
    assert audit == before


@pytest.mark.parametrize("target", ["contract", "candidate", "canonical", "audit"])
def test_export_rejects_binding_mismatch_despite_repaired_outer_hashes(tmp_path, target):
    paths = export_fixture(tmp_path)
    approved = EXPORT.MODULE.load(paths["approved_preflight_path"])
    audit = EXPORT.MODULE.load(paths["canonical_audit_path"])
    values = {"contract": approved["command_contract"],
              "candidate": approved["candidate_readiness"]["candidate_identity"],
              "canonical": approved["canonical_preflight"], "audit": audit}
    values[target]["dual_control_binding"]["flow_model_sha256"] = "0" * 64
    EXPORT.dump(paths["approved_preflight_path"], approved)
    EXPORT.dump(paths["canonical_audit_path"], audit)
    repair_binding(paths)
    with pytest.raises(ValueError, match="dual control bindings"):
        EXPORT.MODULE.adapt(**paths)


def test_export_rejects_dual_disguised_as_legacy(tmp_path):
    paths = export_fixture(tmp_path)
    approved = EXPORT.MODULE.load(paths["approved_preflight_path"])
    approved["command_contract"]["candidate_kind"] = "fixture_h100"
    approved["candidate_readiness"]["candidate_identity"]["candidate_kind"] = "fixture_h100"
    EXPORT.dump(paths["approved_preflight_path"], approved)
    repair_binding(paths)
    with pytest.raises(ValueError, match="single-model"):
        EXPORT.MODULE.adapt(**paths)


def p013_gate_fixture(tmp_path, monkeypatch):
    """Stub only validated P013 provenance; exercise unchanged shared numeric gates."""
    values = READY.fixture(tmp_path)
    lineage = READY.MODULE.load(values["lineage_path"])
    lineage.update(candidate_kind=KIND, status="FC_P013_DUAL_CANDIDATE_LINEAGE_PASS_NOT_ADMISSION")
    READY.write_json(values["lineage_path"], lineage)
    receipt = READY.MODULE.load(values["posteval_receipt_path"])
    receipt["sha256"]["lineage.json"] = READY.digest(values["lineage_path"])
    READY.write_json(values["posteval_receipt_path"], receipt)
    def provenance(**kwargs):
        assert kwargs["lineage"]["candidate_kind"] == KIND
        return {"config": values["candidate_root"] / "resolved_config.yaml", "epoch": 1,
                "checkpoint_relative_directory": Path("candidate/aerodynamic"),
                "model_file": "FNO.0.1.mdlus", "state_file": "checkpoint.0.1.pt",
                "dual_control_binding": binding(),
                "dual_manifest_path": "artifacts/candidate/candidate/dual_model_manifest.json",
                "dual_posteval_receipt_path": "artifacts/candidate/posteval_fc_p013/receipt.json"}
    monkeypatch.setattr(READY.MODULE, "p013_candidate_identity", provenance)
    return values


def test_p013_branch_preserves_shared_gate_success_semantics(tmp_path, monkeypatch):
    result = READY.MODULE.audit(**p013_gate_fixture(tmp_path, monkeypatch))
    assert result["status"] == "CANDIDATE_PPO_CPU_DRY_RUN_READY"
    assert result["candidate_identity"]["dual_control_binding"] == binding()
    assert result["ppo_execution_authorized"] is False


@pytest.mark.parametrize("key", ["endpoint_gate_path", "window_gate_path", "dynamic_gate_path", "development_gate_path"])
def test_p013_does_not_bypass_any_legacy_numeric_gate(tmp_path, monkeypatch, key):
    values = p013_gate_fixture(tmp_path, monkeypatch)
    gate = READY.MODULE.load(values[key])
    gate["status"] = "FAIL"
    READY.write_json(values[key], gate)
    receipt = READY.MODULE.load(values["posteval_receipt_path"])
    root = values["posteval_receipt_path"].parent
    if values[key].is_relative_to(root):
        receipt["sha256"][str(values[key].relative_to(root))] = READY.digest(values[key])
    READY.write_json(values["posteval_receipt_path"], receipt)
    result = READY.MODULE.audit(**values)
    assert result["status"] == "CANDIDATE_PPO_CPU_DRY_RUN_BLOCKED"
    assert result["blockers"][0]["kind"] == "SCIENTIFIC_FAIL"


def provenance_fixture(tmp_path, monkeypatch):
    """Real evidence-file checks with mocked official identity and numeric verifier.

    The existing dual manifest/control suites test those verifiers separately.
    This fixture cannot authorize a real policy or claim CFD accuracy.
    """
    import audit_fcp011_candidate
    from fluid_control import dual_fno, dual_control_contract
    values = READY.fixture(tmp_path)
    root = values["candidate_root"]
    original = root / "best/FNO.0.2.mdlus"
    roles = {}
    for role, epoch in (("flow", 0), ("aerodynamic", 1)):
        directory = root / "candidate" / role
        directory.mkdir(parents=True)
        model = directory / f"FNO.0.{epoch}.mdlus"
        state = directory / f"checkpoint.0.{epoch}.pt"
        shutil.copyfile(original, model)
        state.write_bytes(role.encode())
        roles[role] = SimpleNamespace(directory=directory, model=model, state=state,
                                      model_sha256=READY.digest(model), state_sha256=READY.digest(state))
    config = tmp_path / READY.MODULE.FC_P013_CONFIG
    config.parent.mkdir(parents=True)
    shutil.copyfile(root / "resolved_config.yaml", config)
    manifest = root / "candidate/dual_model_manifest.json"
    READY.write_json(manifest, {"fixture_only": True})
    identity = SimpleNamespace(**roles, manifest_sha256=READY.digest(manifest))
    expected_inputs = {}
    data_artifacts = {}
    for name in ("base_manifest", "train8_manifest", "train16_manifest", "normalization"):
        path = tmp_path / (name + ".fixture")
        path.write_text(name)
        data_artifacts[name] = path
        expected_inputs[name] = READY.digest(path)
    monkeypatch.setattr(audit_fcp011_candidate, "INPUT_SHA", expected_inputs)
    result = root / "candidate/result.json"
    READY.write_json(result, {"input_sha256": expected_inputs})
    lineage = {
        "status": "FC_P013_DUAL_CANDIDATE_LINEAGE_PASS_NOT_ADMISSION", "candidate_kind": KIND,
        "candidate_root": str(root), "checkpoint_epoch": 1,
        "checkpoint_relative_directory": "candidate/aerodynamic",
        "checkpoint_model_file": "FNO.0.1.mdlus", "checkpoint_state_file": "checkpoint.0.1.pt",
        "training_performed": True, "optimizer_training_performed": True,
        "calibration_fit_performed": False, "optimizer_steps": 1368,
        "validation_or_frozen_accessed": False, "ppo_auto_launch": False,
        "official_image_id": READY.MODULE.IMAGE_ID,
        "dual_manifest_sha256": identity.manifest_sha256,
        "checkpoint_sha256": identity.aerodynamic.model_sha256,
        "checkpoint_state_sha256": identity.aerodynamic.state_sha256,
        "flow_model_sha256": identity.flow.model_sha256,
        "flow_state_sha256": identity.flow.state_sha256,
        "resolved_config_sha256": READY.digest(config),
        "candidate_result_sha256": READY.digest(result),
        "sha256": {str(path.relative_to(root)): READY.digest(path)
                   for path in (manifest, result, identity.flow.model, identity.flow.state,
                                identity.aerodynamic.model, identity.aerodynamic.state)},
    }
    candidate_audit = root / "candidate_audit.json"
    READY.write_json(candidate_audit, lineage)
    completion = root / "completion_receipt.json"
    READY.write_json(completion, {
        "status": "FC_P013_TRAINING_COMPLETE_NOT_ADMISSION",
        "candidate_audit_sha256": READY.digest(candidate_audit), "optimizer_steps": 1368,
        "scientific_admission": False, "ppo_executed": False,
        "dual_manifest_sha256": identity.manifest_sha256,
        "flow_model_sha256": identity.flow.model_sha256, "flow_state_sha256": identity.flow.state_sha256,
        "aerodynamic_model_sha256": identity.aerodynamic.model_sha256,
        "aerodynamic_state_sha256": identity.aerodynamic.state_sha256,
        "candidate_result_sha256": READY.digest(result),
    })
    posteval = values["posteval_receipt_path"]
    READY.write_json(posteval.parent / "precision.json", {
        "status": "FC_P013_FORMAL_EVALUATION_DEFAULT_TF32_HIGH",
        "official_image_id": READY.MODULE.IMAGE_ID, **READY.MODULE.FC_P008_PRECISION})
    READY.write_json(posteval.parent / "evidence/formal_evaluation_approval.json", {
        "candidate_kind": KIND, "candidate_completion_receipt_sha256": READY.digest(completion),
        "candidate_result_sha256": READY.digest(result), "dual_manifest_sha256": identity.manifest_sha256})

    def manifest_verifier(path, *, expected_sha256):
        if READY.digest(path) != expected_sha256:
            raise ValueError("manifest SHA differs")
        for role in roles.values():
            if READY.digest(role.model) != role.model_sha256 or READY.digest(role.state) != role.state_sha256:
                raise ValueError("checkpoint SHA differs")
        return identity
    config_hash = READY.digest(config)
    norm_hash = READY.digest(values["normalization_path"])
    def runtime_verifier(identity, *, config_path, normalization_path):
        if READY.digest(config_path) != config_hash or READY.digest(normalization_path) != norm_hash:
            raise ValueError("runtime identity differs")
    monkeypatch.setattr(dual_fno, "validate_dual_fno_manifest", manifest_verifier)
    monkeypatch.setattr(dual_fno, "validate_dual_runtime_files", runtime_verifier)
    monkeypatch.setattr(dual_control_contract, "verify_dual_control_binding", lambda **kwargs: binding())
    return dict(repo_root=tmp_path, candidate_root=root, lineage=lineage, receipt_path=posteval,
                normalization_path=values["normalization_path"], data_artifacts=data_artifacts)


def test_p013_uses_actual_terminal_schema_without_legacy_fields(tmp_path, monkeypatch):
    args = provenance_fixture(tmp_path, monkeypatch)
    result = READY.MODULE.p013_candidate_identity(**args)
    assert result["epoch"] == 1
    assert result["checkpoint_relative_directory"] == Path("candidate/aerodynamic")
    assert result["dual_control_binding"] == binding()
    assert "data_lineage" not in args["lineage"]
    assert "launch_receipt_sha256" not in args["lineage"]


@pytest.mark.parametrize("relative", [
    "candidate/dual_model_manifest.json", "candidate/flow/FNO.0.0.mdlus",
    "candidate/flow/checkpoint.0.0.pt", "candidate/aerodynamic/FNO.0.1.mdlus",
    "candidate/aerodynamic/checkpoint.0.1.pt", "candidate/result.json",
])
def test_p013_rejects_each_changed_identity_file(tmp_path, monkeypatch, relative):
    args = provenance_fixture(tmp_path, monkeypatch)
    path = args["candidate_root"] / relative
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        READY.MODULE.p013_candidate_identity(**args)


@pytest.mark.parametrize("fault", ["config", "normalization", "data", "completion", "approval", "precision", "epoch"])
def test_p013_rejects_incompatible_provenance(tmp_path, monkeypatch, fault):
    args = provenance_fixture(tmp_path, monkeypatch)
    if fault in ("config", "normalization", "data"):
        path = (tmp_path / READY.MODULE.FC_P013_CONFIG if fault == "config" else
                args["normalization_path"] if fault == "normalization" else args["data_artifacts"]["train8_manifest"])
        path.write_bytes(path.read_bytes() + b"changed")
    elif fault == "epoch":
        args["lineage"]["checkpoint_epoch"] = 0
    else:
        path = {"completion": args["candidate_root"] / "completion_receipt.json",
                "approval": args["receipt_path"].parent / "evidence/formal_evaluation_approval.json",
                "precision": args["receipt_path"].parent / "precision.json"}[fault]
        value = READY.MODULE.load(path)
        if fault == "completion":
            value["aerodynamic_state_sha256"] = "0" * 64
        elif fault == "approval":
            value["candidate_kind"] = "old_p009"
        else:
            value["cuda_matmul_allow_tf32"] = False
        READY.write_json(path, value)
    with pytest.raises(ValueError):
        READY.MODULE.p013_candidate_identity(**args)
