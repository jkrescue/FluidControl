"""Adversarial software receipts only; no real candidate/model/data reads."""
import ast
import copy
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parent/"scripts/run_fcp026_posteval.py"
if not SCRIPT.is_file():
    SCRIPT = HERE/"run_fcp026_posteval.py"
spec = importlib.util.spec_from_file_location("p026_proof_runner", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture(autouse=True)
def isolation(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)


def fixture(tmp_path, k=1):
    candidate = tmp_path/"candidate"
    candidate.mkdir()
    result = dict(flow_tensor_sha256="1"*64, aerodynamic_terminal_tensor_sha256="2"*64)
    (candidate/"result.json").write_text(json.dumps(result))
    files = {name: "3"*64 for name in ("training_protocol.json", "dual_model_manifest.json",
        "flow/FNO.0.0.mdlus", "flow/checkpoint.0.0.pt", "aerodynamic/FNO.0.1.mdlus",
        "aerodynamic/checkpoint.0.1.pt")}
    files["result.json"] = runner.sha(candidate/"result.json")
    unit = f"fluid-control-fcp026-history-k{k}-20261005.service"
    invocation = "a"*32
    a = dict(history_k=k, candidate_sha256=files, training_unit=unit,
             training_invocation=invocation, training_config_sha256="4"*64,
             source_sha256={"src/fluid_control/dual_fno.py": "5"*64})
    common = dict(history_k=k, training_protocol_sha256=files["training_protocol.json"],
        dual_manifest_sha256=files["dual_model_manifest.json"], candidate_result_sha256=files["result.json"],
        candidate_sha256={"candidate/"+name: value for name,value in files.items()},
        scientific_admission=False, ppo_authorized=False,
        tensor_sha256=dict(flow="1"*64, aerodynamic="2"*64))
    audit = dict(**copy.deepcopy(common), status=f"FC_P026_K{k}_CANDIDATE_INTEGRITY_VERIFIED_NOT_ADMISSION",
        actual_optimizer_steps=171, actual_training_windows=1368, accumulation_windows=8,
        training_unit=unit, training_invocation=invocation, dual_adapter_fresh_reload_verified=False,
        role_loader_sha256="5"*64, terminal_evidence=dict(LoadState="loaded", InvocationID=invocation,
        ActiveState="active", SubState="exited", Result="success", ExecMainCode="1", ExecMainStatus="0", MainPID="0"))
    reload = dict(**copy.deepcopy(common), status=f"FC_P026_K{k}_OFFICIAL_CPU_DUAL_RELOAD_VERIFIED_NOT_ADMISSION",
        candidate_audit_sha256="pending", config_sha256="4"*64, required_official_image_id=runner.IMAGE,
        device="cpu", official_dual_reload_verified=True, forward_performed=False,
        optimizer_created=False, model_saved=False, gpu_used=False,
        runtime_source_sha256={"src/fluid_control/dual_fno.py": "5"*64})
    save(tmp_path, a, audit, reload)
    return candidate, a, audit, reload


def save(root, approval, audit, reload):
    audit_path, reload_path = root/"audit.json", root/"reload.json"
    audit_path.write_text(json.dumps(audit))
    approval["independent_terminal_audit"] = dict(path="audit.json", sha256=runner.sha(audit_path), reviewed_by_lead=True)
    reload["candidate_audit_sha256"] = runner.sha(audit_path)
    reload_path.write_text(json.dumps(reload))
    approval["official_dual_reload"] = dict(path="reload.json", sha256=runner.sha(reload_path), reviewed_by_lead=True)


@pytest.mark.parametrize("k", [1, 4])
def test_exact_arm_receipts_accept_literal_prefix_normalization(tmp_path, k):
    candidate, approval, audit, reload = fixture(tmp_path, k)
    result = runner.validate_terminal_proofs(tmp_path, candidate, approval)
    assert result == dict(independent_terminal_audit=audit, official_dual_reload=reload)


@pytest.mark.parametrize("side", ["audit", "reload"])
@pytest.mark.parametrize("mutation", ["cross_arm", "status", "protocol", "result", "manifest", "missing_prefix",
    "double_prefix", "extra", "missing", "file_digest", "tensor", "admission", "ppo", "bool_arm"])
def test_content_rejected_even_when_lead_file_sha_refreshed(tmp_path, side, mutation):
    candidate, approval, audit, reload = fixture(tmp_path)
    proof = audit if side == "audit" else reload
    if mutation == "cross_arm":
        proof["history_k"] = 4
    elif mutation == "status":
        proof["status"] = "FC_P018_COMPLETE"
    elif mutation in ("protocol", "result", "manifest"):
        key = {"protocol": "training_protocol_sha256", "result": "candidate_result_sha256", "manifest": "dual_manifest_sha256"}[mutation]
        proof[key] = "9"*64
    elif mutation == "missing_prefix":
        proof["candidate_sha256"] = {key[len("candidate/"):]: value for key,value in proof["candidate_sha256"].items()}
    elif mutation == "double_prefix":
        proof["candidate_sha256"] = {"candidate/"+key: value for key,value in proof["candidate_sha256"].items()}
    elif mutation == "extra":
        proof["candidate_sha256"]["candidate/extra.json"] = "9"*64
    elif mutation == "missing":
        del proof["candidate_sha256"]["candidate/result.json"]
    elif mutation == "file_digest":
        proof["candidate_sha256"]["candidate/flow/FNO.0.0.mdlus"] = "9"*64
    elif mutation == "tensor":
        proof["tensor_sha256"]["flow"] = "9"*64
    elif mutation == "admission":
        proof["scientific_admission"] = 0  # Numeric false must not substitute for an explicit flag.
    elif mutation == "ppo":
        proof["ppo_authorized"] = True
    else:
        proof["history_k"] = True
    save(tmp_path, approval, audit, reload)
    with pytest.raises(ValueError):
        runner.validate_terminal_proofs(tmp_path, candidate, approval)


@pytest.mark.parametrize("key,value", [("actual_optimizer_steps", 1368), ("actual_training_windows", 171),
    ("accumulation_windows", 1), ("training_unit", "wrong.service"), ("training_invocation", "b"*32),
    ("dual_adapter_fresh_reload_verified", True)])
def test_actual_audit_execution_and_counts(tmp_path, key, value):
    candidate, approval, audit, reload = fixture(tmp_path)
    audit[key] = value
    save(tmp_path, approval, audit, reload)
    with pytest.raises(ValueError):
        runner.validate_terminal_proofs(tmp_path, candidate, approval)


@pytest.mark.parametrize("key,value", [("SubState", "start"), ("MainPID", "123"),
                                      ("LoadState", "not-found"), ("InvocationID", "b"*32)])
def test_terminal_receipt_cannot_be_running_or_stale(tmp_path, key, value):
    candidate, approval, audit, reload = fixture(tmp_path)
    audit["terminal_evidence"][key] = value
    save(tmp_path, approval, audit, reload)
    with pytest.raises(ValueError):
        runner.validate_terminal_proofs(tmp_path, candidate, approval)


@pytest.mark.parametrize("key,value", [("official_dual_reload_verified", False), ("device", "cuda"),
    ("gpu_used", True), ("optimizer_created", True), ("model_saved", True),
    ("forward_performed", True), ("config_sha256", "9"*64)])
def test_actual_reload_scope(tmp_path, key, value):
    candidate, approval, audit, reload = fixture(tmp_path)
    reload[key] = value
    save(tmp_path, approval, audit, reload)
    with pytest.raises(ValueError):
        runner.validate_terminal_proofs(tmp_path, candidate, approval)


def test_reload_cannot_bind_a_different_audit(tmp_path):
    candidate, approval, audit, reload = fixture(tmp_path)
    reload["candidate_audit_sha256"] = "9"*64
    (tmp_path/"reload.json").write_text(json.dumps(reload))
    approval["official_dual_reload"]["sha256"] = runner.sha(tmp_path/"reload.json")
    with pytest.raises(ValueError):
        runner.validate_terminal_proofs(tmp_path, candidate, approval)


@pytest.mark.parametrize("mutation", ["file_after_pin", "result_after_pin", "unreviewed", "loader"])
def test_byte_pin_review_and_loader_binding(tmp_path, mutation):
    candidate, approval, audit, reload = fixture(tmp_path)
    if mutation == "file_after_pin":
        (tmp_path/"reload.json").write_text("{}")
    elif mutation == "result_after_pin":
        (candidate/"result.json").write_text("{}")
    elif mutation == "unreviewed":
        approval["official_dual_reload"]["reviewed_by_lead"] = False
    else:
        reload["runtime_source_sha256"]["src/fluid_control/dual_fno.py"] = "9"*64
        save(tmp_path, approval, audit, reload)
    with pytest.raises(ValueError):
        runner.validate_terminal_proofs(tmp_path, candidate, approval)


def test_all_existing_functions_except_preflight_remain_bytecode_ast_identical():
    # Compare staged patch with the exact committed preparation baseline on Main.
    baseline = Path("/workspace/fluid_control/scripts/run_fcp026_posteval.py")
    if baseline.resolve() == SCRIPT.resolve():
        pytest.skip("baseline comparison applies before integration only")
    old = ast.parse(baseline.read_text())
    new = ast.parse(SCRIPT.read_text())
    old_functions = {node.name: ast.dump(node, include_attributes=False) for node in old.body if isinstance(node, ast.FunctionDef)}
    new_functions = {node.name: ast.dump(node, include_attributes=False) for node in new.body if isinstance(node, ast.FunctionDef)}
    assert set(new_functions) == set(old_functions) | {"validate_terminal_proofs"}
    for name, node in old_functions.items():
        if name != "preflight":
            assert new_functions[name] == node, name
    assert [ast.dump(n) for n in old.body if isinstance(n, (ast.Assign, ast.AnnAssign))] == [
        ast.dump(n) for n in new.body if isinstance(n, (ast.Assign, ast.AnnAssign))]
