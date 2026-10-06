"""Project provenance gate for a dual-FNO controller; not a new physics model.

This supplements, never replaces, canonical endpoint/window/dynamic admission.
It does not create a policy or prove real-CFD control performance.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from .dual_fno import (
    P015_SYSTEM_KIND, P018_SYSTEM_KIND, P018_PROTOCOL_SHA256, P018_LEARNING_RATE,
    P026_K1_SYSTEM_KIND, P026_K4_SYSTEM_KIND,
    P064_SYSTEM_KIND,
    P026_HISTORY_STATE_SHA256, P026_HISTORY_INFERENCE_SHA256,
    SYSTEM_KIND, sha256, validate_dual_fno_manifest,
    validate_dual_runtime_files,
)

DEVELOPMENT_AUDITOR_SHA = "ca6da0afdce5859be1c060eb48ba2cdd1ccc5ee3aeb2570d9c9b53067d5bc412"
P026_FORMAL_RUNNER_SHA256 = "03c5862e34a648a1254284d1709bd74c3b995d3a91ae06c4a6f92a945029c0f3"
P018_APPROVAL_SHA256 = "eea5bd5c6a3fe585ae1104600421e50b335f61299c4014c5e3136722af3d4d39"
P018_OBSERVATION_SHA256 = "c040eae25fa31a98164e08e10dc4f007eb6a3ef38329ad0dcfaddd734c7eb53c"
P018_INVOCATION = "1ca4654aab074278bb2efdfff8dbc1eb"
PROTOCOL = [
    "validation10_H1_H10_H50_H100_stride25_batch4",
    "dynamic6_H1_H10_H50_H100_stride1_batch8",
    "force_window6", "unchanged_development_gate",
]


def p026_runtime_binding(identity) -> dict | None:
    """Derive HydroGym routing only from an already validated dual identity."""
    profiles = {
        P026_K1_SYSTEM_KIND: ("p026_k1", 1),
        P026_K4_SYSTEM_KIND: ("p026_k4", 4),
        P064_SYSTEM_KIND["A"]: ("p026_k1", 1),
        P064_SYSTEM_KIND["B"]: ("p026_k1", 1),
    }
    kind = identity.payload.get("kind")
    if kind not in profiles:
        return None
    profile, history_length = profiles[kind]
    history_input = identity.payload.get("history_input")
    exact_history = {
        "schema_version": 1,
        "profile": profile,
        "history_length": history_length,
        "flow_input_channels": 6,
        "aerodynamic_input_channels": 6 if history_length == 1 else 18,
        "left_padding": "trajectory_frame0",
        "autoregressive_state_source": "frozen_flow_prediction",
        "future_state_inputs": False,
        "future_force_inputs": False,
    }
    if history_input != exact_history:
        raise ValueError("P026 HydroGym history input differs from validated profile")
    if (
        identity.payload.get("history_state_module_sha256")
        != P026_HISTORY_STATE_SHA256
        or identity.payload.get("history_inference_module_sha256")
        != P026_HISTORY_INFERENCE_SHA256
    ):
        raise ValueError("P026 HydroGym history source identity differs")
    return {
        **exact_history,
        "manifest_kind": kind,
        "dual_manifest_sha256": identity.manifest_sha256,
        "flow_model_sha256": identity.flow.model_sha256,
        "flow_state_sha256": identity.flow.state_sha256,
        "aerodynamic_model_sha256": identity.aerodynamic.model_sha256,
        "aerodynamic_state_sha256": identity.aerodynamic.state_sha256,
        "history_state_module_sha256": P026_HISTORY_STATE_SHA256,
        "history_inference_module_sha256": P026_HISTORY_INFERENCE_SHA256,
        "training_protocol_sha256": identity.payload["training_protocol_sha256"],
        "config_sha256": identity.payload["config_sha256"],
        "normalization_sha256": identity.payload["normalization_sha256"],
    }


def verify_receipt_files(root: Path, files: dict) -> None:
    if not isinstance(files, dict) or not files:
        raise ValueError("post-evaluation file table is absent")
    root = root.resolve()
    required = {
        "lineage.json", "precision.json", "development_gate.json",
        "force_window/result.json", "validation10/evaluation.json",
        "validation10/segments.json", "validation10/endpoint_gate.json",
        "validation10/diagnostic.json", "dynamic6/diagnostic.json",
        "evidence/formal_evaluation_approval.json",
        "dynamic6/evaluation.json", "dynamic6/segments.json",
        "step_receipts/validation10.json", "step_receipts/dynamic6.json",
        "step_receipts/force_window.json",
    }
    if not required.issubset(files):
        raise ValueError("post-evaluation evidence is incomplete")
    for relative, digest in files.items():
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise ValueError("invalid post-evaluation file identity")
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file() or sha256(path) != digest:
            raise ValueError("post-evaluation file changed or escapes its root")


def verify_p026_terminal_chain(root: Path, receipt: dict, identity) -> dict:
    """Bind copied approval and both reviewed terminal proofs at low level."""
    approval_path = root / "evidence/formal_approval.json"
    files = receipt.get("sha256", {})
    if (
        not approval_path.is_file()
        or receipt.get("formal_approval_sha256") != sha256(approval_path)
        or files.get("evidence/formal_approval.json") != sha256(approval_path)
    ):
        raise ValueError("P026 copied formal approval differs")
    candidate = identity.manifest_path.parent.resolve()
    artifact_parent = next(
        (parent for parent in candidate.parents if parent.name == "artifacts"), None
    )
    if artifact_parent is None:
        raise ValueError("P026 candidate is outside the repository artifact root")
    repo = artifact_parent.parent
    runner_path = repo / "scripts/run_fcp026_posteval.py"
    if not runner_path.is_file() or sha256(runner_path) != P026_FORMAL_RUNNER_SHA256:
        raise ValueError("reviewed P026 formal runner differs")
    spec = importlib.util.spec_from_file_location(
        "p026_dual_control_formal_runner", runner_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load reviewed P026 formal runner")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    if not isinstance(approval, dict):
        raise TypeError("P026 formal approval is not a JSON object")
    proofs = runner.validate_terminal_proofs(repo, candidate, approval)
    if approval.get("candidate_sha256", {}).get("dual_model_manifest.json") != identity.manifest_sha256:
        raise ValueError("P026 approval belongs to another dual manifest")
    return proofs


def verify_p026_posteval(root: Path, receipt: dict, expected: dict, identity) -> None:
    """Bind the standalone P026 original-formal receipt without legacy aliases."""
    kinds = {P026_K1_SYSTEM_KIND: 1, P026_K4_SYSTEM_KIND: 4}
    history_k = kinds.get(identity.payload.get("kind"))
    if history_k is None:
        raise ValueError("unsupported P026 history kind")
    required = {
        "status": "FC_P026_ORIGINAL_FORMAL_COMPLETE_NOT_ADMISSION",
        "history_k": history_k,
        "protocol": PROTOCOL,
        "scientific_admission": False,
        "ppo_auto_launched": False,
        "frozen_test_accessed": False,
    }
    if any(receipt.get(key) != value for key, value in required.items()):
        raise ValueError("P026 formal receipt identity differs")
    verify_p026_terminal_chain(root, receipt, identity)
    candidate = receipt.get("candidate_sha256")
    exact_candidate = {
        "result.json": sha256(identity.manifest_path.parent / "result.json"),
        "training_protocol.json": identity.payload["training_protocol_sha256"],
        "dual_model_manifest.json": identity.manifest_sha256,
        "flow/FNO.0.0.mdlus": identity.flow.model_sha256,
        "flow/checkpoint.0.0.pt": identity.flow.state_sha256,
        "aerodynamic/FNO.0.1.mdlus": identity.aerodynamic.model_sha256,
        "aerodynamic/checkpoint.0.1.pt": identity.aerodynamic.state_sha256,
    }
    if candidate != exact_candidate:
        raise ValueError("P026 formal receipt candidate bytes differ")
    files = receipt.get("sha256")
    required_outputs = {
        "validation10/evaluation.json",
        "validation10/segments.json",
        "validation10/endpoint_gate.json",
        "validation10/diagnostic.json",
        "dynamic6/evaluation.json",
        "dynamic6/segments.json",
        "dynamic6/diagnostic.json",
        "force_window/result.json",
        "development_gate.json",
    }
    if not isinstance(files, dict) or not required_outputs.issubset(files):
        raise ValueError("P026 formal outputs are incomplete")
    resolved_root = root.resolve()
    for relative, digest in files.items():
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise ValueError("invalid P026 formal output identity")
        path = (resolved_root / relative).resolve()
        if (
            resolved_root not in path.parents
            or not path.is_file()
            or sha256(path) != digest
        ):
            raise ValueError("P026 formal output changed or escapes its root")
    # The endpoint/window/dynamic gate content is still checked by canonical PPO
    # readiness. This function only proves that those exact outputs and both model
    # roles belong to this P026 formal execution.
    if expected != {
        "checkpoint_sha256": identity.aerodynamic.model_sha256,
        "checkpoint_state_sha256": identity.aerodynamic.state_sha256,
        "dual_manifest_sha256": identity.manifest_sha256,
        "flow_model_sha256": identity.flow.model_sha256,
        "flow_state_sha256": identity.flow.state_sha256,
    }:
        raise ValueError("P026 expected dual identity differs")


def receipt_profile(manifest_kind: str) -> tuple[str, str, str]:
    """Select only from the already validated manifest, never receipt claims."""
    profiles = {
        SYSTEM_KIND: ("FC_P013_POSTEVAL_COMPLETE", "fcp013_independent_force_dual_fno",
                      "FC_P013_POSTEVAL_STEP_COMPLETE"),
        P015_SYSTEM_KIND: ("FC_P015_POSTEVAL_COMPLETE", "fcp015_window_accumulation_dual_fno",
                           "FC_P015_POSTEVAL_STEP_COMPLETE"),
        P018_SYSTEM_KIND: ("FC_P018_POSTEVAL_COMPLETE", "fcp018_reduced_rate_dual_fno",
                           "FC_P018_POSTEVAL_STEP_COMPLETE"),
    }
    if manifest_kind not in profiles:
        raise ValueError("unsupported dual control experiment")
    return profiles[manifest_kind]


def check_receipt_identity(receipt: dict, expected: dict, *, manifest_kind: str = SYSTEM_KIND) -> None:
    status, kind, _ = receipt_profile(manifest_kind)
    required = {
        "status": status,
        "candidate_kind": kind,
        "checkpoint_epoch": 1, "protocol": PROTOCOL,
        "frozen_test_accessed": False, "ppo_auto_launched": False,
        **expected,
    }
    if any(receipt.get(key) != value for key, value in required.items()):
        raise ValueError("dual post-evaluation receipt identity differs")


def verify_p015_evidence(root: Path, receipt: dict, expected: dict, identity) -> None:
    """Verify P015's extra provenance without changing any numerical gate."""
    load = lambda path: json.loads(path.read_text())
    experiment = dict(training_experiment="FC-P015", accumulation_windows=8,
                      training_windows=1368, optimizer_steps=171)
    lineage = load(root / "lineage.json")
    required = {"status": "FC_P015_DUAL_CANDIDATE_LINEAGE_PASS_NOT_ADMISSION",
                "candidate_kind": "fcp015_window_accumulation_dual_fno", "checkpoint_epoch": 1,
                **experiment, **expected}
    if any(lineage.get(k) != v for k, v in required.items()):
        raise ValueError("P015 lineage experiment/identity differs")
    hashes = {"lineage_sha256": sha256(root / "lineage.json"),
              "formal_evaluation_approval_sha256": sha256(root / "evidence/formal_evaluation_approval.json"),
              "precision_sha256": sha256(root / "precision.json")}
    if any(receipt.get(k) != v for k, v in hashes.items()):
        raise ValueError("P015 receipt provenance differs")
    chain_sha = receipt.get("posteval_chain_receipt_sha256")
    if not isinstance(chain_sha, str) or len(chain_sha) != 64 or any(c not in "0123456789abcdef" for c in chain_sha):
        raise ValueError("P015 numerical chain SHA absent")
    hashes["posteval_chain_receipt_sha256"] = chain_sha
    for name in ("validation10", "dynamic6", "force_window"):
        step = load(root / "step_receipts" / f"{name}.json")
        if (step.get("candidate_kind") != required["candidate_kind"] or step.get("checkpoint_epoch") != 1
                or any(step.get(k) != v for k, v in hashes.items())):
            raise ValueError("P015 step provenance differs")
    approval = load(root / "evidence/formal_evaluation_approval.json")
    reload_path = root.parent / "dual_reload_receipt.json"
    result_path = root.parent / "candidate/result.json"
    result_sha = sha256(result_path)
    approval_expected = {"status": "FC_P015_FORMAL_EVALUATION_APPROVED", **experiment,
        "candidate_kind": required["candidate_kind"], "checkpoint_epoch": 1,
        "candidate_model_sha256": expected["checkpoint_sha256"],
        "candidate_state_sha256": expected["checkpoint_state_sha256"],
        "dual_manifest_sha256": expected["dual_manifest_sha256"],
        "flow_model_sha256": expected["flow_model_sha256"], "flow_state_sha256": expected["flow_state_sha256"],
        "candidate_result_sha256": result_sha, "dual_reload_receipt_sha256": sha256(reload_path),
        "candidate_completion_receipt_sha256": sha256(root.parent / "completion_receipt.json"),
        "formal_evaluation_authorized": True, "protocol": PROTOCOL,
        "frozen_test_accessed": False, "ppo_auto_launch": False}
    if any(approval.get(k) != v for k, v in approval_expected.items()) or lineage.get("candidate_result_sha256") != result_sha:
        raise ValueError("P015 formal approval/reload/result binding differs")
    reloaded = load(reload_path)
    result = load(result_path)
    reload_expected = {"status": "FC_P015_OFFICIAL_CPU_DUAL_RELOAD_VERIFIED_NOT_ADMISSION",
        **experiment, "device": "cpu", "dual_manifest_sha256": expected["dual_manifest_sha256"],
        "flow_model_sha256": expected["flow_model_sha256"], "flow_state_sha256": expected["flow_state_sha256"],
        "aerodynamic_model_sha256": expected["checkpoint_sha256"],
        "aerodynamic_state_sha256": expected["checkpoint_state_sha256"],
        "config_sha256": identity.payload["config_sha256"], "training_result_sha256": result_sha,
        "tensor_sha256": {role: result.get(role + "_tensor_sha256_after") for role in ("flow", "aerodynamic")},
        "posteval_chain_receipt_sha256": chain_sha, "forward_performed": False,
        "optimizer_created": False, "model_saved": False, "gpu_used": False,
        "scientific_admission": False, "ppo_authorized": False}
    if any(reloaded.get(k) != v for k, v in reload_expected.items()):
        raise ValueError("P015 actual CPU reload identity differs")


def verify_p018_evidence(root: Path, receipt: dict, expected: dict, identity) -> None:
    """Verify P018's extra provenance without changing any numerical gate."""
    load = lambda path: json.loads(path.read_text())
    experiment = dict(training_experiment="FC-P018", accumulation_windows=8,
                      training_windows=1368, optimizer_steps=171,
                      actual_learning_rate=P018_LEARNING_RATE, training_protocol_sha256=P018_PROTOCOL_SHA256)
    protocol_path = root.parent / "candidate/training_protocol.json"
    if sha256(protocol_path) != P018_PROTOCOL_SHA256:
        raise ValueError("P018 candidate training protocol bytes differ")
    lineage = load(root / "lineage.json")
    required = {"status": "FC_P018_DUAL_CANDIDATE_LINEAGE_PASS_NOT_ADMISSION",
                "candidate_kind": "fcp018_reduced_rate_dual_fno", "checkpoint_epoch": 1,
                **experiment, **expected}
    if any(lineage.get(k) != v for k, v in required.items()):
        raise ValueError("P018 lineage experiment/identity differs")
    hashes = {"lineage_sha256": sha256(root / "lineage.json"),
              "formal_evaluation_approval_sha256": sha256(root / "evidence/formal_evaluation_approval.json"),
              "precision_sha256": sha256(root / "precision.json")}
    if any(receipt.get(k) != v for k, v in hashes.items()):
        raise ValueError("P018 receipt provenance differs")
    chain_sha = receipt.get("posteval_chain_receipt_sha256")
    if not isinstance(chain_sha, str) or len(chain_sha) != 64 or any(c not in "0123456789abcdef" for c in chain_sha):
        raise ValueError("P018 numerical chain SHA absent")
    hashes["posteval_chain_receipt_sha256"] = chain_sha
    for name in ("validation10", "dynamic6", "force_window"):
        step = load(root / "step_receipts" / f"{name}.json")
        if (step.get("candidate_kind") != required["candidate_kind"] or step.get("checkpoint_epoch") != 1
                or any(step.get(k) != v for k, v in hashes.items())):
            raise ValueError("P018 step provenance differs")
    approval = load(root / "evidence/formal_evaluation_approval.json")
    reload_path = root.parent / "dual_reload_receipt.json"
    result_path = root.parent / "candidate/result.json"
    result_sha = sha256(result_path)
    approval_expected = {"status": "FC_P018_FORMAL_EVALUATION_APPROVED", **experiment,
        "candidate_kind": required["candidate_kind"], "checkpoint_epoch": 1,
        "candidate_model_sha256": expected["checkpoint_sha256"],
        "candidate_state_sha256": expected["checkpoint_state_sha256"],
        "dual_manifest_sha256": expected["dual_manifest_sha256"],
        "flow_model_sha256": expected["flow_model_sha256"], "flow_state_sha256": expected["flow_state_sha256"],
        "candidate_result_sha256": result_sha, "dual_reload_receipt_sha256": sha256(reload_path),
        "candidate_completion_receipt_sha256": sha256(root.parent / "completion_receipt.json"),
        "formal_evaluation_authorized": True, "protocol": PROTOCOL,
        "frozen_test_accessed": False, "ppo_auto_launch": False}
    if any(approval.get(k) != v for k, v in approval_expected.items()) or lineage.get("candidate_result_sha256") != result_sha:
        raise ValueError("P018 formal approval/reload/result binding differs")
    reloaded = load(reload_path)
    result = load(result_path)
    reload_expected = {"status": "FC_P018_OFFICIAL_CPU_DUAL_RELOAD_VERIFIED_NOT_ADMISSION",
        **experiment, "device": "cpu", "dual_manifest_sha256": expected["dual_manifest_sha256"],
        "flow_model_sha256": expected["flow_model_sha256"], "flow_state_sha256": expected["flow_state_sha256"],
        "aerodynamic_model_sha256": expected["checkpoint_sha256"],
        "aerodynamic_state_sha256": expected["checkpoint_state_sha256"],
        "config_sha256": identity.payload["config_sha256"], "training_result_sha256": result_sha,
        "tensor_sha256": {role: result.get(role + "_tensor_sha256_after") for role in ("flow", "aerodynamic")},
        "posteval_chain_receipt_sha256": chain_sha, "forward_performed": False,
        "optimizer_created": False, "model_saved": False, "gpu_used": False,
        "scientific_admission": False, "ppo_authorized": False}
    if any(reloaded.get(k) != v for k, v in reload_expected.items()):
        raise ValueError("P018 actual CPU reload identity differs")


    execution = {"training_approval_sha256": P018_APPROVAL_SHA256,
                 "execution_observation_sha256": P018_OBSERVATION_SHA256}
    if sha256(root.parent / "execution_approval.json") != P018_APPROVAL_SHA256:
        raise ValueError("P018 actual training execution approval differs")
    if any(lineage.get(k) != v for k,v in {**execution,"invocation_id":P018_INVOCATION}.items()):
        raise ValueError("P018 lineage execution evidence differs")
    result_expected = {"status":"FC_P018_REDUCED_RATE_TRAINING_COMPLETE_NOT_ADMISSION",
        **experiment,"training_protocol_file":"training_protocol.json",
        "dual_model_manifest_sha256":expected["dual_manifest_sha256"],
        "config_sha256":identity.payload["config_sha256"],"base_config_sha256":identity.payload["config_sha256"],
        "official_pair_fresh_reload_verified":True,
        "selection_performed":False,"validation_accessed":False,"frozen_test_accessed":False,"ppo_executed":False}
    if any(result.get(k) != v for k,v in result_expected.items()):
        raise ValueError("P018 training result effective protocol differs")
    tensor_hashes = [result.get(role+"_tensor_sha256_after") for role in ("flow","aerodynamic")]
    if (any(not isinstance(v,str) or len(v)!=64 or any(ch not in "0123456789abcdef" for ch in v) for v in tensor_hashes)
            or result.get("flow_tensor_sha256_before") != tensor_hashes[0]
            or result.get("aerodynamic_tensor_sha256_before") != tensor_hashes[0]):
        raise ValueError("P018 frozen-flow/initial tensor evidence differs")
    completion = load(root.parent / "completion_receipt.json")
    completion_expected = {"status":"FC_P018_TRAINING_COMPLETE_NOT_ADMISSION",**experiment,**execution,**expected,
        "training_protocol_file":"training_protocol.json","candidate_result_sha256":result_sha,
        "unit":"fluid-control-fcp018-reduced-rate-20261005.service",
        "scientific_admission":False,"ppo_executed":False}
    if any(completion.get(k) != v for k,v in completion_expected.items()):
        raise ValueError("P018 training completion identity differs")
    terminal = completion.get("terminal_unit",{})
    if any(terminal.get(k) != v for k,v in {"LoadState":"loaded","InvocationID":P018_INVOCATION,
            "ActiveState":"active","SubState":"exited","MainPID":"0","Result":"success",
            "ExecMainCode":"1","ExecMainStatus":"0"}.items()):
        raise ValueError("P018 retained terminal service proof differs")
    audit_path = root.parent / "candidate_audit.json"
    if completion.get("candidate_audit_sha256") != sha256(audit_path) or load(audit_path) != lineage:
        raise ValueError("P018 completed audit differs from formal lineage")
    if reloaded.get("training_protocol_file") != "training_protocol.json":
        raise ValueError("P018 reload effective protocol file differs")

def verify_dual_control_binding(
    *, manifest_path: Path, expected_manifest_sha256: str,
    training_config: Path, normalization_path: Path,
    checkpoint_dir: Path, expected_checkpoint_sha256: str,
    posteval_receipt: Path, expected_posteval_receipt_sha256: str,
    development_auditor: Path,
) -> dict:
    """Require exact complete dual evidence before canonical PPO can use it."""
    identity = validate_dual_fno_manifest(manifest_path, expected_sha256=expected_manifest_sha256)
    # Real validated manifests always contain kind; default preserves the legacy
    # helper's P013 call contract and historical synthetic fixtures.
    manifest_kind = identity.payload.get("kind", SYSTEM_KIND)
    p026_kind = manifest_kind in (P026_K1_SYSTEM_KIND, P026_K4_SYSTEM_KIND)
    step_status = None if p026_kind else receipt_profile(manifest_kind)[2]
    validate_dual_runtime_files(identity, config_path=training_config,
                               normalization_path=normalization_path)
    if identity.aerodynamic.directory != checkpoint_dir.resolve() or identity.aerodynamic.model_sha256 != expected_checkpoint_sha256:
        raise ValueError("canonical primary checkpoint is not this dual system")
    if sha256(posteval_receipt) != expected_posteval_receipt_sha256:
        raise ValueError("expected complete post-evaluation receipt differs")
    receipt = json.loads(posteval_receipt.read_text())
    expected = {
        "checkpoint_sha256": identity.aerodynamic.model_sha256,
        "checkpoint_state_sha256": identity.aerodynamic.state_sha256,
        "dual_manifest_sha256": identity.manifest_sha256,
        "flow_model_sha256": identity.flow.model_sha256,
        "flow_state_sha256": identity.flow.state_sha256,
    }
    root = posteval_receipt.parent
    if p026_kind:
        verify_p026_posteval(root, receipt, expected, identity)
    else:
        check_receipt_identity(receipt, expected, manifest_kind=manifest_kind)
        verify_receipt_files(root, receipt.get("sha256"))
        for name in ("validation10", "dynamic6", "force_window"):
            step = json.loads((root / "step_receipts" / f"{name}.json").read_text())
            if step.get("status") != step_status or step.get("step") != name or any(step.get(k) != v for k, v in expected.items()):
                raise ValueError("step receipt does not bind the same complete dual system")
    if manifest_kind == P015_SYSTEM_KIND:
        verify_p015_evidence(root, receipt, expected, identity)
    elif manifest_kind == P018_SYSTEM_KIND:
        verify_p018_evidence(root, receipt, expected, identity)
    if sha256(development_auditor) != DEVELOPMENT_AUDITOR_SHA:
        raise ValueError("original numerical development auditor changed")
    spec = importlib.util.spec_from_file_location("dual_control_original_development_audit", development_auditor)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load original numerical gate")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    recomputed = module.audit(root / "force_window/result.json", expected_checkpoint_sha256)
    if recomputed != json.loads((root / "development_gate.json").read_text()):
        raise ValueError("stored development result differs from original recomputation")
    if recomputed.get("status") != "DYNAMIC_FNO_DEVELOPMENT_ADMISSION_PASS":
        raise ValueError("dual surrogate has not passed original development admission")
    history_runtime = p026_runtime_binding(identity)
    return {
        "status": "DUAL_CONTROL_IDENTITY_VERIFIED_NOT_CONTROL_SUCCESS",
        **expected, "posteval_receipt_sha256": expected_posteval_receipt_sha256,
        "training_config_sha256": identity.payload["config_sha256"],
        "normalization_sha256": identity.payload["normalization_sha256"],
        "canonical_endpoint_window_dynamic_gates_still_required": True,
        "policy_trained": False, "real_cfd_control_validated": False,
        **(
            {
                "dual_system_kind": manifest_kind,
                "training_experiment": "FC-P026",
                "fno_history_runtime": history_runtime,
            }
            if history_runtime is not None
            else {}
        ),
        **({"dual_system_kind":P018_SYSTEM_KIND,"training_experiment":"FC-P018",
            "actual_learning_rate":P018_LEARNING_RATE,"training_protocol_sha256":P018_PROTOCOL_SHA256,
            "training_protocol_file":"training_protocol.json"} if manifest_kind == P018_SYSTEM_KIND else {}),
    }
