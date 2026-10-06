#!/usr/bin/env python3
"""P064-B identity layer over the unchanged P026 full-formal numerics."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys


P028_SHA = "f3acd198e2cc0a977f93b99b523470d46e4b07f64ad66044c94a37f2170de5e6"
P028_PATH = Path(__file__).with_name("run_fcp028_posteval.py")
if P028_PATH.is_symlink() or hashlib.sha256(P028_PATH.read_bytes()).hexdigest() != P028_SHA:
    raise ValueError("reviewed identity wrapper changed")
spec = importlib.util.spec_from_file_location("_p064_formal_identity_base", P028_PATH)
if spec is None or spec.loader is None:
    raise ImportError(P028_PATH)
p028 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = p028
spec.loader.exec_module(p028)

from flow_repair_profiles import FlowRepairProfile, PROFILES

EXPERIMENT = "FC-P064"
KIND = "FC_P064_ARM_B_CONTROLLED_AERO_FORCE_FNO"
PROFILES[EXPERIMENT] = FlowRepairProfile(
    EXPERIMENT, KIND, "FC_P009_TRAIN_ONLY_JOINT_FORCE_ROW_CANDIDATE",
    "equal_H1_AR_half_equal_four_half_rearCl_normalized_MSE", True,
)
FILES = {
    "result.json", "training_protocol.json", "dual_model_manifest.json",
    "flow/FNO.0.0.mdlus", "flow/checkpoint.0.0.pt",
    "aerodynamic/FNO.0.1.mdlus", "aerodynamic/checkpoint.0.1.pt",
}


def candidate_contract(base, candidate: Path, approval: dict, experiment=EXPERIMENT):
    del experiment
    files = approval.get("candidate_sha256")
    base.require(isinstance(files, dict) and set(files) == FILES, "candidate proof incomplete")
    for name, digest in files.items():
        base.bound(candidate, name, digest)
    result = json.loads((candidate / "result.json").read_text())
    protocol = json.loads((candidate / "training_protocol.json").read_text())
    manifest = json.loads((candidate / "dual_model_manifest.json").read_text())
    protocol_sha = files["training_protocol.json"]
    base.fields(result, {
        "status": "FC_P064_ARM_B_TRAINING_COMPLETE_NOT_ADMISSION", "arm": "B",
        "history_k": 1, "optimizer_steps": 32, "training_windows": 256,
        "protocol_sha256": protocol_sha, "official_fresh_reload_verified": True,
        "scientific_admission": False,
    })
    base.fields(protocol, {
        "training_experiment": EXPERIMENT, "arm": "B", "parent_experiment": "FC-P026-K1",
        "training_windows": 256, "optimizer_steps": 32, "accumulation_windows": 8,
        "learning_rate": 1.5625e-7, "validation_accessed": False,
        "frozen_test_accessed": False, "selection_performed": False,
    })
    base.fields(manifest, {
        "status": "FC_P064_ARM_B_DUAL_FNO_MANIFEST_VERIFIED", "kind": KIND,
        "training_experiment": EXPERIMENT, "arm": "B", "training_windows": 256,
        "optimizer_steps": 32, "training_protocol_sha256": protocol_sha,
    })
    base.require(manifest["training_semantics"] == protocol, "training protocol differs")
    base.fields(manifest["history_input"], {
        "profile": "p026_k1", "history_length": 1, "flow_input_channels": 6,
        "aerodynamic_input_channels": 6, "future_state_inputs": False,
        "future_force_inputs": False,
    })
    records = result.get("records")
    base.require(isinstance(records, list) and len(records) == 32, "incomplete updates")
    base.require(
        [(row.get("update"), row.get("consumed_windows")) for row in records]
        == [(index, index * 8) for index in range(1, 33)], "update sequence differs",
    )
    base.fields(manifest["flow"], {
        "role": "flow", "frozen": True, "checkpoint_epoch": 0,
        "model_sha256": files["flow/FNO.0.0.mdlus"],
        "state_sha256": files["flow/checkpoint.0.0.pt"],
    })
    base.fields(manifest["aerodynamic"], {
        "role": "aerodynamic", "frozen": False, "checkpoint_epoch": 1,
        "metadata_kind": "FC_P064_ARM_B_CONTROLLED_AERO_CHECKPOINT",
        "model_sha256": files["aerodynamic/FNO.0.1.mdlus"],
        "state_sha256": files["aerodynamic/checkpoint.0.1.pt"],
    })
    return manifest


def validate_candidate_loader(base, source: Path, candidate: Path, approval: dict, experiment=EXPERIMENT):
    del experiment
    loader = source / "src/fluid_control/dual_fno.py"
    base.require(base.sha(loader) == approval["source_sha256"]["src/fluid_control/dual_fno.py"], "loader bytes differ")
    module_spec = importlib.util.spec_from_file_location("_p064_frozen_loader", loader)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    try:
        module_spec.loader.exec_module(module)
        identity = module.validate_dual_fno_manifest(
            candidate / "dual_model_manifest.json",
            expected_sha256=approval["candidate_sha256"]["dual_model_manifest.json"],
        )
    finally:
        sys.modules.pop(module_spec.name, None)
    base.fields(identity.payload, {"kind": KIND, "arm": "B"})
    base.require(identity.flow.directory == (candidate / "flow").resolve(), "flow role differs")
    base.require(identity.aerodynamic.directory == (candidate / "aerodynamic").resolve(), "aero role differs")
    return identity


def validate_terminal_proofs(base, repo: Path, candidate: Path, approval: dict, experiment=EXPERIMENT):
    del experiment
    expected = approval["candidate_sha256"]
    prefixed = {"candidate/" + name: digest for name, digest in expected.items()}
    proofs = {}
    for label in ("independent_terminal_audit", "official_dual_reload"):
        binding = approval[label]
        base.require(binding.get("reviewed_by_lead") is True, "terminal proof is unreviewed")
        path = base.bound(repo, binding["path"], binding["sha256"])
        payload = json.loads(path.read_text())
        base.require(payload.get("candidate_sha256") == prefixed, "terminal proof candidate bytes differ")
        base.fields(payload, {
            "training_protocol_sha256": expected["training_protocol.json"],
            "dual_manifest_sha256": expected["dual_model_manifest.json"],
            "candidate_result_sha256": expected["result.json"],
            "scientific_admission": False, "ppo_authorized": False,
        })
        proofs[label] = payload
    audit, reload = proofs["independent_terminal_audit"], proofs["official_dual_reload"]
    base.fields(audit, {
        "status": "FC_P064_ARM_B_CANDIDATE_INTEGRITY_VERIFIED_NOT_ADMISSION",
        "actual_optimizer_steps": 32, "actual_training_windows": 256,
        "accumulation_windows": 8, "training_unit": approval["training_unit"],
        "training_invocation": approval["training_invocation"],
        "flow_trained": False, "aerodynamic_trained": True,
        "dual_adapter_fresh_reload_verified": False,
    })
    base.terminal(audit["terminal_evidence"], approval["training_invocation"])
    base.fields(reload, {
        "status": "FC_P064_ARM_B_OFFICIAL_CPU_DUAL_RELOAD_VERIFIED_NOT_ADMISSION",
        "candidate_audit_sha256": approval["independent_terminal_audit"]["sha256"],
        "config_sha256": approval["training_config_sha256"],
        "required_official_image_id": base.IMAGE, "device": "cpu",
        "official_dual_reload_verified": True, "forward_performed": False,
        "optimizer_created": False, "model_saved": False, "gpu_used": False,
    })
    result = json.loads((candidate / "result.json").read_text())
    tensors = {"flow": result["flow_tensor_sha256"], "aerodynamic": result["aerodynamic_terminal_tensor_sha256"]}
    base.require(audit.get("tensor_sha256") == reload.get("tensor_sha256") == tensors, "terminal tensor identity differs")
    base.require(
        audit.get("role_loader_sha256")
        == reload.get("runtime_source_sha256", {}).get("src/fluid_control/dual_fno.py")
        == approval["source_sha256"]["src/fluid_control/dual_fno.py"],
        "audit/reload/formal loader differs",
    )


def commands(base, manifest_sha: str, aero_model_sha: str, experiment=EXPERIMENT):
    del experiment
    plan = base.commands(1, aero_model_sha, manifest_sha)
    replaced = 0
    for name, _, command in plan:
        if name == "validation_diagnostic":
            index = command.index("--candidate-kind") + 1
            base.require(command[index] == "FC_P026_K1_HISTORY_FORCE_FNO", "base candidate identity drifted")
            command[index] = KIND
            replaced += 1
    base.require(replaced == 1, "P064 diagnostic identity was not routed")
    return plan


_inherited_preflight = p028.preflight


def preflight(base, args):
    """Bind every orchestration file before delegating to reviewed numerics."""
    approval = _inherited_preflight(base, args)
    expected = approval.get("orchestration_sha256")
    base.require(isinstance(expected, dict), "P064 orchestration proof absent")
    actual = {
        "scripts/run_fcp026_posteval.py": Path(args.numerical_runner),
        "scripts/run_fcp028_posteval.py": P028_PATH,
        "scripts/flow_repair_profiles.py": Path(p028.flow_repair_profiles.__file__),
        "scripts/run_fcp064_posteval.py": Path(args.entry_file),
    }
    base.require(set(expected) == set(actual), "P064 orchestration proof incomplete")
    for name, path in actual.items():
        base.require(expected[name] == base.sha(path), f"P064 orchestration bytes differ: {name}")
    return approval


p028.candidate_contract = candidate_contract
p028.validate_candidate_loader = validate_candidate_loader
p028.validate_terminal_proofs = validate_terminal_proofs
p028.commands = commands
p028.preflight = preflight

if __name__ == "__main__":
    p028.main(EXPERIMENT, entry_file=__file__)
