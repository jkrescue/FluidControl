#!/usr/bin/env python3
"""Independent CPU lineage checks for P018; never scientific admission.

P013's experiment-neutral archive, train-data/order and guard checks are reused.
P018 optimizer and execution identities are separate and fail closed.
"""
from __future__ import annotations

import math
import argparse
import hashlib
import json
from pathlib import Path

from audit_fcp011_candidate import INPUT_SHA, PROTOCOL, load_model_state
from fluid_control.dual_fno import (
    P018_AERO_KIND, P018_SYSTEM_KIND, PRECISION_PROTOCOL, validate_dual_fno_manifest,
)
from train_fcp013_independent_force_fno import (
    CONFIG_SHA, NORMALIZATION_SHA, OFFICIAL_FROZEN_PARAMETER_NAMES,
    PARENT_MODEL_SHA, PARENT_STATE_SHA, sha256, tensor_sha256,
)
import diagnose_fcp014_train_objective as diagnostic

from audit_fcp013_dual_candidate import (
    IMAGE, ORDER_SHA, expected_train_hdf_hashes, observed_order_from_hdf,
    state_digest, validate_guard, validate_state_pair,
)

from audit_fcp015_candidate import (
    require_finite, require_fields, checked_hash, validate_resource_watch, validate_panels,
    validate_records as validate_accumulation_records,
)

PROTOCOL_SHA = "310f0bdf8563a2a70b844a32852791fa1b1dc20278a3098418942e1dab204d2d"
LR = 1.5625e-7
P015_HELPER_SHA = "2f5de1a946b040c42c21f8164da41a5afc642fd6174e7bb35372bb7ba98eb996"
APPROVAL_SHA = "eea5bd5c6a3fe585ae1104600421e50b335f61299c4014c5e3136722af3d4d39"
OBSERVATION_SHA = "c040eae25fa31a98164e08e10dc4f007eb6a3ef38329ad0dcfaddd734c7eb53c"
INVOCATION = "1ca4654aab074278bb2efdfff8dbc1eb"
UNIT = "fluid-control-fcp018-reduced-rate-20261005.service"

WINDOWS = 1368
GROUP_SIZE = 8
UPDATES = 171
P014_SHA = "849570afd814faeaa92af99b1cc26cf71182439aa5c4c42f76b9e3b90bb1c30d"
TRAINER_KEY = "scripts/train_fcp018_reduced_rate.py"
EXPERIMENT = {"training_experiment": "FC-P018", "accumulation_windows": GROUP_SIZE,
              "training_windows": WINDOWS, "optimizer_steps": UPDATES,
              "actual_learning_rate": LR, "training_protocol_sha256": PROTOCOL_SHA,
              "training_protocol_file": "training_protocol.json",
              "config_role": "base_model_data_configuration_not_full_effective_training_configuration"}


def validate_optimizer(state: dict) -> None:
    """Inspect persisted AdamW moments, not a reported update count."""
    import torch

    if type(state.get("epoch")) is not int or state.get("epoch") != 1:
        raise ValueError("P018 terminal epoch differs")
    optimizer = state.get("optimizer_state_dict", {})
    groups = optimizer.get("param_groups", [])
    if len(groups) != 1:
        raise ValueError("P018 AdamW group count differs")
    group = groups[0]
    expected = {"lr": 1.5625e-7, "weight_decay": 1e-4,
                "betas": (0.9, 0.999), "eps": 1e-8}
    if any(group.get(k) != v for k, v in expected.items()):
        raise ValueError("P018 AdamW numerical protocol differs")
    if any(group.get(k, False) for k in ("amsgrad", "maximize", "differentiable")):
        raise ValueError("P018 AdamW variant differs")
    ids, moments = group.get("params", []), optimizer.get("state", {})
    if len(ids) != 28 or any(type(i) is not int for i in ids) or len(set(ids)) != 28 or set(ids) != set(moments):
        raise ValueError("P018 AdamW parameter identity/count differs")
    for item in moments.values():
        step = item.get("step")
        if isinstance(step, torch.Tensor):
            if step.numel() != 1:
                raise ValueError("P018 AdamW step must be scalar")
            step = float(step)
        if isinstance(step, bool) or not isinstance(step, (int, float)) or step != UPDATES:
            raise ValueError("P018 actual AdamW step differs from 171")
        first, second = item.get("exp_avg"), item.get("exp_avg_sq")
        if (not isinstance(first, torch.Tensor) or not isinstance(second, torch.Tensor)
                or first.shape != second.shape or first.dtype != second.dtype
                or not first.numel() or not torch.isfinite(first).all()
                or not torch.isfinite(second).all() or ((torch.view_as_real(second) if second.is_complex() else second) < 0).any()):
            raise ValueError("P018 AdamW moments missing, nonfinite or inconsistent")


def validate_records(records):
    flattened = validate_accumulation_records(records)
    for record in records:
        require_fields(record, {"actual_learning_rate": LR, "training_protocol_sha256": PROTOCOL_SHA}, "update protocol")
    return flattened


def validate_training_result(result: dict, manifest_sha: str) -> list[dict]:
    require_fields(result, {**EXPERIMENT,
        "status": "FC_P018_REDUCED_RATE_TRAINING_COMPLETE_NOT_ADMISSION",
        "dual_model_manifest_sha256": manifest_sha,
        "official_pair_fresh_reload_verified": True,
        "dual_adapter_fresh_reload_verified": False,
        "dual_adapter_reload_status": "SEPARATE_P018_PROFILE_VERIFICATION_REQUIRED",
        "input_sha256": INPUT_SHA, "config_sha256": CONFIG_SHA,
        "sampler_order_sha256": ORDER_SHA, "precision": PRECISION_PROTOCOL,
        "p014_diagnostic_sha256": P014_SHA, "source_sha256": diagnostic.SOURCE_SHA,
        "p015_helper_sha256": P015_HELPER_SHA, "base_config_sha256": CONFIG_SHA,
        "selection_performed": False, "validation_accessed": False,
        "frozen_test_accessed": False, "ppo_executed": False}, "training result contract")
    require_finite(result)
    records = validate_records(result.get("records"))
    validate_panels(result.get("fixed_train_panels"), result["flow_tensor_sha256_before"],
                    result["aerodynamic_tensor_sha256_before"], result["aerodynamic_tensor_sha256_after"])
    return records


def validate_execution(candidate: Path, approval_sha: str, observation_sha: str) -> dict:
    repo = candidate.parent.parent
    if approval_sha != APPROVAL_SHA or observation_sha != OBSERVATION_SHA:
        raise ValueError("P018 execution pin differs")
    checked_hash(candidate / "execution_approval.json", approval_sha)
    observation_path = repo / "docs/FC_P018_RUNNING_EXECUTION_20261005.json"
    checked_hash(observation_path, observation_sha)
    approval = json.loads((candidate / "execution_approval.json").read_text())
    observation = json.loads(observation_path.read_text())
    require_fields(approval, {"status": "FC_P018_APPROVED_REDUCED_RATE_TRAINING_NOT_ADMISSION",
                             "training_protocol_sha256": PROTOCOL_SHA}, "approval")
    require_fields(observation, {"status": "FC_P018_RUNNING_EXECUTION_NOT_COMPLETION",
        "unit": UNIT, "invocation": INVOCATION, "approval_sha256": approval_sha}, "running observation")
    container = observation["container"]
    require_fields(container, {"image": IMAGE, "user": "1000:1000"}, "actual container")
    require_fields(container["state"], {"Running": True, "Paused": False, "OOMKilled": False}, "observed container state")
    if int(container["state"].get("Pid",0)) <= 0:
        raise ValueError("P018 initial container was not live")
    for name,digest in approval["source_sha256"].items():
        checked_hash(repo / "artifacts/fcp018_reduced_rate_source_20261005_immutable" / name, digest)
    trainer_sha = approval["source_sha256"][TRAINER_KEY]
    checked_hash(candidate / "immutable_launcher.sh", approval["launcher_sha256"])
    checked_hash(candidate / "training_protocol.json", PROTOCOL_SHA)
    for name in ("resource_or_deadline_violation", "resource_watcher_unexpected_exit"):
        if (candidate / name).exists(): raise ValueError("P018 external resource watchdog reported failure")
    return {"trainer_sha256": trainer_sha, "invocation_id": INVOCATION, "container_id": container["id"],
            "execution_observation_path": "docs/FC_P018_RUNNING_EXECUTION_20261005.json",
            "guard": validate_guard((candidate / "run.log").read_text()),
            "resource_watch": validate_resource_watch((candidate / "resource_watch.jsonl").read_text())}


def validate_candidate(repo: Path, candidate: Path, *, approval_sha: str, observation_sha: str) -> dict:
    import torch

    repo, candidate = repo.resolve(), candidate.resolve()
    if candidate != repo / "artifacts/fcp018_reduced_rate_training_20261005":
        raise ValueError("P018 candidate root differs")
    checked_hash(Path(diagnostic.__file__), P014_SHA)
    execution = validate_execution(candidate, approval_sha, observation_sha)
    checked_hash(candidate / "candidate/training_protocol.json", PROTOCOL_SHA)
    manifest = candidate / "candidate/dual_model_manifest.json"
    identity = validate_dual_fno_manifest(manifest)
    if identity.payload["kind"] != P018_SYSTEM_KIND:
        raise ValueError("P018 auditor cannot admit another experiment identity")
    require_fields(identity.payload.get("training_semantics"), {**EXPERIMENT,
        "window_count": WINDOWS, "rollout_steps": 100, "batch_size": 1, "seed": 20261003,
        "sampler_order_sha256": ORDER_SHA, "h1_force_weight": .5, "frozen_flow_ar_force_weight": .5,
        "force_objective": "0.5_equal_four_normalized_mse_plus_0.5_rear_cl_normalized_mse",
        "field_loss_used": False, "chunk_size": 10, "optimizer": "AdamW", "learning_rate": 1.5625e-7,
        "weight_decay": 1e-4, "gradient_clip_norm": 1.,
        "validation_accessed": False, "frozen_test_accessed": False}, "manifest training semantics")
    require_fields(identity.payload, {**EXPERIMENT, "input_sha256": INPUT_SHA,
        "base_config_sha256": CONFIG_SHA, "accumulation_helper_sha256": P015_HELPER_SHA}, "manifest train inputs")
    result_path = candidate / "candidate/result.json"
    result = json.loads(result_path.read_text())
    records = validate_training_result(result, identity.manifest_sha256)
    if result.get("trainer_sha256") != execution["trainer_sha256"]:
        raise ValueError("P018 reported trainer differs from observed executable")
    order, hdf_sha = observed_order_from_hdf(repo, records)
    parent, aero = load_model_state(identity.flow.model), load_model_state(identity.aerodynamic.model)
    changed = validate_state_pair(parent, aero)
    expected_tensors = {"flow_tensor_sha256_before": state_digest(parent),
                        "flow_tensor_sha256_after": state_digest(parent),
                        "aerodynamic_tensor_sha256_before": state_digest(parent),
                        "aerodynamic_tensor_sha256_after": state_digest(aero)}
    require_fields(result, expected_tensors, "persisted tensor identity")
    state = torch.load(identity.aerodynamic.state, map_location="cpu", weights_only=True)
    validate_optimizer(state)
    frozen_hashes = {name: tensor_sha256(parent[name]) for name in OFFICIAL_FROZEN_PARAMETER_NAMES}
    require_fields(result, {"frozen_parameter_sha256": frozen_hashes}, "actual frozen parameter hashes")
    require_fields(state.get("metadata"), {**EXPERIMENT,
        "status": P018_AERO_KIND, "checkpoint_epoch": 1,
        "base_config_sha256": CONFIG_SHA, "accumulation_helper_sha256": P015_HELPER_SHA,
        "flow_parent_model_sha256": PARENT_MODEL_SHA, "flow_parent_state_sha256": PARENT_STATE_SHA,
        "aerodynamic_initial_model_sha256": PARENT_MODEL_SHA, "aerodynamic_initial_state_sha256": PARENT_STATE_SHA,
        "sampler_order_sha256": ORDER_SHA, "input_sha256": INPUT_SHA,
        "selection_performed": False, "validation_accessed": False,
        "frozen_test_accessed": False, "ppo_executed": False}, "saved checkpoint metadata")
    paths = [candidate / n for n in ("execution_approval.json", "training_protocol.json", "candidate/training_protocol.json",
                                    "immutable_launcher.sh", "run.log", "resource_watch.jsonl")]
    paths += [manifest, result_path, identity.flow.model, identity.flow.state,
              identity.aerodynamic.model, identity.aerodynamic.state]
    return {"status": "FC_P018_DUAL_CANDIDATE_LINEAGE_PASS_NOT_ADMISSION",
        "candidate_kind": "fcp018_reduced_rate_dual_fno", **EXPERIMENT,
        "candidate_root": str(candidate), "execution_attempt": 1,
        "checkpoint_relative_directory": "candidate/aerodynamic", "checkpoint_epoch": 1,
        "checkpoint_model_file": identity.aerodynamic.model.name,
        "checkpoint_state_file": identity.aerodynamic.state.name,
        "checkpoint_sha256": identity.aerodynamic.model_sha256,
        "checkpoint_state_sha256": identity.aerodynamic.state_sha256,
        "dual_manifest_sha256": identity.manifest_sha256,
        "flow_model_sha256": identity.flow.model_sha256, "flow_state_sha256": identity.flow.state_sha256,
        "official_image_id": IMAGE, "resolved_config_sha256": CONFIG_SHA,
        "normalization_sha256": NORMALIZATION_SHA, "train_order_sha256": order,
        "training_approval_sha256": approval_sha, "execution_observation_sha256": observation_sha,
        "candidate_result_sha256": sha256(result_path), "train_hdf_sha256": hdf_sha,
        "changed_aerodynamic_tensor_names": changed, **execution,
        "official_pair_fresh_reload_verified": True, "dual_adapter_fresh_reload_verified": False,
        "formal_profile_verification_required": True, "formal_protocol": PROTOCOL,
        "training_performed": True, "optimizer_training_performed": True,
        "calibration_fit_performed": False, "validation_or_frozen_accessed": False,
        "ppo_auto_launch": False,
        "sha256": {str(p.relative_to(candidate)): sha256(p) for p in paths}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--execution-approval-sha256", required=True)
    parser.add_argument("--execution-observation-sha256", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = validate_candidate(args.repo, args.candidate,
        approval_sha=args.execution_approval_sha256, observation_sha=args.execution_observation_sha256)
    if args.output:
        diagnostic.write_exclusive(args.output, payload)
    print(json.dumps(payload, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
