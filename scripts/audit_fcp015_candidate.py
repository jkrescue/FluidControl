#!/usr/bin/env python3
"""Independent CPU lineage checks for P015; never scientific admission.

P013's experiment-neutral archive, train-data/order and guard checks are reused.
P015 optimizer and execution identities are separate and fail closed.
"""
from __future__ import annotations

import math
import argparse
import hashlib
import json
from pathlib import Path

from audit_fcp011_candidate import INPUT_SHA, PROTOCOL, load_model_state
from fluid_control.dual_fno import (
    P015_AERO_KIND, P015_SYSTEM_KIND, PRECISION_PROTOCOL, validate_dual_fno_manifest,
)
from train_fcp013_independent_force_fno import (
    CONFIG_SHA, NORMALIZATION_SHA, OFFICIAL_FROZEN_PARAMETER_NAMES,
    PARENT_MODEL_SHA, PARENT_STATE_SHA, sha256,
)
import diagnose_fcp014_train_objective as diagnostic

from audit_fcp013_dual_candidate import (
    IMAGE, ORDER_SHA, expected_train_hdf_hashes, observed_order_from_hdf,
    state_digest, validate_guard, validate_state_pair,
)

WINDOWS = 1368
GROUP_SIZE = 8
UPDATES = 171
P014_SHA = "849570afd814faeaa92af99b1cc26cf71182439aa5c4c42f76b9e3b90bb1c30d"
TRAINER_KEY = "scripts/train_fcp015_window_accumulation.py"
EXPERIMENT = {"training_experiment": "FC-P015", "accumulation_windows": GROUP_SIZE,
              "training_windows": WINDOWS, "optimizer_steps": UPDATES}


def validate_optimizer(state: dict) -> None:
    """Inspect persisted AdamW moments, not a reported update count."""
    import torch

    if state.get("epoch") != 1:
        raise ValueError("P015 terminal epoch differs")
    optimizer = state.get("optimizer_state_dict", {})
    groups = optimizer.get("param_groups", [])
    if len(groups) != 1:
        raise ValueError("P015 AdamW group count differs")
    group = groups[0]
    expected = {"lr": 1e-5, "weight_decay": 1e-4,
                "betas": (0.9, 0.999), "eps": 1e-8}
    if any(group.get(k) != v for k, v in expected.items()):
        raise ValueError("P015 AdamW numerical protocol differs")
    if any(group.get(k, False) for k in ("amsgrad", "maximize", "differentiable")):
        raise ValueError("P015 AdamW variant differs")
    ids, moments = group.get("params", []), optimizer.get("state", {})
    if len(ids) != 28 or len(set(ids)) != 28 or set(ids) != set(moments):
        raise ValueError("P015 AdamW parameter identity/count differs")
    for item in moments.values():
        step = item.get("step")
        if isinstance(step, torch.Tensor):
            if step.numel() != 1:
                raise ValueError("P015 AdamW step must be scalar")
            step = float(step)
        if isinstance(step, bool) or not isinstance(step, (int, float)) or step != UPDATES:
            raise ValueError("P015 actual AdamW step differs from 171")
        first, second = item.get("exp_avg"), item.get("exp_avg_sq")
        if (not isinstance(first, torch.Tensor) or not isinstance(second, torch.Tensor)
                or first.shape != second.shape or first.dtype != second.dtype
                or not first.numel() or not torch.isfinite(first).all()
                or not torch.isfinite(second).all() or (second < 0).any()):
            raise ValueError("P015 AdamW moments missing, nonfinite or inconsistent")


def require_finite(value, label="result") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            require_finite(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            require_finite(item, f"{label}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"nonfinite P015 value: {label}")


def require_fields(value, expected, label):
    if not isinstance(value, dict) or any(value.get(k) != v or
            (type(v) in (bool, int) and type(value.get(k)) is not type(v))
            for k, v in expected.items()):
        raise ValueError(f"P015 {label} differs")


def validate_records(records: list) -> list[dict]:
    """Return the ordered 1368 real consumed windows for the physical-data audit."""
    if not isinstance(records, list) or len(records) != UPDATES:
        raise ValueError("P015 needs exactly 171 update records")
    flattened = []
    for update, group in enumerate(records, 1):
        require_fields(group, {"update": update, "consumed_windows": update * GROUP_SIZE,
                              "windows": GROUP_SIZE, "optimizer_steps": 1}, "update boundary")
        norm = group.get("preclip_mean_gradient_norm")
        if isinstance(norm, bool) or not isinstance(norm, (int, float)) or not math.isfinite(norm) or norm < 0:
            raise ValueError("P015 aggregated gradient norm invalid")
        require_fields(group.get("gradient_audit"), {
            "official_frozen_parameter_names": list(OFFICIAL_FROZEN_PARAMETER_NAMES),
            "official_frozen_parameter_shapes": {
                "spec_encoder.lift_network.0.conv.bias": [24],
                "spec_encoder.lift_network.2.conv.bias": [48]},
            "trainable_parameter_tensor_count": 28,
            "trainable_gradient_all_finite": True}, "gradient audit")
        rows = group.get("records")
        if not isinstance(rows, list) or len(rows) != GROUP_SIZE:
            raise ValueError("P015 update does not contain eight windows")
        for row in rows:
            require_fields(row, {"chunk_size": 10, "chunks": 10}, "window chunk schedule")
            require_fields(row.get("identity"), {"split": "train", "rollout_steps": 100}, "train window")
            for domain in ("h1", "ar"):
                channels = row.get(domain + "_channel_mse")
                if not isinstance(channels, list) or len(channels) != 4 or any(
                        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in channels):
                    raise ValueError("P015 four-force objective channels invalid")
                expected = .125 * sum(channels) + .5 * channels[3]
                if not math.isclose(row[domain + "_balanced"], expected, rel_tol=2e-6, abs_tol=1e-10):
                    raise ValueError("P015 balanced objective normalization differs")
            if not math.isclose(row["total"], .5 * (row["h1_balanced"] + row["ar_balanced"]),
                                rel_tol=2e-6, abs_tol=1e-10):
                raise ValueError("P015 H1/AR objective mixing differs")
        for key in ("h1_balanced", "ar_balanced", "total"):
            if group.get("mean_objective", {}).get(key) != sum(row[key] for row in rows) / GROUP_SIZE:
                raise ValueError("P015 update mean objective differs")
        flattened.extend(rows)
    require_finite(records)
    return flattened


def validate_panels(panels: list, flow_hash: str, initial_hash: str, terminal_hash: str) -> None:
    if not isinstance(panels, list) or len(panels) != 4:
        raise ValueError("P015 fixed diagnostic snapshots missing")
    for panel, consumed in zip(panels, (0, 456, 912, 1368), strict=True):
        require_fields(panel, {"consumed_windows": consumed, "optimizer_steps": consumed // GROUP_SIZE,
                               "no_grad": True, "model_selection": False}, "diagnostic boundary")
        before, after = panel.get("tensor_sha256_before"), panel.get("tensor_sha256_after")
        if (not isinstance(before, list) or len(before) != 2 or before != after or before[0] != flow_hash
                or (consumed == 0 and before[1] != initial_hash)
                or (consumed == WINDOWS and before[1] != terminal_hash)):
            raise ValueError("P015 diagnostic tensor mutation/identity differs")
        rows = panel.get("rows", [])
        diagnostic.validate_windows(rows)
        for row in rows:
            data = row["panel"]
            require_fields(data, {"force_forward_batch_sizes": [20] * 10,
                                  "force_predictor_training_mode": True, "autograd_enabled": False},
                           "diagnostic force schedule/mode")
            for domain in ("h1", "ar"):
                values = data["domains"][domain]
                residual = values["rear_cl_physical_residual"]
                for name, vector in (("h100", residual), ("tail62", residual[38:])):
                    if diagnostic.residual_statistics(vector) != values[name]:
                        raise ValueError("P015 residual decomposition differs")
    require_finite(panels)


def validate_training_result(result: dict, manifest_sha: str) -> list[dict]:
    require_fields(result, {**EXPERIMENT,
        "status": "FC_P015_WINDOW_ACCUMULATION_TRAINING_COMPLETE_NOT_ADMISSION",
        "dual_model_manifest_sha256": manifest_sha,
        "official_pair_fresh_reload_verified": True,
        "dual_adapter_fresh_reload_verified": False,
        "dual_adapter_reload_status": "SEPARATE_P015_PROFILE_VERIFICATION_REQUIRED",
        "input_sha256": INPUT_SHA, "config_sha256": CONFIG_SHA,
        "sampler_order_sha256": ORDER_SHA, "precision": PRECISION_PROTOCOL,
        "p014_diagnostic_sha256": P014_SHA, "source_sha256": diagnostic.SOURCE_SHA,
        "selection_performed": False, "validation_accessed": False,
        "frozen_test_accessed": False, "ppo_executed": False}, "training result contract")
    require_finite(result)
    records = validate_records(result.get("records"))
    validate_panels(result.get("fixed_train_panels"), result["flow_tensor_sha256_before"],
                    result["aerodynamic_tensor_sha256_before"], result["aerodynamic_tensor_sha256_after"])
    return records


def checked_hash(path: Path, expected: str) -> None:
    if (not isinstance(expected, str) or len(expected) != 64
            or any(c not in "0123456789abcdef" for c in expected)
            or sha256(path) != expected):
        raise ValueError(f"P015 pinned evidence SHA differs: {path}")


def validate_resource_watch(text: str) -> dict:
    """P015's actual host sampler schema; progress timeout is enforced by launcher."""
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not rows:
        raise ValueError("P015 resource watch is empty")
    previous = 0
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"timestamp", "mem_available_kib", "mem_free_kib"}:
            raise ValueError("P015 resource watch schema/failure differs")
        if any(type(v) is not int for v in row.values()) or row["timestamp"] <= previous:
            raise ValueError("P015 resource timestamp or sample differs")
        if min(row["mem_available_kib"], row["mem_free_kib"]) < 20 * 1024**2:
            raise ValueError("P015 host resource floor violated")
        previous = row["timestamp"]
    return {"samples": len(rows),
            "min_mem_available_gib": min(r["mem_available_kib"] for r in rows) / 1024**2,
            "min_mem_free_gib": min(r["mem_free_kib"] for r in rows) / 1024**2}


def validate_execution(candidate: Path, approval_sha: str, observation_sha: str) -> dict:
    checked_hash(candidate / "execution_approval.json", approval_sha)
    checked_hash(candidate / "running_execution_evidence.json", observation_sha)
    approval = json.loads((candidate / "execution_approval.json").read_text())
    observation = json.loads((candidate / "running_execution_evidence.json").read_text())
    require_fields(approval, {"status": "FC_P015_APPROVED_FIXED_ACCUMULATION_TRAINING_NOT_ADMISSION"}, "approval kind")
    require_fields(observation, {
        "status": "FC_P015_RUNNING_EXECUTION_OBSERVED", "image_id": IMAGE,
        "approval_sha256": approval_sha, "attempt": 1,
        "retrospective_launch_approval": False, "validation_or_frozen_mounted": False,
        "training_complete": False}, "observed runtime")
    trainer_sha = approval.get("source_sha256", {}).get(TRAINER_KEY)
    if not trainer_sha or observation.get("source_sha256", {}).get(TRAINER_KEY) != trainer_sha:
        raise ValueError("P015 observed trainer differs from approval")
    launcher_sha = approval.get("launcher_sha256")
    if observation.get("executed_launcher_sha256") != launcher_sha:
        raise ValueError("P015 observed launcher differs from approval")
    checked_hash(candidate / "immutable_launcher.sh", launcher_sha)
    unit = observation.get("unit", {})
    if (unit.get("ActiveState") != "active" or unit.get("SubState") != "running"
            or not unit.get("InvocationID") or int(unit.get("MainPID", 0)) <= 0):
        raise ValueError("P015 original running observation is not live evidence")
    for name in ("resource_or_deadline_violation", "resource_watcher_unexpected_exit"):
        if (candidate / name).exists():
            raise ValueError("P015 external resource watchdog reported failure")
    return {"trainer_sha256": trainer_sha, "invocation_id": unit["InvocationID"],
            "container_id": observation["container_id"],
            "guard": validate_guard((candidate / "run.log").read_text()),
            "resource_watch": validate_resource_watch((candidate / "resource_watch.jsonl").read_text())}


def validate_candidate(repo: Path, candidate: Path, *, approval_sha: str, observation_sha: str) -> dict:
    import torch

    repo, candidate = repo.resolve(), candidate.resolve()
    if candidate != repo / "artifacts/fcp015_window_accumulation_training_20261005":
        raise ValueError("P015 candidate root differs")
    checked_hash(Path(diagnostic.__file__), P014_SHA)
    execution = validate_execution(candidate, approval_sha, observation_sha)
    manifest = candidate / "candidate/dual_model_manifest.json"
    identity = validate_dual_fno_manifest(manifest)
    if identity.payload["kind"] != P015_SYSTEM_KIND:
        raise ValueError("P015 auditor cannot admit another experiment identity")
    require_fields(identity.payload.get("training_semantics"), {**EXPERIMENT,
        "window_count": WINDOWS, "rollout_steps": 100, "batch_size": 1, "seed": 20261003,
        "sampler_order_sha256": ORDER_SHA, "h1_force_weight": .5, "frozen_flow_ar_force_weight": .5,
        "force_objective": "0.5_equal_four_normalized_mse_plus_0.5_rear_cl_normalized_mse",
        "field_loss_used": False, "chunk_size": 10, "optimizer": "AdamW", "learning_rate": 1e-5,
        "weight_decay": 1e-4, "gradient_clip_norm": 1.,
        "validation_accessed": False, "frozen_test_accessed": False}, "manifest training semantics")
    require_fields(identity.payload, {"input_sha256": INPUT_SHA}, "manifest train inputs")
    result_path = candidate / "candidate/result.json"
    result = json.loads(result_path.read_text())
    records = validate_training_result(result, identity.manifest_sha256)
    if result.get("trainer_sha256") != execution["trainer_sha256"]:
        raise ValueError("P015 reported trainer differs from observed executable")
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
    require_fields(state.get("metadata"), {**EXPERIMENT,
        "status": P015_AERO_KIND, "checkpoint_epoch": 1,
        "flow_parent_model_sha256": PARENT_MODEL_SHA, "flow_parent_state_sha256": PARENT_STATE_SHA,
        "aerodynamic_initial_model_sha256": PARENT_MODEL_SHA, "aerodynamic_initial_state_sha256": PARENT_STATE_SHA,
        "sampler_order_sha256": ORDER_SHA, "input_sha256": INPUT_SHA,
        "selection_performed": False, "validation_accessed": False,
        "frozen_test_accessed": False, "ppo_executed": False}, "saved checkpoint metadata")
    paths = [candidate / n for n in ("execution_approval.json", "running_execution_evidence.json",
                                    "immutable_launcher.sh", "run.log", "resource_watch.jsonl")]
    paths += [manifest, result_path, identity.flow.model, identity.flow.state,
              identity.aerodynamic.model, identity.aerodynamic.state]
    return {"status": "FC_P015_DUAL_CANDIDATE_LINEAGE_PASS_NOT_ADMISSION",
        "candidate_kind": "fcp015_window_accumulation_dual_fno", **EXPERIMENT,
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
