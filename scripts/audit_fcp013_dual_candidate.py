#!/usr/bin/env python3
"""Independently audit persisted P013 tensors, updates and execution evidence.

This project auditor does not evaluate scientific accuracy or authorize PPO.
It reads official checkpoint archives on CPU; it never changes a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from audit_fcp011_candidate import INPUT_SHA, PROTOCOL, atomic_json, load_model_state
from evaluate_fcp013_fixed_train_windows import check_training_result
from fluid_control.dual_fno import validate_dual_fno_manifest
from train_fcp013_independent_force_fno import (
    CONFIG_SHA, NORMALIZATION_SHA, OFFICIAL_FROZEN_PARAMETER_NAMES,
    PARENT_MODEL_SHA, PARENT_STATE_SHA, sha256,
)

APPROVAL_SHA = "1bdcfcf71d1581bbae66fc6551dde61a500bd95a464d9f61d510cbac1721a120"
TRAINER_SHA = "f3cf4b9a745cc0cbee39db9385cbfc398e4834e25bc487d8fb2d6eca07b483d7"
IMAGE = "sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
ORDER_SHA = "177ebd9523cde918eb0c1fb8026286dac7e95f3d228ae0e349757a2a9a288f9f"
OBSERVATION_SHA = "a106555efbe2e6b6ab1046a94ed05f58d903785b223716cdd7075a73bc432f63"
ROOT_NAME = "fcp013_independent_force_fno_training_20261005"


def state_digest(state: dict) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def validate_state_pair(parent: dict, aerodynamic: dict) -> list[str]:
    import torch
    if parent.keys() != aerodynamic.keys():
        raise ValueError("official FNO state keys differ")
    changed = []
    for name, left in parent.items():
        right = aerodynamic[name]
        if left.shape != right.shape or left.dtype != right.dtype or not torch.isfinite(right).all():
            raise ValueError("aerodynamic tensor structure or finiteness differs")
        if not torch.equal(left, right):
            if name in OFFICIAL_FROZEN_PARAMETER_NAMES:
                raise ValueError("official frozen lifting bias changed")
            changed.append(name)
    if not changed:
        raise ValueError("independent force FNO was not updated")
    return sorted(changed)


def validate_optimizer(state: dict) -> None:
    if state.get("epoch") != 1:
        raise ValueError("terminal epoch differs")
    optimizer = state.get("optimizer_state_dict", {})
    groups = optimizer.get("param_groups", [])
    if len(groups) != 1:
        raise ValueError("AdamW group count differs")
    group = groups[0]
    if group.get("lr") != 1e-5 or group.get("weight_decay") != 1e-4 or group.get("betas") != (0.9, 0.999) or group.get("eps") != 1e-8:
        raise ValueError("AdamW protocol differs")
    ids = group.get("params", [])
    values = optimizer.get("state", {})
    if len(ids) != 28 or len(set(ids)) != 28 or set(ids) != set(values):
        raise ValueError("AdamW trainable parameter count differs")
    import torch
    for item in values.values():
        if float(item.get("step", -1)) != 1368:
            raise ValueError("actual persisted AdamW step differs")
        for name in ("exp_avg", "exp_avg_sq"):
            tensor = item.get(name)
            if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
                raise ValueError("AdamW moments missing or nonfinite")


def validate_guard(text: str) -> dict:
    events = []
    for line in text.splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict):
            events.append(item)
    if any(item.get("event") == "gpu_floor_violation" for item in events):
        raise ValueError("memory floor was violated")
    completed = [item for item in events if item.get("event") == "gpu_guard_complete"]
    if len(completed) != 1:
        raise ValueError("one successful completed guard required")
    guard = completed[0]
    memory = guard.get("min_observed_mem_available_gib")
    if guard.get("exit_code") != 0 or guard.get("min_required_mem_available_gib") != 20 or not isinstance(memory, (int, float)) or not math.isfinite(memory) or memory < 20 or guard.get("memory_samples", 0) < 1:
        raise ValueError("execution or memory guard did not pass")
    return guard


def observed_order_from_hdf(repo: Path, records: list[dict]) -> tuple[str, dict]:
    """Reconstruct global windows from actual HDF shapes, not reported counts."""
    import h5py
    roots = (
        ("tandem_cylinders_matched_start_full40_dev30_v1", 20, 20, "base_manifest"),
        ("tandem_cylinders_dynamic_train8_v1", 2, 8, "train8_manifest"),
        ("tandem_cylinders_directppo_train16_v1", 2, 16, "train16_manifest"),
    )
    mapping, source_sha = {}, {}
    for family, (name, stride, count, manifest_key) in enumerate(roots):
        root = repo / "data/curated" / name
        if sha256(root / "manifest.json") != INPUT_SHA[manifest_key] or sha256(root / "normalization.json") != NORMALIZATION_SHA:
            raise ValueError("source manifest or normalization differs")
        files = sorted((root / "train").glob("*.h5"))
        if len(files) != count:
            raise ValueError("training trajectory count differs")
        for path in files:
            with h5py.File(path, "r") as handle:
                frames = len(handle["state"])
            source_sha[str(path.relative_to(repo))] = sha256(path)
            for start in range(0, frames - 100, stride):
                key = (path.stem, start, family)
                if key in mapping:
                    raise ValueError("duplicate source identity")
                mapping[key] = len(mapping)
    order = []
    for row in records:
        identity = row["identity"]
        key = (identity["case"], identity["start"], identity["dataset_index"])
        if key not in mapping:
            raise ValueError("training identity absent from physical HDF")
        order.append(mapping[key])
    if len(mapping) != 1368 or sorted(order) != list(range(1368)):
        raise ValueError("training pass not complete and unique")
    digest = hashlib.sha256(json.dumps(order, separators=(",", ":")).encode()).hexdigest()
    if digest != ORDER_SHA:
        raise ValueError("actual consumed window order differs")
    return digest, source_sha


def validate_candidate(repo: Path, candidate: Path) -> dict:
    import torch
    if candidate.resolve() != (repo / "artifacts" / ROOT_NAME).resolve():
        raise ValueError("candidate root differs")
    if sha256(candidate / "execution_approval.json") != APPROVAL_SHA or sha256(candidate / "running_execution_evidence.json") != OBSERVATION_SHA:
        raise ValueError("execution evidence differs")
    observation = json.loads((candidate / "running_execution_evidence.json").read_text())
    if observation.get("image_id") != IMAGE or observation.get("source_sha256", {}).get("scripts/train_fcp013_independent_force_fno.py") != TRAINER_SHA:
        raise ValueError("observed runtime source differs")
    if sha256(candidate / "immutable_launcher.sh") != observation["source_sha256"]["scripts/run_fcp013_training_spark.sh"]:
        raise ValueError("observed launcher differs")
    guard = validate_guard((candidate / "run.log").read_text())
    manifest = candidate / "candidate/dual_model_manifest.json"
    identity = validate_dual_fno_manifest(manifest)
    result_path = candidate / "candidate/result.json"
    result = json.loads(result_path.read_text())
    check_training_result(result, identity.manifest_sha256)
    if result.get("input_sha256") != INPUT_SHA:
        raise ValueError("training data identities differ")
    order, hdf_sha = observed_order_from_hdf(repo, result["records"])
    parent = load_model_state(identity.flow.model)
    aero = load_model_state(identity.aerodynamic.model)
    changed = validate_state_pair(parent, aero)
    for key, expected in (("flow_tensor_sha256_before", state_digest(parent)),
                          ("flow_tensor_sha256_after", state_digest(parent)),
                          ("aerodynamic_tensor_sha256_before", state_digest(parent)),
                          ("aerodynamic_tensor_sha256_after", state_digest(aero))):
        if result.get(key) != expected:
            raise ValueError("reported tensor identity differs from saved archive")
    state = torch.load(identity.aerodynamic.state, map_location="cpu", weights_only=False)
    validate_optimizer(state)
    metadata = state.get("metadata", {})
    required = {
        "status": "FC_P013_INDEPENDENT_FORCE_FNO_AERODYNAMIC_CHECKPOINT",
        "checkpoint_epoch": 1, "optimizer_steps": 1368,
        "flow_parent_model_sha256": PARENT_MODEL_SHA, "flow_parent_state_sha256": PARENT_STATE_SHA,
        "aerodynamic_initial_model_sha256": PARENT_MODEL_SHA,
        "aerodynamic_initial_state_sha256": PARENT_STATE_SHA,
        "sampler_order_sha256": ORDER_SHA, "input_sha256": INPUT_SHA,
        "selection_performed": False, "validation_accessed": False,
        "frozen_test_accessed": False, "ppo_executed": False,
    }
    if any(metadata.get(key) != value for key, value in required.items()):
        raise ValueError("saved checkpoint metadata differs")
    files = [candidate / "execution_approval.json", candidate / "running_execution_evidence.json",
             candidate / "immutable_launcher.sh", candidate / "run.log", result_path, manifest,
             identity.flow.model, identity.flow.state, identity.aerodynamic.model, identity.aerodynamic.state]
    return {
        "status": "FC_P013_DUAL_CANDIDATE_LINEAGE_PASS_NOT_ADMISSION",
        "candidate_kind": "fcp013_independent_force_dual_fno", "candidate_root": str(candidate.resolve()),
        "checkpoint_relative_directory": "candidate/aerodynamic", "checkpoint_epoch": 1,
        "checkpoint_model_file": identity.aerodynamic.model.name,
        "checkpoint_state_file": identity.aerodynamic.state.name,
        "checkpoint_sha256": identity.aerodynamic.model_sha256,
        "checkpoint_state_sha256": identity.aerodynamic.state_sha256,
        "dual_manifest_sha256": identity.manifest_sha256,
        "flow_model_sha256": identity.flow.model_sha256, "flow_state_sha256": identity.flow.state_sha256,
        "official_image_id": IMAGE, "resolved_config_sha256": CONFIG_SHA,
        "normalization_sha256": NORMALIZATION_SHA, "optimizer_steps": 1368,
        "training_performed": True, "optimizer_training_performed": True,
        "calibration_fit_performed": False, "train_order_sha256": order,
        "changed_aerodynamic_tensor_names": changed, "guard": guard,
        "validation_or_frozen_accessed": False, "ppo_auto_launch": False,
        "formal_protocol": PROTOCOL, "training_approval_sha256": APPROVAL_SHA,
        "candidate_result_sha256": sha256(result_path), "train_hdf_sha256": hdf_sha,
        "sha256": {str(path.relative_to(candidate)): sha256(path) for path in files},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--execution-approval-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.execution_approval_sha256 not in (None, APPROVAL_SHA):
        raise ValueError("training approval differs")
    candidate = args.candidate or args.repo / "artifacts" / ROOT_NAME
    if not (candidate / "candidate/result.json").is_file() and args.candidate is None and args.output is None:
        print(json.dumps({"status": "FC_P013_CANDIDATE_AUDITOR_READY_NO_TERMINAL_YET"}))
        return
    result = validate_candidate(args.repo, candidate)
    if args.output is not None:
        atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
