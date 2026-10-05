#!/usr/bin/env python3
"""Strict lineage audit for the train-only FC-P009 joint force-row candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from fluid_control.calibrated_checkpoint import (
    FC_P009_STATUS,
    validate_calibrated_epoch_zero,
)

IMAGE_ID = "sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
CANDIDATE_NAME = "fcp009_joint_force_row_candidate_20261005"
CANDIDATE_KIND = "joint_h1_free_ar_force_row_recalibration"
LINEAGE_STATUS = "FC_P009_CANDIDATE_LINEAGE_PASS"
RESULT_STATUS = f"{FC_P009_STATUS}_COMPLETE_NOT_ADMISSION"
LAUNCH_STATUS = "FC_P009_JOINT_CANDIDATE_LAUNCH_VERIFIED"
COMPLETION_STATUS = "FC_P009_JOINT_FORCE_ROW_CANDIDATE_VERIFIED_NOT_ADMISSION"
APPROVAL_STATUS = "FC_P009_JOINT_FORCE_ROW_CANDIDATE_EXECUTION_APPROVED"
IMPLEMENTATION = "dbe75763f8c480feefc54cb657f53848c24055e88ecc70a9a0aaaad544d81ecf"
MODEL = "dc41fc91d42476e052970b39fc66aed22fa72aa8b6f218a341a3abb095f42e31"
STATE = "4998e534d4b82b17393c217357ed18220fb8e739166a88147483bb9cc5fb771e"
RESULT = "131428d4a1b1d4920a879281bcb91c3ca9937f63a88a8ddef1d08f2a3d5da60d"
LAUNCH = "d4dd2ebe2540c9a2688996772a795acd00be8f3dd296b8b3d1a316494fe860d0"
COMPLETION = "26ac241f07e94329b86bb54dae4f0339fcc9cd7145bf71b9030393161543b9b5"
EXECUTION_APPROVAL = "f7140169283c8b91f3f4ddada4db2db3a7a54dd19caf845824a449937361e46d"
CONFIG = "07e55fd11df8030313338cef0344490c3453e515aae9c6b6122e997bad5085d9"
NORM = "f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1"
PRECISION = {
    "NVIDIA_TF32_OVERRIDE": None,
    "cuda_matmul_allow_tf32": True,
    "cudnn_allow_tf32": True,
    "float32_matmul_precision": "high",
}
PROTOCOL = [
    "validation10_H1_H10_H50_H100_stride25_batch4",
    "dynamic6_H1_H10_H50_H100_stride1_batch8",
    "force_window6",
    "unchanged_development_gate",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def validate_source_snapshot(candidate: Path, manifest: Path) -> None:
    root = (candidate / "source_snapshot").resolve()
    entries = {}
    for line in manifest.read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        relative = relative.lstrip("* ")
        path = (root / relative).resolve()
        if root not in path.parents or relative in entries or not path.is_file():
            raise ValueError("FC-P009 source snapshot manifest differs")
        entries[relative] = digest
        if sha256(path) != digest:
            raise ValueError("FC-P009 source snapshot member differs")
    if not entries or entries.get("scripts/build_fcp009_joint_force_row_candidate.py") != IMPLEMENTATION:
        raise ValueError("FC-P009 reviewed implementation is absent from snapshot")


def validate_candidate(repo: Path, candidate: Path, execution_approval_sha256: str) -> dict:
    if candidate.resolve() != (repo / "artifacts" / CANDIDATE_NAME).resolve():
        raise ValueError("FC-P009 candidate root differs")
    paths = {
        "launch": candidate / "launch_receipt.json",
        "completion": candidate / "completion_receipt.json",
        "result": candidate / "candidate_build/result.json",
        "model": candidate / "candidate_build/candidate/FNO.0.0.mdlus",
        "state": candidate / "candidate_build/candidate/checkpoint.0.0.pt",
        "approval": candidate / "launch_evidence/execution_approval.json",
        "source_manifest": candidate / "source_snapshot.sha256",
    }
    expected = {
        "launch": LAUNCH,
        "completion": COMPLETION,
        "result": RESULT,
        "model": MODEL,
        "state": STATE,
        "approval": EXECUTION_APPROVAL,
    }
    if any(not path.is_file() or sha256(path) != expected[name] for name, path in paths.items() if name in expected):
        raise ValueError("FC-P009 immutable candidate artifact differs")
    if execution_approval_sha256 != EXECUTION_APPROVAL:
        raise ValueError("FC-P009 requested execution approval differs")
    validate_source_snapshot(candidate, paths["source_manifest"])
    launch, completion, result, approval = (
        load(paths[name]) for name in ("launch", "completion", "result", "approval")
    )
    approval_required = {
        "status": APPROVAL_STATUS,
        "implementation_sha256": IMPLEMENTATION,
        "candidate_kind": FC_P009_STATUS,
        "candidate_build_authorized": True,
        "alpha": 0.0,
        "domain_mix": {"free_ar": 0.5, "matched_weight_h1": 0.5},
        "validation_or_frozen_accessed": False,
        "ppo_executed": False,
        "formal_evaluation_authorized": False,
    }
    if any(approval.get(key) != value for key, value in approval_required.items()):
        raise ValueError("FC-P009 execution approval content differs")
    if (
        launch.get("status") != LAUNCH_STATUS
        or launch.get("implementation_sha256") != IMPLEMENTATION
        or launch.get("execution_approval_sha256") != EXECUTION_APPROVAL
        or launch.get("runtime_image_id") != IMAGE_ID
        or launch.get("validation_accessed") is not False
        or launch.get("frozen_test_accessed") is not False
        or launch.get("ppo_executed") is not False
    ):
        raise ValueError("FC-P009 launch identity differs")
    completion_files = completion.get("sha256", {})
    exact_files = {
        "candidate_build/candidate/FNO.0.0.mdlus": MODEL,
        "candidate_build/candidate/checkpoint.0.0.pt": STATE,
        "candidate_build/result.json": RESULT,
        "launch_receipt.json": LAUNCH,
        "run.log": sha256(candidate / "run.log"),
        "source_snapshot.sha256": sha256(paths["source_manifest"]),
    }
    if (
        completion.get("status") != COMPLETION_STATUS
        or completion.get("runtime_image_id") != IMAGE_ID
        or completion.get("implementation_sha256") != IMPLEMENTATION
        or completion.get("execution_approval_sha256") != EXECUTION_APPROVAL
        or completion.get("candidate_checkpoint_epoch") != 0
        or completion.get("validation_accessed") is not False
        or completion.get("frozen_test_accessed") is not False
        or completion.get("ppo_executed") is not False
        or completion.get("formal_evaluation_authorized") is not False
        or completion_files != exact_files
    ):
        raise ValueError("FC-P009 completion receipt differs")
    if (
        result.get("status") != RESULT_STATUS
        or result.get("candidate_kind") != FC_P009_STATUS
        or result.get("candidate_model_sha256") != MODEL
        or result.get("candidate_state_sha256") != STATE
        or result.get("candidate_checkpoint_epoch") != 0
        or result.get("parent_checkpoint_epoch") != 2
        or result.get("calibration_generation") != 1
        or result.get("alpha") != 0.0
        or result.get("domain_mix") != {"free_ar": 0.5, "matched_weight_h1": 0.5}
        or result.get("precision_protocol") != PRECISION
        or result.get("state_rows_0_3_byte_identical") is not True
        or result.get("all_other_tensors_byte_identical") is not True
        or result.get("calibration_fit_performed") is not True
        or result.get("optimizer_training_performed") is not False
        or result.get("validation_accessed") is not False
        or result.get("frozen_test_accessed") is not False
        or result.get("ppo_executed") is not False
        or result.get("formal_evaluation_authorized") is not False
    ):
        raise ValueError("FC-P009 result identity differs")
    if result.get("input_sha256", {}).get("resolved_config") != CONFIG or result.get("input_sha256", {}).get("normalization") != NORM:
        raise ValueError("FC-P009 config/normalization identity differs")
    checkpoint = paths["model"].parent
    calibrated = validate_calibrated_epoch_zero(
        checkpoint,
        0,
        allow=True,
        expected_model_sha256=MODEL,
        expected_state_sha256=STATE,
        expected_kind=FC_P009_STATUS,
    )
    if calibrated != result.get("calibrated_epoch_zero_identity"):
        raise ValueError("FC-P009 stored epoch-zero identity differs")
    return {
        "status": LINEAGE_STATUS,
        "candidate_kind": CANDIDATE_KIND,
        "candidate_root": str(candidate.resolve()),
        "checkpoint_relative_directory": "candidate_build/candidate",
        "checkpoint_model_file": "FNO.0.0.mdlus",
        "checkpoint_state_file": "checkpoint.0.0.pt",
        "checkpoint_epoch": 0,
        "checkpoint_sha256": MODEL,
        "checkpoint_state_sha256": STATE,
        "official_image_id": IMAGE_ID,
        "precision_protocol": PRECISION,
        "resolved_config_sha256": CONFIG,
        "normalization_sha256": NORM,
        "parent_checkpoint_epoch": 2,
        "calibration_generation": 1,
        "training_performed": False,
        "calibration_fit_performed": True,
        "optimizer_training_performed": False,
        "validation_or_frozen_accessed": False,
        "ppo_auto_launch": False,
        "formal_protocol": PROTOCOL,
        "candidate_result_sha256": RESULT,
        "candidate_completion_receipt_sha256": COMPLETION,
        "execution_approval_sha256": EXECUTION_APPROVAL,
    }


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--execution-approval-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    candidate = args.candidate or args.repo / "artifacts" / CANDIDATE_NAME
    if args.execution_approval_sha256 is None:
        if args.candidate is not None or args.output is not None:
            raise ValueError("execution approval SHA required for persisted lineage")
        print(json.dumps({"status": "FC_P009_CANDIDATE_AUDITOR_DRY_RUN_READY", "candidate": str(candidate)}))
        return
    value = validate_candidate(args.repo, candidate, args.execution_approval_sha256)
    if args.output is not None:
        atomic_json(args.output, value)
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
