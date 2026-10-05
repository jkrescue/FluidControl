#!/usr/bin/env python3
"""Audit one completed FC-P011 positive-epoch training arm for formal evaluation."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import zipfile


IMAGE_ID = "sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
TRAINER_SHA = "9c761cfcb3d4f18dbe35aed1b3defe0614b29db867006055dae63d9fa94485a5"
TRAINING_APPROVAL_SHA = "1cd0dce1bb4ea4c2e580e147a0a440f2d528ea16e914479a629eb8185709338a"
CONFIG_SHA = "07e55fd11df8030313338cef0344490c3453e515aae9c6b6122e997bad5085d9"
NORMALIZATION_SHA = "f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1"
PARENT_MODEL_SHA = "dc41fc91d42476e052970b39fc66aed22fa72aa8b6f218a341a3abb095f42e31"
PARENT_STATE_SHA = "4998e534d4b82b17393c217357ed18220fb8e739166a88147483bb9cc5fb771e"
ORDER_SHA = "177ebd9523cde918eb0c1fb8026286dac7e95f3d228ae0e349757a2a9a288f9f"
RESULT_STATUS = "FC_P011_DECODER_SCOPE_TRAINING_COMPLETE_NOT_ADMISSION"
CHECKPOINT_STATUS = "FC_P011_DECODER_SCOPE_TERMINAL_CHECKPOINT"
COMPLETION_STATUS = "FC_P011_DECODER_SCOPE_TRAINING_EXECUTION_COMPLETE_NOT_ADMISSION"
FINAL_WEIGHT = "decoder_net.final_layer.linear.weight"
FINAL_BIAS = "decoder_net.final_layer.linear.bias"
HIDDEN_WEIGHT = "decoder_net.layers.1.linear.weight"
HIDDEN_BIAS = "decoder_net.layers.1.linear.bias"
PROTOCOL = [
    "validation10_H1_H10_H50_H100_stride25_batch4",
    "dynamic6_H1_H10_H50_H100_stride1_batch8",
    "force_window6",
    "unchanged_development_gate",
]
PRECISION = {
    "NVIDIA_TF32_OVERRIDE": None,
    "cuda_matmul_allow_tf32": True,
    "cudnn_allow_tf32": True,
    "float32_matmul_precision": "high",
}
PROFILES = {
    "head_only": {
        "root": "fcp011_head_only_training_20261005",
        "kind": "fcp011_head_only_epoch1",
        "lineage": "FC_P011_HEAD_ONLY_CANDIDATE_LINEAGE_PASS",
        "changed": [FINAL_BIAS, FINAL_WEIGHT],
    },
    "decoder_tail": {
        "root": "fcp011_decoder_tail_training_worker_20261005",
        "kind": "fcp011_decoder_tail_epoch1",
        "lineage": "FC_P011_DECODER_TAIL_CANDIDATE_LINEAGE_PASS",
        "changed": [HIDDEN_BIAS, HIDDEN_WEIGHT, FINAL_BIAS, FINAL_WEIGHT],
    },
}
INPUT_SHA = {
    "base_manifest": "5213c7bb07c824c6e601636c3cfa974b80ec7c051633b0c8368a045cea41ddd2",
    "train8_manifest": "a0bd0e3b79d4d43ace407292c31a595d552dbd6e1a5de1fd149b1ca446f53f35",
    "train16_manifest": "7c62dab94e317442cecf7ea3be547cf6e03c594a2c7ec2a1a695caba2445cf5b",
    "normalization": NORMALIZATION_SHA,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def all_finite(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(all_finite(item) for item in value)
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    return False


def load_model_state(path: Path) -> dict:
    import torch

    with zipfile.ZipFile(path) as archive:
        if set(archive.namelist()) != {"model.pt", "args.json", "metadata.json"}:
            raise ValueError("FC-P011 model archive members differ")
        value = torch.load(
            io.BytesIO(archive.read("model.pt")), map_location="cpu", weights_only=False
        )
    if not isinstance(value, dict) or not value:
        raise TypeError("FC-P011 model state differs")
    return value


def validate_tensor_confinement(parent_path: Path, candidate_path: Path, scope: str) -> list[str]:
    import torch

    parent, candidate = load_model_state(parent_path), load_model_state(candidate_path)
    if parent.keys() != candidate.keys():
        raise ValueError("FC-P011 model tensor keys differ")
    changed = []
    allowed_whole = {HIDDEN_WEIGHT, HIDDEN_BIAS} if scope == "decoder_tail" else set()
    for name in parent:
        left, right = parent[name], candidate[name]
        if torch.equal(left, right):
            continue
        changed.append(name)
        if name in allowed_whole:
            continue
        if name in {FINAL_WEIGHT, FINAL_BIAS}:
            if torch.equal(left[:6], right[:6]) and not torch.equal(left[6], right[6]):
                continue
        raise ValueError(f"FC-P011 tensor changed outside approved scope: {name}")
    expected = sorted(PROFILES[scope]["changed"])
    if sorted(changed) != expected:
        raise ValueError("FC-P011 exact changed tensor set differs")
    return sorted(changed)


def validate_checkpoint_metadata(path: Path, scope: str) -> None:
    import torch

    value = torch.load(path, map_location="cpu", weights_only=False)
    metadata = value.get("metadata") if isinstance(value, dict) else None
    expected = {
        "status": CHECKPOINT_STATUS,
        "scope": scope,
        "optimizer_steps": 1368,
        "parent_model_sha256": PARENT_MODEL_SHA,
        "parent_state_sha256": PARENT_STATE_SHA,
        "selection_performed": False,
        "validation_accessed": False,
        "frozen_test_accessed": False,
        "ppo_executed": False,
    }
    if metadata != expected:
        raise ValueError("FC-P011 positive-epoch checkpoint metadata differs")


def validate_candidate(repo: Path, candidate: Path, scope: str) -> dict:
    profile = PROFILES[scope]
    if candidate.resolve() != (repo / "artifacts" / profile["root"]).resolve():
        raise ValueError("FC-P011 candidate root differs")
    paths = {
        "launch": candidate / "launch_receipt.json",
        "completion": candidate / "completion_receipt.json",
        "result": candidate / "result.json",
        "model": candidate / "final/FNO.0.1.mdlus",
        "state": candidate / "final/checkpoint.0.1.pt",
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("FC-P011 terminal candidate is incomplete")
    launch, completion, result = (load(paths[name]) for name in ("launch", "completion", "result"))
    if (
        launch.get("status") != "FC_P011_TRAINING_LAUNCH_VERIFIED"
        or launch.get("scope") != scope
        or launch.get("trainer_sha256") != TRAINER_SHA
        or launch.get("approval_sha256") != TRAINING_APPROVAL_SHA
        or launch.get("config_sha256") != CONFIG_SHA
        or launch.get("parent_model_sha256") != PARENT_MODEL_SHA
        or launch.get("parent_state_sha256") != PARENT_STATE_SHA
        or launch.get("image_id") != IMAGE_ID
        or launch.get("expected_updates") != 1368
        or launch.get("resource_probe") is not False
        or launch.get("validation_or_frozen_mounted") is not False
    ):
        raise ValueError("FC-P011 launch identity differs")
    actual_hashes = {
        "launch_receipt.json": sha256(paths["launch"]),
        "result.json": sha256(paths["result"]),
        "final/FNO.0.1.mdlus": sha256(paths["model"]),
        "final/checkpoint.0.1.pt": sha256(paths["state"]),
    }
    receipt_hashes = completion.get("sha256")
    if not isinstance(receipt_hashes, dict) or any(
        receipt_hashes.get(name) != digest for name, digest in actual_hashes.items()
    ):
        raise ValueError("FC-P011 completion hashes differ")
    if (
        completion.get("status") != COMPLETION_STATUS
        or completion.get("scope") != scope
        or completion.get("training_exit_code") != 0
        or completion.get("trainer_sha256") != TRAINER_SHA
        or completion.get("approval_sha256") != TRAINING_APPROVAL_SHA
        or completion.get("config_sha256") != CONFIG_SHA
        or completion.get("parent_model_sha256") != PARENT_MODEL_SHA
        or completion.get("parent_state_sha256") != PARENT_STATE_SHA
        or completion.get("optimizer_steps") != 1368
        or completion.get("train_order_sha256") != ORDER_SHA
        or completion.get("validation_accessed") is not False
        or completion.get("frozen_test_accessed") is not False
        or completion.get("ppo_executed") is not False
    ):
        raise ValueError("FC-P011 completion identity differs")
    expected_changed = sorted(profile["changed"])
    expected_scope = {
        "optimizer_parameter_names": sorted(
            [FINAL_BIAS, FINAL_WEIGHT]
            + ([HIDDEN_BIAS, HIDDEN_WEIGHT] if scope == "decoder_tail" else [])
        ),
        "optimizer_tensor_elements": 17415 if scope == "decoder_tail" else 903,
        "effective_trainable_coefficients": 16641 if scope == "decoder_tail" else 129,
    }
    expected_loss = {
        "field_weight": 1.0,
        "force_weight": 0.2,
        "equal_four_share": 0.5,
        "rear_cl_share": 0.5,
        "effective_force_channel_weights": [0.125, 0.125, 0.125, 0.625],
        "rollout_steps": 100,
        "step_weights": "uniform",
        "teacher_forcing": 0.0,
    }
    parent_model = repo / "artifacts/fcp009_joint_force_row_candidate_20261005/candidate_build/candidate/FNO.0.0.mdlus"
    if not parent_model.is_file() or sha256(parent_model) != PARENT_MODEL_SHA:
        raise ValueError("FC-P011 parent model differs")
    if validate_tensor_confinement(parent_model, paths["model"], scope) != expected_changed:
        raise ValueError("FC-P011 tensor confinement differs")
    validate_checkpoint_metadata(paths["state"], scope)
    if (
        result.get("status") != RESULT_STATUS
        or result.get("scope") != scope
        or result.get("parent_model_sha256") != PARENT_MODEL_SHA
        or result.get("parent_state_sha256") != PARENT_STATE_SHA
        or result.get("train_order_sha256") != ORDER_SHA
        or result.get("expected_train_order_sha256") != ORDER_SHA
        or result.get("family_window_counts") != {"base20": 720, "train8": 408, "train16": 240}
        or result.get("optimizer_steps") != 1368
        or result.get("batch_size") != 1
        or result.get("trainable_scope") != expected_scope
        or result.get("loss_contract") != expected_loss
        or result.get("input_sha256") != INPUT_SHA
        or result.get("diagnostic_steps_predeclared") != [0, 32, 128, 512, 1368]
        or sorted(result.get("changed_tensor_names", [])) != expected_changed
        or result.get("selection_performed") is not False
        or result.get("validation_accessed") is not False
        or result.get("frozen_test_accessed") is not False
        or result.get("ppo_executed") is not False
        or result.get("terminal_model_sha256") != actual_hashes["final/FNO.0.1.mdlus"]
        or result.get("terminal_state_sha256") != actual_hashes["final/checkpoint.0.1.pt"]
        or result.get("fresh_reload_tensor_sha256") != result.get("model_state_after_sha256")
        or result.get("config_sha256") != CONFIG_SHA
        or result.get("implementation_sha256") != TRAINER_SHA
    ):
        raise ValueError("FC-P011 terminal result identity differs")
    records = result.get("training_records")
    identities = result.get("train_identities")
    diagnostics = result.get("train_only_diagnostics")
    if (
        not isinstance(records, list)
        or len(records) != 1368
        or not isinstance(identities, list)
        or len(identities) != 1368
        or not isinstance(diagnostics, dict)
        or set(diagnostics) != {"0", "32", "128", "512", "1368"}
        or not all_finite(records)
        or not all_finite(diagnostics)
    ):
        raise ValueError("FC-P011 training evidence differs or is non-finite")
    for index, (record, identity) in enumerate(zip(records, identities, strict=True), 1):
        if record.get("step") != index or record.get("identity") != identity:
            raise ValueError("FC-P011 training record sequence differs")
        if identity.get("split") != "train" or identity.get("rollout_steps") != 100:
            raise ValueError("FC-P011 training identity escaped train H100")
        hidden = record.get("decoder_hidden_gradient_norm")
        if scope == "head_only" and hidden != 0.0:
            raise ValueError("FC-P011 head-only arm has a hidden-layer gradient")
        if scope == "decoder_tail" and (not isinstance(hidden, (int, float)) or hidden < 0):
            raise ValueError("FC-P011 decoder-tail hidden gradient differs")
    if len({(row["case"], row["start"], row["dataset_index"]) for row in identities}) != 1368:
        raise ValueError("FC-P011 training identities are not unique")
    observed_counts = {
        "base20": sum(row["dataset_index"] == 0 for row in identities),
        "train8": sum(row["dataset_index"] == 1 for row in identities),
        "train16": sum(row["dataset_index"] == 2 for row in identities),
    }
    if observed_counts != {"base20": 720, "train8": 408, "train16": 240}:
        raise ValueError("FC-P011 observed family counts differ")
    return {
        "status": profile["lineage"],
        "candidate_kind": profile["kind"],
        "training_scope": scope,
        "candidate_root": str(candidate.resolve()),
        "checkpoint_relative_directory": "final",
        "checkpoint_model_file": "FNO.0.1.mdlus",
        "checkpoint_state_file": "checkpoint.0.1.pt",
        "checkpoint_epoch": 1,
        "checkpoint_sha256": actual_hashes["final/FNO.0.1.mdlus"],
        "checkpoint_state_sha256": actual_hashes["final/checkpoint.0.1.pt"],
        "official_image_id": IMAGE_ID,
        "precision_protocol": PRECISION,
        "resolved_config_sha256": CONFIG_SHA,
        "normalization_sha256": NORMALIZATION_SHA,
        "parent_checkpoint_epoch": 0,
        "parent_model_sha256": PARENT_MODEL_SHA,
        "parent_state_sha256": PARENT_STATE_SHA,
        "training_performed": True,
        "calibration_fit_performed": False,
        "optimizer_training_performed": True,
        "optimizer_steps": 1368,
        "train_order_sha256": ORDER_SHA,
        "allowed_changed_tensor_names": expected_changed,
        "validation_or_frozen_accessed": False,
        "ppo_auto_launch": False,
        "formal_protocol": PROTOCOL,
        "candidate_result_sha256": actual_hashes["result.json"],
        "candidate_completion_receipt_sha256": sha256(paths["completion"]),
        "training_approval_sha256": TRAINING_APPROVAL_SHA,
    }


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--scope", choices=tuple(PROFILES), required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--execution-approval-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    candidate = args.candidate or args.repo / "artifacts" / PROFILES[args.scope]["root"]
    if (
        args.execution_approval_sha256 is not None
        and args.execution_approval_sha256 != TRAINING_APPROVAL_SHA
    ):
        raise ValueError("FC-P011 training execution approval differs")
    if not (candidate / "completion_receipt.json").is_file():
        if args.candidate is not None or args.output is not None:
            raise FileNotFoundError("FC-P011 terminal completion receipt is absent")
        print(json.dumps({"status": "FC_P011_CANDIDATE_AUDITOR_DRY_RUN_READY", "scope": args.scope}))
        return
    value = validate_candidate(args.repo, candidate, args.scope)
    if args.output is not None:
        atomic_json(args.output, value)
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
