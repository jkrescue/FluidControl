#!/usr/bin/env python3
"""Audit the FC-P003C training contract and a completed candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import yaml

IMAGE_ID = "sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
APPROVAL = "e0733bcceb131018089584f1daed5193e3ac6aeff6d933c8ef32ca3b0a8cdea4"
PARENT_MODEL = "8466bd47f2de188f5e741197832ec3bee1223f72f54d8956e584c586a2774240"
PARENT_STATE = "1e5d4c055812d8f92bc55f58708e839f7cc776071849544f6a26d3d62053cbd0"
NORM = "f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1"
DEV30 = "5213c7bb07c824c6e601636c3cfa974b80ec7c051633b0c8368a045cea41ddd2"
TRAIN8 = "a0bd0e3b79d4d43ace407292c31a595d552dbd6e1a5de1fd149b1ca446f53f35"
TRAIN16 = "7c62dab94e317442cecf7ea3be547cf6e03c594a2c7ec2a1a695caba2445cf5b"
DYNAMIC_PAIR = "b756c6d777b68fe9dc6c5a81e40733e24d2c248b315723ddb40cc3273902d28c"
PAIR_PROBE = "e549121553e4e1fa0fffe5c0f3080dc1f03db615342530a7a7fcb0a0e45ac7aa"
SAMPLING = "da078a1c43f03bf86ee71a5010632c05a4868b169dc8f8c7c341c78134873242"
BASELINE_ORDER = "fab043a70652475e0b03aa869eac3445eaec1ec6c66a74cf976a0342a3dbca88"
MIXED_COMPLETION = "f95a6f554e20964125628a2d33a88709827daf298e935de653471aaa1708b1cd"
MIXED_RESULT = "614264a6626d5411df23d6994c025796c043d8c29f1e2aeb5c5b9f948a0a30ea"
CANDIDATE = "tandem_fno_true_state_paired_step_lambda10_20261005"
EXPECTED_INDICES = [index * 1367 // 15 for index in range(16)]


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


def archive_payload(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != {"model.pt", "args.json", "metadata.json"}:
            raise ValueError("model archive member set differs")
        return {name: hashlib.sha256(archive.read(name)).hexdigest() for name in sorted(names)}


def write_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_static(repo: Path) -> dict:
    repo = repo.resolve()
    files = {
        "approval": (repo / "docs/FC-P003C_APPROVAL.md", APPROVAL),
        "dev30": (repo / "data/curated/tandem_cylinders_matched_start_full40_dev30_v1/manifest.json", DEV30),
        "train8": (repo / "data/curated/tandem_cylinders_dynamic_train8_v1/manifest.json", TRAIN8),
        "train16": (repo / "data/curated/tandem_cylinders_directppo_train16_v1/manifest.json", TRAIN16),
        "dynamic_pair": (repo / "artifacts/fc_p003_dynamic8_pair_candidate_20261005/manifest.json", DYNAMIC_PAIR),
        "pair_probe": (repo / "artifacts/fc_p003_dynamic8_pair_candidate_20261005/cpu_probe.json", PAIR_PROBE),
        "sampling": (repo / "artifacts/fc_p003b_real_sampling_contract_20261005/result.json", SAMPLING),
        "baseline_order": (repo / "artifacts/fc_p003_dataloader_order_20261005/result.json", BASELINE_ORDER),
        "parent_model": (repo / "artifacts/tandem_fno_control_train16_h100_20261004/best/FNO.0.2.mdlus", PARENT_MODEL),
        "parent_state": (repo / "artifacts/tandem_fno_control_train16_h100_20261004/best/checkpoint.0.2.pt", PARENT_STATE),
    }
    for name, (path, expected) in files.items():
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"{name} identity differs")
    for root in (
        repo / "data/curated/tandem_cylinders_matched_start_full40_dev30_v1",
        repo / "data/curated/tandem_cylinders_dynamic_train8_v1",
        repo / "data/curated/tandem_cylinders_directppo_train16_v1",
    ):
        if sha256(root / "normalization.json") != NORM:
            raise ValueError(f"normalization differs: {root}")
    sampling = load(files["sampling"][0])
    if sampling.get("status") != "FC_P003B_REAL_OFFICIAL_DATALOADER_SAMPLING_PASS":
        raise ValueError("real sampling contract is not PASS")
    if sampling.get("fc_p003_order_receipt_sha256") != BASELINE_ORDER:
        raise ValueError("sampling contract does not bind FC-P003 order")
    return {"status": "FC_P003C_STATIC_PREFLIGHT_PASS", "sha256": {name: expected for name, (_, expected) in files.items()}}


def validate_resolved(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    training, data, model = config["training"], config["data"], config["model"]
    expected_training = {
        "epochs": 2, "batch_size": 1, "rollout_steps": 100,
        "validation_rollout_steps": 100, "seed": 20261003,
        "train_stride": 20, "additional_train_stride": 2,
        "initial_checkpoint": "/workspace/parent", "paired_batch_size": 1,
        "paired_batches_per_epoch": 16, "max_paired_eval_batches": 8,
        "paired_batch_schedule": "interleaved", "expected_regular_batches": 1368,
        "paired_dataset_repetitions": 2, "paired_objective_kind": "true_state_step_force",
        "paired_step_chunk_size": 10, "force_channel_weights": [1.0, 1.0, 4.0, 1.0],
    }
    if any(training.get(key) != value for key, value in expected_training.items()):
        raise ValueError("resolved FC-P003C training contract differs")
    if any(float(training.get(key, math.nan)) != value for key, value in {
        "paired_stat_loss_weight": 10.0, "force_loss_weight": 0.2,
        "teacher_forcing_start": 0.0, "teacher_forcing_end": 0.0,
        "learning_rate": 1e-5, "gradient_clip_norm": 1.0,
    }.items()):
        raise ValueError("resolved numeric training contract differs")
    expected_data = {
        "root": "/workspace/base", "additional_train_roots": ["/workspace/train8", "/workspace/train16"],
        "paired_dataset_kind": "dynamic8", "paired_action_root": "/workspace/train8",
        "paired_zero_root": "/workspace/base", "paired_manifest": "/workspace/dynamic_pair_manifest.json",
    }
    if any(data.get(key) != value for key, value in expected_data.items()):
        raise ValueError("resolved data contract differs")
    expected_model = {"in_channels": 6, "out_channels": 7, "latent_channels": 48, "num_fno_layers": 5, "num_fno_modes": [32, 32], "padding": 8, "coord_features": True}
    if any(model.get(key) != value for key, value in expected_model.items()):
        raise ValueError("resolved FNO architecture differs")
    return {"status": "FC_P003C_RESOLVED_CONFIG_PASS", "resolved_config_sha256": sha256(config_path), "training": expected_training}


def _finite(value: object) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def validate_execution_evidence(candidate: Path, launch: dict) -> None:
    evidence = candidate / "launch_evidence"
    mixed_completion = evidence / "mixed_probe_completion_receipt.json"
    mixed_result = evidence / "mixed_probe_result.json"
    full_approval = evidence / "full_training_approval.json"
    if sha256(mixed_completion) != MIXED_COMPLETION or sha256(mixed_result) != MIXED_RESULT:
        raise ValueError("candidate mixed-probe evidence differs")
    if sha256(full_approval) != launch.get("full_execution_approval_sha256"):
        raise ValueError("candidate full-approval evidence differs")
    approval_payload = load(full_approval)
    if approval_payload.get("status") != "FC_P003C_FULL_TRAINING_APPROVED" or approval_payload.get("gpu_execution_authorized") is not True:
        raise ValueError("candidate full approval contract differs")
    mixed_payload, result_payload = load(mixed_completion), load(mixed_result)
    if mixed_payload.get("status") != "FC_P003C_MIXED_LOSS_TECHNICAL_PROBE_COMPLETE" or mixed_payload.get("result_sha256") != MIXED_RESULT:
        raise ValueError("candidate mixed-probe completion differs")
    if result_payload.get("status") != "FC_P003C_MIXED_LOSS_TECHNICAL_PROBE_PASS" or result_payload.get("optimizer_steps") != 1 or result_payload.get("candidate_saved") is not False or result_payload.get("validation_or_frozen_accessed") is not False:
        raise ValueError("candidate mixed-probe result differs")


def validate_immutable_chain(receipt_path: Path, chain_root: Path, commit: str) -> dict:
    receipt, chain_root = load(receipt_path), chain_root.resolve()
    actual = {
        str(path.relative_to(chain_root.parent)): sha256(path)
        for path in sorted(chain_root.rglob("*"))
        if path.is_file()
    }
    if receipt.get("status") != "FC_P003C_IMMUTABLE_POSTEVAL_CHAIN_STAGED" or receipt.get("git_commit") != commit or receipt.get("sha256") != actual:
        raise ValueError("immutable posteval chain differs")
    return receipt


def validate_candidate(repo: Path, candidate: Path) -> dict:
    repo, candidate = repo.resolve(), candidate.resolve()
    if candidate != (repo / "artifacts" / CANDIDATE).resolve():
        raise ValueError("candidate root differs")
    static = validate_static(repo)
    resolved = validate_resolved(candidate / "resolved_config.yaml")
    launch, completion = load(candidate / "launch_receipt.json"), load(candidate / "completion_receipt.json")
    if launch.get("status") != "FC_P003C_LAUNCH_STAGED" or launch.get("mode") != "--execute":
        raise ValueError("launch receipt status differs")
    for field, expected in {
        "approval_sha256": APPROVAL, "parent_model_sha256": PARENT_MODEL,
        "parent_state_sha256": PARENT_STATE, "dynamic_pair_manifest_sha256": DYNAMIC_PAIR,
        "real_sampling_receipt_sha256": SAMPLING, "baseline_order_receipt_sha256": BASELINE_ORDER,
        "official_image_id": IMAGE_ID, "resolved_config_sha256": resolved["resolved_config_sha256"],
        "mixed_probe_receipt_sha256": MIXED_COMPLETION,
        "mixed_probe_result_sha256": MIXED_RESULT,
    }.items():
        if launch.get(field) != expected:
            raise ValueError(f"launch receipt {field} differs")
    if launch.get("single_factor") != "paired_objective_paired_statistics_to_true_state_step_force" or launch.get("ppo_auto_launch") is not False:
        raise ValueError("launch single-factor contract differs")
    validate_execution_evidence(candidate, launch)
    snapshot = candidate / "source_snapshot"
    actual_snapshot = {str(path.relative_to(snapshot)): sha256(path) for path in sorted(snapshot.rglob("*")) if path.is_file()}
    if launch.get("source_snapshot_sha256") != actual_snapshot:
        raise ValueError("source snapshot differs")
    if completion.get("status") != "FC_P003C_TRAINING_COMPLETE" or completion.get("epochs") != 2:
        raise ValueError("completion receipt differs")
    for name, digest in completion.get("sha256", {}).items():
        path = candidate / name
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"completion artifact differs: {name}")
    history = json.loads((candidate / "training_history.json").read_text(encoding="utf-8"))
    if not isinstance(history, list) or [row.get("epoch") for row in history] != [1, 2] or not _finite(history):
        raise ValueError("training history differs")
    manifest = load(repo / "artifacts/fc_p003_dynamic8_pair_candidate_20261005/manifest.json")
    expected_ids = {f"{row['phase']}:{row['profile']}" for row in manifest["pairs"]}
    for row in history:
        if row.get("paired_objective_kind") != "true_state_step_force" or row.get("paired_batch_indices") != EXPECTED_INDICES:
            raise ValueError("actual paired objective/schedule differs")
        identities = row.get("paired_identities")
        if len(identities or []) != 16 or Counter(identities) != Counter({pair_id: 2 for pair_id in expected_ids}):
            raise ValueError("actual dynamic pair identity trace differs")
        if row.get("train_true_state_paired_optimizer_steps") != 16:
            raise ValueError("paired optimizer step count differs")
        for key in ("train_true_state_paired_step_force_per_channel_mse", "train_true_state_paired_step_force_per_channel_weighted_contribution"):
            values = row.get(key)
            if not isinstance(values, list) or len(values) != 4 or not _finite(values):
                raise ValueError(f"per-channel training evidence differs: {key}")
    selected = min(history, key=lambda row: row["selection_score"])
    models, states = list((candidate / "best").glob("FNO.0.*.mdlus")), list((candidate / "best").glob("checkpoint.0.*.pt"))
    if len(models) != 1 or len(states) != 1:
        raise ValueError("best artifact count differs")
    model, state = models[0], states[0]
    epoch = int(model.name.split(".")[2])
    if epoch != selected["epoch"] or state.name.split(".")[2] != str(epoch):
        raise ValueError("best epoch differs from selection minimum")
    checkpoint_model, checkpoint_state = candidate / f"checkpoints/FNO.0.{epoch}.mdlus", candidate / f"checkpoints/checkpoint.0.{epoch}.pt"
    if archive_payload(model) != archive_payload(checkpoint_model) or sha256(state) != sha256(checkpoint_state):
        raise ValueError("best/checkpoint generation differs")
    required = [candidate / name for name in ("launch_receipt.json", "resolved_config.yaml", "runtime_metadata.json", "training_data_sources.json", "training_history.json", "train.log")]
    required += [model, state, checkpoint_model, checkpoint_state]
    if set(completion.get("sha256", {})) != {str(path.relative_to(candidate)) for path in required}:
        raise ValueError("completion artifact set differs")
    return {
        "status": "FC_P003C_CANDIDATE_LINEAGE_PASS", "candidate_kind": "true_state_paired_step_lambda10",
        "candidate_root": str(candidate.relative_to(repo)), "checkpoint_epoch": epoch,
        "checkpoint_sha256": sha256(model), "checkpoint_state_sha256": sha256(state),
        "checkpoint_generation_payload_sha256": archive_payload(model),
        "resolved_config_sha256": resolved["resolved_config_sha256"],
        "launch_receipt_sha256": sha256(candidate / "launch_receipt.json"),
        "completion_receipt_sha256": sha256(candidate / "completion_receipt.json"),
        "training_history_sha256": sha256(candidate / "training_history.json"),
        "data_lineage": static["sha256"], "training_contract": resolved["training"],
        "formal_protocol": ["validation10_H1_H10_H50_H100_stride25_batch4", "dynamic6_H1_H10_H50_H100_stride1_batch8", "force_window6", "unchanged_development_gate"],
        "training_performed": True, "frozen_test_opened_or_enumerated": False, "ppo_auto_launch": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--chain-receipt", type=Path)
    parser.add_argument("--chain-root", type=Path)
    parser.add_argument("--reviewed-commit")
    args = parser.parse_args()
    if args.chain_receipt:
        if not args.chain_root or not args.reviewed_commit:
            parser.error("--chain-root and --reviewed-commit are required with --chain-receipt")
        result = validate_immutable_chain(args.chain_receipt, args.chain_root, args.reviewed_commit)
    elif args.candidate:
        result = validate_candidate(args.repo, args.candidate)
    elif args.resolved_config:
        result = {**validate_static(args.repo), "resolved": validate_resolved(args.resolved_config)}
    else:
        result = validate_static(args.repo)
    if args.output:
        write_exclusive(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
