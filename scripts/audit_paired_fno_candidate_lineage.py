#!/usr/bin/env python3
"""Fail-closed lineage audit for FC-P001 paired-stat FNO candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

import yaml

IMAGE_ID = "sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
DEV30 = "5213c7bb07c824c6e601636c3cfa974b80ec7c051633b0c8368a045cea41ddd2"
TRAIN8 = "a0bd0e3b79d4d43ace407292c31a595d552dbd6e1a5de1fd149b1ca446f53f35"
TRAIN16 = "7c62dab94e317442cecf7ea3be547cf6e03c594a2c7ec2a1a695caba2445cf5b"
PAIR = "15bfa7a47e3195afad59884f96fd4e305c8ba0001dd6eb043be2412b2b9ce2b7"
PAIR_PROBE = "8453a2836a14fb271ad9fdfe3205915d15b6414ade916434a5713e068a185ec4"
NORM = "f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1"
PARENT_MODEL = "8466bd47f2de188f5e741197832ec3bee1223f72f54d8956e584c586a2774240"
PARENT_STATE = "1e5d4c055812d8f92bc55f58708e839f7cc776071849544f6a26d3d62053cbd0"
CONFIG_SHA = "cec8941fbe47ef1830b99e38fb2f2ca16b41dcd9405d58248763c66ece55fd55"
COMMIT = "b6aada926942161da4430d52664c92db1a2269c7"
TREE = "6e46d152d2d96017707ac2b3d874debbee46eaf4"
ROOTS = {
    "tandem_fno_paired_stats_lambda0_20261004": (
        "lambda0", 0.0, "d6738ab0c44317e20493cf02412d72b4098dfda9f994f8d7ac1ca84a40596d4e",
        "3a222827c9f0b3d39d72cec035c20c7d35411ecfc119415109c9528ea7876de7",
    ),
    "tandem_fno_paired_stats_lambda10_20261004": (
        "lambda10", 10.0, "66a88b3c80751eaa92722ce7f7622eb1f815d7d7d4260798217934c1997b76d8",
        "b7bd654933c31ca9066cb9c37f2f9071c56b75f09e495c224b996acb1079dc55",
    ),
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


def archive_payload(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("model archive has duplicate members")
        if set(names) != {"model.pt", "args.json", "metadata.json"}:
            raise ValueError("model archive member set differs")
        names = sorted(names)
        return {name: hashlib.sha256(archive.read(name)).hexdigest() for name in names}


def build(repo: Path, candidate: Path) -> dict:
    repo, candidate = repo.resolve(), candidate.resolve()
    if candidate.name not in ROOTS or (repo / "artifacts").resolve() not in candidate.parents:
        raise ValueError("candidate is not an approved FC-P001 root")
    branch, weight, resolved_sha, expected_model_sha = ROOTS[candidate.name]
    launch, completion = load(candidate / "launch_receipt.json"), load(candidate / "completion_receipt.json")
    config = yaml.safe_load((candidate / "resolved_config.yaml").read_text())
    training, data = config["training"], config["data"]
    if (
        launch.get("branch") != branch
        or float(launch.get("paired_stat_loss_weight", math.nan)) != weight
        or launch.get("single_factor") != "paired_stat_loss_weight_0_vs_10"
        or launch.get("git_commit") != COMMIT
        or launch.get("git_tree") != TREE
        or launch.get("pair_manifest_sha256") != PAIR
        or launch.get("pair_probe_sha256") != PAIR_PROBE
        or launch.get("parent_model_sha256") != PARENT_MODEL
        or launch.get("parent_state_sha256") != PARENT_STATE
        or launch.get("ppo_auto_launch") is not False
    ):
        raise ValueError("launch receipt differs from approved paired contract")
    actual_tree = subprocess.run(
        ["git", "rev-parse", f"{COMMIT}^{{tree}}"], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if actual_tree != TREE:
        raise ValueError("launch commit tree differs")
    if (
        sha256(candidate / "resolved_config.yaml") != resolved_sha
        or completion["sha256"].get("resolved_config.yaml") != resolved_sha
        or completion.get("status") != "PAIRED_STATS_CONTROLLED_TRAINING_COMPLETE"
        or completion.get("epochs") != 2
        or completion.get("ppo_auto_launched") is not False
    ):
        raise ValueError("completion receipt differs")
    for name, digest in completion["sha256"].items():
        if sha256(candidate / name) != digest:
            raise ValueError(f"completion artifact differs: {name}")
    expected_training = {
        "epochs": 2, "batch_size": 1, "rollout_steps": 100,
        "validation_rollout_steps": 100, "seed": 20261003,
        "train_stride": 20, "additional_train_stride": 2,
        "initial_checkpoint": "/workspace/parent", "paired_batch_size": 1,
        "paired_batches_per_epoch": 16, "max_paired_eval_batches": 16,
        "paired_stat_horizons": [20, 50, 100],
    }
    if any(training.get(key) != value for key, value in expected_training.items()):
        raise ValueError("paired batch1/H100 training contract differs")
    if (
        float(training.get("paired_stat_loss_weight", math.nan)) != weight
        or training.get("teacher_forcing_start") != 0.0
        or training.get("teacher_forcing_end") != 0.0
        or data.get("root") != "/workspace/base"
        or data.get("additional_train_roots") != ["/workspace/train8", "/workspace/train16"]
        or data.get("paired_manifest") != "/workspace/pair_manifest.json"
    ):
        raise ValueError("paired resolved config differs")
    sources = load(candidate / "training_data_sources.json")
    additional = sources.get("additional_sources")
    if sources.get("base_windows") != 720 or sources.get("total_windows") != 1368 or len(additional or []) != 2:
        raise ValueError("training source window counts differ")
    expected_sources = [("/workspace/train8", 408, TRAIN8), ("/workspace/train16", 240, TRAIN16)]
    for row, (root, windows, manifest) in zip(additional, expected_sources, strict=True):
        if row.get("root") != root or row.get("windows") != windows or row.get("manifest_sha256") != manifest or row.get("normalization_sha256") != NORM:
            raise ValueError("training source identity differs")
    manifests = {
        "dev30": repo / "data/curated/tandem_cylinders_matched_start_full40_dev30_v1/manifest.json",
        "train8": repo / "data/curated/tandem_cylinders_dynamic_train8_v1/manifest.json",
        "train16": repo / "data/curated/tandem_cylinders_directppo_train16_v1/manifest.json",
        "paired": repo / "artifacts/train20_paired_stat_datapipe_v1/manifest.json",
    }
    expected_manifest = {"dev30": DEV30, "train8": TRAIN8, "train16": TRAIN16, "paired": PAIR}
    if any(sha256(path) != expected_manifest[name] for name, path in manifests.items()):
        raise ValueError("data manifest differs")
    normalization_paths = [
        repo / "data/curated/tandem_cylinders_matched_start_full40_dev30_v1/normalization.json",
        repo / "data/curated/tandem_cylinders_dynamic_train8_v1/normalization.json",
        repo / "data/curated/tandem_cylinders_directppo_train16_v1/normalization.json",
    ]
    if any(sha256(path) != NORM for path in normalization_paths):
        raise ValueError("actual training normalization differs")
    if sha256(repo / "artifacts/train20_paired_stat_datapipe_v1/cpu_probe_v3.json") != PAIR_PROBE:
        raise ValueError("paired DataPipe probe differs")
    snapshot = candidate / "source_snapshot"
    recorded = launch.get("source_snapshot_sha256")
    actual_files = {str(path.relative_to(snapshot)): sha256(path) for path in snapshot.rglob("*") if path.is_file()}
    if not isinstance(recorded, dict) or actual_files != recorded or recorded.get("conf/tandem_fno_paired_stats_h100.yaml") != CONFIG_SHA:
        raise ValueError("launch source snapshot differs")
    history = json.loads((candidate / "training_history.json").read_text())
    if not isinstance(history, list) or [row.get("epoch") for row in history] != [1, 2]:
        raise ValueError("training history differs")
    if not all(all(math.isfinite(float(v)) for v in row.values() if isinstance(v, (int, float))) for row in history):
        raise ValueError("training history is nonfinite")
    selected = min(history, key=lambda row: row["selection_score"])
    if selected["epoch"] != 2:
        raise ValueError("best epoch differs from unique selection minimum")
    model = candidate / "best/FNO.0.2.mdlus"
    state = candidate / "best/checkpoint.0.2.pt"
    checkpoint_model = candidate / "checkpoints/FNO.0.2.mdlus"
    checkpoint_state = candidate / "checkpoints/checkpoint.0.2.pt"
    if sha256(model) != expected_model_sha:
        raise ValueError("branch best model SHA differs")
    if archive_payload(model) != archive_payload(checkpoint_model) or sha256(state) != sha256(checkpoint_state):
        raise ValueError("best files differ from checkpoint generation payload")
    parent_model = next((candidate / "immutable_parent").glob("FNO.*.mdlus"))
    parent_state = next((candidate / "immutable_parent").glob("checkpoint.*.pt"))
    if sha256(parent_model) != PARENT_MODEL or sha256(parent_state) != PARENT_STATE:
        raise ValueError("immutable parent differs")
    return {
        "status": "PAIRED_STATS_CANDIDATE_LINEAGE_PASS",
        "candidate_kind": f"paired_stats_{branch}", "branch": branch,
        "paired_stat_loss_weight": weight, "candidate_root": str(candidate.relative_to(repo)),
        "checkpoint_epoch": 2, "checkpoint_sha256": sha256(model),
        "checkpoint_state_sha256": sha256(state),
        "checkpoint_generation_payload_sha256": archive_payload(model),
        "resolved_config_sha256": sha256(candidate / "resolved_config.yaml"),
        "launch_receipt_sha256": sha256(candidate / "launch_receipt.json"),
        "completion_receipt_sha256": sha256(candidate / "completion_receipt.json"),
        "training_history_sha256": sha256(candidate / "training_history.json"),
        "source_snapshot_commit": COMMIT, "source_snapshot_tree": TREE,
        "data_lineage": {**expected_manifest, "normalization": NORM},
        "training_contract": {**expected_training, "paired_stat_loss_weight": weight},
        "formal_protocol": ["validation10_H1_H10_H50_H100", "dynamic6_H1_H10_H50_H100", "force_window6", "development_gate"],
        "training_performed": False, "ppo_auto_launch": False,
        "frozen_test_opened_or_enumerated": False,
    }


def write_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name); json.dump(payload, stream, indent=2, sort_keys=True); stream.write("\n")
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = build(args.repo, args.candidate_root)
    if args.output:
        write_exclusive(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
