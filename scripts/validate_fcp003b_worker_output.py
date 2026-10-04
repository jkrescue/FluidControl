#!/usr/bin/env python3
"""Fail-closed validation for FC-P003B Worker source and training artifacts."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import tempfile
import zipfile
from pathlib import Path

import yaml

EXPECTED_IDENTITIES = {
    f"b{phase:02d}:{profile}"
    for phase in (0, 2, 4, 6)
    for profile in ("multisine", "prbs")
}
EXPECTED_POSITIONS = [i * 1367 // 15 for i in range(16)]
EXPECTED_MANIFESTS = {
    "/workspace/train8": "a0bd0e3b79d4d43ace407292c31a595d552dbd6e1a5de1fd149b1ca446f53f35",
    "/workspace/train16": "7c62dab94e317442cecf7ea3be547cf6e03c594a2c7ec2a1a695caba2445cf5b",
}
NORMALIZATION_SHA = "f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_map(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def require_finite(value, label: str = "root") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            require_finite(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            require_finite(item, f"{label}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"nonfinite numeric value: {label}")


def archive_payload(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate model members: {path}")
        if set(names) != {"model.pt", "args.json", "metadata.json"}:
            raise ValueError(f"model member set differs: {path}")
        return {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in sorted(names)
        }


def validate_source(
    source: Path,
    receipt_path: Path,
    expected_commit: str,
    required_hashes_path: Path,
) -> dict:
    receipt = load_json(receipt_path)
    required = load_json(required_hashes_path)
    actual = file_map(source)
    if receipt.get("status") != "FC_P003B_IMMUTABLE_SOURCE_STAGED_V2":
        raise ValueError("source receipt status differs")
    if receipt.get("git_commit") != expected_commit:
        raise ValueError("source commit differs")
    if receipt.get("source_files") != actual:
        raise ValueError("source receipt does not cover exact snapshot")
    if not isinstance(required, dict) or not required:
        raise ValueError("required source hash map missing")
    for relative, expected in required.items():
        if actual.get(relative) != expected:
            raise ValueError(f"required source dependency differs: {relative}")
    return {"file_count": len(actual), "source_tree_sha256": sha256(receipt_path)}


def validate_sampling(real_path: Path, baseline_path: Path) -> dict:
    real, baseline = load_json(real_path), load_json(baseline_path)
    if real.get("status") != "FC_P003B_REAL_OFFICIAL_DATALOADER_SAMPLING_PASS":
        raise ValueError("real sampling audit status differs")
    if baseline.get("status") != "FC_P003_OFFICIAL_DATALOADER_ORDER_COUNTERFACTUAL_PASS":
        raise ValueError("baseline order audit status differs")
    sequence = baseline.get("regular_sequence_sha256")
    if real.get("regular_sequence_sha256") != sequence:
        raise ValueError("real regular sequence is not bound to FC-P003 baseline")
    if real.get("fc_p003_regular_sequence_sha256") != sequence:
        raise ValueError("real audit FC-P003 reference differs")
    if real.get("global_torch_rng_sha256") != baseline.get("global_torch_rng_sha256"):
        raise ValueError("global RNG digest differs from FC-P003 baseline")
    passes = real.get("pair_identity_passes")
    if not isinstance(passes, list) or len(passes) != 2:
        raise ValueError("real sampling audit must contain two pair passes")
    if any(len(row) != 8 or set(row) != EXPECTED_IDENTITIES for row in passes):
        raise ValueError("real sampling pair pass coverage differs")
    if real.get("paired_batch_indices") != EXPECTED_POSITIONS:
        raise ValueError("real sampling interleaved positions differ")
    if real.get("regular_count") != 1368:
        raise ValueError("real regular count differs")
    if real.get("validation_or_frozen_accessed") is not False:
        raise ValueError("real sampling accessed held-out data")
    return {"regular_sequence_sha256": sequence, "pair_identity_passes": passes}


def expected_config(mode: str) -> dict:
    probe = mode == "--probe"
    return {
        "epochs": 1 if probe else 2,
        "max_train_batches": 8 if probe else None,
        "max_validation_batches": 1 if probe else None,
        "expected_regular_batches": 8 if probe else 1368,
        "paired_dataset_repetitions": 1 if probe else 2,
        "paired_batches_per_epoch": 8 if probe else 16,
        "max_paired_eval_batches": 1 if probe else 8,
    }


def validate_output(root: Path, mode: str) -> dict:
    if mode not in {"--probe", "--execute"}:
        raise ValueError("invalid mode")
    required_names = [
        "launch_receipt.json",
        "resolved_config.yaml",
        "runtime_metadata.json",
        "training_data_sources.json",
        "training_history.json",
        "train.log",
    ]
    required = [root / name for name in required_names]
    if any(not path.is_file() for path in required):
        raise ValueError("FC-P003B outputs incomplete")
    history = load_json(root / "training_history.json")
    require_finite(history, "training_history")
    expected = expected_config(mode)
    if not isinstance(history, list) or len(history) != expected["epochs"]:
        raise ValueError("epoch count differs")
    expected_indices = list(range(8)) if mode == "--probe" else EXPECTED_POSITIONS
    repetitions = expected["paired_dataset_repetitions"]
    for epoch, row in enumerate(history, 1):
        if row.get("epoch") != epoch:
            raise ValueError("epoch numbering differs")
        if row.get("paired_dataset_kind") != "dynamic8":
            raise ValueError("paired dataset kind differs")
        if row.get("paired_batch_schedule") != "interleaved":
            raise ValueError("paired schedule differs")
        if row.get("paired_batch_indices") != expected_indices:
            raise ValueError("dynamic schedule differs")
        passes = row.get("paired_identity_passes")
        if not isinstance(passes, list) or len(passes) != repetitions:
            raise ValueError("paired pass count differs")
        if any(len(item) != 8 or set(item) != EXPECTED_IDENTITIES for item in passes):
            raise ValueError("paired pass coverage differs")
        counts = collections.Counter(row.get("paired_identities", []))
        if counts != collections.Counter({key: repetitions for key in EXPECTED_IDENTITIES}):
            raise ValueError("paired multiplicity differs")

    cfg = yaml.safe_load((root / "resolved_config.yaml").read_text(encoding="utf-8"))
    train, data = cfg["training"], cfg["data"]
    contract = {
        "batch_size": 1,
        "rollout_steps": 100,
        "validation_rollout_steps": 100,
        "seed": 20261003,
        "train_stride": 20,
        "additional_train_stride": 2,
        "initial_checkpoint": "/workspace/parent",
        "paired_batch_size": 1,
        "paired_stat_horizons": [20, 50, 100],
        "paired_stat_loss_weight": 10.0,
        "paired_batch_schedule": "interleaved",
        **expected,
    }
    if any(train.get(key) != value for key, value in contract.items()):
        raise ValueError("resolved training contract differs")
    expected_data = {
        "root": "/workspace/base",
        "additional_train_roots": ["/workspace/train8", "/workspace/train16"],
        "paired_dataset_kind": "dynamic8",
        "paired_action_root": "/workspace/train8",
        "paired_zero_root": "/workspace/base",
        "paired_manifest": "/workspace/dynamic_pair_manifest.json",
    }
    if any(data.get(key) != value for key, value in expected_data.items()):
        raise ValueError("resolved data contract differs")
    runtime = load_json(root / "runtime_metadata.json")
    require_finite(runtime, "runtime_metadata")
    if (
        runtime.get("rollout_steps") != 100
        or runtime.get("initial_checkpoint") != "/workspace/parent"
        or runtime.get("paired_stat_loss_weight") != 10.0
        or runtime.get("paired_stat_horizons") != [20, 50, 100]
    ):
        raise ValueError("runtime metadata differs")
    sources = load_json(root / "training_data_sources.json")
    require_finite(sources, "training_data_sources")
    if sources.get("base_windows") != 720 or sources.get("total_windows") != 1368:
        raise ValueError("regular source window counts differ")
    additional = sources.get("additional_sources")
    if not isinstance(additional, list) or len(additional) != 2:
        raise ValueError("additional sources differ")
    for row in additional:
        path = row.get("root")
        if row.get("manifest_sha256") != EXPECTED_MANIFESTS.get(path):
            raise ValueError("additional source manifest differs")
        if row.get("normalization_sha256") != NORMALIZATION_SHA:
            raise ValueError("additional source normalization differs")

    selected = min(history, key=lambda item: item["selection_score"])
    epoch = selected["epoch"]
    model_name, state_name = f"FNO.0.{epoch}.mdlus", f"checkpoint.0.{epoch}.pt"
    best_model, best_state = root / "best" / model_name, root / "best" / state_name
    checkpoint_model = root / "checkpoints" / model_name
    checkpoint_state = root / "checkpoints" / state_name
    for path in (best_model, best_state, checkpoint_model, checkpoint_state):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"checkpoint member missing: {path}")
    if archive_payload(best_model) != archive_payload(checkpoint_model):
        raise ValueError("best model differs from selected checkpoint payload")
    if sha256(best_state) != sha256(checkpoint_state):
        raise ValueError("best state differs from selected checkpoint generation")
    artifacts = {path.relative_to(root).as_posix(): sha256(path) for path in required}
    for path in (best_model, best_state, checkpoint_model, checkpoint_state):
        artifacts[path.relative_to(root).as_posix()] = sha256(path)
    return {
        "status": "FC_P003B_DYNAMIC_PAIR_PROBE_PASS"
        if mode == "--probe"
        else "FC_P003B_DYNAMIC_PAIR_TRAINING_COMPLETE",
        "mode": mode,
        "epochs": len(history),
        "paired_updates_per_epoch": 8 if mode == "--probe" else 16,
        "paired_dataset_repetitions": repetitions,
        "checkpoint_epoch": epoch,
        "checkpoint_model_sha256": sha256(best_model),
        "checkpoint_state_sha256": sha256(best_state),
        "checkpoint_generation_payload_sha256": archive_payload(best_model),
        "sha256": artifacts,
        "training_exit_code": 0,
        "frozen_test_accessed": False,
        "ppo_auto_launched": False,
    }


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    source_parser = sub.add_parser("source")
    source_parser.add_argument("--source", type=Path, required=True)
    source_parser.add_argument("--receipt", type=Path, required=True)
    source_parser.add_argument("--commit", required=True)
    source_parser.add_argument("--required-hashes", type=Path, required=True)
    sampling_parser = sub.add_parser("sampling")
    sampling_parser.add_argument("--real", type=Path, required=True)
    sampling_parser.add_argument("--baseline", type=Path, required=True)
    output_parser = sub.add_parser("output")
    output_parser.add_argument("--root", type=Path, required=True)
    output_parser.add_argument("--mode", choices=("--probe", "--execute"), required=True)
    output_parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "source":
        result = validate_source(args.source, args.receipt, args.commit, args.required_hashes)
    elif args.command == "sampling":
        result = validate_sampling(args.real, args.baseline)
    else:
        result = validate_output(args.root, args.mode)
        write_exclusive(args.receipt, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
