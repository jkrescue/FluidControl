#!/usr/bin/env python3
"""Recompute the approved FC-P003B dynamic-pair candidate lineage."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_validator(path: Path):
    spec = importlib.util.spec_from_file_location("fcp003b_validator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(args) -> dict:
    validator = load_validator(args.validator)
    recomputed = validator.validate_output(args.candidate, "--execute")
    stored = load(args.candidate / "completion_receipt.json")
    if stored != recomputed:
        raise ValueError("stored completion receipt differs from strict recomputation")
    source = validator.validate_source(
        args.source, args.source_receipt, validator.SOURCE_COMMIT, args.source_required
    )
    sampling = validator.validate_sampling(args.real_sampling, args.baseline_order)
    if sha256(args.approval) != validator.APPROVAL_SHA:
        raise ValueError("FC-P003B approval differs")
    if sha256(args.dynamic_manifest) != validator.DYNAMIC_MANIFEST_SHA:
        raise ValueError("dynamic pair manifest differs")
    if sha256(args.parent / "FNO.0.2.mdlus") != validator.PARENT_MODEL_SHA:
        raise ValueError("immutable parent model differs")
    if sha256(args.parent / "checkpoint.0.2.pt") != validator.PARENT_STATE_SHA:
        raise ValueError("immutable parent state differs")
    history = load(args.candidate / "training_history.json")
    best_epoch = recomputed["checkpoint_epoch"]
    return {
        "status": "FC_P003B_DYNAMIC_PAIR_CANDIDATE_LINEAGE_PASS",
        "candidate_kind": "dynamic_paired_interleaved_lambda10",
        "checkpoint_epoch": best_epoch,
        "checkpoint_sha256": recomputed["checkpoint_model_sha256"],
        "checkpoint_state_sha256": recomputed["checkpoint_state_sha256"],
        "checkpoint_generation_payload_sha256": recomputed[
            "checkpoint_generation_payload_sha256"
        ],
        "completion_receipt_sha256": sha256(
            args.candidate / "completion_receipt.json"
        ),
        "launch_receipt_sha256": sha256(args.candidate / "launch_receipt.json"),
        "resolved_config_sha256": sha256(args.candidate / "resolved_config.yaml"),
        "training_history_sha256": sha256(args.candidate / "training_history.json"),
        "source_commit": validator.SOURCE_COMMIT,
        "source_receipt_sha256": sha256(args.source_receipt),
        "source_file_count": source["file_count"],
        "approval_sha256": sha256(args.approval),
        "dynamic_pair_manifest_sha256": sha256(args.dynamic_manifest),
        "real_sampling_receipt_sha256": sha256(args.real_sampling),
        "baseline_order_receipt_sha256": sha256(args.baseline_order),
        "regular_sequence_sha256": sampling["regular_sequence_sha256"],
        "epochs": [row["epoch"] for row in history],
        "paired_updates_per_epoch": [len(row["paired_identities"]) for row in history],
        "paired_identity_passes_per_epoch": [
            row["paired_identity_passes"] for row in history
        ],
        "parent_model_sha256": validator.PARENT_MODEL_SHA,
        "parent_state_sha256": validator.PARENT_STATE_SHA,
        "formal_protocol": [
            "validation10_H1_H10_H50_H100_stride25",
            "dynamic6_H1_H10_H50_H100_stride1",
            "force_window6",
            "unchanged_development_gate",
        ],
        "training_performed": False,
        "frozen_test_opened_or_enumerated": False,
        "ppo_auto_launch": False,
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
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--source-required", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--dynamic-manifest", type=Path, required=True)
    parser.add_argument("--real-sampling", type=Path, required=True)
    parser.add_argument("--baseline-order", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = build(args)
    if args.output:
        write_exclusive(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
