#!/usr/bin/env python3
"""Fail-closed completion validator for the FC-P003C technical probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

REGULAR_METADATA = {
    "case": "matched_start_acquisition_train_b04_m075",
    "step": 180,
    "rollout_steps": 100,
    "split": "train",
    "dataset_index": 0,
}
PAIR_METADATA = {
    "pair_id": "b00:multisine",
    "phase": "b00",
    "profile": "multisine",
    "split": "train",
    "start": 0,
    "horizon": 100,
}
CHANNELS = ["front_cd", "front_cl", "rear_cd", "rear_cl"]
WEIGHTS = [1 / 7, 1 / 7, 4 / 7, 1 / 7]
REGULAR_HDF_SHA = "9fd391090802f7f6a7189d1cbb2ccedf3e667fa4ee78b2f8f1c2c20c1dea0345"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256(Path(path).read_bytes())
    return digest.hexdigest()


def valid_sha(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def validate(root: Path, model_sha: str, state_sha: str, image_id: str) -> dict:
    result_path = root / "container_output/result_bundle/result.json"
    result = json.loads(result_path.read_text())
    expected_bindings = {
        "config_sha256": sha256(root / "resolved_config.yaml"),
        "source_manifest_sha256": sha256(root / "source_snapshot.sha256"),
        "launch_receipt_sha256": sha256(root / "launch_receipt.json"),
        "cpu_equivalence_receipt_sha256": sha256(
            root / "cpu_equivalence_receipt.json"
        ),
        "image_id": image_id,
        "physicsnemo_version": "2.2.2",
        "regular_hdf_sha256": REGULAR_HDF_SHA,
        "regular_metadata": REGULAR_METADATA,
        "pair_metadata": PAIR_METADATA,
        "force_channels": CHANNELS,
    }
    for key, expected in expected_bindings.items():
        if result.get(key) != expected:
            raise ValueError(f"result binding differs: {key}")
    if (
        result.get("status") != "FC_P003C_MIXED_LOSS_TECHNICAL_PROBE_PASS"
        or result.get("optimizer_steps") != 1
        or result.get("candidate_saved") is not False
        or result.get("validation_or_frozen_accessed") is not False
    ):
        raise ValueError("result status/scope differs")
    weights = result.get("force_channel_weights_normalized", [])
    if len(weights) != 4 or any(
        not math.isclose(value, expected, rel_tol=2e-7, abs_tol=2e-8)
        for value, expected in zip(weights, WEIGHTS, strict=True)
    ):
        raise ValueError("normalized force weights differ")
    before = result.get("scratch_model_state_sha256_before")
    after = result.get("scratch_model_state_sha256_after")
    if not valid_sha(before) or not valid_sha(after) or before == after:
        raise ValueError("scratch model state-change evidence differs")
    parent = {"model": model_sha, "state": state_sha}
    if (
        result.get("parent_file_sha256_before") != parent
        or result.get("parent_file_sha256_after") != parent
    ):
        raise ValueError("parent before/after SHA differs")
    metrics = result.get("metrics", {})
    raw = metrics.get("paired_step_force_per_channel_mse", [])
    weighted = metrics.get(
        "paired_step_force_per_channel_weighted_contribution", []
    )
    values = [
        result.get("regular_field_loss"),
        result.get("regular_force_loss"),
        metrics.get("loss"),
        metrics.get("base_loss"),
        metrics.get("paired_step_force_loss"),
        metrics.get("paired_step_force_weighted_loss"),
        metrics.get("preclip_gradient_norm"),
        *raw,
        *weighted,
        result.get("cuda_peak_allocated_gib"),
        result.get("cuda_peak_reserved_gib"),
        result.get("minimum_mem_available_gib"),
    ]
    if (
        len(raw) != 4
        or len(weighted) != 4
        or metrics.get("optimizer_steps") != 1
        or metrics.get("chunk_size") != 10
        or metrics.get("chunk_count") != 10
        or any(
            not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in values
        )
        or result["minimum_mem_available_gib"] < 20
    ):
        raise ValueError("finite numeric/step contract differs")
    for suffix in ("*.pt", "*.mdlus", "*.ckpt", "*.pth"):
        if list((root / "container_output").rglob(suffix)):
            raise ValueError("probe wrote a candidate/checkpoint")
    return {
        "status": "FC_P003C_MIXED_LOSS_TECHNICAL_PROBE_COMPLETE",
        "scientific_result": False,
        "optimizer_steps": 1,
        "candidate_saved": False,
        "validation_or_frozen_accessed": False,
        "result_sha256": sha256(result_path),
        "launch_receipt_sha256": sha256(root / "launch_receipt.json"),
        "source_manifest_sha256": sha256(root / "source_snapshot.sha256"),
        "immutable_launcher_sha256": sha256(root / "immutable_launcher.sh"),
        "immutable_guard_sha256": sha256(root / "immutable_guard.py"),
        "validator_sha256": sha256(Path(__file__)),
        "checkpoint_model_sha256": model_sha,
        "checkpoint_state_sha256": state_sha,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--state-sha256", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"exclusive completion receipt exists: {output}")
    value = validate(Path(args.root), args.model_sha256, args.state_sha256, args.image_id)
    temporary = output.with_name(f"{output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.link(temporary, output)
    temporary.unlink()


if __name__ == "__main__":
    main()
