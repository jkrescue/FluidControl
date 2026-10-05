#!/usr/bin/env python3
"""Derive D012 step bindings from one complete FC-P008 post-evaluation.

This CPU-only compatibility adapter verifies the complete native FC-P008
bundle, its calibrated epoch-zero lineage, and its native step receipts.  It
does not recompute metrics, copy a scientific PASS, authorize PPO, or modify
the source bundle.  Derived receipts are written exclusively outside it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path


CANDIDATE_KIND = "full_train_force_row_recalibration"
COMPLETE_STATUS = "FC_P008_POSTEVAL_COMPLETE"
SOURCE_STEP_STATUS = "FC_P008_POSTEVAL_STEP_COMPLETE"
DERIVED_STEP_STATUS = "CONTROL_TRAIN16_POSTEVAL_STEP_COMPLETE"
LINEAGE_STATUS = "FC_P008_CANDIDATE_LINEAGE_PASS"
IMAGE_ID = "sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
PROTOCOL = [
    "validation10_H1_H10_H50_H100_stride25_batch4",
    "dynamic6_H1_H10_H50_H100_stride1_batch8",
    "force_window6",
    "unchanged_development_gate",
]
PRECISION = {
    "status": "FC_P008_FORMAL_EVALUATION_DEFAULT_TF32_HIGH",
    "official_image_id": IMAGE_ID,
    "NVIDIA_TF32_OVERRIDE": None,
    "cuda_matmul_allow_tf32": True,
    "cudnn_allow_tf32": True,
    "float32_matmul_precision": "high",
}
REQUIRED = {
    "lineage.json",
    "precision.json",
    "evidence/formal_evaluation_approval.json",
    "validation10/evaluation.json",
    "validation10/segments.json",
    "validation10/diagnostic.json",
    "validation10/endpoint_gate.json",
    "dynamic6/evaluation.json",
    "dynamic6/segments.json",
    "dynamic6/diagnostic.json",
    "force_window/result.json",
    "development_gate.json",
    "step_receipts/dynamic6.json",
    "step_receipts/force_window.json",
}
STEP_FILES = {
    "dynamic6": (
        "dynamic6/evaluation.json",
        "dynamic6/segments.json",
        "dynamic6/diagnostic.json",
    ),
    "force_window": ("force_window/result.json", "development_gate.json"),
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


def _require_sha(value: str, label: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")


def _actual_bundle_table(root: Path, receipt_path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    resolved_root = root.resolve()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.resolve() == receipt_path or path.name == "outer.log":
            continue
        try:
            path.resolve().relative_to(resolved_root)
        except ValueError as error:
            raise ValueError(f"FC-P008 bundle file escapes source root: {path}") from error
        result[str(path.relative_to(root))] = sha256(path)
    return result


def _verify_lineage(
    path: Path, *, checkpoint: str, state: str, lineage_sha: str
) -> dict:
    if sha256(path) != lineage_sha:
        raise ValueError("FC-P008 lineage SHA differs")
    lineage = load(path)
    required = {
        "status": LINEAGE_STATUS,
        "candidate_kind": CANDIDATE_KIND,
        "checkpoint_relative_directory": "candidate_build/candidate",
        "checkpoint_model_file": "FNO.0.0.mdlus",
        "checkpoint_state_file": "checkpoint.0.0.pt",
        "checkpoint_epoch": 0,
        "parent_checkpoint_epoch": 2,
        "calibration_generation": 1,
        "checkpoint_sha256": checkpoint,
        "checkpoint_state_sha256": state,
        "calibration_fit_performed": True,
        "optimizer_training_performed": False,
        "training_performed": False,
        "validation_or_frozen_accessed": False,
        "ppo_auto_launch": False,
        "official_image_id": IMAGE_ID,
        "formal_protocol": PROTOCOL,
        "precision_protocol": {
            key: value
            for key, value in PRECISION.items()
            if key not in {"status", "official_image_id"}
        },
    }
    if any(lineage.get(key) != value for key, value in required.items()):
        raise ValueError("FC-P008 calibrated epoch-zero lineage differs")
    return lineage


def _verify_source_step(
    root: Path,
    step: str,
    *,
    checkpoint: str,
    state: str,
    lineage_sha: str,
) -> None:
    path = root / f"step_receipts/{step}.json"
    receipt = load(path)
    required = {
        "status": SOURCE_STEP_STATUS,
        "candidate_kind": CANDIDATE_KIND,
        "step": step,
        "checkpoint_epoch": 0,
        "checkpoint_sha256": checkpoint,
        "checkpoint_state_sha256": state,
        "lineage_sha256": lineage_sha,
    }
    if any(receipt.get(key) != value for key, value in required.items()):
        raise ValueError(f"FC-P008 {step} source step identity differs")
    expected = {
        relative: sha256(root / relative) for relative in STEP_FILES[step]
    }
    if receipt.get("sha256") != expected:
        raise ValueError(f"FC-P008 {step} source step SHA table differs")


def verify_bundle(
    receipt_path: Path,
    checkpoint: str,
    state: str,
    lineage_sha: str,
) -> tuple[Path, dict]:
    for value, label in (
        (checkpoint, "checkpoint SHA"),
        (state, "checkpoint-state SHA"),
        (lineage_sha, "lineage SHA"),
    ):
        _require_sha(value, label)
    receipt_path = receipt_path.resolve()
    root = receipt_path.parent
    receipt = load(receipt_path)
    required_identity = {
        "status": COMPLETE_STATUS,
        "candidate_kind": CANDIDATE_KIND,
        "checkpoint_epoch": 0,
        "checkpoint_sha256": checkpoint,
        "checkpoint_state_sha256": state,
        "lineage_sha256": lineage_sha,
        "official_image_id": IMAGE_ID,
        "protocol": PROTOCOL,
        "frozen_test_accessed": False,
        "ppo_auto_launched": False,
    }
    if any(receipt.get(key) != value for key, value in required_identity.items()):
        raise ValueError("FC-P008 completion identity or scope differs")
    recorded = receipt.get("sha256")
    if not isinstance(recorded, dict) or not recorded:
        raise ValueError("FC-P008 completion SHA table is absent")
    actual = _actual_bundle_table(root, receipt_path)
    if recorded != actual:
        raise ValueError("FC-P008 completion SHA table differs from bundle")
    if not REQUIRED.issubset(actual):
        raise ValueError("FC-P008 required post-evaluation evidence is incomplete")

    precision_path = root / "precision.json"
    if load(precision_path) != PRECISION:
        raise ValueError("FC-P008 post-evaluation precision protocol differs")
    precision_sha = sha256(precision_path)
    formal_path = root / "evidence/formal_evaluation_approval.json"
    formal = load(formal_path)
    formal_contract = {
        "status": "FC_P008_FORMAL_EVALUATION_APPROVED",
        "candidate_model_sha256": checkpoint,
        "candidate_state_sha256": state,
        "formal_evaluation_authorized": True,
        "frozen_test_accessed": False,
        "ppo_auto_launch": False,
    }
    if any(formal.get(key) != value for key, value in formal_contract.items()):
        raise ValueError("FC-P008 formal evaluation approval contract differs")
    if (
        receipt.get("precision_sha256") != precision_sha
        or receipt.get("formal_evaluation_approval_sha256") != sha256(formal_path)
    ):
        raise ValueError("FC-P008 completion precision or formal approval binding differs")

    _verify_lineage(
        root / "lineage.json",
        checkpoint=checkpoint,
        state=state,
        lineage_sha=lineage_sha,
    )
    identity_fields = {
        "dynamic6/diagnostic.json": ("checkpoint_sha256", checkpoint),
        "development_gate.json": ("checkpoint_sha256", checkpoint),
        "force_window/result.json": ("model_sha256", checkpoint),
    }
    for relative, (key, expected) in identity_fields.items():
        if load(root / relative).get(key) != expected:
            raise ValueError(f"FC-P008 evidence checkpoint differs: {relative}")
    for step in STEP_FILES:
        native_step = load(root / f"step_receipts/{step}.json")
        if (
            native_step.get("precision_sha256") != precision_sha
            or native_step.get("formal_evaluation_approval_sha256")
            != receipt["formal_evaluation_approval_sha256"]
            or native_step.get("posteval_chain_receipt_sha256")
            != receipt.get("posteval_chain_receipt_sha256")
        ):
            raise ValueError(f"FC-P008 {step} execution binding differs")
        _verify_source_step(
            root,
            step,
            checkpoint=checkpoint,
            state=state,
            lineage_sha=lineage_sha,
        )
    return root, receipt


def payloads(
    receipt_path: Path,
    checkpoint: str,
    state: str,
    lineage_sha: str,
) -> dict[str, dict]:
    root, receipt = verify_bundle(receipt_path, checkpoint, state, lineage_sha)
    source_sha = sha256(receipt_path.resolve())
    return {
        step: {
            "status": DERIVED_STEP_STATUS,
            "step": step,
            "checkpoint_epoch": 0,
            "checkpoint_sha256": checkpoint,
            "checkpoint_state_sha256": state,
            "lineage_sha256": lineage_sha,
            "receipt_kind": "POST_HOC_D012_COMPATIBILITY_BINDING",
            "source_candidate_kind": CANDIDATE_KIND,
            "source_posteval_receipt": str(receipt_path.resolve()),
            "source_posteval_receipt_sha256": source_sha,
            "source_posteval_status": receipt["status"],
            "source_step_receipt": str(
                (root / f"step_receipts/{step}.json").resolve()
            ),
            "source_step_receipt_sha256": sha256(
                root / f"step_receipts/{step}.json"
            ),
            "scientific_status_reused": False,
            "ppo_authorized": False,
            "sha256": {
                str((root / name).resolve()): receipt["sha256"][name]
                for name in names
            },
        }
        for step, names in STEP_FILES.items()
    }


def exclusive_json(path: Path, payload: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_output_paths(
    receipt_path: Path, dynamic_output: Path, force_output: Path
) -> None:
    source_root = receipt_path.resolve().parent
    lexical_source_root = receipt_path.absolute().parent
    outputs = (dynamic_output.resolve(), force_output.resolve())
    lexical_outputs = (dynamic_output.absolute(), force_output.absolute())
    if outputs[0] == outputs[1]:
        raise ValueError("derived receipt outputs must differ")
    for output, lexical_output in zip(outputs, lexical_outputs, strict=True):
        for candidate, root in (
            (output, source_root),
            (lexical_output, lexical_source_root),
        ):
            try:
                candidate.relative_to(root)
            except ValueError:
                continue
            raise ValueError("derived receipt output must be outside source bundle")
    if any(output.exists() for output in outputs):
        raise FileExistsError("refusing to overwrite derived receipt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posteval-receipt", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--checkpoint-state-sha256", required=True)
    parser.add_argument("--lineage-sha256", required=True)
    parser.add_argument("--dynamic-output", type=Path, required=True)
    parser.add_argument("--force-output", type=Path, required=True)
    args = parser.parse_args()
    validate_output_paths(
        args.posteval_receipt, args.dynamic_output, args.force_output
    )
    derived = payloads(
        args.posteval_receipt,
        args.checkpoint_sha256,
        args.checkpoint_state_sha256,
        args.lineage_sha256,
    )
    exclusive_json(args.dynamic_output, derived["dynamic6"])
    exclusive_json(args.force_output, derived["force_window"])
    print("D012_FC_P008_COMPATIBILITY_RECEIPTS_COMPLETE")


if __name__ == "__main__":
    main()
