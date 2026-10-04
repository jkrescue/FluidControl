#!/usr/bin/env python3
"""Derive D012 step bindings from one complete FC-P003C post-evaluation.

This CPU-only compatibility adapter verifies the original bundle and its native
step receipts.  It does not recompute metrics, copy a PASS, authorize PPO, or
modify the source bundle.  Derived receipts are written exclusively outside it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


CANDIDATE_KIND = "true_state_paired_step_lambda10"
COMPLETE_STATUS = "FC_P003C_POSTEVAL_COMPLETE"
SOURCE_STEP_STATUS = "FC_P003C_POSTEVAL_STEP_COMPLETE"
DERIVED_STEP_STATUS = "CONTROL_TRAIN16_POSTEVAL_STEP_COMPLETE"
REQUIRED = {
    "lineage.json",
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


def _actual_bundle_table(root: Path, receipt_path: Path) -> dict[str, str]:
    result = {}
    resolved_root = root.resolve()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.resolve() == receipt_path or path.name == "outer.log":
            continue
        try:
            path.resolve().relative_to(resolved_root)
        except ValueError as error:
            raise ValueError(f"FC-P003C bundle file escapes source root: {path}") from error
        result[str(path.relative_to(root))] = sha256(path)
    return result


def _verify_source_step(root: Path, step: str, checkpoint: str) -> None:
    path = root / f"step_receipts/{step}.json"
    receipt = load(path)
    if (
        receipt.get("status") != SOURCE_STEP_STATUS
        or receipt.get("step") != step
        or receipt.get("checkpoint_sha256") != checkpoint
    ):
        raise ValueError(f"FC-P003C {step} source step identity differs")
    expected = {
        str((root / relative).resolve()): sha256(root / relative)
        for relative in STEP_FILES[step]
    }
    if receipt.get("sha256") != expected:
        raise ValueError(f"FC-P003C {step} source step SHA table differs")


def verify_bundle(receipt_path: Path, checkpoint: str) -> tuple[Path, dict]:
    receipt_path = receipt_path.resolve()
    root = receipt_path.parent
    receipt = load(receipt_path)
    if (
        receipt.get("status") != COMPLETE_STATUS
        or receipt.get("candidate_kind") != CANDIDATE_KIND
        or receipt.get("checkpoint_sha256") != checkpoint
        or receipt.get("frozen_test_accessed") is not False
        or receipt.get("ppo_auto_launched") is not False
    ):
        raise ValueError("FC-P003C completion identity or scope differs")
    recorded = receipt.get("sha256")
    if not isinstance(recorded, dict) or not recorded:
        raise ValueError("FC-P003C completion SHA table is absent")
    actual = _actual_bundle_table(root, receipt_path)
    if recorded != actual:
        raise ValueError("FC-P003C completion SHA table differs from bundle")
    if not REQUIRED.issubset(actual):
        raise ValueError("FC-P003C required post-evaluation evidence is incomplete")

    identity_fields = {
        "lineage.json": ("checkpoint_sha256", checkpoint),
        "dynamic6/diagnostic.json": ("checkpoint_sha256", checkpoint),
        "development_gate.json": ("checkpoint_sha256", checkpoint),
        "force_window/result.json": ("model_sha256", checkpoint),
    }
    for relative, (key, expected) in identity_fields.items():
        if load(root / relative).get(key) != expected:
            raise ValueError(f"FC-P003C evidence checkpoint differs: {relative}")
    lineage = load(root / "lineage.json")
    if (
        lineage.get("status") != "FC_P003C_CANDIDATE_LINEAGE_PASS"
        or lineage.get("candidate_kind") != CANDIDATE_KIND
        or lineage.get("training_performed") is not True
        or lineage.get("frozen_test_opened_or_enumerated") is not False
        or lineage.get("ppo_auto_launch") is not False
    ):
        raise ValueError("FC-P003C candidate lineage scope differs")
    for step in STEP_FILES:
        _verify_source_step(root, step, checkpoint)
    return root, receipt


def payloads(receipt_path: Path, checkpoint: str) -> dict[str, dict]:
    root, receipt = verify_bundle(receipt_path, checkpoint)
    source_sha = sha256(receipt_path.resolve())
    return {
        step: {
            "status": DERIVED_STEP_STATUS,
            "step": step,
            "checkpoint_sha256": checkpoint,
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
    parser.add_argument("--dynamic-output", type=Path, required=True)
    parser.add_argument("--force-output", type=Path, required=True)
    args = parser.parse_args()
    validate_output_paths(
        args.posteval_receipt, args.dynamic_output, args.force_output
    )
    derived = payloads(args.posteval_receipt, args.checkpoint_sha256)
    exclusive_json(args.dynamic_output, derived["dynamic6"])
    exclusive_json(args.force_output, derived["force_window"])
    print("D012_FC_P003C_COMPATIBILITY_RECEIPTS_COMPLETE")


if __name__ == "__main__":
    main()
