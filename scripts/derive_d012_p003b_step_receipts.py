#!/usr/bin/env python3
"""Derive D012 compatibility receipts from a complete FC-P003B bundle.

This CPU-only adapter does not recompute or copy any scientific PASS.  It
strictly verifies the immutable post-evaluation receipt and binds the original
dynamic/force evidence for the existing D012 numerical producer.

Outputs are exclusive.  If the second filesystem link fails after the first
is created, recovery must use a new output directory; this tool never resumes
or overwrites a partial derived-receipt pair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


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


def verify_bundle(receipt_path: Path, checkpoint: str) -> tuple[Path, dict]:
    receipt_path = receipt_path.resolve()
    root = receipt_path.parent
    receipt = load(receipt_path)
    if (
        receipt.get("status") != "FC_P003B_POSTEVAL_COMPLETE"
        or receipt.get("candidate_kind") != "dynamic_paired_interleaved_lambda10"
        or receipt.get("checkpoint_sha256") != checkpoint
        or receipt.get("frozen_test_accessed") is not False
        or receipt.get("ppo_auto_launched") is not False
    ):
        raise ValueError("FC-P003B completion identity or scope differs")
    recorded = receipt.get("sha256")
    if not isinstance(recorded, dict) or not recorded:
        raise ValueError("FC-P003B completion SHA table is absent")
    actual = {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.resolve() != receipt_path
        and not path.name.endswith(".outer.log")
    }
    if recorded != actual:
        raise ValueError("FC-P003B completion SHA table differs from bundle")
    if not REQUIRED.issubset(actual):
        raise ValueError("FC-P003B required post-evaluation evidence is incomplete")

    identity_fields = {
        "lineage.json": ("checkpoint_sha256", checkpoint),
        "dynamic6/diagnostic.json": ("checkpoint_sha256", checkpoint),
        "development_gate.json": ("checkpoint_sha256", checkpoint),
        "force_window/result.json": ("model_sha256", checkpoint),
    }
    for relative, (key, expected) in identity_fields.items():
        if load(root / relative).get(key) != expected:
            raise ValueError(f"FC-P003B evidence checkpoint differs: {relative}")
    return root, receipt


def payloads(receipt_path: Path, checkpoint: str) -> dict[str, dict]:
    root, receipt = verify_bundle(receipt_path, checkpoint)
    source_sha = sha256(receipt_path.resolve())
    result = {}
    for step, names in STEP_FILES.items():
        result[step] = {
            "status": "CONTROL_TRAIN16_POSTEVAL_STEP_COMPLETE",
            "step": step,
            "checkpoint_sha256": checkpoint,
            "receipt_kind": "POST_HOC_D012_COMPATIBILITY_BINDING",
            "source_posteval_receipt": str(receipt_path.resolve()),
            "source_posteval_receipt_sha256": source_sha,
            "source_posteval_status": receipt["status"],
            "scientific_status_reused": False,
            "sha256": {str((root / name).resolve()): receipt["sha256"][name] for name in names},
        }
    return result


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posteval-receipt", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--dynamic-output", type=Path, required=True)
    parser.add_argument("--force-output", type=Path, required=True)
    args = parser.parse_args()
    if args.dynamic_output.resolve() == args.force_output.resolve():
        raise ValueError("derived receipt outputs must differ")
    if args.dynamic_output.exists() or args.force_output.exists():
        raise FileExistsError("refusing to overwrite derived receipt")
    derived = payloads(args.posteval_receipt, args.checkpoint_sha256)
    exclusive_json(args.dynamic_output, derived["dynamic6"])
    exclusive_json(args.force_output, derived["force_window"])
    print("D012_P003B_COMPATIBILITY_RECEIPTS_COMPLETE")


if __name__ == "__main__":
    main()
