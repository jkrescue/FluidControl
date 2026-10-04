#!/usr/bin/env python3
"""Strictly validate reusable FC-P003 post-evaluation evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def base_module(repo: Path):
    path = repo / "scripts/validate_control_train16_posteval_step_fc_p003_immutable.py"
    spec = importlib.util.spec_from_file_location("base_posteval_validator", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def validate_step_receipt(
    receipt_path: Path,
    *,
    step: str,
    checkpoint: str,
    expected_paths: tuple[Path, ...],
    output_root: Path,
) -> None:
    step_receipt = load(receipt_path)
    if (
        step_receipt.get("status") != "CONTROL_TRAIN16_POSTEVAL_STEP_COMPLETE"
        or step_receipt.get("step") != step
        or step_receipt.get("checkpoint_sha256") != checkpoint
    ):
        raise ValueError(f"FC-P003 {step} receipt contract differs")
    hashes = step_receipt.get("sha256")
    expected = {str(path): path for path in expected_paths}
    if not isinstance(hashes, dict) or set(hashes) != set(expected):
        raise ValueError(f"FC-P003 {step} receipt artifact set differs")
    root = output_root.resolve()
    for name, path in expected.items():
        resolved = path.resolve()
        if root not in resolved.parents or Path(name).resolve() != resolved:
            raise ValueError(f"FC-P003 {step} receipt path escapes output")
        if not path.is_file() or sha256(path) != hashes[name]:
            raise ValueError(f"FC-P003 {step} receipt artifact differs")


def validate_complete(out: Path, checkpoint: str) -> None:
    receipt = load(out / "receipt.json")
    if (
        receipt.get("status") != "FC_P003_POSTEVAL_COMPLETE"
        or receipt.get("checkpoint_sha256") != checkpoint
        or receipt.get("ppo_auto_launched") is not False
        or receipt.get("frozen_test_accessed") is not False
    ):
        raise ValueError("paired completion receipt differs")
    required = {
        "lineage.json",
        "step_receipts/validation10.json",
        "step_receipts/dynamic6.json",
        "step_receipts/force_window.json",
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
    actual = {
        str(path.relative_to(out)): sha256(path)
        for path in sorted(out.rglob("*"))
        if path.is_file() and path.name not in {"receipt.json", "outer.log"}
    }
    if receipt.get("sha256") != actual:
        raise ValueError("paired completion hash table differs")
    if not required.issubset(actual):
        raise ValueError("FC-P003 completion required files are missing")
    lineage = load(out / "lineage.json")
    if lineage.get("checkpoint_sha256") != checkpoint:
        raise ValueError("FC-P003 lineage checkpoint differs")
    expected_by_step = {
        "validation10": tuple(out / name for name in (
            "validation10/evaluation.json", "validation10/segments.json",
            "validation10/diagnostic.json", "validation10/endpoint_gate.json",
        )),
        "dynamic6": tuple(out / name for name in (
            "dynamic6/evaluation.json", "dynamic6/segments.json",
            "dynamic6/diagnostic.json",
        )),
        "force_window": (
            out / "force_window/result.json", out / "development_gate.json",
        ),
    }
    for step, expected_paths in expected_by_step.items():
        validate_step_receipt(
            out / "step_receipts" / f"{step}.json",
            step=step,
            checkpoint=checkpoint,
            expected_paths=expected_paths,
            output_root=out,
        )
    development = load(out / "development_gate.json")
    if (
        development.get("status")
        not in {
            "DYNAMIC_FNO_DEVELOPMENT_ADMISSION_PASS",
            "DYNAMIC_FNO_DEVELOPMENT_ADMISSION_FAIL",
        }
        or development.get("ppo_authorized") is not False
        or development.get("frozen_test_accessed") is not False
    ):
        raise ValueError("FC-P003 development gate contract differs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--step", choices=("validation10", "dynamic6", "force_window", "complete"), required=True)
    args = parser.parse_args()
    if args.step == "complete":
        validate_complete(args.output, args.checkpoint_sha256)
    else:
        base = base_module(args.repo)
        if args.step == "validation10":
            base.validate_validation(args.repo, args.candidate, args.output, args.checkpoint_sha256)
        elif args.step == "dynamic6":
            base.validate_dynamic(args.repo, args.candidate, args.output, args.checkpoint_sha256)
        else:
            base.validate_force(args.candidate, args.output, args.checkpoint_sha256)
    print(f"FC_P003_POSTEVAL_REUSE_VALID step={args.step}")


if __name__ == "__main__":
    main()
