#!/usr/bin/env python3
"""Strictly revalidate reusable control-train16 post-evaluation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

IMAGE_ID = "sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def require_checkpoint(document: dict, expected: str) -> None:
    if document.get("checkpoint_sha256") != expected:
        raise ValueError("post-evaluation checkpoint SHA differs")


def validate_validation(repo: Path, candidate: Path, out: Path, checkpoint: str) -> None:
    launch = candidate / "source_snapshot/scripts/audit_dev30_validation_diagnostic.py"
    diagnostic_module = module(launch, "train16_dev30_diagnostic")
    recomputed = diagnostic_module.audit(
        out / "validation10/evaluation.json",
        out / "validation10/segments.json",
        repo / "data/curated/tandem_cylinders_matched_start_full40_dev30_v1",
        candidate / "best",
        "dev30_free_ar_development",
    )
    stored = load_object(out / "validation10/diagnostic.json")
    if stored != recomputed:
        raise ValueError("validation10 diagnostic differs from strict recomputation")
    require_checkpoint(stored, checkpoint)

    endpoint_module = module(
        repo / "scripts/audit_full40_validation_gate.py", "train16_endpoint_gate"
    )
    report = endpoint_module.load(out / "validation10/evaluation.json")
    predecl = repo / "artifacts/tandem_cylinders/matched_start_full40_predeclared_20261003.json"
    expected = endpoint_module.validate_predeclaration(predecl)
    force = endpoint_module.pooled_h100(report, expected)
    action = endpoint_module.strict_start0_differences(
        endpoint_module.load(out / "validation10/segments.json"), expected
    )
    models = sorted((candidate / "best").glob("FNO.*.mdlus"))
    if len(models) != 1 or report.get("checkpoint_dir") != "/workspace/checkpoint":
        raise ValueError("validation checkpoint runtime binding differs")
    checkpoint_fields = {
        "checkpoint_dir": "/workspace/checkpoint",
        "checkpoint_epoch": int(models[0].name.split(".")[2]),
        "checkpoint_model_file": models[0].name,
        "checkpoint_sha256": sha256(models[0]),
    }
    endpoint = load_object(out / "validation10/endpoint_gate.json")
    if (
        endpoint.get("h100_force_gate") != force
        or endpoint.get("h100_start0_action_difference") != action
        or any(endpoint.get(key) != value for key, value in checkpoint_fields.items())
        or endpoint.get("physicsnemo_image_id") != IMAGE_ID
        or endpoint.get("report_sha256")
        != sha256(out / "validation10/evaluation.json")
        or endpoint.get("segments_sha256")
        != sha256(out / "validation10/segments.json")
        or endpoint.get("predeclaration_sha256") != sha256(predecl)
        or endpoint.get("frozen_test_accessed") is not False
    ):
        raise ValueError("validation endpoint Gate differs from strict recomputation")
    require_checkpoint(endpoint, checkpoint)


def validate_dynamic(repo: Path, candidate: Path, out: Path, checkpoint: str) -> None:
    audit_module = module(
        repo / "cfd/tandem_cylinders/audit_full40_dynamic6_fno.py",
        "train16_dynamic6_audit",
    )
    model = next((candidate / "best").glob("FNO.0.*.mdlus"))
    epoch = int(model.name.split(".")[2])
    args = SimpleNamespace(
        data=repo / "data/curated/tandem_cylinders_full40_dynamic_validation_v1",
        checkpoint=candidate / "best",
        checkpoint_epoch=epoch,
        expected_model_sha=checkpoint,
        report=out / "dynamic6/evaluation.json",
        segments=out / "dynamic6/segments.json",
        physical_qc=repo
        / "artifacts/tandem_cylinders/full40_dynamic_validation_real_openfoam_qc_20261003.json",
    )
    recomputed = audit_module.audit(args)
    stored = load_object(out / "dynamic6/diagnostic.json")
    if stored != recomputed:
        raise ValueError("dynamic6 diagnostic differs from strict recomputation")
    require_checkpoint(stored, checkpoint)


def validate_force(candidate: Path, out: Path, checkpoint: str) -> None:
    audit_module = module(
        candidate / "source_snapshot/scripts/audit_dynamic_fno_development_gates.py",
        "train16_development_gate",
    )
    recomputed = audit_module.audit(out / "force_window/result.json", checkpoint)
    stored = load_object(out / "development_gate.json")
    if stored != recomputed:
        raise ValueError("development Gate differs from strict recomputation")
    require_checkpoint(stored, checkpoint)


def validate_complete(out: Path, checkpoint: str) -> None:
    receipt = load_object(out / "receipt.json")
    require_checkpoint(receipt, checkpoint)
    if receipt.get("status") != "CONTROL_TRAIN16_POSTEVAL_COMPLETE":
        raise ValueError("post-evaluation completion status differs")
    recorded = receipt.get("sha256")
    actual = {
        str(path.relative_to(out)): sha256(path)
        for path in sorted(out.rglob("*"))
        if path.is_file() and path.name != "receipt.json"
    }
    if recorded != actual:
        raise ValueError("post-evaluation completion file hash table differs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument(
        "--step", choices=("validation10", "dynamic6", "force_window", "complete"), required=True
    )
    args = parser.parse_args()
    if args.step == "validation10":
        validate_validation(args.repo, args.candidate, args.output, args.checkpoint_sha256)
    elif args.step == "dynamic6":
        validate_dynamic(args.repo, args.candidate, args.output, args.checkpoint_sha256)
    elif args.step == "force_window":
        validate_force(args.candidate, args.output, args.checkpoint_sha256)
    else:
        validate_complete(args.output, args.checkpoint_sha256)
    print(f"CONTROL_TRAIN16_POSTEVAL_REUSE_VALID step={args.step}")


if __name__ == "__main__":
    main()
