#!/usr/bin/env python3
"""Strictly validate FC-P003C post-evaluation steps and completion."""

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


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_checkpoint(value: dict, checkpoint: str) -> None:
    if value.get("checkpoint_sha256") != checkpoint:
        raise ValueError("checkpoint identity differs")


def validate_validation(repo: Path, candidate: Path, out: Path, checkpoint: str) -> None:
    snapshot = candidate / "source_snapshot"
    diagnostic = module(snapshot / "scripts/audit_dev30_validation_diagnostic.py", "fcp003c_validation_diagnostic")
    recomputed = diagnostic.audit(out / "validation10/evaluation.json", out / "validation10/segments.json", repo / "data/curated/tandem_cylinders_matched_start_full40_dev30_v1", candidate / "best", "dev30_free_ar_development")
    stored = load(out / "validation10/diagnostic.json")
    if stored != recomputed:
        raise ValueError("validation diagnostic differs")
    require_checkpoint(stored, checkpoint)
    endpoint_module = module(snapshot / "scripts/audit_full40_validation_gate.py", "fcp003c_endpoint_gate")
    report = endpoint_module.load(out / "validation10/evaluation.json")
    predecl = repo / "artifacts/tandem_cylinders/matched_start_full40_predeclared_20261003.json"
    expected = endpoint_module.validate_predeclaration(predecl)
    force = endpoint_module.pooled_h100(report, expected)
    action = endpoint_module.strict_start0_differences(endpoint_module.load(out / "validation10/segments.json"), expected)
    model = next((candidate / "best").glob("FNO.0.*.mdlus"))
    endpoint = load(out / "validation10/endpoint_gate.json")
    fields = {"checkpoint_dir": "/workspace/checkpoint", "checkpoint_epoch": int(model.name.split(".")[2]), "checkpoint_model_file": model.name, "checkpoint_sha256": checkpoint}
    if endpoint.get("h100_force_gate") != force or endpoint.get("h100_start0_action_difference") != action or any(endpoint.get(key) != value for key, value in fields.items()) or endpoint.get("physicsnemo_image_id") != IMAGE_ID or endpoint.get("report_sha256") != sha256(out / "validation10/evaluation.json") or endpoint.get("segments_sha256") != sha256(out / "validation10/segments.json") or endpoint.get("predeclaration_sha256") != sha256(predecl) or endpoint.get("frozen_test_accessed") is not False:
        raise ValueError("endpoint gate differs")


def validate_dynamic(repo: Path, candidate: Path, out: Path, checkpoint: str) -> None:
    audit = module(candidate / "source_snapshot/cfd/tandem_cylinders/audit_full40_dynamic6_fno.py", "fcp003c_dynamic6")
    model = next((candidate / "best").glob("FNO.0.*.mdlus"))
    args = SimpleNamespace(data=repo / "data/curated/tandem_cylinders_full40_dynamic_validation_v1", checkpoint=candidate / "best", checkpoint_epoch=int(model.name.split(".")[2]), expected_model_sha=checkpoint, report=out / "dynamic6/evaluation.json", segments=out / "dynamic6/segments.json", physical_qc=repo / "artifacts/tandem_cylinders/full40_dynamic_validation_real_openfoam_qc_20261003.json")
    recomputed = audit.audit(args)
    stored = load(out / "dynamic6/diagnostic.json")
    if stored != recomputed:
        raise ValueError("dynamic diagnostic differs")
    require_checkpoint(stored, checkpoint)


def validate_force(candidate: Path, out: Path, checkpoint: str) -> None:
    audit = module(candidate / "source_snapshot/scripts/audit_dynamic_fno_development_gates.py", "fcp003c_development")
    stored = load(out / "development_gate.json")
    if stored != audit.audit(out / "force_window/result.json", checkpoint):
        raise ValueError("development gate differs")
    require_checkpoint(stored, checkpoint)


EXPECTED = {
    "validation10": ("validation10/evaluation.json", "validation10/segments.json", "validation10/diagnostic.json", "validation10/endpoint_gate.json"),
    "dynamic6": ("dynamic6/evaluation.json", "dynamic6/segments.json", "dynamic6/diagnostic.json"),
    "force_window": ("force_window/result.json", "development_gate.json"),
}


def validate_step_receipt(path: Path, out: Path, step: str, checkpoint: str) -> None:
    value = load(path)
    expected_paths = {str(out / name): out / name for name in EXPECTED[step]}
    if value.get("status") != "FC_P003C_POSTEVAL_STEP_COMPLETE" or value.get("step") != step or value.get("checkpoint_sha256") != checkpoint:
        raise ValueError(f"{step} receipt contract differs")
    if value.get("sha256") != {name: sha256(item) for name, item in expected_paths.items()}:
        raise ValueError(f"{step} receipt hashes differ")
    if any(out.resolve() not in item.resolve().parents for item in expected_paths.values()):
        raise ValueError(f"{step} receipt path escapes output")


def validate_complete(candidate: Path, out: Path, checkpoint: str) -> None:
    receipt = load(out / "receipt.json")
    if receipt.get("status") != "FC_P003C_POSTEVAL_COMPLETE" or receipt.get("checkpoint_sha256") != checkpoint or receipt.get("ppo_auto_launched") is not False or receipt.get("frozen_test_accessed") is not False:
        raise ValueError("completion receipt contract differs")
    required = {"lineage.json", *(name for names in EXPECTED.values() for name in names), *(f"step_receipts/{step}.json" for step in EXPECTED)}
    actual = {str(path.relative_to(out)): sha256(path) for path in sorted(out.rglob("*")) if path.is_file() and path.name not in {"receipt.json", "outer.log"}}
    if receipt.get("sha256") != actual or not required.issubset(actual):
        raise ValueError("completion artifact table differs")
    lineage = load(out / "lineage.json")
    if lineage.get("status") != "FC_P003C_CANDIDATE_LINEAGE_PASS" or lineage.get("checkpoint_sha256") != checkpoint:
        raise ValueError("lineage differs")
    for step in EXPECTED:
        validate_step_receipt(out / "step_receipts" / f"{step}.json", out, step, checkpoint)
    development = load(out / "development_gate.json")
    if development.get("status") not in {"DYNAMIC_FNO_DEVELOPMENT_ADMISSION_PASS", "DYNAMIC_FNO_DEVELOPMENT_ADMISSION_FAIL"} or development.get("ppo_authorized") is not False or development.get("frozen_test_accessed") is not False:
        raise ValueError("development gate contract differs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--step", choices=(*EXPECTED, "complete"), required=True)
    args = parser.parse_args()
    if args.step == "complete":
        validate_complete(args.candidate, args.output, args.checkpoint_sha256)
    else:
        if args.step == "validation10":
            validate_validation(args.repo, args.candidate, args.output, args.checkpoint_sha256)
        elif args.step == "dynamic6":
            validate_dynamic(args.repo, args.candidate, args.output, args.checkpoint_sha256)
        else:
            validate_force(args.candidate, args.output, args.checkpoint_sha256)
    print(f"FC_P003C_POSTEVAL_REUSE_VALID step={args.step}")


if __name__ == "__main__":
    main()
