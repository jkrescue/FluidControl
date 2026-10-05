#!/usr/bin/env python3
"""Validate FC-P008 post-evaluation without copying numerical gate logic."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import tempfile
from pathlib import Path
from types import ModuleType

IMAGE_ID = "sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
CANDIDATE_KIND = "full_train_force_row_recalibration"
LINEAGE_STATUS = "FC_P008_CANDIDATE_LINEAGE_PASS"
STEP_STATUS = "FC_P008_POSTEVAL_STEP_COMPLETE"
COMPLETE_STATUS = "FC_P008_POSTEVAL_COMPLETE"
PROTOCOL = [
    "validation10_H1_H10_H50_H100_stride25_batch4",
    "dynamic6_H1_H10_H50_H100_stride1_batch8",
    "force_window6",
    "unchanged_development_gate",
]
PRECISION = {
    "NVIDIA_TF32_OVERRIDE": None,
    "cuda_matmul_allow_tf32": True,
    "cudnn_allow_tf32": True,
    "float32_matmul_precision": "high",
}
CALIBRATED_KIND = "FC_P008_TRAIN_ONLY_FORCE_ROW_CANDIDATE"
DIAGNOSTIC_KIND = "fc_p008_force_row_calibrated_epoch0"
PRECISION_STATUS = "FC_P008_FORMAL_EVALUATION_DEFAULT_TF32_HIGH"
CHAIN_STATUS = "FC_P008_IMMUTABLE_POSTEVAL_CHAIN_STAGED"


def configure_profile(name: str) -> None:
    """Select an exact identity profile; numerical validation stays shared."""
    global CANDIDATE_KIND, LINEAGE_STATUS, STEP_STATUS, COMPLETE_STATUS
    global CALIBRATED_KIND, DIAGNOSTIC_KIND, PRECISION_STATUS, CHAIN_STATUS
    profiles = {
        "p008": (
            "full_train_force_row_recalibration",
            "FC_P008_CANDIDATE_LINEAGE_PASS",
            "FC_P008_POSTEVAL_STEP_COMPLETE",
            "FC_P008_POSTEVAL_COMPLETE",
            "FC_P008_TRAIN_ONLY_FORCE_ROW_CANDIDATE",
            "fc_p008_force_row_calibrated_epoch0",
            "FC_P008_FORMAL_EVALUATION_DEFAULT_TF32_HIGH",
            "FC_P008_IMMUTABLE_POSTEVAL_CHAIN_STAGED",
        ),
        "p009": (
            "joint_h1_free_ar_force_row_recalibration",
            "FC_P009_CANDIDATE_LINEAGE_PASS",
            "FC_P009_POSTEVAL_STEP_COMPLETE",
            "FC_P009_POSTEVAL_COMPLETE",
            "FC_P009_TRAIN_ONLY_JOINT_FORCE_ROW_CANDIDATE",
            "fc_p009_joint_force_row_calibrated_epoch0",
            "FC_P009_FORMAL_EVALUATION_DEFAULT_TF32_HIGH",
            "FC_P009_IMMUTABLE_POSTEVAL_CHAIN_STAGED",
        ),
    }
    if name not in profiles:
        raise ValueError("unknown calibrated post-evaluation profile")
    (
        CANDIDATE_KIND,
        LINEAGE_STATUS,
        STEP_STATUS,
        COMPLETE_STATUS,
        CALIBRATED_KIND,
        DIAGNOSTIC_KIND,
        PRECISION_STATUS,
        CHAIN_STATUS,
    ) = profiles[name]
EXPECTED = {
    "validation10": (
        "validation10/evaluation.json",
        "validation10/segments.json",
        "validation10/diagnostic.json",
        "validation10/endpoint_gate.json",
    ),
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


def numerically_equivalent(left: object, right: object) -> bool:
    """Compare recomputed JSON while tolerating only machine-roundoff floats."""
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-14, abs_tol=1e-14)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            numerically_equivalent(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            numerically_equivalent(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def checkpoint_files(candidate: Path, lineage: dict) -> tuple[Path, Path]:
    directory = candidate / lineage.get("checkpoint_relative_directory", "")
    model = directory / lineage.get("checkpoint_model_file", "")
    state = directory / lineage.get("checkpoint_state_file", "")
    if not model.is_file() or not state.is_file():
        raise ValueError("FC-P008 checkpoint pair is incomplete")
    expected_epoch = lineage.get("checkpoint_epoch")
    try:
        model_epoch = int(model.name.split(".")[2])
        state_epoch = int(state.name.split(".")[2])
    except (IndexError, ValueError) as error:
        raise ValueError("FC-P008 checkpoint names do not encode an epoch") from error
    if model_epoch != expected_epoch or state_epoch != expected_epoch:
        raise ValueError("FC-P008 checkpoint epoch differs")
    if sha256(model) != lineage.get("checkpoint_sha256"):
        raise ValueError("FC-P008 checkpoint model differs")
    if sha256(state) != lineage.get("checkpoint_state_sha256"):
        raise ValueError("FC-P008 checkpoint state differs")
    return model, state


def validate_lineage(candidate: Path, lineage_path: Path) -> tuple[dict, Path, Path]:
    lineage = load(lineage_path)
    if lineage.get("status") != LINEAGE_STATUS:
        raise ValueError("FC-P008 candidate lineage is not PASS")
    if lineage.get("candidate_kind") != CANDIDATE_KIND:
        raise ValueError("FC-P008 candidate kind differs")
    if lineage.get("official_image_id") != IMAGE_ID:
        raise ValueError("FC-P008 image identity differs")
    if lineage.get("formal_protocol") != PROTOCOL:
        raise ValueError("FC-P008 formal protocol differs")
    if lineage.get("training_performed") is not False:
        raise ValueError("FC-P008 must not claim a new training run")
    if (
        lineage.get("calibration_fit_performed") is not True
        or lineage.get("optimizer_training_performed") is not False
    ):
        raise ValueError("FC-P008 calibration/optimizer scope differs")
    if lineage.get("validation_or_frozen_accessed") is not False:
        raise ValueError("FC-P008 candidate creation accessed held-out data")
    if lineage.get("ppo_auto_launch") is not False:
        raise ValueError("FC-P008 candidate cannot auto-launch PPO")
    model, state = checkpoint_files(candidate, lineage)
    return lineage, model, state


def calibrated_kwargs(lineage: dict) -> dict:
    value = {
        "allow_calibrated_epoch_zero": True,
        "expected_calibrated_model_sha256": lineage["checkpoint_sha256"],
        "expected_calibrated_state_sha256": lineage["checkpoint_state_sha256"],
    }
    if CALIBRATED_KIND != "FC_P008_TRAIN_ONLY_FORCE_ROW_CANDIDATE":
        value["expected_calibrated_kind"] = CALIBRATED_KIND
    return value


def validate_science_step(
    repo: Path,
    candidate: Path,
    out: Path,
    lineage: dict,
    step: str,
    numerical_source: Path,
) -> None:
    checkpoint = candidate / lineage["checkpoint_relative_directory"]
    calibrated = calibrated_kwargs(lineage)
    if step == "validation10":
        diagnostic = module(
            numerical_source / "scripts/audit_dev30_validation_diagnostic.py",
            "fcp008_validation_diagnostic",
        )
        recomputed = diagnostic.audit(
            out / "validation10/evaluation.json",
            out / "validation10/segments.json",
            repo / "data/curated/tandem_cylinders_matched_start_full40_dev30_v1",
            checkpoint,
            DIAGNOSTIC_KIND,
            **calibrated,
        )
        if load(out / "validation10/diagnostic.json") != recomputed:
            raise ValueError("FC-P008 validation diagnostic differs")
        endpoint = module(
            numerical_source / "scripts/audit_full40_validation_gate.py",
            "fcp008_endpoint_gate",
        )
        report_path = out / "validation10/evaluation.json"
        report = load(report_path)
        host_data = repo / "data/curated/tandem_cylinders_matched_start_full40_v1"
        # The immutable report correctly records its container paths.  Give the
        # host-side auditor an ephemeral path view, then restore the immutable
        # container identity and original report digest before comparison.
        host_view = dict(report)
        host_view.update(
            checkpoint_dir=str(checkpoint),
            evaluation_data=str(host_data),
            normalization_data=str(host_data),
        )
        with tempfile.TemporaryDirectory(prefix="fcp008-endpoint-view-") as directory:
            view_path = Path(directory) / "evaluation.json"
            view_path.write_text(
                json.dumps(host_view, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            recomputed = endpoint.audit(
                view_path,
                out / "validation10/segments.json",
                repo
                / "artifacts/tandem_cylinders/matched_start_full40_predeclared_20261003.json",
                checkpoint,
                host_data,
                numerical_source / "conf/tandem_fno_full40_h20.yaml",
                IMAGE_ID,
                **calibrated,
            )
        recomputed["checkpoint_dir"] = report["checkpoint_dir"]
        recomputed["report_sha256"] = sha256(report_path)
        if not numerically_equivalent(
            load(out / "validation10/endpoint_gate.json"), recomputed
        ):
            raise ValueError("FC-P008 endpoint gate differs")
    elif step == "dynamic6":
        dynamic = module(
            numerical_source / "cfd/tandem_cylinders/audit_full40_dynamic6_fno.py",
            "fcp008_dynamic6",
        )
        args = type(
            "Args",
            (),
            {
                "data": repo
                / "data/curated/tandem_cylinders_full40_dynamic_validation_v1",
                "checkpoint": checkpoint,
                "checkpoint_epoch": 0,
                "expected_model_sha": lineage["checkpoint_sha256"],
                "report": out / "dynamic6/evaluation.json",
                "segments": out / "dynamic6/segments.json",
                "physical_qc": repo
                / "artifacts/tandem_cylinders/full40_dynamic_validation_real_openfoam_qc_20261003.json",
            },
        )()
        if load(out / "dynamic6/diagnostic.json") != dynamic.audit(args):
            raise ValueError("FC-P008 dynamic diagnostic differs")
    elif step == "force_window":
        development = module(
            numerical_source / "scripts/audit_dynamic_fno_development_gates.py",
            "fcp008_development",
        )
        recomputed = development.audit(
            out / "force_window/result.json", lineage["checkpoint_sha256"]
        )
        if load(out / "development_gate.json") != recomputed:
            raise ValueError("FC-P008 development gate differs")
    else:
        raise ValueError(f"unknown science step: {step}")


def validate_step_receipt(path: Path, out: Path, step: str, lineage: dict) -> None:
    value = load(path)
    expected_paths = {name: out / name for name in EXPECTED[step]}
    expected_identity = {
        "status": STEP_STATUS,
        "candidate_kind": CANDIDATE_KIND,
        "step": step,
        "checkpoint_epoch": lineage["checkpoint_epoch"],
        "checkpoint_sha256": lineage["checkpoint_sha256"],
        "checkpoint_state_sha256": lineage["checkpoint_state_sha256"],
        "lineage_sha256": lineage["lineage_sha256"],
        "posteval_chain_receipt_sha256": lineage["posteval_chain_receipt_sha256"],
        "formal_evaluation_approval_sha256": lineage["formal_evaluation_approval_sha256"],
        "precision_sha256": sha256(out / "precision.json"),
    }
    if any(value.get(key) != expected for key, expected in expected_identity.items()):
        raise ValueError(f"{step} receipt identity differs")
    if value.get("sha256") != {
        name: sha256(item) for name, item in expected_paths.items()
    }:
        raise ValueError(f"{step} receipt hashes differ")
    if any(out.resolve() not in item.resolve().parents for item in expected_paths.values()):
        raise ValueError(f"{step} receipt path escapes output")


def validate_complete(
    candidate: Path, out: Path, lineage_path: Path, lineage: dict
) -> None:
    receipt = load(out / "receipt.json")
    expected_identity = {
        "status": COMPLETE_STATUS,
        "candidate_kind": CANDIDATE_KIND,
        "checkpoint_epoch": lineage["checkpoint_epoch"],
        "checkpoint_sha256": lineage["checkpoint_sha256"],
        "checkpoint_state_sha256": lineage["checkpoint_state_sha256"],
        "official_image_id": IMAGE_ID,
        "lineage_sha256": sha256(lineage_path),
        "posteval_chain_receipt_sha256": lineage["posteval_chain_receipt_sha256"],
        "formal_evaluation_approval_sha256": lineage["formal_evaluation_approval_sha256"],
        "precision_sha256": sha256(out / "precision.json"),
        "protocol": PROTOCOL,
        "frozen_test_accessed": False,
        "ppo_auto_launched": False,
    }
    if any(receipt.get(key) != expected for key, expected in expected_identity.items()):
        raise ValueError("FC-P008 completion receipt identity differs")
    required = {
        "lineage.json",
        "precision.json",
        *(name for names in EXPECTED.values() for name in names),
        *(f"step_receipts/{step}.json" for step in EXPECTED),
    }
    actual = {
        str(path.relative_to(out)): sha256(path)
        for path in sorted(out.rglob("*"))
        if path.is_file() and path.name not in {"receipt.json", "outer.log"}
    }
    if receipt.get("sha256") != actual or not required.issubset(actual):
        raise ValueError("FC-P008 completion artifact table differs")
    receipt_lineage = load(out / "lineage.json")
    persisted_lineage = {
        key: value
        for key, value in lineage.items()
        if key
        not in {
            "posteval_chain_receipt_sha256",
            "formal_evaluation_approval_sha256",
        }
    }
    if receipt_lineage != persisted_lineage or sha256(out / "lineage.json") != sha256(lineage_path):
        raise ValueError("FC-P008 stored lineage differs")
    lineage_with_sha = {**lineage, "lineage_sha256": sha256(lineage_path)}
    for step in EXPECTED:
        validate_step_receipt(
            out / "step_receipts" / f"{step}.json", out, step, lineage_with_sha
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
        raise ValueError("FC-P008 development gate contract differs")
    checkpoint_files(candidate, lineage)


def validate_precision(path: Path) -> None:
    value = load(path)
    if value != {
        "status": PRECISION_STATUS,
        "official_image_id": IMAGE_ID,
        **PRECISION,
    }:
        raise ValueError("FC-P008 formal evaluation precision differs")


def validate_chain_receipt(path: Path, numerical_source: Path) -> str:
    value = load(path)
    root = path.parent.resolve()
    if value.get("status") != CHAIN_STATUS:
        raise ValueError("FC-P008 posteval chain status differs")
    for key in ("git_commit", "git_tree", "numerical_source_commit", "numerical_source_tree"):
        item = value.get(key)
        if not isinstance(item, str) or len(item) != 40:
            raise ValueError(f"FC-P008 posteval chain {key} differs")
    actual = {
        str(item.relative_to(root)): sha256(item)
        for item in sorted(root.rglob("*"))
        if item.is_file() and item != path
    }
    if value.get("sha256") != actual:
        raise ValueError("FC-P008 posteval chain files differ")
    if numerical_source.resolve() != (root / "numerical_source").resolve():
        raise ValueError("FC-P008 numerical source does not belong to chain")
    return sha256(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lineage", type=Path, required=True)
    parser.add_argument("--numerical-source", type=Path, required=True)
    parser.add_argument("--chain-receipt", type=Path, required=True)
    parser.add_argument("--formal-approval-sha256", required=True)
    parser.add_argument("--step", choices=(*EXPECTED, "complete"), required=True)
    parser.add_argument("--profile", choices=("p008", "p009"), default="p008")
    args = parser.parse_args()
    configure_profile(args.profile)
    lineage, _, _ = validate_lineage(args.candidate, args.lineage)
    chain_sha = validate_chain_receipt(args.chain_receipt, args.numerical_source)
    if len(args.formal_approval_sha256) != 64:
        raise ValueError("FC-P008 formal approval SHA differs")
    approval_copy = args.output / "evidence/formal_evaluation_approval.json"
    if not approval_copy.is_file() or sha256(approval_copy) != args.formal_approval_sha256:
        raise ValueError("FC-P008 stored formal approval differs")
    lineage = {
        **lineage,
        "posteval_chain_receipt_sha256": chain_sha,
        "formal_evaluation_approval_sha256": args.formal_approval_sha256,
    }
    validate_precision(args.output / "precision.json")
    if args.step == "complete":
        validate_complete(args.candidate, args.output, args.lineage, lineage)
    else:
        validate_science_step(
            args.repo,
            args.candidate,
            args.output,
            lineage,
            args.step,
            args.numerical_source,
        )
        validate_step_receipt(
            args.output / "step_receipts" / f"{args.step}.json",
            args.output,
            args.step,
            {**lineage, "lineage_sha256": sha256(args.lineage)},
        )
    print(f"{COMPLETE_STATUS}_REUSE_VALID step={args.step}")


if __name__ == "__main__":
    main()
