"""Project provenance gate for a dual-FNO controller; not a new physics model.

This supplements, never replaces, canonical endpoint/window/dynamic admission.
It does not create a policy or prove real-CFD control performance.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from .dual_fno import sha256, validate_dual_fno_manifest, validate_dual_runtime_files

DEVELOPMENT_AUDITOR_SHA = "ca6da0afdce5859be1c060eb48ba2cdd1ccc5ee3aeb2570d9c9b53067d5bc412"
PROTOCOL = [
    "validation10_H1_H10_H50_H100_stride25_batch4",
    "dynamic6_H1_H10_H50_H100_stride1_batch8",
    "force_window6", "unchanged_development_gate",
]


def verify_receipt_files(root: Path, files: dict) -> None:
    if not isinstance(files, dict) or not files:
        raise ValueError("post-evaluation file table is absent")
    root = root.resolve()
    required = {
        "lineage.json", "precision.json", "development_gate.json",
        "force_window/result.json", "validation10/evaluation.json",
        "validation10/segments.json", "validation10/endpoint_gate.json",
        "validation10/diagnostic.json", "dynamic6/diagnostic.json",
        "evidence/formal_evaluation_approval.json",
        "dynamic6/evaluation.json", "dynamic6/segments.json",
        "step_receipts/validation10.json", "step_receipts/dynamic6.json",
        "step_receipts/force_window.json",
    }
    if not required.issubset(files):
        raise ValueError("post-evaluation evidence is incomplete")
    for relative, digest in files.items():
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise ValueError("invalid post-evaluation file identity")
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file() or sha256(path) != digest:
            raise ValueError("post-evaluation file changed or escapes its root")


def check_receipt_identity(receipt: dict, expected: dict) -> None:
    required = {
        "status": "FC_P013_POSTEVAL_COMPLETE",
        "candidate_kind": "fcp013_independent_force_dual_fno",
        "checkpoint_epoch": 1, "protocol": PROTOCOL,
        "frozen_test_accessed": False, "ppo_auto_launched": False,
        **expected,
    }
    if any(receipt.get(key) != value for key, value in required.items()):
        raise ValueError("dual post-evaluation receipt identity differs")


def verify_dual_control_binding(
    *, manifest_path: Path, expected_manifest_sha256: str,
    training_config: Path, normalization_path: Path,
    checkpoint_dir: Path, expected_checkpoint_sha256: str,
    posteval_receipt: Path, expected_posteval_receipt_sha256: str,
    development_auditor: Path,
) -> dict:
    """Require exact complete dual evidence before canonical PPO can use it."""
    identity = validate_dual_fno_manifest(manifest_path, expected_sha256=expected_manifest_sha256)
    validate_dual_runtime_files(identity, config_path=training_config,
                               normalization_path=normalization_path)
    if identity.aerodynamic.directory != checkpoint_dir.resolve() or identity.aerodynamic.model_sha256 != expected_checkpoint_sha256:
        raise ValueError("canonical primary checkpoint is not this dual system")
    if sha256(posteval_receipt) != expected_posteval_receipt_sha256:
        raise ValueError("expected complete post-evaluation receipt differs")
    receipt = json.loads(posteval_receipt.read_text())
    expected = {
        "checkpoint_sha256": identity.aerodynamic.model_sha256,
        "checkpoint_state_sha256": identity.aerodynamic.state_sha256,
        "dual_manifest_sha256": identity.manifest_sha256,
        "flow_model_sha256": identity.flow.model_sha256,
        "flow_state_sha256": identity.flow.state_sha256,
    }
    check_receipt_identity(receipt, expected)
    root = posteval_receipt.parent
    verify_receipt_files(root, receipt.get("sha256"))
    for name in ("validation10", "dynamic6", "force_window"):
        step = json.loads((root / "step_receipts" / f"{name}.json").read_text())
        if step.get("status") != "FC_P013_POSTEVAL_STEP_COMPLETE" or step.get("step") != name or any(step.get(k) != v for k, v in expected.items()):
            raise ValueError("step receipt does not bind the same complete dual system")
    if sha256(development_auditor) != DEVELOPMENT_AUDITOR_SHA:
        raise ValueError("original numerical development auditor changed")
    spec = importlib.util.spec_from_file_location("dual_control_original_development_audit", development_auditor)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load original numerical gate")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    recomputed = module.audit(root / "force_window/result.json", expected_checkpoint_sha256)
    if recomputed != json.loads((root / "development_gate.json").read_text()):
        raise ValueError("stored development result differs from original recomputation")
    if recomputed.get("status") != "DYNAMIC_FNO_DEVELOPMENT_ADMISSION_PASS":
        raise ValueError("dual surrogate has not passed original development admission")
    return {
        "status": "DUAL_CONTROL_IDENTITY_VERIFIED_NOT_CONTROL_SUCCESS",
        **expected, "posteval_receipt_sha256": expected_posteval_receipt_sha256,
        "training_config_sha256": identity.payload["config_sha256"],
        "normalization_sha256": identity.payload["normalization_sha256"],
        "canonical_endpoint_window_dynamic_gates_still_required": True,
        "policy_trained": False, "real_cfd_control_validated": False,
    }
