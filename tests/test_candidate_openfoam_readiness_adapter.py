from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/adapt_candidate_ppo_openfoam_readiness.py"
SPEC = importlib.util.spec_from_file_location("candidate_openfoam_adapter", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "candidate"
    policy = root / "checkpoints/ppo_00002048.zip"
    vec = root / "vecnormalize.pkl"
    policy.parent.mkdir(parents=True)
    policy.write_bytes(b"policy")
    vec.write_bytes(b"identity-vec")
    checkpoint = "a" * 64
    contract = {
        "candidate_kind": "dynamic_paired_interleaved_lambda10",
        "checkpoint_sha256": checkpoint,
        "observation_dimensions": 69,
        "max_abs_omega": 0.75,
        "max_delta_omega": 0.1,
        "vecnormalize": {
            "norm_obs": False,
            "norm_reward": False,
            "reason": "preserve existing canonical PPO numerical contract",
        },
    }
    readiness = {
        "status": MODULE.READY,
        "checkpoint_sha256": checkpoint,
        "dev30_full40_promotion_lineage": {"full40_manifest_sha256": "m"},
    }
    approved = root / "approved.json"
    candidate_identity = {
        "candidate_kind": contract["candidate_kind"],
        "checkpoint_sha256": checkpoint,
        "checkpoint_state_sha256": "b" * 64,
        "resolved_config_sha256": "c" * 64,
    }
    contract.update(
        checkpoint_state_sha256=candidate_identity["checkpoint_state_sha256"],
        resolved_config_sha256=candidate_identity["resolved_config_sha256"],
    )
    dump(
        approved,
        {
            "status": MODULE.APPROVED,
            "command_contract": contract,
            "canonical_preflight": readiness,
            "candidate_readiness": {
                "status": "CANDIDATE_PPO_CPU_DRY_RUN_READY",
                "candidate_identity": candidate_identity,
                "blockers": [],
                "cpu_only": True,
                "training_executed": False,
                "policy_created": False,
                "ppo_execution_authorized": False,
                "frozen_test_directory_enumerated_or_opened": False,
            },
            "training_executed": False,
        },
    )
    audit = root / "audit.json"
    dump(
        audit,
        {
            "status": MODULE.COMPLETE,
            "physicsnemo_checkpoint_sha256": checkpoint,
            "vecnormalize_contract": MODULE.IDENTITY_VEC,
            "vecnormalize_sha256": MODULE.sha256(vec),
            "iterations": [
                {
                    "checkpoint": "/workspace/artifacts/run/checkpoints/ppo_00002048.zip",
                    "checkpoint_sha256": MODULE.sha256(policy),
                }
            ],
            "frozen_test_accessed": False,
        },
    )
    binding = root / "candidate_binding_receipt.json"
    dump(
        binding,
        {
            "status": MODULE.BOUND,
            "candidate_readiness_sha256": MODULE.sha256(approved),
            "command_contract": contract,
            "canonical_audit_sha256": MODULE.sha256(audit),
            "final_policy": "checkpoints/ppo_00002048.zip",
            "final_policy_sha256": MODULE.sha256(policy),
            "vecnormalize": "vecnormalize.pkl",
            "vecnormalize_sha256": MODULE.sha256(vec),
            "training_executed": True,
            "evaluated_on_surrogate_only": True,
            "real_cfd_validation_complete": False,
            "frozen_test_accessed": False,
        },
    )
    return {
        "candidate_root": root,
        "approved_preflight_path": approved,
        "binding_receipt_path": binding,
        "canonical_audit_path": audit,
        "policy_path": policy,
        "vecnormalize_path": vec,
    }


def test_exports_old_entry_contract_without_changing_science(tmp_path: Path) -> None:
    values = fixture(tmp_path)
    readiness, audit = MODULE.adapt(**values)
    assert readiness["status"] == MODULE.READY
    evidence = readiness["candidate_openfoam_compatibility"]
    assert evidence["status"] == "CANDIDATE_PPO_OPENFOAM_COMPATIBILITY_PASS"
    assert evidence["vecnormalize_contract"] == MODULE.IDENTITY_VEC
    assert audit["iterations"][-1]["checkpoint"] == str(values["policy_path"].resolve())
    assert audit["candidate_openfoam_compatibility"]["scientific_values_changed"] is False


@pytest.mark.parametrize(
    ("target", "mutate", "message"),
    [
        (
            "approved_preflight_path",
            lambda value: value["command_contract"]["vecnormalize"].update(norm_obs=True),
            "not identity",
        ),
        (
            "binding_receipt_path",
            lambda value: value.update(final_policy_sha256="0" * 64),
            "policy binding differs",
        ),
        (
            "canonical_audit_path",
            lambda value: value.update(physicsnemo_checkpoint_sha256="b" * 64),
            "checkpoint lineage differs",
        ),
    ],
)
def test_rejects_tampered_contract(
    tmp_path: Path, target: str, mutate, message: str
) -> None:
    values = fixture(tmp_path)
    path = values[target]
    payload = MODULE.load(path)
    mutate(payload)
    dump(path, payload)
    # When a source changes, repairing an outer hash must still expose the
    # semantic mismatch rather than making the adapter accept it.
    binding = MODULE.load(values["binding_receipt_path"])
    if target == "approved_preflight_path":
        binding["candidate_readiness_sha256"] = MODULE.sha256(path)
        binding["command_contract"] = payload["command_contract"]
    elif target == "canonical_audit_path":
        binding["canonical_audit_sha256"] = MODULE.sha256(path)
    dump(values["binding_receipt_path"], binding)
    with pytest.raises(ValueError, match=message):
        MODULE.adapt(**values)


def test_rejects_nonfinal_policy_path_even_with_same_name(tmp_path: Path) -> None:
    values = fixture(tmp_path)
    audit = MODULE.load(values["canonical_audit_path"])
    audit["iterations"][-1]["checkpoint"] = "/workspace/wrong/ppo_00002048.zip"
    dump(values["canonical_audit_path"], audit)
    binding = MODULE.load(values["binding_receipt_path"])
    binding["canonical_audit_sha256"] = MODULE.sha256(values["canonical_audit_path"])
    dump(values["binding_receipt_path"], binding)
    with pytest.raises(ValueError, match="policy binding differs"):
        MODULE.adapt(**values)


def test_rejects_tampered_candidate_readiness_with_repaired_outer_hash(
    tmp_path: Path,
) -> None:
    values = fixture(tmp_path)
    approved = MODULE.load(values["approved_preflight_path"])
    approved["candidate_readiness"]["candidate_identity"][
        "checkpoint_state_sha256"
    ] = "d" * 64
    dump(values["approved_preflight_path"], approved)
    binding = MODULE.load(values["binding_receipt_path"])
    binding["candidate_readiness_sha256"] = MODULE.sha256(
        values["approved_preflight_path"]
    )
    dump(values["binding_receipt_path"], binding)
    with pytest.raises(ValueError, match="checkpoint_state_sha256 differs"):
        MODULE.adapt(**values)


def test_rejects_approved_preflight_that_claims_training(tmp_path: Path) -> None:
    values = fixture(tmp_path)
    approved = MODULE.load(values["approved_preflight_path"])
    approved["training_executed"] = True
    dump(values["approved_preflight_path"], approved)
    binding = MODULE.load(values["binding_receipt_path"])
    binding["candidate_readiness_sha256"] = MODULE.sha256(
        values["approved_preflight_path"]
    )
    dump(values["binding_receipt_path"], binding)
    with pytest.raises(ValueError, match="approved preflight status/schema"):
        MODULE.adapt(**values)
