#!/usr/bin/env python3
"""Export candidate PPO evidence for the existing real-CFD feedback entry.

This is an evidence adapter only.  It neither trains a policy nor runs CFD.  A
candidate can be exported only when its approved preflight, final PPO audit,
identity VecNormalize artifact, and binding receipt agree byte-for-byte.  The
export retains the canonical readiness object expected by the existing
OpenFOAM runner and embeds the additional candidate evidence in the same
SHA-bound file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

READY = "FULL40_CANONICAL_PPO_EXECUTION_READY"
APPROVED = "CANDIDATE_CANONICAL_PPO_DRY_RUN_READY"
BOUND = "CANDIDATE_CANONICAL_PPO_SURROGATE_RUN_BOUND"
COMPLETE = "FULL40_CANONICAL_PPO_SURROGATE_RUN_COMPLETE"
IDENTITY_VEC = (
    "identity: norm_obs=false, norm_reward=false; preserves legacy PPO numerics"
)


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


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")


def adapt(
    *,
    candidate_root: Path,
    approved_preflight_path: Path,
    binding_receipt_path: Path,
    canonical_audit_path: Path,
    policy_path: Path,
    vecnormalize_path: Path,
) -> tuple[dict, dict]:
    """Validate evidence and return compatible readiness and audit documents."""
    candidate_root = candidate_root.resolve()
    paths = {
        "approved candidate preflight": approved_preflight_path.resolve(),
        "candidate binding receipt": binding_receipt_path.resolve(),
        "canonical PPO audit": canonical_audit_path.resolve(),
        "final PPO policy": policy_path.resolve(),
        "identity VecNormalize": vecnormalize_path.resolve(),
    }
    if not candidate_root.is_dir():
        raise FileNotFoundError(f"candidate root: {candidate_root}")
    for label, path in paths.items():
        _require_file(path, label)
        try:
            path.relative_to(candidate_root)
        except ValueError as error:
            raise ValueError(f"{label} escapes candidate root") from error

    approved = load(paths["approved candidate preflight"])
    binding = load(paths["candidate binding receipt"])
    audit = load(paths["canonical PPO audit"])
    readiness = approved.get("canonical_preflight")
    contract = approved.get("command_contract")
    candidate_readiness = approved.get("candidate_readiness")
    if (
        approved.get("status") != APPROVED
        or approved.get("training_executed") is not False
        or not isinstance(readiness, dict)
    ):
        raise ValueError("candidate approved preflight status/schema differs")
    if readiness.get("status") != READY:
        raise ValueError("nested canonical readiness did not pass")
    if not isinstance(contract, dict):
        raise TypeError("candidate command contract is absent")
    if not isinstance(candidate_readiness, dict):
        raise TypeError("candidate readiness evidence is absent")
    candidate_identity = candidate_readiness.get("candidate_identity")
    if (
        candidate_readiness.get("status") != "CANDIDATE_PPO_CPU_DRY_RUN_READY"
        or not isinstance(candidate_identity, dict)
        or candidate_readiness.get("blockers") != []
        or candidate_readiness.get("cpu_only") is not True
        or candidate_readiness.get("training_executed") is not False
        or candidate_readiness.get("policy_created") is not False
        or candidate_readiness.get("ppo_execution_authorized") is not False
        or candidate_readiness.get("frozen_test_directory_enumerated_or_opened")
        is not False
    ):
        raise ValueError("candidate readiness scope/status differs")
    for key in (
        "candidate_kind",
        "checkpoint_sha256",
        "checkpoint_state_sha256",
        "resolved_config_sha256",
    ):
        if candidate_identity.get(key) != contract.get(key):
            raise ValueError(f"candidate readiness {key} differs from command contract")
    if contract.get("vecnormalize") != {
        "norm_obs": False,
        "norm_reward": False,
        "reason": "preserve existing canonical PPO numerical contract",
    }:
        raise ValueError("candidate VecNormalize contract is not identity")
    if contract.get("observation_dimensions") != 69:
        raise ValueError("candidate observation dimension differs")
    if contract.get("max_abs_omega") != 0.75 or contract.get("max_delta_omega") != 0.1:
        raise ValueError("candidate action contract differs")

    if binding.get("status") != BOUND:
        raise ValueError("candidate PPO binding receipt did not pass")
    if binding.get("candidate_readiness_sha256") != sha256(
        paths["approved candidate preflight"]
    ):
        raise ValueError("binding receipt does not bind the approved preflight")
    if binding.get("command_contract") != contract:
        raise ValueError("binding and approved command contracts differ")
    if binding.get("canonical_audit_sha256") != sha256(paths["canonical PPO audit"]):
        raise ValueError("binding receipt does not bind the canonical audit")
    if binding.get("training_executed") is not True:
        raise ValueError("candidate PPO training is not complete")
    if binding.get("evaluated_on_surrogate_only") is not True:
        raise ValueError("candidate PPO scope differs")
    if binding.get("real_cfd_validation_complete") is not False:
        raise ValueError("candidate receipt improperly claims real-CFD validation")
    if binding.get("frozen_test_accessed") is not False:
        raise ValueError("candidate receipt frozen-test contract differs")

    if audit.get("status") != COMPLETE or audit.get("frozen_test_accessed") is not False:
        raise ValueError("canonical PPO audit status/split contract differs")
    checkpoint_sha = contract.get("checkpoint_sha256")
    if (
        not isinstance(checkpoint_sha, str)
        or len(checkpoint_sha) != 64
        or readiness.get("checkpoint_sha256") != checkpoint_sha
        or audit.get("physicsnemo_checkpoint_sha256") != checkpoint_sha
    ):
        raise ValueError("candidate FNO checkpoint lineage differs")
    if audit.get("vecnormalize_contract") != IDENTITY_VEC:
        raise ValueError("canonical audit VecNormalize contract is not identity")

    vec_sha = sha256(paths["identity VecNormalize"])
    if (
        audit.get("vecnormalize_sha256") != vec_sha
        or binding.get("vecnormalize_sha256") != vec_sha
        or binding.get("vecnormalize")
        != str(paths["identity VecNormalize"].relative_to(candidate_root))
    ):
        raise ValueError("identity VecNormalize artifact binding differs")

    policy_sha = sha256(paths["final PPO policy"])
    iterations = audit.get("iterations")
    if not isinstance(iterations, list) or not iterations:
        raise ValueError("canonical PPO iterations are absent")
    final = iterations[-1]
    policy_relative = Path(str(binding.get("final_policy", "")))
    audit_checkpoint = Path(str(final.get("checkpoint", "")))
    if (
        policy_relative.is_absolute()
        or ".." in policy_relative.parts
        or tuple(audit_checkpoint.parts[-len(policy_relative.parts) :])
        != policy_relative.parts
        or final.get("checkpoint_sha256") != policy_sha
        or binding.get("final_policy_sha256") != policy_sha
        or policy_relative
        != paths["final PPO policy"].relative_to(candidate_root)
    ):
        raise ValueError("final candidate policy binding differs")

    exported = dict(readiness)
    exported["candidate_openfoam_compatibility"] = {
        "status": "CANDIDATE_PPO_OPENFOAM_COMPATIBILITY_PASS",
        "candidate_kind": contract.get("candidate_kind"),
        "approved_preflight_sha256": sha256(paths["approved candidate preflight"]),
        "binding_receipt_sha256": sha256(paths["candidate binding receipt"]),
        "canonical_audit_sha256": sha256(paths["canonical PPO audit"]),
        "final_policy_sha256": policy_sha,
        "fno_checkpoint_sha256": checkpoint_sha,
        "vecnormalize_sha256": vec_sha,
        "vecnormalize_contract": IDENTITY_VEC,
        "observation_contract": (
            "32*(u,v),front_cd,front_cl,rear_cd,rear_cl,applied_omega"
        ),
        "action_contract": {
            "max_abs_omega": 0.75,
            "max_delta_omega_per_0.1_D_over_U": 0.1,
        },
        "execution_performed": False,
        "real_cfd_validation_complete": False,
        "frozen_test_accessed": False,
    }
    # The canonical trainer runs in /workspace, so its immutable audit records
    # the container path.  The historical CFD entry compares the final path to
    # the host policy path.  Export a derived copy with only that location
    # translated; hashes, metrics, and scientific status remain untouched.
    adapted_audit = json.loads(json.dumps(audit))
    adapted_audit["iterations"][-1]["checkpoint"] = str(paths["final PPO policy"])
    adapted_audit["candidate_openfoam_compatibility"] = {
        "source_canonical_audit_sha256": sha256(paths["canonical PPO audit"]),
        "translation": "final policy container path to byte-identical host path",
        "translated_policy_sha256": policy_sha,
        "scientific_values_changed": False,
    }
    return exported, adapted_audit


def write_exclusive(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--approved-preflight", type=Path, required=True)
    parser.add_argument("--binding-receipt", type=Path, required=True)
    parser.add_argument("--canonical-audit", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--vecnormalize", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    readiness, audit = adapt(
        candidate_root=args.candidate_root,
        approved_preflight_path=args.approved_preflight,
        binding_receipt_path=args.binding_receipt,
        canonical_audit_path=args.canonical_audit,
        policy_path=args.policy,
        vecnormalize_path=args.vecnormalize,
    )
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    readiness_path = output / "ppo_readiness.json"
    audit_path = output / "ppo_audit.json"
    write_exclusive(readiness_path, readiness)
    write_exclusive(audit_path, audit)
    receipt = {
        "status": "CANDIDATE_PPO_OPENFOAM_COMPATIBILITY_PASS",
        "ppo_readiness": readiness_path.name,
        "ppo_readiness_sha256": sha256(readiness_path),
        "ppo_audit": audit_path.name,
        "ppo_audit_sha256": sha256(audit_path),
        "execution_performed": False,
        "real_cfd_validation_complete": False,
        "frozen_test_accessed": False,
    }
    write_exclusive(output / "receipt.json", receipt)
    print(receipt["status"])


if __name__ == "__main__":
    main()
