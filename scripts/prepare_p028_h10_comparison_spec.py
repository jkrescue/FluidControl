#!/usr/bin/env python3
"""Prepare a non-authorizing P028 matched-H10 spec from reviewed small receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PENDING = "P028_MATCHED_H10_COMPARISON_PENDING_LEAD_APPROVAL"
APPROVED = "P028_MATCHED_H10_COMPARISON_EXECUTION_APPROVED"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"missing/linked JSON: {path}")
    value = json.loads(path.read_text())
    require(isinstance(value, dict), "JSON object required")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-p027-spec", type=Path, required=True)
    parser.add_argument("--base-p027-spec-sha256", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--p028-candidate-audit", type=Path, required=True)
    parser.add_argument("--p028-candidate-audit-sha256", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "output must be new")
    require(sha(args.base_p027_spec) == args.base_p027_spec_sha256, "base P027 spec differs")
    require(sha(args.source_manifest) == args.source_manifest_sha256, "source manifest differs")
    require(sha(args.p028_candidate_audit) == args.p028_candidate_audit_sha256, "P028 audit differs")
    spec = read(args.base_p027_spec)
    require(spec.get("status") == "P027_OFFLINE_DIAGNOSTIC_EXECUTION_APPROVED", "base P027 spec status differs")
    require(set(spec.get("candidates", {})) == {"1", "4"}, "base K1/K4 arms differ")
    audit = read(args.p028_candidate_audit)
    require(audit.get("status") == "FC_P028_CANDIDATE_INTEGRITY_VERIFIED_NOT_ADMISSION", "P028 audit status differs")
    require(audit.get("scientific_admission") is False and audit.get("ppo_authorized") is False,
            "P028 audit claims admission/control")
    candidate_map = audit.get("candidate_sha256")
    require(isinstance(candidate_map, dict), "P028 candidate map missing")
    manifest_key = "candidate/dual_model_manifest.json"
    require(manifest_key in candidate_map, "P028 manifest proof missing")
    manifest_path = args.candidate / "dual_model_manifest.json"
    require(manifest_path.is_file() and sha(manifest_path) == candidate_map[manifest_key],
            "P028 candidate manifest differs")
    manifest = read(args.source_manifest)
    require(manifest.get("status") == "P028_H10_SOURCE_CLOSURE_FROZEN", "source closure status differs")
    files = manifest.get("files_sha256")
    require(isinstance(files, dict) and files, "source closure empty")
    root = args.source_manifest.parent.resolve()
    spec["status"] = PENDING
    spec["intended_authorized_status"] = APPROVED
    spec["execution_authorized"] = False
    spec["source_files"] = [
        {"path": str(root / name), "sha256": digest}
        for name, digest in sorted(files.items())
    ]
    spec["p028_candidate"] = {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": candidate_map[manifest_key],
    }
    spec["p028_terminal_proof"] = {
        "path": str(args.p028_candidate_audit.resolve()),
        "sha256": args.p028_candidate_audit_sha256,
        "candidate_sha256": candidate_map,
        "reviewed_by_lead": False,
    }
    spec["comparison_contract"] = {
        "original_p027_k1_k4_arms_preserved": True,
        "new_arm": "P028_flow_plus_byte_identical_P026_K1_aerodynamic",
        "families": ["base", "train8", "train16"],
        "origins": 44,
        "origin": 51,
        "horizon": 10,
        "recorded_actions": True,
        "formal_or_scientific_admission": False,
        "lead_must_review_and_authorize": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": PENDING, "sha256": sha(args.output)}))


if __name__ == "__main__":
    main()
