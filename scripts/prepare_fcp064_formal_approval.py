#!/usr/bin/env python3
"""Build a non-authorizing P064-B full-formal draft from actual proofs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


BASE = "7216214b545fbbd50b2fb5ed866f231039b06b18"
IMAGE = "sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
NUMERICAL_RUNNER_SHA = "03c5862e34a648a1254284d1709bd74c3b995d3a91ae06c4a6f92a945029c0f3"
CONFIG_SHA = "07e55fd11df8030313338cef0344490c3453e515aae9c6b6122e997bad5085d9"
PROTOCOL = ["validation10_H1_H10_H50_H100_stride25_batch4",
            "dynamic6_H1_H10_H50_H100_stride1_batch8", "force_window6",
            "unchanged_development_gate"]
RESOURCE_CONTRACT = {
    "startup_mem_available_gib": 50, "runtime_mem_available_gib": 22,
    "reserved_mem_available_gib": 20, "container_memory_gib": 72,
    "container_memory_swap_gib": 72, "container_pids_limit": 1024,
    "allocator_fraction": 0.06, "deadline_seconds": 10800,
}
FILES = {"result.json", "training_protocol.json", "dual_model_manifest.json",
         "flow/FNO.0.0.mdlus", "flow/checkpoint.0.0.pt",
         "aerodynamic/FNO.0.1.mdlus", "aerodynamic/checkpoint.0.1.pt"}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"regular file required: {path}")
    value = json.loads(path.read_text())
    require(isinstance(value, dict), "JSON object required")
    return value


def relative(repo, path):
    path = Path(path).resolve()
    require(path.is_relative_to(repo), "path escapes repo")
    return str(path.relative_to(repo))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "candidate", "candidate-audit", "official-reload", "formal-runner", "resource-adapter",
                 "numerical-runner", "source-receipt", "freeze-receipt", "runtime-manifest",
                 "draft-output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--training-unit", required=True)
    parser.add_argument("--training-invocation", required=True)
    parser.add_argument("--output-relative-directory", required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    require(re.fullmatch(r"[0-9a-f]{32}", args.training_invocation), "training invocation")
    require(not args.draft_output.exists(), "draft exists")
    require(not Path(args.output_relative_directory).is_absolute(), "relative output required")
    runner_sha = sha(args.formal_runner)
    require(sha(args.numerical_runner) == NUMERICAL_RUNNER_SHA, "numerical runner differs")
    script_dir = args.formal_runner.resolve().parent
    orchestration = {
        "scripts/run_fcp026_posteval.py": sha(args.numerical_runner),
        "scripts/run_fcp028_posteval.py": sha(script_dir / "run_fcp028_posteval.py"),
        "scripts/flow_repair_profiles.py": sha(script_dir / "flow_repair_profiles.py"),
        "scripts/run_fcp064_posteval.py": runner_sha,
    }
    source = read(args.source_receipt)
    freeze = read(args.freeze_receipt)
    runtime = read(args.runtime_manifest)
    require(source.get("status") == "FC_P026_FORMAL_SOURCE_CHAIN_FROZEN"
            and source.get("numerical_base_commit") == BASE
            and source.get("training_config_sha256") == CONFIG_SHA, "source chain differs")
    require(freeze.get("status") == "FC_P064_FORMAL_SOURCE_FREEZE_COMPLETE"
            and freeze.get("final_numerical_files_sha256") == source.get("final_files_sha256"),
            "formal freeze differs")
    require(isinstance(runtime, dict) and len(runtime) == 10, "CPU runtime manifest differs")
    audit = read(args.candidate_audit)
    reload = read(args.official_reload)
    require(audit.get("status") == "FC_P064_ARM_B_CANDIDATE_INTEGRITY_VERIFIED_NOT_ADMISSION"
            and audit.get("actual_optimizer_steps") == 32
            and audit.get("actual_training_windows") == 256
            and audit.get("training_unit") == args.training_unit
            and audit.get("training_invocation") == args.training_invocation,
            "candidate audit differs")
    require(reload.get("status") == "FC_P064_ARM_B_OFFICIAL_CPU_DUAL_RELOAD_VERIFIED_NOT_ADMISSION"
            and reload.get("candidate_audit_sha256") == sha(args.candidate_audit)
            and reload.get("official_dual_reload_verified") is True
            and reload.get("forward_performed") is False
            and reload.get("optimizer_created") is False
            and reload.get("model_saved") is False and reload.get("gpu_used") is False,
            "official reload differs")
    mapping = audit.get("candidate_sha256")
    require(set(mapping or {}) == {"candidate/" + name for name in FILES}
            and mapping == reload.get("candidate_sha256"), "candidate proof map differs")
    candidate = {name.removeprefix("candidate/"): digest for name, digest in mapping.items()}
    for name, digest in candidate.items():
        require(sha(args.candidate / name) == digest, "candidate bytes differ")
    manifest = read(args.candidate / "dual_model_manifest.json")
    require(manifest.get("kind") == "FC_P064_ARM_B_CONTROLLED_AERO_FORCE_FNO"
            and manifest.get("arm") == "B", "candidate identity differs")
    require(reload.get("runtime_source_sha256") == runtime
            and reload.get("runtime_source_manifest_sha256") == sha(args.runtime_manifest),
            "reload runtime source differs")
    require(audit.get("tensor_sha256") == reload.get("tensor_sha256"), "tensor proof differs")
    payload = {
        "status": "FC_P064_FORMAL_EVALUATION_APPROVAL_PENDING_LEAD_REVIEW",
        "intended_authorized_status": "FC_P064_APPROVED_ORIGINAL_FORMAL_EVALUATION",
        "formal_evaluation_authorized": False, "reviewed_by_lead": False,
        "candidate_relative_directory": relative(repo, args.candidate),
        "output_relative_directory": args.output_relative_directory,
        "training_unit": args.training_unit, "training_invocation": args.training_invocation,
        "frozen_test_accessed": False, "ppo_auto_launch": False, "protocol": PROTOCOL,
        "numerical_base_commit": BASE, "official_image_id": IMAGE,
        "runner_sha256": runner_sha, "numerical_runner_sha256": NUMERICAL_RUNNER_SHA,
        "resource_adapter_sha256": sha(args.resource_adapter),
        "resource_contract": RESOURCE_CONTRACT,
        "orchestration_sha256": orchestration,
        "training_config_sha256": CONFIG_SHA,
        "parent_manifest_sha256": manifest["parent_manifest_sha256"],
        "source_chain_receipt": {"path": relative(repo, args.source_receipt), "sha256": sha(args.source_receipt)},
        "formal_source_freeze_receipt": {"path": relative(repo, args.freeze_receipt), "sha256": sha(args.freeze_receipt)},
        "cpu_reload_source_manifest": {"path": relative(repo, args.runtime_manifest), "sha256": sha(args.runtime_manifest)},
        "reviewed_overlay_sha256": source["overlay_sha256"],
        "source_sha256": source["final_files_sha256"], "candidate_sha256": candidate,
        "independent_terminal_audit": {"path": relative(repo, args.candidate_audit),
            "sha256": sha(args.candidate_audit), "reviewed_by_lead": False},
        "official_dual_reload": {"path": relative(repo, args.official_reload),
            "sha256": sha(args.official_reload), "reviewed_by_lead": False},
        "generation_constraints": {"generator_authorizes_execution": False,
            "lead_must_review_both_proofs": True, "ppo_authorized": False},
    }
    args.draft_output.parent.mkdir(parents=True, exist_ok=True)
    args.draft_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "draft": str(args.draft_output),
                      "sha256": sha(args.draft_output)}))


if __name__ == "__main__":
    main()
