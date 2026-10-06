#!/usr/bin/env python3
"""Prepare a non-authorizing FC-P028 formal approval draft from terminal proofs.

This utility reads only small JSON receipts and source files.  It does not load
checkpoints or HDF data and can never authorize or execute formal evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


PENDING_STATUS = "FC_P028_FORMAL_EVALUATION_APPROVAL_PENDING_LEAD_REVIEW"
FINAL_STATUS = "FC_P028_APPROVED_ORIGINAL_FORMAL_EVALUATION"
BASE = "7216214b545fbbd50b2fb5ed866f231039b06b18"
IMAGE = "sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
RUNNER_SHA = "7ba31bc02e892b860a2ec9edd9e7187661bedc36c7437c44aa7a078ed221bbd4"
NUMERICAL_RUNNER_SHA = "03c5862e34a648a1254284d1709bd74c3b995d3a91ae06c4a6f92a945029c0f3"
SOURCE_RECEIPT_SHA = "e4645a4d49359fabe22c6040de0d0ab76667b138f9879d3a24487dfc2a782152"
FREEZE_RECEIPT_SHA = "fb5fd1ef87a09188d78453d0c5f93e49cf1a795dc7fa9fcee5fd77bf14cc910d"
RUNTIME_MANIFEST_SHA = "2899934d3f822749e1821088ae8dcb0e369e488fa8e170ffa03143219d26ad25"
CONFIG_SHA = "07e55fd11df8030313338cef0344490c3453e515aae9c6b6122e997bad5085d9"
PROTOCOL = [
    "validation10_H1_H10_H50_H100_stride25_batch4",
    "dynamic6_H1_H10_H50_H100_stride1_batch8",
    "force_window6",
    "unchanged_development_gate",
]
FILES = {
    "result.json",
    "training_protocol.json",
    "dual_model_manifest.json",
    "flow/FNO.0.1.mdlus",
    "flow/checkpoint.0.1.pt",
    "aerodynamic/FNO.0.1.mdlus",
    "aerodynamic/checkpoint.0.1.pt",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict:
    require(path.is_file() and not path.is_symlink(), f"missing/linked JSON: {path}")
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def relative(repo: Path, path: Path) -> str:
    resolved = path.resolve()
    require(resolved.is_relative_to(repo), f"path escapes repo: {path}")
    return str(resolved.relative_to(repo))


def proof(repo: Path, path: Path, expected_status: str) -> tuple[dict, dict]:
    payload = read_json(path)
    require(payload.get("status") == expected_status, "terminal proof status differs")
    require(payload.get("scientific_admission") is False, "proof claims admission")
    require(payload.get("ppo_authorized") is False, "proof authorizes PPO")
    mapping = payload.get("candidate_sha256")
    require(isinstance(mapping, dict), "candidate proof map missing")
    expected = {"candidate/" + name for name in FILES}
    require(set(mapping) == expected, "candidate proof map differs")
    require(
        all(isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) for v in mapping.values()),
        "candidate SHA is invalid",
    )
    return payload, {
        "path": relative(repo, path),
        "sha256": sha(path),
        "reviewed_by_lead": False,
    }


def build(args: argparse.Namespace, experiment="FC-P028") -> dict:
    require(experiment in ("FC-P028", "FC-P029"), "unsupported formal profile")
    prefix = experiment.replace("-", "_")
    pins = dict(runner=RUNNER_SHA, source=SOURCE_RECEIPT_SHA,
                freeze=FREEZE_RECEIPT_SHA, runtime=RUNTIME_MANIFEST_SHA)
    if experiment == "FC-P029":
        pins = {key: getattr(args, key + "_sha256") for key in pins}
        require(all(isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v)
                    for v in pins.values()), "actual P029 source pins required")
    repo = args.repo.resolve()
    require(repo.is_dir(), "repo missing")
    runner = args.runner.resolve()
    numerical_runner = args.numerical_runner.resolve()
    require(sha(runner) == pins["runner"], "flow repair runner differs")
    require(sha(numerical_runner) == NUMERICAL_RUNNER_SHA, "numerical runner differs")

    source_receipt = args.source_receipt.resolve()
    freeze_receipt = args.freeze_receipt.resolve()
    runtime_manifest = args.runtime_manifest.resolve()
    require(sha(source_receipt) == pins["source"], "source-chain receipt differs")
    require(sha(freeze_receipt) == pins["freeze"], "formal freeze receipt differs")
    require(sha(runtime_manifest) == pins["runtime"], "CPU runtime manifest differs")
    source = read_json(source_receipt)
    freeze = read_json(freeze_receipt)
    require(source.get("status") == "FC_P026_FORMAL_SOURCE_CHAIN_FROZEN", "source status differs")
    require(freeze.get("status") == prefix + "_FORMAL_SOURCE_FREEZE_COMPLETE", "freeze status differs")
    require(source.get("numerical_base_commit") == BASE, "numerical base differs")
    require(source.get("training_config_sha256") == CONFIG_SHA, "training config differs")
    require(source.get("final_files_sha256") == freeze.get("final_numerical_files_sha256"), "source maps differ")

    audit, audit_binding = proof(
        repo,
        args.candidate_audit,
        prefix + "_CANDIDATE_INTEGRITY_VERIFIED_NOT_ADMISSION",
    )
    reload, reload_binding = proof(
        repo,
        args.official_reload,
        prefix + "_OFFICIAL_CPU_DUAL_RELOAD_VERIFIED_NOT_ADMISSION",
    )
    require(audit["candidate_sha256"] == reload["candidate_sha256"], "terminal proof maps differ")
    require(
        reload.get("candidate_audit_sha256") == audit_binding["sha256"],
        "reload does not bind terminal audit",
    )
    require(reload.get("official_dual_reload_verified") is True, "official reload not verified")
    require(reload.get("forward_performed") is False, "reload performed forward")
    require(reload.get("optimizer_created") is False, "reload created optimizer")
    require(reload.get("model_saved") is False, "reload saved model")
    require(reload.get("gpu_used") is False, "reload used GPU")

    candidate = {name.removeprefix("candidate/"): digest for name, digest in audit["candidate_sha256"].items()}
    manifest = read_json(args.candidate / "dual_model_manifest.json")
    require(sha(args.candidate / "dual_model_manifest.json") == candidate["dual_model_manifest.json"], "manifest differs")
    kind = "FC_P029_CONTROL_AWARE_FLOW_REPAIR" if experiment == "FC-P029" else "FC_P028_FLOW_ROLLOUT_REPAIR"
    require(manifest.get("kind") == kind, "candidate kind differs")
    require(manifest.get("scientific_admission") is False, "candidate claims admission")
    parent = manifest.get("parent_manifest_sha256")
    require(isinstance(parent, str) and re.fullmatch(r"[0-9a-f]{64}", parent), "parent manifest SHA missing")

    extra = {}
    if experiment == "FC-P029":
        require(audit.get("training_unit") == args.training_unit
                and audit.get("training_invocation") == args.training_invocation,
                "actual P029 training identity differs")
        require(audit.get("actual_optimizer_steps") == 171 and audit.get("actual_training_windows") == 1368,
                "P029 training counts differ")
        expected_terminal = dict(LoadState="loaded", ActiveState="active", SubState="exited",
                                 Result="success", ExecMainCode="1", ExecMainStatus="0", MainPID="0",
                                 InvocationID=args.training_invocation)
        require(audit.get("terminal_evidence") == expected_terminal, "P029 terminal evidence differs")
        tensors = audit.get("tensor_sha256")
        require(isinstance(tensors, dict) and set(tensors) == {"flow", "aerodynamic"}
                and all(isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) for v in tensors.values())
                and tensors == reload.get("tensor_sha256"), "P029 reloaded tensor map differs")
        for proof_payload in (audit, reload):
            for key, filename in (("training_protocol_sha256", "training_protocol.json"),
                                  ("dual_manifest_sha256", "dual_model_manifest.json"),
                                  ("candidate_result_sha256", "result.json")):
                require(proof_payload.get(key) == candidate[filename], "P029 terminal file identity differs")
        runtime = read_json(runtime_manifest)
        require(len(runtime) == 12 and reload.get("runtime_source_sha256") == runtime
                and reload.get("runtime_source_manifest_sha256") == pins["runtime"],
                "P029 actual CPU source binding differs")
        require(audit.get("role_loader_sha256") == runtime.get("src/fluid_control/dual_fno.py")
                == source["final_files_sha256"].get("src/fluid_control/dual_fno.py"),
                "P029 audit/reload/formal loader differs")
        orchestration = {}
        for name in ("run_fcp029_posteval.py", "run_fcp028_posteval.py", "flow_repair_profiles.py"):
            path = runner.parent / name
            digest = sha(path)
            require(freeze.get("external_orchestration", {}).get("scripts/" + name) == digest,
                    "P029 actual orchestration source differs")
            orchestration[name] = digest
        extra["orchestration_sha256"] = orchestration

    pending_status = prefix + "_FORMAL_EVALUATION_APPROVAL_PENDING_LEAD_REVIEW"
    final_status = prefix + "_APPROVED_ORIGINAL_FORMAL_EVALUATION"
    return {
        **extra,
        "status": pending_status,
        "intended_authorized_status": final_status,
        "formal_evaluation_authorized": False,
        "reviewed_by_lead": False,
        "candidate_relative_directory": relative(repo, args.candidate),
        "output_relative_directory": args.output_relative_directory,
        "training_unit": args.training_unit,
        "training_invocation": args.training_invocation,
        "frozen_test_accessed": False,
        "ppo_auto_launch": False,
        "protocol": PROTOCOL,
        "numerical_base_commit": BASE,
        "official_image_id": IMAGE,
        "runner_sha256": pins["runner"],
        "numerical_runner_sha256": NUMERICAL_RUNNER_SHA,
        "training_config_sha256": CONFIG_SHA,
        "parent_manifest_sha256": parent,
        "source_chain_receipt": {
            "path": relative(repo, source_receipt),
            "sha256": pins["source"],
        },
        "formal_source_freeze_receipt": {
            "path": relative(repo, freeze_receipt),
            "sha256": pins["freeze"],
        },
        "cpu_reload_source_manifest": {
            "path": relative(repo, runtime_manifest),
            "sha256": pins["runtime"],
        },
        "reviewed_overlay_sha256": source["overlay_sha256"],
        "source_sha256": source["final_files_sha256"],
        "candidate_sha256": candidate,
        "independent_terminal_audit": audit_binding,
        "official_dual_reload": reload_binding,
        "generation_constraints": {
            "generator_authorizes_execution": False,
            "lead_must_independently_review_candidate_and_both_proofs": True,
            "lead_must_set_status_to": final_status,
            "lead_must_set_formal_evaluation_authorized_true": True,
            "lead_must_set_both_proof_reviewed_by_lead_true": True,
            "ppo_authorized": False,
        },
    }


def parse_args(experiment="FC-P028") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-audit", type=Path, required=True)
    parser.add_argument("--official-reload", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--numerical-runner", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--training-unit", required=True)
    parser.add_argument("--training-invocation", required=True)
    parser.add_argument("--output-relative-directory", required=True)
    parser.add_argument("--draft-output", type=Path, required=True)
    if experiment == "FC-P029":
        for key in ("runner", "source", "freeze", "runtime"):
            parser.add_argument("--" + key + "-sha256", required=True)
    return parser.parse_args()


def main(experiment="FC-P028") -> None:
    args = parse_args(experiment)
    unit_prefix = experiment.lower().replace("-", "")
    require(re.fullmatch(r"fluid-control-" + unit_prefix + r"-[a-zA-Z0-9-]+\.service", args.training_unit) is not None, "training unit differs")
    require(re.fullmatch(r"[0-9a-f]{32}", args.training_invocation) is not None, "training invocation differs")
    require(not Path(args.output_relative_directory).is_absolute(), "formal output must be relative")
    require(not args.draft_output.exists(), "draft output must be new")
    payload = build(args, experiment)
    args.draft_output.parent.mkdir(parents=True, exist_ok=True)
    args.draft_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "draft": str(args.draft_output), "sha256": sha(args.draft_output)}))


if __name__ == "__main__":
    main()
