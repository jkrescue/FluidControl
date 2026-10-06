#!/usr/bin/env python3
"""Thin FC-P028 identity profile over the unchanged P026 numerical formal runner.

This module does not authorize execution.  A future approval must bind the
terminal P028 training evidence, the reviewed source closure, and this runner.
All numerical commands and container/resource handling are delegated to the
reviewed P026 runner; only candidate-role identity and receipt labels differ.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


P028_STATUS = "FC_P028_DUAL_FNO_MANIFEST_VERIFIED"
P028_KIND = "FC_P028_FLOW_ROLLOUT_REPAIR"
P028_FLOW_KIND = "FC_P028_FLOW_ROLLOUT_CHECKPOINT"
P026_AERO_KIND = "FC_P026_K1_HISTORY_AERODYNAMIC_CHECKPOINT"
FORMAL_APPROVAL = "FC_P028_APPROVED_ORIGINAL_FORMAL_EVALUATION"
FORMAL_COMPLETE = "FC_P028_ORIGINAL_FORMAL_COMPLETE_NOT_ADMISSION"
TRAINING_COMPLETE = "FC_P028_TRAINING_COMPLETE_NOT_ADMISSION"
NUMERICAL_RUNNER_SHA256 = "03c5862e34a648a1254284d1709bd74c3b995d3a91ae06c4a6f92a945029c0f3"
FILES = {
    "result.json",
    "training_protocol.json",
    "dual_model_manifest.json",
    "flow/FNO.0.1.mdlus",
    "flow/checkpoint.0.1.pt",
    "aerodynamic/FNO.0.1.mdlus",
    "aerodynamic/checkpoint.0.1.pt",
}


def _base_runner(path: Path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    if path.is_symlink() or digest != NUMERICAL_RUNNER_SHA256:
        raise ValueError("external numerical runner bytes differ before import")
    spec = importlib.util.spec_from_file_location("_fcp028_numerical_base", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def candidate_contract(base, candidate: Path, approval: dict) -> dict:
    """Bind the trained-flow/frozen-aero P028 role pair without admission."""
    files = approval.get("candidate_sha256")
    base.require(isinstance(files, dict) and set(files) == FILES, "candidate proof incomplete")
    for name, digest in files.items():
        base.bound(candidate, name, digest)
    result = json.loads((candidate / "result.json").read_text())
    protocol = json.loads((candidate / "training_protocol.json").read_text())
    manifest = json.loads((candidate / "dual_model_manifest.json").read_text())
    protocol_sha = files["training_protocol.json"]
    base.fields(
        result,
        {
            "status": TRAINING_COMPLETE,
            "mode": "train",
            "training_windows": 1368,
            "optimizer_steps": 171,
            "training_protocol_sha256": protocol_sha,
            "scientific_admission": False,
            "validation_accessed": False,
            "frozen_test_accessed": False,
            "official_fresh_reload_verified": True,
        },
    )
    base.fields(
        protocol,
        {
            "experiment": "FC-P028",
            "optimized_role": "flow",
            "fixed_role": "aerodynamic",
            "horizon": 10,
            "training_windows": 1368,
            "optimizer_steps": 171,
            "accumulation_windows": 8,
            "learning_rate": 1e-5,
            "objective": "ten_equal_masked_normalized_state_MSE",
            "force_loss": False,
            "terminal_selection": False,
            "future_truth_inputs": False,
        },
    )
    base.fields(
        manifest,
        {
            "status": P028_STATUS,
            "kind": P028_KIND,
            "training_experiment": "FC-P028",
            "training_protocol_sha256": protocol_sha,
            "parent_manifest_sha256": approval["parent_manifest_sha256"],
            "scientific_admission": False,
        },
    )
    base.fields(
        manifest["history_input"],
        {
            "profile": "p026_k1",
            "history_length": 1,
            "flow_input_channels": 6,
            "aerodynamic_input_channels": 6,
            "future_state_inputs": False,
            "future_force_inputs": False,
        },
    )
    base.require(manifest["training_semantics"] == protocol, "training protocol differs")
    base.fields(
        manifest["flow"],
        {
            "role": "flow",
            "frozen": False,
            "checkpoint_epoch": 1,
            "metadata_kind": P028_FLOW_KIND,
            "model_sha256": files["flow/FNO.0.1.mdlus"],
            "state_sha256": files["flow/checkpoint.0.1.pt"],
        },
    )
    base.fields(
        manifest["aerodynamic"],
        {
            "role": "aerodynamic",
            "frozen": True,
            "checkpoint_epoch": 1,
            "metadata_kind": P026_AERO_KIND,
            "model_sha256": files["aerodynamic/FNO.0.1.mdlus"],
            "state_sha256": files["aerodynamic/checkpoint.0.1.pt"],
        },
    )
    records = result["records"]
    base.require(len(records) == 171, "incomplete updates")
    base.require(
        [(row["update"], row["consumed_windows"]) for row in records]
        == [(index, index * 8) for index in range(1, 172)],
        "update sequence differs",
    )
    base.require(result["sampler_order_sha256"] == protocol["sampler_order_sha256"], "order differs")
    return manifest


def validate_candidate_loader(base, source: Path, candidate: Path, approval: dict):
    loader = source / "src/fluid_control/dual_fno.py"
    base.require(base.sha(loader) == approval["source_sha256"]["src/fluid_control/dual_fno.py"], "loader bytes differ")
    spec = importlib.util.spec_from_file_location("_fcp028_frozen_loader", loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        identity = module.validate_dual_fno_manifest(
            candidate / "dual_model_manifest.json",
            expected_sha256=approval["candidate_sha256"]["dual_model_manifest.json"],
        )
    finally:
        sys.modules.pop(spec.name, None)
    base.fields(identity.payload, {"kind": P028_KIND})
    base.require(identity.flow.directory == (candidate / "flow").resolve(), "flow role differs")
    base.require(identity.aerodynamic.directory == (candidate / "aerodynamic").resolve(), "aero role differs")
    return identity


def validate_terminal_proofs(base, repo: Path, candidate: Path, approval: dict) -> None:
    """Require externally reviewed audit/reload receipts; never infer admission."""
    expected = approval["candidate_sha256"]
    expected_prefixed = {"candidate/" + name: digest for name, digest in expected.items()}
    proofs = {}
    for label in ("independent_terminal_audit", "official_dual_reload"):
        proof = approval[label]
        base.require(proof.get("reviewed_by_lead") is True, "terminal proof is unreviewed")
        path = base.bound(repo, proof["path"], proof["sha256"])
        payload = json.loads(path.read_text())
        base.require(payload.get("candidate_sha256") == expected_prefixed, "terminal proof candidate bytes differ")
        base.fields(
            payload,
            {
                "training_protocol_sha256": expected["training_protocol.json"],
                "dual_manifest_sha256": expected["dual_model_manifest.json"],
                "candidate_result_sha256": expected["result.json"],
                "scientific_admission": False,
                "ppo_authorized": False,
            },
        )
        proofs[label] = payload
    audit = proofs["independent_terminal_audit"]
    reload = proofs["official_dual_reload"]
    base.fields(
        audit,
        {
            "status": "FC_P028_CANDIDATE_INTEGRITY_VERIFIED_NOT_ADMISSION",
            "actual_optimizer_steps": 171,
            "actual_training_windows": 1368,
            "accumulation_windows": 8,
            "training_unit": approval["training_unit"],
            "training_invocation": approval["training_invocation"],
            "flow_trained": True,
            "aerodynamic_frozen": True,
            "dual_adapter_fresh_reload_verified": False,
        },
    )
    base.require(isinstance(audit.get("terminal_evidence"), dict), "terminal evidence missing")
    base.terminal(audit["terminal_evidence"], approval["training_invocation"])
    base.fields(audit["terminal_evidence"], {"LoadState": "loaded"})
    base.fields(
        reload,
        {
            "status": "FC_P028_OFFICIAL_CPU_DUAL_RELOAD_VERIFIED_NOT_ADMISSION",
            "candidate_audit_sha256": approval["independent_terminal_audit"]["sha256"],
            "config_sha256": approval["training_config_sha256"],
            "required_official_image_id": base.IMAGE,
            "device": "cpu",
            "official_dual_reload_verified": True,
            "forward_performed": False,
            "optimizer_created": False,
            "model_saved": False,
            "gpu_used": False,
        },
    )
    result = json.loads((candidate / "result.json").read_text())
    tensors = {
        "flow": result.get("flow_terminal_tensor_sha256"),
        "aerodynamic": result.get("frozen_aerodynamic_tensor_sha256"),
    }
    base.require(
        all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in tensors.values()),
        "result tensor digest differs",
    )
    base.require(
        audit.get("tensor_sha256") == reload.get("tensor_sha256") == tensors,
        "terminal/reloaded/saved tensor identity differs",
    )
    base.require(
        audit.get("role_loader_sha256")
        == reload.get("runtime_source_sha256", {}).get("src/fluid_control/dual_fno.py")
        == approval["source_sha256"]["src/fluid_control/dual_fno.py"],
        "audited/reloaded/formal role loader differs",
    )


def commands(base, manifest_sha: str, aero_model_sha: str):
    plan = base.commands(1, aero_model_sha, manifest_sha)
    expected = "FC_P026_K1_HISTORY_FORCE_FNO"
    replaced = 0
    for name, _, command in plan:
        if name == "validation_diagnostic":
            index = command.index("--candidate-kind") + 1
            base.require(command[index] == expected, "numerical base candidate identity drifted")
            command[index] = P028_KIND
            replaced += 1
    base.require(replaced == 1, "P028 diagnostic identity was not routed")
    return plan


def preflight(base, args) -> dict:
    base.require(base.sha(args.approval) == args.approval_sha256, "formal approval SHA differs")
    approval = json.loads(args.approval.read_text())
    base.fields(
        approval,
        {
            "status": FORMAL_APPROVAL,
            "formal_evaluation_authorized": True,
            "frozen_test_accessed": False,
            "ppo_auto_launch": False,
            "protocol": base.PROTOCOL,
            "numerical_base_commit": base.BASE,
            "official_image_id": base.IMAGE,
            "runner_sha256": base.sha(Path(__file__)),
            "numerical_runner_sha256": NUMERICAL_RUNNER_SHA256,
        },
    )
    base.require(
        args.numerical_runner.is_file()
        and not args.numerical_runner.is_symlink()
        and base.sha(args.numerical_runner) == NUMERICAL_RUNNER_SHA256,
        "external numerical runner bytes differ",
    )
    base.require(args.candidate.resolve() == (args.repo / approval["candidate_relative_directory"]).resolve(), "candidate path differs")
    base.require(args.output.resolve() == (args.repo / approval["output_relative_directory"]).resolve(), "output path differs")
    base.require(not args.output.exists(), "output must be new")
    base.require(
        args.repo.resolve() in args.output.resolve().parents
        and args.repo.resolve() in args.candidate.resolve().parents,
        "output/candidate escapes repo",
    )
    base.validate_source_chain(args, approval)
    base.require(base.REQUIRED_SOURCES <= set(approval["source_sha256"]), "source closure incomplete")
    base.fields(approval["source_sha256"], base.UNCHANGED_AUDITS)
    actual = {
        str(path.relative_to(args.source))
        for directory in ("src", "scripts", "conf", "cfd")
        for path in (args.source / directory).rglob("*")
        if path.is_file()
    }
    base.require(actual == set(approval["source_sha256"]), "source tree extra/missing files")
    for name, digest in approval["source_sha256"].items():
        base.bound(args.source, name, digest)
    base.bound(args.source, "training_config.yaml", approval["training_config_sha256"])
    for name, digest in base.INPUTS.items():
        role, filename = name.split("/", 1)
        base.bound(args.repo, base.DATA[role] + "/" + filename, digest)
    base.bound(args.repo, base.PREDECL, base.PREDECL_SHA)
    base.bound(args.repo, base.QC, base.QC_SHA)
    candidate_contract(base, args.candidate, approval)
    validate_candidate_loader(base, args.source, args.candidate, approval)
    validate_terminal_proofs(base, args.repo, args.candidate, approval)
    base.require(re.fullmatch(r"fluid-control-fcp028-[a-zA-Z0-9-]+\.service", approval["training_unit"]) is not None, "training unit differs")
    base.require(
        re.fullmatch(r"[0-9a-f]{32}", approval["training_invocation"]) is not None,
        "invalid invocation",
    )
    unit = subprocess.check_output(
        ["systemctl", "--user", "show", approval["training_unit"], "-p", "InvocationID", "-p", "ActiveState", "-p", "SubState", "-p", "Result", "-p", "ExecMainCode", "-p", "ExecMainStatus", "-p", "MainPID"],
        text=True,
        timeout=10,
    )
    base.terminal(dict(line.split("=", 1) for line in unit.splitlines() if "=" in line), approval["training_invocation"])
    return approval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "source", "candidate", "output", "approval"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--approval-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--numerical-runner",
        type=Path,
        required=True,
        help="Reviewed unchanged run_fcp026_posteval.py used for numerical orchestration",
    )
    args = parser.parse_args()
    base = _base_runner(args.numerical_runner)
    approval = preflight(base, args)
    plan = commands(
        base,
        approval["candidate_sha256"]["dual_model_manifest.json"],
        approval["candidate_sha256"]["aerodynamic/FNO.0.1.mdlus"],
    )
    if not args.execute:
        print(
            json.dumps(
                {"status": "FC_P028_FORMAL_PREFLIGHT_ONLY", "commands": plan},
                indent=2,
            )
        )
        return
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True,
        timeout=10,
    )
    base.require(not gpu.strip(), "GPU compute already active")
    base.require(min(base.memory().values()) >= 20 * 1024**3, "startup dual memory guard")
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("validation10", "dynamic6", "force_window", "evidence"):
        (args.output / name).mkdir()
    (args.output / "evidence/formal_approval.json").write_bytes(args.approval.read_bytes())

    def interrupt(signum, frame):
        del frame
        raise RuntimeError(f"interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    deadline = time.monotonic() + 10800
    precision = (
        "import json,os,torch; "
        "x={'NVIDIA_TF32_OVERRIDE':os.environ.get('NVIDIA_TF32_OVERRIDE'),"
        "'cuda_matmul_allow_tf32':bool(torch.backends.cuda.matmul.allow_tf32),"
        "'cudnn_allow_tf32':bool(torch.backends.cudnn.allow_tf32),"
        "'float32_matmul_precision':torch.get_float32_matmul_precision()}; "
        "assert x=={'NVIDIA_TF32_OVERRIDE':None,'cuda_matmul_allow_tf32':True,"
        "'cudnn_allow_tf32':True,'float32_matmul_precision':'high'},x; "
        "print(json.dumps(x,sort_keys=True))"
    )
    base.run_container(args, approval, "precision", True, ["python", "-c", precision], deadline)
    for name, gpu_required, command in plan:
        base.run_container(args, approval, name, gpu_required, command, deadline)
    required_outputs = (
        "validation10/evaluation.json",
        "validation10/segments.json",
        "validation10/diagnostic.json",
        "validation10/endpoint_gate.json",
        "dynamic6/evaluation.json",
        "dynamic6/segments.json",
        "dynamic6/diagnostic.json",
        "force_window/result.json",
        "development_gate.json",
    )
    for name in required_outputs:
        base.require((args.output / name).is_file(), f"missing formal output: {name}")
    candidate_contract(base, args.candidate, approval)
    outputs = {
        str(path.relative_to(args.output)): base.sha(path)
        for path in args.output.rglob("*")
        if path.is_file()
    }
    receipt = {
        "status": FORMAL_COMPLETE,
        "formal_approval_sha256": args.approval_sha256,
        "candidate_sha256": approval["candidate_sha256"],
        "source_sha256": approval["source_sha256"],
        "protocol": base.PROTOCOL,
        "sha256": outputs,
        "scientific_admission": False,
        "ppo_auto_launched": False,
        "frozen_test_accessed": False,
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
