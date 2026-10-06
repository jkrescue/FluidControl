#!/usr/bin/env python3
"""Resume P064 formal evaluation after a proven completed validation10 stage."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


FORMAL_SHA = "aac728bcf7f568073b723fb640a84f2fcad1e671b55b68ba8a331b77856089c9"
RESOURCE_SHA = "ac1b00563f7af30bc350421d0938fb61d9bf1cea5f8b0fff065e3e75f87e4e05"
FORMAL = Path(__file__).with_name("run_fcp064_posteval.py")
RESOURCE = Path(__file__).with_name("run_fcp064_formal_resource_adapter.py")
REUSED = (
    "precision.log",
    "evidence/precision_container.json",
    "evidence/precision_container_terminal.json",
    "validation10.log",
    "validation10/evaluation.json",
    "validation10/segments.json",
    "evidence/validation10_container.json",
    "evidence/validation10_container_terminal.json",
    "memory.jsonl",
)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def load(name: str, path: Path, digest: str):
    require(path.is_file() and not path.is_symlink() and sha(path) == digest, f"{name} bytes differ")
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"{name} import unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def terminal_unit(unit: str, invocation: str) -> dict:
    raw = subprocess.check_output(
        ["systemctl", "--user", "show", unit, "-p", "InvocationID", "-p", "ActiveState",
         "-p", "SubState", "-p", "Result", "-p", "ExecMainStatus", "-p", "MainPID"],
        text=True, timeout=10,
    )
    state = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
    require(state == {"MainPID": "0", "Result": "exit-code", "ExecMainStatus": "1",
                     "ActiveState": "failed", "SubState": "failed", "InvocationID": invocation},
            "prior R2 terminal identity differs")
    return state


def execution_unit(unit: str) -> dict:
    invocation = os.environ.get("INVOCATION_ID", "")
    require(len(invocation) == 32 and all(c in "0123456789abcdef" for c in invocation),
            "actual R3 invocation identity absent")
    raw = subprocess.check_output(
        ["systemctl", "--user", "show", unit, "-p", "InvocationID", "-p", "ActiveState",
         "-p", "SubState", "-p", "MainPID"], text=True, timeout=10,
    )
    state = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
    require(state.get("InvocationID") == invocation and state.get("ActiveState") == "active"
            and state.get("SubState") in {"running", "start"}
            and int(state.get("MainPID", "0")) == os.getpid(), "actual R3 unit identity differs")
    return {"unit": unit, "invocation": invocation, "main_pid": os.getpid()}


def validate_reused(prior: Path, expected: dict[str, str]) -> None:
    require(set(expected) == set(REUSED), "reused stage proof incomplete")
    for relative in REUSED:
        source = prior / relative
        require(source.is_file() and not source.is_symlink() and sha(source) == expected[relative],
                f"reused source differs: {relative}")
    for relative in ("evidence/precision_container_terminal.json", "evidence/validation10_container_terminal.json"):
        state = json.loads((prior / relative).read_text())["State"]
        require(state["ExitCode"] == 0 and state["OOMKilled"] is False and state["Running"] is False,
                f"reused terminal container differs: {relative}")


def copy_reused(prior: Path, output: Path, expected: dict[str, str]) -> None:
    validate_reused(prior, expected)
    for relative in REUSED:
        source = prior / relative
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        require(sha(target) == expected[relative], f"reused copy differs: {relative}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "source", "candidate", "output", "approval", "numerical-runner", "prior-output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--approval-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    args.source = args.source.resolve()
    args.candidate = args.candidate.resolve()
    args.output = args.output.resolve()
    args.approval = args.approval.resolve()
    args.numerical_runner = args.numerical_runner.resolve()
    args.prior_output = args.prior_output.resolve()
    formal = load("_p064_resume_formal", FORMAL, FORMAL_SHA)
    resource = load("_p064_resume_resource", RESOURCE, RESOURCE_SHA)
    inherited = formal.p028._base_runner

    def adapted_base(path):
        base = inherited(path)
        def startup_memory():
            available = resource.available_memory()["MemAvailable"]
            require(available >= resource.STARTUP_AVAILABLE, "P064 resume startup MemAvailable below 50 GiB")
            return {"MemAvailable": available}
        base.memory = startup_memory
        base.run_container = lambda call_args, approval, name, gpu, command, deadline: resource.adapted_run_container(
            base, call_args, approval, name, gpu, command, deadline
        )
        return base

    base = adapted_base(args.numerical_runner)
    args.experiment = formal.EXPERIMENT
    args.entry_file = FORMAL
    approval = formal.preflight(base, args)
    require(approval.get("resource_adapter_sha256") == RESOURCE_SHA,
            "resume resource adapter approval differs")
    require(approval.get("resource_contract") == resource.RESOURCE_CONTRACT,
            "resume resource contract differs")
    require(approval.get("resume_runner_sha256") == sha(Path(__file__)), "resume runner bytes differ")
    resume = approval.get("resume_from")
    require(isinstance(resume, dict), "resume proof absent")
    require(args.prior_output == (args.repo / resume["relative_directory"]).resolve(), "prior output differs")
    terminal_unit(resume["unit"], resume["invocation"])
    validate_reused(args.prior_output, resume["sha256"])
    plan = formal.commands(base, approval["candidate_sha256"]["dual_model_manifest.json"],
                           approval["candidate_sha256"]["aerodynamic/FNO.0.1.mdlus"])
    require([row[0] for row in plan] == ["validation10", "validation_diagnostic", "endpoint_gate",
            "dynamic6", "dynamic_diagnostic", "force_window", "development_gate"], "formal plan differs")
    remaining = plan[1:]
    if not args.execute:
        print(json.dumps({"status": "FC_P064_FORMAL_RESUME_PREFLIGHT_ONLY", "reused": ["precision", "validation10"],
                          "commands": remaining}, indent=2))
        return
    execution = execution_unit(approval.get("planned_execution_unit", ""))
    startup_samples = []
    for index in range(2):
        row = resource.available_memory()
        require(row["MemAvailable"] >= resource.STARTUP_AVAILABLE,
                "P064 resume startup MemAvailable below 50 GiB")
        startup_samples.append({"time_unix": time.time(), **row})
        if index == 0:
            time.sleep(2)
    # preflight above proves output absent; create it only after every prior-stage check.
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("validation10", "dynamic6", "force_window", "evidence"):
        (args.output / name).mkdir(exist_ok=True)
    copy_reused(args.prior_output, args.output, resume["sha256"])
    (args.output / "evidence/formal_approval.json").write_bytes(args.approval.read_bytes())

    def interrupt(signum, _frame):
        raise RuntimeError(f"interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    deadline = time.monotonic() + 10800
    for name, gpu, command in remaining:
        base.run_container(args, approval, name, gpu, command, deadline)
    required = ("validation10/evaluation.json", "validation10/segments.json", "validation10/diagnostic.json",
                "validation10/endpoint_gate.json", "dynamic6/evaluation.json", "dynamic6/segments.json",
                "dynamic6/diagnostic.json", "force_window/result.json", "development_gate.json")
    for relative in required:
        require((args.output / relative).is_file(), f"missing formal output: {relative}")
    formal.candidate_contract(base, args.candidate, approval)
    outputs = {str(path.relative_to(args.output)): sha(path) for path in args.output.rglob("*") if path.is_file()}
    receipt = {
        "status": "FC_P064_ORIGINAL_FORMAL_COMPLETE_NOT_ADMISSION",
        "formal_approval_sha256": args.approval_sha256,
        "candidate_sha256": approval["candidate_sha256"], "source_sha256": approval["source_sha256"],
        "protocol": base.PROTOCOL, "sha256": outputs, "scientific_admission": False,
        "ppo_auto_launched": False, "frozen_test_accessed": False,
        "stage_provenance": {
            "reused": {"stages": ["precision", "validation10"], "unit": resume["unit"],
                       "invocation": resume["invocation"], "sha256": resume["sha256"]},
            "executed": {"stages": [row[0] for row in remaining],
                         "resume_runner_sha256": sha(Path(__file__)),
                         **execution, "startup_memory_samples": startup_samples},
        },
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
