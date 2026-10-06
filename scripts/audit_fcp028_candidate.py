#!/usr/bin/env python3
"""FC-P028 terminal identity audit; no model load, forward, or admission."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

# Pin the new shared identity dependency even for legacy P028 approvals.
_PROFILE_SHA = "e6d333b4b2630d06151d9fcf557c85f95fb4d1be5760a1aa5b9199227215e553"
_PROFILE_PATH = Path(__file__).with_name("flow_repair_profiles.py")
if (_PROFILE_PATH.is_symlink()
        or hashlib.sha256(_PROFILE_PATH.read_bytes()).hexdigest() != _PROFILE_SHA):
    raise ValueError("shared flow repair identity source differs before import")
from flow_repair_profiles import repair_profile, validate_p029_result_binding
if hashlib.sha256(Path(sys.modules["flow_repair_profiles"].__file__).read_bytes()).hexdigest() != _PROFILE_SHA:
    raise ValueError("imported flow repair identity source differs")


STATUS = "FC_P028_CANDIDATE_INTEGRITY_VERIFIED_NOT_ADMISSION"
KIND = "FC_P028_FLOW_ROLLOUT_REPAIR"
FILES = (
    "result.json",
    "training_protocol.json",
    "dual_model_manifest.json",
    "flow/FNO.0.1.mdlus",
    "flow/checkpoint.0.1.pt",
    "aerodynamic/FNO.0.1.mdlus",
    "aerodynamic/checkpoint.0.1.pt",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    value = json.loads(Path(path).read_text())
    require(isinstance(value, dict), "JSON object required")
    return value


def candidate_files(candidate):
    root = Path(candidate).resolve()
    result = {}
    for name in FILES:
        path = root / name
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root), "candidate file missing/escaped")
        result["candidate/" + name] = sha(path)
    return result


def terminal(unit, invocation):
    fields = ("LoadState", "ActiveState", "SubState", "Result", "ExecMainCode", "ExecMainStatus", "MainPID", "InvocationID")
    text = subprocess.check_output(
        ["systemctl", "--user", "show", unit, *["--property=" + field for field in fields]],
        text=True,
        timeout=15,
    )
    value = dict(line.split("=", 1) for line in text.splitlines() if "=" in line)
    expected = {
        "LoadState": "loaded",
        "ActiveState": "active",
        "SubState": "exited",
        "Result": "success",
        "ExecMainCode": "1",
        "ExecMainStatus": "0",
        "MainPID": "0",
        "InvocationID": invocation,
    }
    require(value == expected, "retained terminal service differs")
    return value


def validate(candidate, loader, loader_sha, unit, invocation, experiment="FC-P028"):
    profile = repair_profile(experiment)
    candidate = Path(candidate).resolve()
    require(re.fullmatch(r"[0-9a-f]{32}", invocation) is not None, "invalid invocation")
    require(Path(loader).is_file() and not Path(loader).is_symlink() and sha(loader) == loader_sha, "reviewed loader differs")
    spec = importlib.util.spec_from_file_location("_p028_audit_loader", loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    files = candidate_files(candidate)
    identity = module.validate_dual_fno_manifest(
        candidate / "dual_model_manifest.json",
        expected_sha256=files["candidate/dual_model_manifest.json"],
    )
    require(identity.payload.get("kind") == profile.kind, "candidate kind differs")
    result = read(candidate / "result.json")
    protocol = read(candidate / "training_protocol.json")
    if experiment == "FC-P029":
        validate_p029_result_binding(result, protocol, identity.payload)
    require(
        result.get("status") == profile.status("TRAINING_COMPLETE_NOT_ADMISSION")
        and result.get("mode") == "train"
        and result.get("optimizer_steps") == 171
        and result.get("training_windows") == 1368
        and result.get("scientific_admission") is False
        and result.get("validation_accessed") is False
        and result.get("frozen_test_accessed") is False,
        "terminal result differs",
    )
    require(
        protocol == identity.payload.get("training_semantics")
        and files["candidate/training_protocol.json"] == identity.payload.get("training_protocol_sha256")
        == result.get("training_protocol_sha256"),
        "training protocol differs",
    )
    records = result.get("records")
    require(
        isinstance(records, list)
        and [(row.get("update"), row.get("consumed_windows")) for row in records]
        == [(index, index * 8) for index in range(1, 172)],
        "actual update sequence differs",
    )
    tensors = {
        "flow": result.get("flow_terminal_tensor_sha256"),
        "aerodynamic": result.get("frozen_aerodynamic_tensor_sha256"),
    }
    require(all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in tensors.values()), "tensor digest differs")
    return {
        "status": profile.status("CANDIDATE_INTEGRITY_VERIFIED_NOT_ADMISSION"),
        "training_protocol_sha256": files["candidate/training_protocol.json"],
        "dual_manifest_sha256": identity.manifest_sha256,
        "candidate_result_sha256": files["candidate/result.json"],
        "candidate_sha256": files,
        "actual_optimizer_steps": 171,
        "actual_training_windows": 1368,
        "accumulation_windows": 8,
        "training_unit": unit,
        "training_invocation": invocation,
        "flow_trained": True,
        "aerodynamic_frozen": True,
        "dual_adapter_fresh_reload_verified": False,
        "terminal_evidence": terminal(unit, invocation),
        "tensor_sha256": tensors,
        "role_loader_sha256": loader_sha,
        "scientific_admission": False,
        "ppo_authorized": False,
    }


def main(experiment="FC-P028"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--loader", type=Path, required=True)
    parser.add_argument("--loader-sha256", required=True)
    parser.add_argument("--training-unit", required=True)
    parser.add_argument("--training-invocation", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "output exists")
    with args.output.open("x") as stream:
        json.dump(
            validate(
                args.candidate,
                args.loader,
                args.loader_sha256,
                args.training_unit,
                args.training_invocation,
                experiment,
            ),
            stream,
            indent=2,
        )


if __name__ == "__main__":
    main()
