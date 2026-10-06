#!/usr/bin/env python3
"""Actual official FC-P028 CPU dual reload; no forward, update, save, or admission."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

# Pin the new shared identity dependency even for legacy P028 approvals.
_PROFILE_SHA = "e6d333b4b2630d06151d9fcf557c85f95fb4d1be5760a1aa5b9199227215e553"
_PROFILE_PATH = Path(__file__).with_name("flow_repair_profiles.py")
if (_PROFILE_PATH.is_symlink()
        or hashlib.sha256(_PROFILE_PATH.read_bytes()).hexdigest() != _PROFILE_SHA):
    raise ValueError("shared flow repair identity source differs before import")
from flow_repair_profiles import repair_profile
if hashlib.sha256(Path(sys.modules["flow_repair_profiles"].__file__).read_bytes()).hexdigest() != _PROFILE_SHA:
    raise ValueError("imported flow repair identity source differs")


IMAGE = "sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
OFFICIAL_FNO_SHA = "e64eb9bef031bfdae5d84f0ed35a1ebb27915f18aed4a333b2dd985a083c71a9"
OFFICIAL_CHECKPOINT_SHA = "0d26a62251c3724a1ceebfa1daa1bb5ba9dcdc5e73a0ded3ccb955355af2f78e"
STATUS = "FC_P028_OFFICIAL_CPU_DUAL_RELOAD_VERIFIED_NOT_ADMISSION"


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


def verify_sources(root, manifest, manifest_sha, experiment="FC-P028", entry_file=__file__):
    require(sha(manifest) == manifest_sha, "runtime source manifest differs")
    mapping = read(manifest)
    required = {
        "src/fluid_control/dual_fno.py",
        "src/fluid_control/calibrated_checkpoint.py",
        "scripts/train_tandem_fno.py",
        "scripts/p026_history_inference.py",
        "scripts/p026_state_history.py",
        "scripts/audit_fcp028_candidate.py",
        "scripts/verify_fcp028_dual_reload.py",
    }
    if experiment == "FC-P029":
        required.update({"scripts/flow_repair_profiles.py", "scripts/audit_fcp029_candidate.py", "scripts/verify_fcp029_dual_reload.py"})
    require(required <= set(mapping), "runtime source closure incomplete")
    root = Path(root).resolve()
    for name, digest in mapping.items():
        path = root / name
        require(path.is_file() and path.resolve().is_relative_to(root) and sha(path) == digest, "runtime source differs")
    require(sha(__file__) == mapping["scripts/verify_fcp028_dual_reload.py"], "executed verifier differs")
    if experiment == "FC-P029":
        import flow_repair_profiles
        require(sha(flow_repair_profiles.__file__) == mapping["scripts/flow_repair_profiles.py"], "imported profile source differs")
        require(sha(entry_file) == mapping["scripts/verify_fcp029_dual_reload.py"], "P029 verifier entry differs")
    return mapping


def bindings(args, experiment="FC-P028", entry_file=__file__):
    profile = repair_profile(experiment)
    sources = verify_sources(args.source_root, args.runtime_source_manifest, args.runtime_source_manifest_sha256, experiment, entry_file)
    audit = read(args.candidate_audit)
    require(sha(args.candidate_audit) == args.candidate_audit_sha256, "candidate audit SHA differs")
    require(
        audit.get("status") == profile.status("CANDIDATE_INTEGRITY_VERIFIED_NOT_ADMISSION")
        and audit.get("scientific_admission") is False
        and audit.get("ppo_authorized") is False,
        "candidate audit contract differs",
    )
    loader = sources["src/fluid_control/dual_fno.py"]
    require(audit.get("role_loader_sha256") == loader, "candidate audit loader differs")
    actual_files = {}
    for relative, digest in audit["candidate_sha256"].items():
        require(relative.startswith("candidate/"), "candidate receipt prefix differs")
        path = args.candidate / relative[len("candidate/"):]
        require(path.is_file() and not path.is_symlink() and sha(path) == digest, "candidate bytes changed")
        actual_files[relative] = digest
    require(actual_files == audit["candidate_sha256"], "candidate receipt map differs")
    require(
        audit.get("training_protocol_sha256") == actual_files["candidate/training_protocol.json"]
        and audit.get("dual_manifest_sha256") == actual_files["candidate/dual_model_manifest.json"]
        and audit.get("candidate_result_sha256") == actual_files["candidate/result.json"],
        "candidate audit key bindings differ",
    )
    manifest = read(args.candidate / "dual_model_manifest.json")
    require(sha(args.config) == manifest.get("config_sha256"), "candidate config identity differs")
    return {
        "status": profile.status("OFFICIAL_CPU_DUAL_RELOAD_VERIFIED_NOT_ADMISSION"),
        "training_protocol_sha256": audit["training_protocol_sha256"],
        "dual_manifest_sha256": audit["dual_manifest_sha256"],
        "candidate_result_sha256": audit["candidate_result_sha256"],
        "candidate_sha256": audit["candidate_sha256"],
        "candidate_audit_sha256": args.candidate_audit_sha256,
        "config_sha256": sha(args.config),
        "runtime_source_manifest_sha256": args.runtime_source_manifest_sha256,
        "runtime_source_sha256": sources,
        "tensor_sha256": audit["tensor_sha256"],
        "required_official_image_id": IMAGE,
        "device": "cpu",
        "official_source_sha256": {"fno": OFFICIAL_FNO_SHA, "checkpoint": OFFICIAL_CHECKPOINT_SHA},
        "official_dual_reload_verified": True,
        "forward_performed": False,
        "optimizer_created": False,
        "model_saved": False,
        "gpu_used": False,
        "scientific_admission": False,
        "ppo_authorized": False,
    }


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def configure_precision(torch):
    torch.set_float32_matmul_precision("high")
    observed = {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }
    require(
        observed
        == {
            "float32_matmul_precision": "high",
            "cuda_matmul_allow_tf32": True,
            "cudnn_allow_tf32": True,
        },
        "runtime precision differs",
    )
    return observed


def execute(args, expected, experiment="FC-P028"):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA must be hidden")
    import torch
    from omegaconf import OmegaConf
    from physicsnemo.utils import load_checkpoint
    import fluid_control.dual_fno as dual
    import fluid_control.calibrated_checkpoint as calibrated
    import p026_history_inference as history
    import p026_state_history as state_history
    import train_tandem_fno as train

    require(not torch.cuda.is_initialized(), "CUDA initialized")
    configure_precision(torch)
    for module, relative in (
        (dual, "src/fluid_control/dual_fno.py"),
        (calibrated, "src/fluid_control/calibrated_checkpoint.py"),
        (train, "scripts/train_tandem_fno.py"),
        (history, "scripts/p026_history_inference.py"),
        (state_history, "scripts/p026_state_history.py"),
    ):
        require(sha(inspect.getfile(module)) == expected["runtime_source_sha256"][relative], "imported runtime source differs")
    require(sha(inspect.getfile(load_checkpoint)) == OFFICIAL_CHECKPOINT_SHA, "official checkpoint source differs")
    cfg = OmegaConf.load(args.config)

    def build(config):
        model = train.build_model(config)
        require(sha(inspect.getfile(type(model))) == OFFICIAL_FNO_SHA, "official FNO source differs")
        return model

    with torch.no_grad():
        adapter, identity = dual.load_dual_fno(
            args.candidate / "dual_model_manifest.json",
            cfg,
            torch.device("cpu"),
            build_model=build,
            load_checkpoint=load_checkpoint,
            expected_manifest_sha256=expected["dual_manifest_sha256"],
        )
    require(identity.payload.get("kind") == repair_profile(experiment).kind, "flow repair kind differs")
    require(getattr(adapter, "history_length", None) == 1, "P028 K1 history differs")
    require(all(not parameter.requires_grad and parameter.grad is None for parameter in adapter.parameters()), "adapter not frozen")
    observed = {
        "flow": state_digest(adapter.flow_model.state_dict()),
        "aerodynamic": state_digest(adapter.aerodynamic_model.state_dict()),
    }
    require(observed == expected["tensor_sha256"], "loaded tensors differ")
    require(not torch.cuda.is_initialized(), "CUDA initialized during CPU reload")


def main(experiment="FC-P028", entry_file=__file__):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "candidate-audit", "config", "source-root", "runtime-source-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--candidate-audit-sha256", required=True)
    parser.add_argument("--runtime-source-manifest-sha256", required=True)
    parser.add_argument("--execute-cpu", action="store_true")
    args = parser.parse_args()
    sys.path[:0] = [str(args.source_root / "src"), str(args.source_root / "scripts")]
    expected = bindings(args, experiment, entry_file)
    require(args.execute_cpu, "explicit CPU execution required")
    require(not args.output.exists(), "output exists")
    execute(args, expected, experiment)
    with args.output.open("x") as stream:
        json.dump(expected, stream, indent=2)


if __name__ == "__main__":
    main()
