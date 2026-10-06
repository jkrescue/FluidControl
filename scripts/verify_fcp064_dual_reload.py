#!/usr/bin/env python3
"""Actual official P064 CPU dual reload, no forward/update/save/admission.

Root must separately capture the actual CPU container and bind this receipt.
The required image below is not a claim about the observed execution environment.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile

IMAGE = "sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
OFFICIAL_FNO_SHA = "e64eb9bef031bfdae5d84f0ed35a1ebb27915f18aed4a333b2dd985a083c71a9"
OFFICIAL_CHECKPOINT_SHA = "0d26a62251c3724a1ceebfa1daa1bb5ba9dcdc5e73a0ded3ccb955355af2f78e"
REQUIRED = {"src/fluid_control/dual_fno.py", "src/fluid_control/calibrated_checkpoint.py",
            "scripts/train_tandem_fno.py", "scripts/p026_history_inference.py",
            "scripts/p026_state_history.py", "scripts/audit_fcp064_candidate.py",
            "scripts/verify_fcp064_dual_reload.py"}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def verify_sources(args):
    require(sha(args.runtime_source_manifest) == args.runtime_source_manifest_sha256,
            "external runtime source manifest SHA differs")
    mapping = read(args.runtime_source_manifest)
    require(REQUIRED <= set(mapping), "CPU reload source closure missing required files")
    root = args.source_root.resolve()
    for name, digest in mapping.items():
        path = root/name
        require(path.resolve().is_relative_to(root) and path.is_file() and sha(path) == digest,
                "CPU reload runtime file differs: "+name)
    require(sha(__file__) == mapping["scripts/verify_fcp064_dual_reload.py"], "executed verifier differs")
    return mapping


def bindings(args):
    source = verify_sources(args)
    import audit_fcp064_candidate as audit
    import fluid_control.dual_fno as dual
    for module, name in ((audit, "scripts/audit_fcp064_candidate.py"),
                         (dual, "src/fluid_control/dual_fno.py")):
        require(sha(module.__file__) == source[name], "actual imported module differs")
    require(sha(dual.__file__) == audit.LOADER_SHA, "reviewed P064 role loader required")
    require(sha(args.candidate_audit) == args.candidate_audit_sha256, "external candidate audit SHA differs")
    checked = read(args.candidate_audit)
    k = args.history_k
    audit.fields(checked, dict(status=f"FC_P064_ARM_{args.arm}_CANDIDATE_INTEGRITY_VERIFIED_NOT_ADMISSION",
        arm=args.arm, history_k=k, actual_optimizer_steps=32, actual_training_windows=256,
        accumulation_windows=8, scientific_admission=False, ppo_authorized=False), "terminal candidate integrity receipt")
    files = audit.candidate_files(args.root)
    audit.equal(files, checked["candidate_sha256"], "candidate files changed since terminal audit")
    identity = dual.validate_dual_fno_manifest(args.root/"dual_model_manifest.json")
    require(identity.payload["kind"] == f"FC_P064_ARM_{args.arm}_CONTROLLED_AERO_FORCE_FNO", "P064 arm differs")
    require(sha(args.config) == identity.payload["config_sha256"], "actual model config differs")
    result = read(args.root/"result.json")
    tensors = dict(flow=result["flow_tensor_sha256"], aerodynamic=result["aerodynamic_terminal_tensor_sha256"])
    audit.equal(tensors, checked["tensor_sha256"], "terminal tensor identities differ")
    require(files["candidate/training_protocol.json"] == checked["training_protocol_sha256"] and
            files["candidate/dual_model_manifest.json"] == checked["dual_manifest_sha256"] and
            files["candidate/result.json"] == checked["candidate_result_sha256"], "audit key bindings differ")
    return dict(status=f"FC_P064_ARM_{args.arm}_OFFICIAL_CPU_DUAL_RELOAD_VERIFIED_NOT_ADMISSION", arm=args.arm, history_k=k,
        training_protocol_sha256=checked["training_protocol_sha256"], dual_manifest_sha256=identity.manifest_sha256,
        candidate_result_sha256=files["candidate/result.json"], candidate_sha256=files,
        candidate_audit_sha256=args.candidate_audit_sha256, config_sha256=sha(args.config),
        runtime_source_manifest_sha256=args.runtime_source_manifest_sha256, runtime_source_sha256=source,
        tensor_sha256=tensors, required_official_image_id=IMAGE, device="cpu",
        official_source_sha256=dict(fno=OFFICIAL_FNO_SHA, checkpoint=OFFICIAL_CHECKPOINT_SHA),
        official_dual_reload_verified=True, forward_performed=False, optimizer_created=False,
        model_saved=False, gpu_used=False, scientific_admission=False, ppo_authorized=False)


def execute_cpu(args, expected):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "explicit CUDA_VISIBLE_DEVICES='' required")
    import torch
    from omegaconf import OmegaConf
    from physicsnemo.utils import load_checkpoint
    import fluid_control.dual_fno as dual
    import fluid_control.calibrated_checkpoint as calibrated
    import train_tandem_fno as train
    import p026_history_inference as history
    import p026_state_history as state_history
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dual.validate_runtime_precision()
    for module, name in ((train, "scripts/train_tandem_fno.py"),
                         (history, "scripts/p026_history_inference.py"),
                         (state_history, "scripts/p026_state_history.py"),
                         (calibrated, "src/fluid_control/calibrated_checkpoint.py")):
        require(sha(module.__file__) == expected["runtime_source_sha256"][name], "imported official-loader dependency differs")
    require(not torch.cuda.is_initialized(), "CUDA initialized before CPU reload")
    require(sha(inspect.getfile(load_checkpoint)) == OFFICIAL_CHECKPOINT_SHA,
            "actual official checkpoint implementation differs")
    cfg = OmegaConf.load(args.config)
    def official_model(config):
        model = train.build_model(config)
        require(sha(inspect.getfile(type(model))) == OFFICIAL_FNO_SHA,
                "actual official FNO implementation differs")
        return model
    with torch.no_grad():
        adapter, identity = dual.load_dual_fno(args.root/"dual_model_manifest.json", cfg,
            torch.device("cpu"), build_model=official_model, load_checkpoint=load_checkpoint,
            expected_manifest_sha256=expected["dual_manifest_sha256"])
    require(getattr(adapter, "history_length", None) == args.history_k, "loaded history adapter kind")
    require(all(p.device.type == "cpu" and not p.requires_grad and p.grad is None for p in adapter.parameters()),
            "dual reload must be frozen CPU-only without gradients")
    def state_digest(state):
        digest = hashlib.sha256()
        for name, value in sorted(state.items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()
    observed = {role: state_digest(getattr(adapter, role+"_model").state_dict()) for role in ("flow", "aerodynamic")}
    require(observed == expected["tensor_sha256"], "actual official dual-loaded tensors differ")
    require(identity.payload["history_input"]["history_length"] == args.history_k and
            not torch.cuda.is_initialized(), "wrong history or CUDA initialized")
    require(bindings(args) == expected, "bound bytes changed during official CPU reload")


def persist(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)  # Atomic and never replaces an earlier receipt.
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "candidate-audit", "config", "source-root", "runtime-source-manifest", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("candidate-audit-sha256", "runtime-source-manifest-sha256"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--history-k", type=int, choices=(1,), default=1)
    parser.add_argument("--arm", choices=("A", "B"), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--execute-cpu", action="store_true")
    mode.add_argument("--check-receipt", action="store_true")
    args = parser.parse_args()
    sys.path[:0] = [str(args.source_root/"src"), str(args.source_root/"scripts")]
    expected = bindings(args)
    if args.check_receipt:
        require(read(args.output) == expected, "actual CPU reload receipt differs")
        print("FC_P064_CPU_RELOAD_RECEIPT_RECHECKED_NO_MODEL_LOAD")
    else:
        require(not args.output.exists(), "CPU reload output exists")
        execute_cpu(args, expected)
        persist(args.output, expected)
        print(json.dumps(dict(status=expected["status"], receipt_sha256=sha(args.output))))


if __name__ == "__main__":
    main()
