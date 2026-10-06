#!/usr/bin/env python3
"""Engineering-only actual PhysicsNeMo P064 save/load contract check.

This performs no optimization, forward pass, data read, or scientific evaluation.
The temporary candidate-shaped files exist only to exercise the reviewed producer
metadata through the official checkpoint API and the staged P064 consumer.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def protocol(dual, arm: str) -> dict:
    return {
        "training_experiment": "FC-P064",
        "arm": arm,
        "parent_experiment": "FC-P026-K1",
        "history_input": dual._p026_history_input(1),
        "training_windows": 256,
        "accumulation_windows": 8,
        "optimizer_steps": 32,
        "learning_rate": dual.P026_LEARNING_RATE,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
        "seed": 20261003,
        "chunk_size": 10,
        "rollout_steps": 100,
        "parent_sampler_order_sha256": dual.P026_ORDER_SHA256,
        "schedule_sha256": dual.P064_SCHEDULE_SHA256[arm],
        "diagnostic_counts": [0, 256],
        "objective": "equal_H1_AR_half_equal_four_half_rearCl_normalized_MSE",
        "history_state_module_sha256": dual.P026_HISTORY_STATE_SHA256,
        "history_objective_sha256": dual.P026_HISTORY_OBJECTIVE_SHA256,
        "history_inference_module_sha256": dual.P026_HISTORY_INFERENCE_SHA256,
        "action_semantics": "stored_prescribed_action_samples_not_exact_nominal_time_commands",
        "controlled_b00_action_semantics": (
            "actual_closed_loop_applied_endpoint_omega_samples"
            if arm == "B"
            else "not_applicable_no_b00_windows"
        ),
        "b00_windows": 0 if arm == "A" else 64,
        "b00_weight": 0.0 if arm == "A" else 0.25,
        "replacement_within_each_update": [] if arm == "A" else [0, 4],
        "allocator_fraction": 0.06,
        "wall_seconds": 3600,
        "validation_accessed": False,
        "frozen_test_accessed": False,
        "selection_performed": False,
    }


def state_sha(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--loader", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--parent-manifest-sha256", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("A", "B"), default="A")
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA must be hidden")
    require(not args.output.exists(), "exclusive engineering fixture output exists")
    require(sha(args.parent_manifest) == args.parent_manifest_sha256, "parent SHA differs")
    sys.path[:0] = [str(args.repo / "src"), str(args.repo / "scripts")]

    import torch
    from omegaconf import OmegaConf
    from physicsnemo.utils import load_checkpoint, save_checkpoint
    import train_tandem_fno as train

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dual = import_file("dual_fno_p064_actual_roundtrip", args.loader)
    cfg = OmegaConf.load(args.config)
    parent = json.loads(args.parent_manifest.read_text())

    def build(config):
        return train.build_model(config)

    with torch.no_grad():
        loaded, identity = dual.load_dual_fno(
            args.parent_manifest,
            cfg,
            torch.device("cpu"),
            build_model=build,
            load_checkpoint=load_checkpoint,
            expected_manifest_sha256=args.parent_manifest_sha256,
        )
    require(identity.payload["kind"] == dual.P026_K1_SYSTEM_KIND, "parent is not K1")
    before = state_sha(loaded.aerodynamic_model)
    args.output.mkdir(parents=True)
    flow_dir = args.output / "flow"
    flow_dir.mkdir()
    parent_root = args.parent_manifest.parent
    for key in ("model_file", "state_file"):
        source = parent_root / parent["flow"]["checkpoint_relative_directory"] / parent["flow"][key]
        shutil.copy2(source, flow_dir / source.name)

    semantics = protocol(dual, args.arm)
    protocol_path = args.output / "training_protocol.json"
    protocol_path.write_text(json.dumps(semantics, sort_keys=True, separators=(",", ":")))
    metadata = {
        "status": dual.P064_AERO_KIND[args.arm],
        "checkpoint_epoch": 1,
        "training_experiment": "FC-P064",
        "arm": args.arm,
        "history_profile": "p026_k1",
        "history_k": 1,
        "model_in_channels": 6,
        "training_protocol_sha256": sha(protocol_path),
        "training_protocol_file": protocol_path.name,
        "history_state_module_sha256": dual.P026_HISTORY_STATE_SHA256,
        "history_inference_module_sha256": dual.P026_HISTORY_INFERENCE_SHA256,
        "training_windows": 256,
        "optimizer_steps": 32,
        "accumulation_windows": 8,
        "actual_learning_rate": dual.P026_LEARNING_RATE,
        "parent_sampler_order_sha256": dual.P026_ORDER_SHA256,
        "schedule_sha256": dual.P064_SCHEDULE_SHA256[args.arm],
        "parent_history_inventory": dual._p026_inventory(),
        "flow_parent_model_sha256": dual.FLOW_MODEL_SHA256,
        "flow_parent_state_sha256": dual.FLOW_STATE_SHA256,
        "aerodynamic_parent_model_sha256": dual.P028_AERO_PARENT_MODEL_SHA256,
        "aerodynamic_parent_state_sha256": dual.P028_AERO_PARENT_STATE_SHA256,
        "selection_performed": False,
        "validation_accessed": False,
        "frozen_test_accessed": False,
        "ppo_executed": False,
    }
    aero_dir = args.output / "aerodynamic"
    save_checkpoint(aero_dir, models=loaded.aerodynamic_model, epoch=1, metadata=metadata)
    manifest = copy.deepcopy(parent)
    manifest.update(
        status=dual.P064_MANIFEST_STATUS[args.arm],
        kind=dual.P064_SYSTEM_KIND[args.arm],
        training_experiment="FC-P064",
        arm=args.arm,
        history_input=dual._p026_history_input(1),
        parent_history_inventory=dual._p026_inventory(),
        training_protocol_file=protocol_path.name,
        training_protocol_sha256=sha(protocol_path),
        training_semantics=semantics,
        history_state_module_sha256=dual.P026_HISTORY_STATE_SHA256,
        history_inference_module_sha256=dual.P026_HISTORY_INFERENCE_SHA256,
        training_windows=256,
        optimizer_steps=32,
        accumulation_windows=8,
        actual_learning_rate=dual.P026_LEARNING_RATE,
        parent_manifest_sha256=args.parent_manifest_sha256,
        aerodynamic_initial_model_sha256=dual.P028_AERO_PARENT_MODEL_SHA256,
        aerodynamic_initial_state_sha256=dual.P028_AERO_PARENT_STATE_SHA256,
        aerodynamic_parent_model_sha256=dual.P028_AERO_PARENT_MODEL_SHA256,
        aerodynamic_parent_state_sha256=dual.P028_AERO_PARENT_STATE_SHA256,
        engineering_fixture_not_candidate=True,
    )
    manifest["flow_architecture"] = copy.deepcopy(dual.ARCHITECTURE)
    manifest["aerodynamic_architecture"] = copy.deepcopy(dual.ARCHITECTURE)
    manifest["flow"].update(
        checkpoint_relative_directory="flow",
        model_file=parent["flow"]["model_file"],
        state_file=parent["flow"]["state_file"],
    )
    manifest["aerodynamic"].update(
        checkpoint_relative_directory="aerodynamic",
        model_file="FNO.0.1.mdlus",
        state_file="checkpoint.0.1.pt",
        checkpoint_epoch=1,
        metadata_kind=dual.P064_AERO_KIND[args.arm],
        model_sha256=sha(aero_dir / "FNO.0.1.mdlus"),
        state_sha256=sha(aero_dir / "checkpoint.0.1.pt"),
        frozen=False,
    )
    manifest_path = args.output / "dual_model_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    with torch.no_grad():
        reloaded, observed = dual.load_dual_fno(
            manifest_path,
            cfg,
            torch.device("cpu"),
            build_model=build,
            load_checkpoint=load_checkpoint,
            expected_manifest_sha256=sha(manifest_path),
            allow_engineering_fixture=True,
        )
    require(state_sha(reloaded.aerodynamic_model) == before, "official aero save/reload differs")
    require(state_sha(reloaded.flow_model) == state_sha(loaded.flow_model), "flow reload differs")
    require(observed.payload["engineering_fixture_not_candidate"] is True, "fixture label lost")
    result = {
        "status": "P064_ACTUAL_OFFICIAL_CPU_SAVE_LOAD_ENGINEERING_FIXTURE_PASS",
        "arm": args.arm,
        "parent_manifest_sha256": args.parent_manifest_sha256,
        "fixture_manifest_sha256": sha(manifest_path),
        "loader_sha256": sha(args.loader),
        "aerodynamic_tensor_sha256": before,
        "optimizer_steps_performed": 0,
        "forward_performed": False,
        "real_data_accessed": False,
        "scientific_candidate": False,
        "gpu_used": False,
    }
    (args.output / "engineering_result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
