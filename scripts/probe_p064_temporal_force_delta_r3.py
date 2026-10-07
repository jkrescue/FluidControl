"""One fixed train-window absolute-vs-temporal-residual probe; never saves a model."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time


LR = 1.5625e-7
BETAS = (0.9, 0.999)
EPS = 1e-8
WEIGHT_DECAY = 1e-4
CLIP = 1.0
ALLOCATOR_BYTES = 16 * 2**30
FIXED_START = 0
FIXED_REASON = "first controlled_b00 slot of the reviewed P064-B schedule"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_path(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def host_memory():
    rows = {
        line.split(":")[0]: int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith(("MemAvailable:", "MemFree:"))
    }
    return rows


def snapshot(model):
    return {name: value.detach().cpu().clone() for name, value in model.named_parameters()}


def update_inventory(before, model):
    import torch

    rows = {}
    for name, parameter in model.named_parameters():
        difference = parameter.detach().cpu() - before[name]
        rows[name] = {
            "changed": not torch.equal(parameter.detach().cpu(), before[name]),
            "l2": float(difference.double().norm()),
            "max_abs": float(difference.abs().max()),
        }
    return rows


def per_tensor_gradient_inventory(model):
    import torch

    rows = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            if parameter.grad is not None:
                raise RuntimeError("frozen parameter received a gradient")
            continue
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"missing/nonfinite gradient: {name}")
        rows[name] = {
            "shape": list(parameter.shape),
            "l2": float(parameter.grad.detach().double().norm()),
            "nonzero": int(torch.count_nonzero(parameter.grad.detach())),
        }
    if len(rows) != 28:
        raise RuntimeError("official aerodynamic trainable tensor count differs")
    return rows


def scalar_metrics(row):
    return {
        key: float(row[key])
        for key in ("h1_balanced", "ar_balanced", "total")
    }


def configure_parent_training_precision(torch):
    """Reproduce the P064 parent-training precision before its fail-closed guard."""
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def execute(args, residual, p013):
    import torch
    from omegaconf import OmegaConf
    from physicsnemo.datapipes.readers.hdf5 import HDF5Reader
    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils import load_checkpoint
    from fluid_control.dual_fno import validate_runtime_precision
    from fluid_control.tandem_datapipe import TandemRolloutDataset
    from train_tandem_fno import build_model, configured_force_indices, predict

    token = os.environ.get("P064_TEMPORAL_DELTA_CHILD_TOKEN", "")
    if len(token) != 64 or any(character not in "0123456789abcdef" for character in token):
        raise RuntimeError("bounded supervisor child token required")
    started = time.monotonic()
    resources = []

    def observe(phase):
        memory = host_memory()
        if memory.get("MemAvailable", 0) < 22 * 2**30:
            raise RuntimeError("runtime MemAvailable below22GiB")
        resources.append({"phase": phase, "elapsed_seconds": time.monotonic() - started, **memory})

    cfg = OmegaConf.load(args.config)
    if tuple(configured_force_indices(cfg)) != (0, 1, 2, 3):
        raise ValueError("four-force schema required")
    if sha(args.hdf) != args.hdf_sha256:
        raise ValueError("fixed b00 HDF SHA differs")
    if sha(args.data_root / "normalization.json") != args.normalization_sha256:
        raise ValueError("train-only normalization SHA differs")
    if sha(args.data_root / "manifest.json") != args.data_manifest_sha256:
        raise ValueError("train-view manifest SHA differs")
    if args.hdf.resolve().parent != (args.data_root / "train").resolve():
        raise ValueError("HDF is not in the bound train-only root")
    observe("startup")
    if resources[-1]["MemAvailable"] < 50 * 2**30:
        raise RuntimeError("startup MemAvailable below50GiB")

    DistributedManager.initialize()
    dist = DistributedManager()
    if dist.distributed or not dist.cuda or torch.cuda.device_count() != 1:
        raise RuntimeError("one approved CUDA device required")
    total_device_memory = torch.cuda.get_device_properties(dist.device).total_memory
    allocator_fraction = min(1.0, ALLOCATOR_BYTES / total_device_memory)
    torch.cuda.set_per_process_memory_fraction(allocator_fraction, dist.device)
    configure_parent_training_precision(torch)
    precision = validate_runtime_precision()

    flow = build_model(cfg)
    parent = build_model(cfg)
    for role, model, epoch in (("flow", flow, 0), ("aerodynamic", parent, 1)):
        if load_checkpoint(args.parent / role, models=model, metadata_dict={}, device="cpu") != epoch:
            raise ValueError(f"{role} parent epoch differs")
    flow.eval()
    for parameter in flow.parameters():
        parameter.requires_grad_(False)
    frozen = set(p013.OFFICIAL_FROZEN_PARAMETER_NAMES)
    for name, parameter in parent.named_parameters():
        parameter.requires_grad_(name not in frozen)
    if sum(parameter.requires_grad for parameter in parent.parameters()) != 28:
        raise ValueError("official 28-tensor aerodynamic scope differs")

    absolute = copy.deepcopy(parent).to(dist.device).train()
    temporal = copy.deepcopy(parent).to(dist.device).train()
    residual_initialization = residual.zero_force_output_rows(temporal)
    flow = flow.to(dist.device)

    dataset = TandemRolloutDataset(
        args.data_root, "train", 100, stride=1, num_workers=0,
        force_indices=(0, 1, 2, 3),
    )
    reader = HDF5Reader(args.hdf, fields=["state", "mask", "omega", "force", "time"])
    try:
        if len(dataset.paths) != 1 or dataset.paths[0].resolve() != args.hdf.resolve():
            raise ValueError("fixed train view must contain exactly the bound HDF")
        sample, metadata = dataset[FIXED_START]
        frame0, _ = reader[FIXED_START]
        if metadata != {"case": args.hdf.stem, "step": 0, "rollout_steps": 100, "split": "train"}:
            raise ValueError("fixed window metadata differs")
        initial_force = (
            frame0["force"].float()[list((0, 1, 2, 3))] - dataset.force_mean
        ) / dataset.force_std
        batch = {key: value[None].to(dist.device) for key, value in sample.items()}
        initial_force = initial_force[None].to(dist.device)
        observe("frozen_flow")
        flow_states = p013.frozen_flow_states(
            flow, batch["state"], batch["mask"], batch["omega"], predict
        ).detach()
        h1_states = p013.true_state_inputs(batch["state"], batch["target_state"]).detach()
        flow.cpu()
        torch.cuda.empty_cache()

        def absolute_eval(model, backward):
            return p013.chunk_force_objective(
                model, flow_states, h1_states, batch["mask"], batch["omega"],
                batch["target_force"], predict, backward=backward,
            )

        def temporal_eval(model, backward):
            return residual.exact_recompute_vjp_objective(
                model, flow_states, h1_states, batch["mask"], batch["omega"],
                batch["target_force"], initial_force, p013.make_inputs, predict,
                p013.balanced_force_objective, chunk_size=10, backward=backward,
            )

        arms = []
        for label, model, evaluate in (
            ("absolute", absolute, absolute_eval),
            ("temporal_residual", temporal, temporal_eval),
        ):
            observe(label + "_initial")
            torch.cuda.reset_peak_memory_stats()
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=LR,
                betas=BETAS, eps=EPS, weight_decay=WEIGHT_DECAY,
            )
            before = snapshot(model)
            initial = evaluate(model, False)
            if label == "temporal_residual":
                truth_current = residual.causal_current_force(batch["target_force"], initial_force)
                if not torch.equal(initial["h1"], truth_current):
                    raise RuntimeError("zero head does not reproduce H1 current-force persistence")
                if not torch.equal(initial["ar"], initial_force[:, None].expand(-1, 100, -1)):
                    raise RuntimeError("zero head does not reproduce AR constant persistence")
            optimizer.zero_grad(set_to_none=True)
            trained = evaluate(model, True)
            gradients = {
                "p013_scope_audit": p013.audit_aerodynamic_gradients(model),
                "per_tensor": per_tensor_gradient_inventory(model),
            }
            if label == "temporal_residual":
                gradients["zero_head_interpretation"] = residual.gradient_inventory(model)
            trainable = [p for p in model.parameters() if p.requires_grad]
            preclip = torch.nn.utils.clip_grad_norm_(trainable, CLIP)
            if not torch.isfinite(preclip):
                raise FloatingPointError("nonfinite gradient norm")
            torch.cuda.synchronize()
            peak_before_step = {
                "allocated": torch.cuda.max_memory_allocated(),
                "reserved": torch.cuda.max_memory_reserved(),
            }
            optimizer.step()
            after = evaluate(model, False)
            torch.cuda.synchronize()
            arms.append({
                "arm": label,
                "initial": scalar_metrics(initial),
                "backward": scalar_metrics(trained),
                "after_one_step_same_window": scalar_metrics(after),
                "preclip_gradient_norm": float(preclip),
                "gradients": gradients,
                "updates": update_inventory(before, model),
                "optimizer_steps": 1,
                "learning_rate": LR,
                "arm_phase_cuda_peak_before_step": peak_before_step,
                "arm_phase_cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "arm_phase_cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            })
            del optimizer, initial, trained, after
            model.zero_grad(set_to_none=True)
            observe(label + "_complete")
        return {
            "status": "P064_TEMPORAL_FORCE_DELTA_ONE_STEP_PROBE_COMPLETE_NOT_A_CANDIDATE",
            "window": {
                "dataset": "b00_projected_ppo_train",
                "start": FIXED_START,
                "reason": FIXED_REASON,
                "metadata": metadata,
                "hdf_sha256": args.hdf_sha256,
            },
            "protocol": {
                "official_fno": True,
                "network_input_channels": 6,
                "force_output_parameterizations": ["absolute", "temporal_residual"],
                "objective": "unchanged_half_H1_half_AR_absolute_force_loss",
                "rollout_steps": 100,
                "chunk_size": 10,
                "gradient_clip_norm": CLIP,
                "one_engineering_optimizer_step_per_arm": True,
                "candidate_saved": False,
                "dev_or_frozen_accessed": False,
                "interpretation": "two real optimizer steps on isolated arm copies; no candidate checkpoint is saved",
            },
            "residual_initialization": residual_initialization,
            "arms": arms,
            "precision": precision,
            "allocator": {
                "cap_bytes": ALLOCATOR_BYTES,
                "device_total_bytes": total_device_memory,
                "fraction": allocator_fraction,
            },
            "resources": resources,
            "optimizer_steps_total": 2,
            "models_saved": 0,
            "scientific_admission": False,
        }
    finally:
        reader.close()
        dataset.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--hdf", type=Path, required=True)
    parser.add_argument("--hdf-sha256", required=True)
    parser.add_argument("--normalization-sha256", required=True)
    parser.add_argument("--data-manifest-sha256", required=True)
    parser.add_argument("--residual-module", type=Path, required=True)
    parser.add_argument("--p013-objective", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print("P064_TEMPORAL_DELTA_PREPARATION_ONLY_NO_GPU")
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    residual = load_path(args.residual_module, "p064_temporal_force_delta")
    p013 = load_path(args.p013_objective, "p064_p013_objective")
    result = execute(args, residual, p013)
    result["source_sha256"] = {
        "worker": sha(Path(__file__)),
        "residual_module": sha(args.residual_module),
        "p013_objective": sha(args.p013_objective),
        "config": sha(args.config),
    }
    args.output.mkdir(parents=True)
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
