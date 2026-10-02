#!/usr/bin/env python3
"""Time official PhysicsNeMo FNO inference on a real CFD action sequence."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import h5py
import torch
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import load_checkpoint

from evaluate_tandem_fno import load_composed_config
from train_tandem_fno import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--normalization-data", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if args.steps < 1 or not torch.cuda.is_available():
        parser.error("positive steps and CUDA are required")

    DistributedManager.initialize()
    dist = DistributedManager()
    if dist.distributed:
        raise RuntimeError("benchmark requires one GPU process")
    torch.cuda.set_per_process_memory_fraction(0.20, device=dist.device)
    cfg = load_composed_config(Path("conf/tandem_fno_total_drag.yaml"))
    network = build_model(cfg).to(dist.device).eval()
    metadata: dict = {}
    epoch = load_checkpoint(
        args.checkpoint_dir, models=network, metadata_dict=metadata,
        device=dist.device,
    )
    if epoch < 1:
        raise FileNotFoundError("missing official PhysicsNeMo checkpoint")

    stats = json.loads((args.normalization_data / "normalization.json").read_text())
    manifest = json.loads((args.normalization_data / "manifest.json").read_text())
    scale = float(manifest["max_abs_omega"])
    with h5py.File(args.case, "r") as handle:
        if len(handle["omega"]) < args.steps + 1:
            raise ValueError("CFD trajectory is shorter than benchmark horizon")
        initial = torch.from_numpy(handle["state"][0]).float().to(dist.device)
        mask = torch.from_numpy(handle["mask"][0]).float().to(dist.device)
        omega = torch.from_numpy(handle["omega"][: args.steps + 1, 0]).float().to(dist.device)
    mean = torch.tensor(stats["state_mean"], device=dist.device)[:, None, None]
    std = torch.tensor(stats["state_std"], device=dist.device)[:, None, None]
    initial = ((initial - mean) / std * mask).unsqueeze(0)
    mask = mask.unsqueeze(0)
    height, width = mask.shape[-2:]

    def step(state: torch.Tensor, index: int) -> torch.Tensor:
        action_now = (omega[index] / scale).expand(1, 1, height, width)
        action_next = (omega[index + 1] / scale).expand(1, 1, height, width)
        raw = network(torch.cat((state, mask, action_now, action_next), dim=1))
        if raw.shape != (1, 7, height, width):
            raise ValueError(f"unexpected FNO output shape {tuple(raw.shape)}")
        return (state + raw[:, :3]) * mask

    with torch.inference_mode():
        state = initial
        for index in range(min(10, args.steps)):
            state = step(state, index)
        torch.cuda.synchronize(dist.device)
        state = initial
        durations_ms = []
        for index in range(args.steps):
            torch.cuda.synchronize(dist.device)
            started = time.perf_counter()
            state = step(state, index)
            torch.cuda.synchronize(dist.device)
            durations_ms.append((time.perf_counter() - started) * 1000)
    if not bool(torch.isfinite(state).all()):
        raise FloatingPointError("non-finite benchmark rollout")
    ordered = sorted(durations_ms)
    report = {
        "status": "FNO_REAL_CFD_INFERENCE_BENCHMARK_OK",
        "scope": "model_forward_only_single_real_CFD_segment_not_end_to_end_control_latency",
        "checkpoint_epoch": epoch,
        "checkpoint_dir": str(args.checkpoint_dir),
        "case": str(args.case),
        "steps": args.steps,
        "batch_size": 1,
        "grid": [height, width],
        "model": "physicsnemo.models.fno.FNO",
        "step_median_ms": statistics.median(durations_ms),
        "step_p95_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "rollout_100_step_seconds": sum(durations_ms) / 1000 if args.steps == 100 else None,
        "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(dist.device) / 1024**3,
        "action_scale": scale,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
