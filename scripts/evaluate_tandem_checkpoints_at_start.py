#!/usr/bin/env python3
"""Evaluate every PPO checkpoint at one held-out real-CFD-derived start."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from physicsnemo.utils import load_checkpoint
from stable_baselines3 import PPO

from evaluate_tandem_fno import load_composed_config
from train_tandem_fno import build_model
from train_tandem_hydrogym_ppo_smoke import evaluate_episode, make_env


def evaluate(
    *, data: Path, split: str, case: str, frame: int, horizon: int,
    network: torch.nn.Module, epoch: int, policy: PPO | None, device: torch.device,
) -> dict:
    env = make_env(
        data=data, split=split, case=case, frame=frame, network=network,
        epoch=epoch, episode_steps=horizon, device=device,
    )
    try:
        return evaluate_episode(env, policy, horizon)
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--policy-run", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if not output.is_relative_to(project / "artifacts" / "hydrogym"):
        parser.error("output must be under artifacts/hydrogym")
    if output.exists():
        raise FileExistsError(output)
    if args.frame < 0 or args.horizon < 2:
        parser.error("frame must be nonnegative and horizon must be at least 2")
    checkpoints = sorted((args.policy_run / "checkpoints").glob("ppo_*.zip"))
    if not checkpoints:
        raise FileNotFoundError("no PPO checkpoints found")

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available() or device.index not in (None, 0):
            parser.error("GPU0 must be the only selected CUDA device")
        torch.cuda.set_per_process_memory_fraction(0.20, device=0)
    torch.set_num_threads(2)
    network = build_model(load_composed_config(args.config)).to(device).eval().requires_grad_(False)
    epoch = load_checkpoint(args.checkpoint_dir, models=network, device=device)
    if epoch != 20:
        raise ValueError(f"expected PhysicsNeMo epoch 20, got {epoch}")

    starts = (
        ("validation", "expanded_validation_00"),
        ("test", "expanded_test_00"),
    )
    rows = []
    for split, case in starts:
        zero = evaluate(
            data=args.data, split=split, case=case, frame=args.frame,
            horizon=args.horizon, network=network, epoch=epoch,
            policy=None, device=device,
        )
        for checkpoint in checkpoints:
            policy = PPO.load(checkpoint, device=device)
            controlled = evaluate(
                data=args.data, split=split, case=case, frame=args.frame,
                horizon=args.horizon, network=network, epoch=epoch,
                policy=policy, device=device,
            )
            rows.append({
                "split": split,
                "case": case,
                "frame": args.frame,
                "timesteps": int(checkpoint.stem.rsplit("_", 1)[1]),
                "checkpoint": str(checkpoint),
                "zero": zero,
                "policy": controlled,
                "reward_change": controlled["reward_sum"] - zero["reward_sum"],
                "cd_mean_change": controlled["cd_mean"] - zero["cd_mean"],
                "cl_rms_change": controlled["cl_rms"] - zero["cl_rms"],
            })
    by_timestep = {}
    for row in rows:
        by_timestep.setdefault(row["timesteps"], []).append(row)
    summary = []
    for timestep, group in sorted(by_timestep.items()):
        reward = [row["reward_change"] for row in group]
        if max(reward) - min(reward) > 1e-7:
            raise ValueError(f"shared start mismatch at checkpoint {timestep}")
        summary.append({
            "timesteps": timestep,
            "shared_start_rows": len(group),
            "reward_change": sum(reward) / len(reward),
            "cd_mean_change": sum(row["cd_mean_change"] for row in group) / len(group),
            "cl_rms_change": sum(row["cl_rms_change"] for row in group) / len(group),
            "passes_positive_reward_gate": reward[0] > 0,
        })
    report = {
        "status": "CHECKPOINT_START_AUDIT_COMPLETE",
        "scientific_scope": "frozen_physicsnemo_fno_surrogate_only",
        "physicsnemo_checkpoint_epoch": epoch,
        "frame": args.frame,
        "horizon": args.horizon,
        "device": str(device),
        "summary": summary,
        "evaluations": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
