#!/usr/bin/env python3
"""Audit one frozen PPO checkpoint on all independent tandem CFD starts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from physicsnemo.utils import load_checkpoint
from stable_baselines3 import PPO

from evaluate_tandem_fno import load_composed_config
from train_tandem_fno import build_model
from train_tandem_hydrogym_ppo_pilot import TEST, VALIDATION, summarize_evaluations
from train_tandem_hydrogym_ppo_smoke import evaluate_episode, make_env

HYDROGYM_COMMIT = "4ab9854dea3d84e38a59c25e0f5835a00cf8225f"


def episode(
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
    parser.add_argument("--policy", type=Path, required=True)
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
    if args.horizon < 2:
        parser.error("horizon must be at least 2")
    if not args.policy.is_file():
        raise FileNotFoundError(args.policy)

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
    policy = PPO.load(args.policy, device=device)

    audits = []
    for split, cases in (("validation", VALIDATION), ("test", TEST)):
        for case in cases:
            for frame in (0, 100, 400):
                zero = episode(
                    data=args.data, split=split, case=case, frame=frame,
                    horizon=args.horizon, network=network, epoch=epoch,
                    policy=None, device=device,
                )
                controlled = episode(
                    data=args.data, split=split, case=case, frame=frame,
                    horizon=args.horizon, network=network, epoch=epoch,
                    policy=policy, device=device,
                )
                audits.append({
                    "split": split,
                    "case": case,
                    "initial_frame": frame,
                    "status": "evaluated",
                    "zero": zero,
                    "policy": controlled,
                    "reward_change": controlled["reward_sum"] - zero["reward_sum"],
                    "cd_mean_change": controlled["cd_mean"] - zero["cd_mean"],
                    "cl_rms_change": controlled["cl_rms"] - zero["cl_rms"],
                })
    summary = summarize_evaluations(audits)
    report = {
        "status": "SURROGATE_RL_PILOT_EVALUATED",
        "scientific_status": "frozen_fno_hydrogym_surrogate_only_not_real_cfd_control",
        "physicsnemo_checkpoint_epoch": epoch,
        "hydrogym_commit": HYDROGYM_COMMIT,
        "policy_checkpoint": str(args.policy),
        "device": str(device),
        "episode_steps": args.horizon,
        "training_environments": 8,
        "summary_weighting": "common_t80_frame0_counted_once_per_split",
        "summary": summary,
        "evaluations": audits,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
