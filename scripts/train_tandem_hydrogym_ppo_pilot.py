#!/usr/bin/env python3
"""Guarded HydroGym PPO pilot on a frozen PhysicsNeMo FNO; surrogate only."""
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


INDEPENDENT_IDS = (0, 1, 2, 4)
VALIDATION = tuple(f"expanded_validation_{index:02d}" for index in INDEPENDENT_IDS)
TEST = tuple(f"expanded_test_{index:02d}" for index in INDEPENDENT_IDS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timesteps", type=int, default=512)
    parser.add_argument("--episode-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()
    if args.timesteps < 256 or args.timesteps % 32 or args.episode_steps < 16:
        parser.error("timesteps must be a multiple of 32 and >=256; episode-steps >=16")
    project = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if not output.is_relative_to(project / "artifacts" / "hydrogym"):
        parser.error("output must be under project artifacts/hydrogym")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    gate = json.loads(args.readiness.read_text(encoding="utf-8"))
    if gate.get("status") != "CANDIDATE_SURROGATE_SCREEN_PASS" or gate.get("checkpoint_epoch") != 20:
        raise ValueError("20-epoch independent held-out surrogate gate has not passed")
    cases = (("train", "expanded_train_00"),) + tuple(
        (split, case) for split, group in (("validation", VALIDATION), ("test", TEST)) for case in group
    )
    for split, case in cases:
        if not (args.data / split / f"{case}.h5").is_file():
            raise FileNotFoundError(f"missing real CFD start: {split}/{case}")
    torch.set_num_threads(2)
    cfg = load_composed_config(args.config)
    network = build_model(cfg).cpu().eval().requires_grad_(False)
    epoch = load_checkpoint(args.checkpoint_dir, models=network, device=torch.device("cpu"))
    if epoch != 20:
        raise ValueError(f"checkpoint epoch {epoch} does not match gate")
    train_env = make_env(
        data=args.data, split="train", case="expanded_train_00",
        frame=0, network=network, epoch=epoch, episode_steps=args.episode_steps,
    )
    model = PPO(
        "MlpPolicy", train_env, seed=args.seed, device="cpu",
        n_steps=32, batch_size=32, n_epochs=2,
        learning_rate=3e-4, gamma=0.99, verbose=0,
    )
    try:
        model.learn(total_timesteps=args.timesteps)
    finally:
        train_env.close()
    output.mkdir(parents=True)
    model.save(output / "ppo_policy.zip")
    audits = []
    # Frame 0 is the common t=80 restart in every trajectory; later frames
    # test independent CFD histories, not eight duplicates of the same state.
    for split, case in cases[1:]:
        for frame in (0, 100, 400):
            env = make_env(
                data=args.data, split=split, case=case, frame=frame,
                network=network, epoch=epoch, episode_steps=args.episode_steps,
            )
            try:
                zero = evaluate_episode(env, None, args.episode_steps)
                policy = evaluate_episode(env, model, args.episode_steps)
                audits.append({
                    "split": split,
                    "case": case,
                    "initial_frame": frame,
                    "status": "evaluated",
                    "zero": zero,
                    "policy": policy,
                    "reward_change": policy["reward_sum"] - zero["reward_sum"],
                    "cd_mean_change": policy["cd_mean"] - zero["cd_mean"],
                    "cl_rms_change": policy["cl_rms"] - zero["cl_rms"],
                })
            except (FloatingPointError, RuntimeError, ValueError) as exc:
                audits.append({
                    "split": split, "case": case, "initial_frame": frame,
                    "status": "evaluation_failed", "error": str(exc),
                })
            finally:
                env.close()
    all_evaluated = all(row["status"] == "evaluated" for row in audits)
    summary = {}
    if all_evaluated:
        for split in ("validation", "test"):
            group = [row for row in audits if row["split"] == split]
            summary[split] = {
                "evaluations": len(group),
                "unique_cases": len(set(row["case"] for row in group)),
                "initial_frames": [0, 100, 400],
                "reward_change_mean": sum(row["reward_change"] for row in group) / len(group),
                "positive_reward_cases": sum(row["reward_change"] > 0 for row in group),
                "cd_mean_change_mean": sum(row["cd_mean_change"] for row in group) / len(group),
                "cl_rms_change_mean": sum(row["cl_rms_change"] for row in group) / len(group),
            }
    report = {
        "status": "SURROGATE_RL_PILOT_EVALUATED" if all_evaluated else "SURROGATE_RL_PILOT_EVALUATION_FAILED",
        "scientific_status": "frozen_fno_hydrogym_surrogate_only_not_real_cfd_control",
        "physicsnemo_checkpoint_epoch": epoch,
        "readiness_gate": str(args.readiness),
        "hydrogym_commit": "4ab9854dea3d84e38a59c25e0f5835a00cf8225f",
        "training_case": "expanded_train_00",
        "initial_frame": 0,
        "ppo_timesteps": args.timesteps,
        "episode_steps": args.episode_steps,
        "seed": args.seed,
        "summary": summary,
        "evaluations": audits,
    }
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "evaluations"}, indent=2))
    if not all_evaluated:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
