#!/usr/bin/env python3
"""Guarded HydroGym PPO pilot on a frozen PhysicsNeMo FNO; surrogate only."""
from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch
from physicsnemo.utils import load_checkpoint
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from evaluate_tandem_fno import load_composed_config
from train_tandem_fno import build_model
from train_tandem_hydrogym_ppo_smoke import evaluate_episode, make_env


INDEPENDENT_IDS = (0, 1, 2, 4)
VALIDATION = tuple(f"expanded_validation_{index:02d}" for index in INDEPENDENT_IDS)
TEST = tuple(f"expanded_test_{index:02d}" for index in INDEPENDENT_IDS)
TRAIN_STARTS = (
    ("expanded_train_00", 100),
    ("expanded_train_03", 200),
    ("expanded_train_06", 300),
    ("expanded_train_09", 400),
    ("expanded_train_12", 500),
    ("expanded_train_15", 600),
    ("expanded_train_18", 700),
    ("expanded_train_21", 250),
)
TARGETED_T80_TRAIN_STARTS = (
    # Four stochastic PPO environments intentionally weight the one physical
    # restart available for matched real-CFD replay; the other four retain
    # distinct real CFD histories to limit single-state overfitting.
    ("expanded_train_00", 0),
    ("expanded_train_03", 0),
    ("expanded_train_06", 0),
    ("expanded_train_09", 0),
    ("expanded_train_12", 200),
    ("expanded_train_15", 400),
    ("expanded_train_18", 600),
    ("expanded_train_21", 300),
)


def summarize_evaluations(audits: list[dict]) -> dict:
    summary = {}
    for split in ("validation", "test"):
        raw = [row for row in audits if row["split"] == split]
        common = [row for row in raw if row["initial_frame"] == 0]
        if len(common) != len(INDEPENDENT_IDS):
            raise ValueError(f"{split} must contain four common frame-0 evaluations")
        keys = ("reward_change", "cd_mean_change", "cl_rms_change")
        for key in keys:
            if max(row[key] for row in common) - min(row[key] for row in common) > 1e-7:
                raise ValueError(f"{split} frame-0 {key} is not a shared restart")
        unique = [common[0]] + [row for row in raw if row["initial_frame"] != 0]
        summary[split] = {
            "raw_evaluations": len(raw),
            "unique_initial_states": len(unique),
            "unique_cases": len(set(row["case"] for row in raw)),
            "initial_frames": [0, 100, 400],
            "common_frame0_deduplicated": True,
            "reward_change_mean": sum(row["reward_change"] for row in unique) / len(unique),
            "positive_reward_initial_states": sum(
                row["reward_change"] > 0 for row in unique
            ),
            "cd_mean_change_mean": sum(row["cd_mean_change"] for row in unique) / len(unique),
            "cl_rms_change_mean": sum(row["cl_rms_change"] for row in unique) / len(unique),
            "raw_reward_change_mean": sum(row["reward_change"] for row in raw) / len(raw),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timesteps", type=int, default=8192)
    parser.add_argument("--episode-steps", type=int, default=32)
    parser.add_argument("--checkpoint-interval", type=int, default=2048)
    parser.add_argument("--checkpoint-eval-steps", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument(
        "--training-profile", choices=("diverse", "t80_targeted"), default="diverse"
    )
    args = parser.parse_args()
    train_starts = (
        TARGETED_T80_TRAIN_STARTS
        if args.training_profile == "t80_targeted"
        else TRAIN_STARTS
    )
    rollout_batch = 32 * len(train_starts)
    if (
        args.timesteps < 2048
        or args.timesteps % rollout_batch
        or args.episode_steps < 16
    ):
        parser.error("timesteps must be >=2048 and divisible by the 256-step rollout")
    if (
        args.checkpoint_interval < rollout_batch
        or args.checkpoint_interval % rollout_batch
        or args.timesteps % args.checkpoint_interval
        or args.checkpoint_eval_steps < 2
    ):
        parser.error("checkpoint interval must divide timesteps and the rollout batch")
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available() or device.index not in (None, 0):
            parser.error("GPU0 must be the only selected CUDA device")
        if not 0 < args.gpu_memory_fraction <= 1:
            parser.error("gpu-memory-fraction must be in (0, 1]")
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device=0)
    project = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if not output.is_relative_to(project / "artifacts" / "hydrogym"):
        parser.error("output must be under project artifacts/hydrogym")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    gate = json.loads(args.readiness.read_text(encoding="utf-8"))
    if gate.get("status") != "CANDIDATE_SURROGATE_SCREEN_PASS" or gate.get("checkpoint_epoch") != 20:
        raise ValueError("20-epoch independent held-out surrogate gate has not passed")
    evaluation_cases = tuple(
        (split, case) for split, group in (("validation", VALIDATION), ("test", TEST)) for case in group
    )
    required_cases = tuple(("train", case) for case, _ in train_starts) + evaluation_cases
    for split, case in required_cases:
        if not (args.data / split / f"{case}.h5").is_file():
            raise FileNotFoundError(f"missing real CFD start: {split}/{case}")
    torch.set_num_threads(2)
    cfg = load_composed_config(args.config)
    network = build_model(cfg).to(device).eval().requires_grad_(False)
    epoch = load_checkpoint(args.checkpoint_dir, models=network, device=device)
    if epoch != 20:
        raise ValueError(f"checkpoint epoch {epoch} does not match gate")
    train_env = DummyVecEnv([
        partial(
            make_env,
            data=args.data,
            split="train",
            case=case,
            frame=frame,
            network=network,
            epoch=epoch,
            episode_steps=args.episode_steps,
            device=device,
        )
        for case, frame in train_starts
    ])
    training_state_bound = float(
        train_env.envs[0].env.env.flow.training_state_bound
    )
    state_divergence_guard = float(train_env.envs[0].env.max_abs_normalized_state)
    model = PPO(
        "MlpPolicy", train_env, seed=args.seed, device=device,
        n_steps=32, batch_size=256, n_epochs=4,
        learning_rate=3e-4, gamma=0.99, verbose=0,
    )
    output.mkdir(parents=True)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir()
    model.save(checkpoint_dir / "ppo_000000.zip")
    completed = 0
    try:
        while completed < args.timesteps:
            model.learn(
                total_timesteps=args.checkpoint_interval,
                reset_num_timesteps=(completed == 0),
            )
            completed += args.checkpoint_interval
            model.save(checkpoint_dir / f"ppo_{completed:06d}.zip")
    finally:
        train_env.close()
    model.save(output / "ppo_policy.zip")
    checkpoint_evaluations = []
    for checkpoint in sorted(checkpoint_dir.glob("ppo_*.zip")):
        checkpoint_policy = PPO.load(checkpoint, device=device)
        timestep = int(checkpoint.stem.split("_")[-1])
        rows = []
        for split, group in (("validation", VALIDATION), ("test", TEST)):
            for case in group:
                env = make_env(
                    data=args.data, split=split, case=case, frame=100,
                    network=network, epoch=epoch,
                    episode_steps=args.checkpoint_eval_steps,
                    device=device,
                )
                try:
                    metrics = evaluate_episode(
                        env, checkpoint_policy, args.checkpoint_eval_steps
                    )
                    rows.append({
                        "split": split, "case": case, "status": "evaluated",
                        **{key: value for key, value in metrics.items() if key != "trajectory"},
                    })
                except (FloatingPointError, RuntimeError, ValueError) as exc:
                    rows.append({
                        "split": split, "case": case,
                        "status": "evaluation_failed", "error": str(exc),
                    })
                finally:
                    env.close()
        stable = [row for row in rows if row["status"] == "evaluated"]
        checkpoint_evaluations.append({
            "timesteps": timestep,
            "panel_frame": 100,
            "horizon": args.checkpoint_eval_steps,
            "evaluated": len(stable),
            "total": len(rows),
            "reward_sum_mean": (
                sum(row["reward_sum"] for row in stable) / len(stable)
                if stable else None
            ),
            "cd_mean": (
                sum(row["cd_mean"] for row in stable) / len(stable)
                if stable else None
            ),
            "cl_rms_mean": (
                sum(row["cl_rms"] for row in stable) / len(stable)
                if stable else None
            ),
            "evaluations": rows,
        })
    (output / "checkpoint_evaluations.json").write_text(
        json.dumps(checkpoint_evaluations, indent=2) + "\n", encoding="utf-8"
    )
    audits = []
    # Frame 0 is the common t=80 restart in every trajectory; later frames
    # test independent CFD histories, not eight duplicates of the same state.
    for split, case in evaluation_cases:
        for frame in (0, 100, 400):
            env = make_env(
                data=args.data, split=split, case=case, frame=frame,
                network=network, epoch=epoch, episode_steps=args.episode_steps,
                device=device,
            )
            try:
                zero = evaluate_episode(env, None, args.episode_steps)
                zero_status = "evaluated"
                zero_error = None
            except (FloatingPointError, RuntimeError, ValueError) as exc:
                zero = None
                zero_status = "evaluation_failed"
                zero_error = str(exc)
            finally:
                env.close()
            env = make_env(
                data=args.data, split=split, case=case, frame=frame,
                network=network, epoch=epoch, episode_steps=args.episode_steps,
                device=device,
            )
            try:
                policy = evaluate_episode(env, model, args.episode_steps)
                policy_status = "evaluated"
                policy_error = None
            except (FloatingPointError, RuntimeError, ValueError) as exc:
                policy = None
                policy_status = "evaluation_failed"
                policy_error = str(exc)
            finally:
                env.close()
            if zero is not None and policy is not None:
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
            else:
                audits.append({
                    "split": split, "case": case, "initial_frame": frame,
                    "status": "evaluation_failed",
                    "zero_status": zero_status, "zero_error": zero_error,
                    "policy_status": policy_status, "policy_error": policy_error,
                })
    all_evaluated = all(row["status"] == "evaluated" for row in audits)
    summary = summarize_evaluations(audits) if all_evaluated else {}
    report = {
        "status": "SURROGATE_RL_PILOT_EVALUATED" if all_evaluated else "SURROGATE_RL_PILOT_EVALUATION_FAILED",
        "scientific_status": "frozen_fno_hydrogym_surrogate_only_not_real_cfd_control",
        "physicsnemo_checkpoint_epoch": epoch,
        "training_state_bound": training_state_bound,
        "state_divergence_guard": state_divergence_guard,
        "device": str(device),
        "gpu_memory_fraction": args.gpu_memory_fraction if device.type == "cuda" else None,
        "readiness_gate": str(args.readiness),
        "hydrogym_commit": "4ab9854dea3d84e38a59c25e0f5835a00cf8225f",
        "training_starts": [
            {"case": case, "initial_frame": frame} for case, frame in train_starts
        ],
        "training_environments": len(train_starts),
        "training_profile": args.training_profile,
        "ppo_timesteps": args.timesteps,
        "checkpoint_interval": args.checkpoint_interval,
        "checkpoint_evaluations": "checkpoint_evaluations.json",
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
