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
    parser.add_argument("--checkpoint-interval", type=int, default=128)
    parser.add_argument("--checkpoint-eval-steps", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()
    if args.timesteps < 256 or args.timesteps % 32 or args.episode_steps < 16:
        parser.error("timesteps must be a multiple of 32 and >=256; episode-steps >=16")
    if (
        args.checkpoint_interval < 32
        or args.checkpoint_interval % 32
        or args.timesteps % args.checkpoint_interval
        or args.checkpoint_eval_steps < 2
    ):
        parser.error("checkpoint interval must divide timesteps and be a multiple of 32")
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
    cases = (("train", "expanded_train_00"),) + tuple(
        (split, case) for split, group in (("validation", VALIDATION), ("test", TEST)) for case in group
    )
    for split, case in cases:
        if not (args.data / split / f"{case}.h5").is_file():
            raise FileNotFoundError(f"missing real CFD start: {split}/{case}")
    torch.set_num_threads(2)
    cfg = load_composed_config(args.config)
    network = build_model(cfg).to(device).eval().requires_grad_(False)
    epoch = load_checkpoint(args.checkpoint_dir, models=network, device=device)
    if epoch != 20:
        raise ValueError(f"checkpoint epoch {epoch} does not match gate")
    train_env = make_env(
        data=args.data, split="train", case="expanded_train_00",
        frame=0, network=network, epoch=epoch, episode_steps=args.episode_steps,
        device=device,
    )
    training_state_bound = float(train_env.env.env.flow.training_state_bound)
    state_divergence_guard = float(train_env.env.max_abs_normalized_state)
    model = PPO(
        "MlpPolicy", train_env, seed=args.seed, device=device,
        n_steps=32, batch_size=32, n_epochs=2,
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
    for split, case in cases[1:]:
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
        "training_state_bound": training_state_bound,
        "state_divergence_guard": state_divergence_guard,
        "device": str(device),
        "gpu_memory_fraction": args.gpu_memory_fraction if device.type == "cuda" else None,
        "readiness_gate": str(args.readiness),
        "hydrogym_commit": "4ab9854dea3d84e38a59c25e0f5835a00cf8225f",
        "training_case": "expanded_train_00",
        "initial_frame": 0,
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
