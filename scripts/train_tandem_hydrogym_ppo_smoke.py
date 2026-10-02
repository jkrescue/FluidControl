#!/usr/bin/env python3
"""Short, explicitly surrogate-only HydroGym PPO software smoke on tandem CFD starts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from hydrogym import FlowEnv
from physicsnemo.utils import load_checkpoint
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from evaluate_tandem_fno import load_composed_config
from fluid_control.tandem_hydrogym import TandemFNOStepper, TandemRewardAudit, TandemSurrogateFlow
from train_tandem_fno import build_model


def make_env(
    *, data: Path, split: str, case: str, frame: int,
    network: torch.nn.Module, epoch: int, episode_steps: int,
    device: torch.device | str = "cpu",
    reward_mode: str = "legacy_rear",
    phase_baseline: Mapping[str, float | str] | None = None,
    shedding_period: float = 6.15,
):
    raw = FlowEnv({
        "flow": TandemSurrogateFlow,
        "flow_config": {
            "data_root": data,
            "split": split,
            "case": case,
            "frame": frame,
            "network": network,
            "device": device,
            "checkpoint_epoch": epoch,
            "max_delta_omega": 0.5,
            "reward_mode": reward_mode,
            "phase_baseline": phase_baseline,
            "shedding_period": shedding_period,
        },
        "solver": TandemFNOStepper,
        "solver_config": {"dt": 0.1},
        "max_steps": episode_steps,
    })
    return Monitor(TandemRewardAudit(raw))


def evaluate_episode(env, policy, horizon: int) -> dict:
    obs, _ = env.reset()
    rows = []
    for step in range(horizon):
        action = (
            np.zeros((1,), dtype=np.float32)
            if policy is None
            else policy.predict(obs, deterministic=True)[0]
        )
        obs, reward, terminated, truncated, info = env.step(action)
        rows.append({
            "step": step + 1,
            "reward": float(reward),
            "cd": info["predicted_cd"],
            "cl": info["predicted_cl"],
            "requested_omega": info["requested_omega"],
            "applied_omega": info["applied_omega"],
            "rate_limited": info["rate_limited"],
            "state_bound": info["max_abs_normalized_state"],
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "termination_reason": info.get("termination_reason"),
            **{key: float(value) for key, value in info.items() if key.startswith("reward_")},
        })
        audited = sum(value for key, value in rows[-1].items() if key.startswith("reward_"))
        if abs(audited - reward) > 1e-7:
            raise AssertionError("audited reward components do not sum to reward")
        if terminated or truncated:
            break
    if len(rows) != horizon or any(row["terminated"] for row in rows):
        raise RuntimeError("surrogate rollout ended early or diverged")
    cd = np.asarray([row["cd"] for row in rows])
    cl = np.asarray([row["cl"] for row in rows])
    omega = np.asarray([row["applied_omega"] for row in rows])
    rewards = np.asarray([row["reward"] for row in rows])
    if not all(np.isfinite(a).all() for a in (cd, cl, omega, rewards)):
        raise FloatingPointError("non-finite surrogate evaluation metric")
    component_errors = [
        abs(sum(v for k, v in row.items() if k.startswith("reward_")) - row["reward"])
        for row in rows
    ]
    return {
        "steps": len(rows),
        "reward_sum": float(rewards.sum()),
        "reward_components_sum_error_max": max(component_errors),
        "cd_mean": float(cd.mean()),
        "cl_rms": float(np.sqrt(np.mean(cl**2))),
        "action_rms": float(np.sqrt(np.mean(omega**2))),
        "max_abs_omega": float(np.abs(omega).max()),
        "max_abs_normalized_state": max(row["state_bound"] for row in rows),
        "rate_limited_steps": sum(row["rate_limited"] for row in rows),
        "trajectory": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timesteps", type=int, default=32)
    parser.add_argument("--episode-steps", type=int, default=16)
    parser.add_argument("--train-case", default="expanded_train_00")
    parser.add_argument("--validation-case", default="expanded_validation_00")
    parser.add_argument("--frame", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()
    if args.timesteps < 16 or args.episode_steps < 2:
        parser.error("timesteps>=16 and episode-steps>=2 required")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    for split, case in (("train", args.train_case), ("validation", args.validation_case)):
        if not (args.data / split / f"{case}.h5").is_file():
            parser.error(f"missing real CFD {split} case: {case}")
    torch.set_num_threads(2)
    cfg = load_composed_config(args.config)
    network = build_model(cfg).cpu().eval()
    epoch = load_checkpoint(
        args.checkpoint_dir, models=network, device=torch.device("cpu")
    )
    if epoch < 1:
        raise FileNotFoundError("missing PhysicsNeMo checkpoint")
    train_env = make_env(
        data=args.data, split="train", case=args.train_case,
        frame=args.frame, network=network, epoch=epoch,
        episode_steps=args.episode_steps,
    )
    model = PPO(
        "MlpPolicy", train_env, seed=args.seed, device="cpu",
        n_steps=16, batch_size=16, n_epochs=1,
        learning_rate=3e-4, gamma=0.99, verbose=0,
    )
    model.learn(total_timesteps=args.timesteps)
    validation_env = make_env(
        data=args.data, split="validation", case=args.validation_case,
        frame=args.frame, network=network, epoch=epoch,
        episode_steps=args.episode_steps,
    )
    zero = evaluate_episode(validation_env, None, args.episode_steps)
    ppo = evaluate_episode(validation_env, model, args.episode_steps)
    args.output.mkdir(parents=True)
    model.save(args.output / "ppo_smoke.zip")
    report = {
        "status": "TANDEM_HYDROGYM_PPO_SMOKE_OK",
        "scientific_status": "surrogate_only_software_smoke_not_cfd_control",
        "hydrogym_commit": "4ab9854dea3d84e38a59c25e0f5835a00cf8225f",
        "physicsnemo_checkpoint_epoch": epoch,
        "real_cfd_train_case": args.train_case,
        "real_cfd_validation_case": args.validation_case,
        "initial_frame": args.frame,
        "seed": args.seed,
        "ppo_timesteps": args.timesteps,
        "episode_steps": args.episode_steps,
        "zero": zero,
        "ppo": ppo,
        "validation_reward_change": ppo["reward_sum"] - zero["reward_sum"],
        "validation_cd_change_pct": 100 * (ppo["cd_mean"] / zero["cd_mean"] - 1),
        "validation_cl_rms_change_pct": 100 * (ppo["cl_rms"] / zero["cl_rms"] - 1),
    }
    (args.output / "audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        key: value for key, value in report.items() if key not in {"zero", "ppo"}
    }, indent=2))
    print("TANDEM_HYDROGYM_PPO_SMOKE_OK", flush=True)
    validation_env.close()
    train_env.close()


if __name__ == "__main__":
    main()
