#!/usr/bin/env python3
"""Train and audit SB3 PPO on HydroGym's official Firedrake rotary cylinder."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import gymnasium as gym
import hydrogym.firedrake as hgym
import numpy as np
from hydrogym import FlowEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

HF_REVISION = "cf5ef6748443ae72e9647f398487e98d421a7243"
PROBES = [
    (float(x), float(y))
    for x in np.linspace(1.0, 8.0, 10)
    for y in np.linspace(-1.0, 1.0, 5)
]


class PhysicalReward(gym.Wrapper):
    """Add auditable drag, lift-variance, actuation and rate terms."""

    def __init__(
        self, env: FlowEnv, lift_weight: float, action_weight: float, rate_weight: float
    ):
        super().__init__(env)
        self.lift_weight = lift_weight
        self.action_weight = action_weight
        self.rate_weight = rate_weight
        self.previous_action = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous_action = float(np.asarray(self.env.flow.control_state[0]).item())
        return obs, info

    def step(self, action):
        obs, default_reward, terminated, truncated, info = self.env.step(action)
        cl, cd = (float(x) for x in self.env.flow.compute_forces())
        applied = float(np.asarray(self.env.flow.control_state[0]).item())
        dt = float(self.env.solver.dt)
        terms = {
            "drag": -dt * cd,
            "lift": -dt * self.lift_weight * cl * cl,
            "actuation": -dt * self.action_weight * applied * applied,
            "rate": -dt * self.rate_weight * (applied - self.previous_action) ** 2,
        }
        self.previous_action = applied
        reward = float(sum(terms.values()))
        info.update(
            cl=cl,
            cd=cd,
            applied_action=applied,
            requested_action=float(np.asarray(action).reshape(-1)[0]),
            default_reward=float(default_reward),
            **{f"reward_{key}": value for key, value in terms.items()},
            physical_reward=reward,
        )
        return obs, reward, terminated, truncated, info


def checkpoints(cache: Path) -> list[Path]:
    pattern = (
        "hub/datasets--dynamicslab--HydroGym-environments/snapshots/"
        f"{HF_REVISION}/RotaryCylinder_2D_Re100_medium_FD/*.ckpt"
    )
    paths = sorted(cache.glob(pattern))
    if len(paths) != 22 or not all(path.exists() for path in paths):
        raise RuntimeError(
            f"Expected 22 official rotary-cylinder checkpoints, found {len(paths)}"
        )
    return paths


def make_env(restarts: list[Path], args: argparse.Namespace) -> gym.Env:
    flow = hgym.RotaryCylinder
    raw = FlowEnv(
        {
            "flow": flow,
            "flow_config": {
                "Re": 100.0,
                "mesh": "medium",
                "observation_type": "pressure_probes",
                "probes": PROBES,
                "restart": [str(path) for path in restarts],
                "use_HF_data_manager": False,
            },
            "solver": hgym.SemiImplicitBDF,
            "solver_config": {"dt": 0.01, "order": 3, "stabilization": "none"},
            "actuation_config": {"num_substeps": 1, "reward_aggregation": "mean"},
            "callbacks": [],
            "max_steps": 1_000_000,
        }
    )
    return PhysicalReward(raw, args.lift_weight, args.action_weight, args.rate_weight)


class SaveEvery(BaseCallback):
    def __init__(self, directory: Path, interval: int = 2048):
        super().__init__()
        self.directory = directory
        self.interval = interval

    def _on_step(self) -> bool:
        if self.num_timesteps % self.interval == 0:
            self.model.save(self.directory / f"model_{self.num_timesteps}.zip")
            self.training_env.save(
                str(self.directory / f"vecnorm_{self.num_timesteps}.pkl")
            )
        return True


def train(args: argparse.Namespace, paths: list[Path]) -> None:
    train_paths = [paths[0]]
    env = VecNormalize(
        DummyVecEnv([lambda: Monitor(make_env(train_paths, args))]),
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
    )
    model = PPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        device="cpu",
        verbose=1,
        n_steps=256,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        tensorboard_log=str(args.output / "tensorboard"),
    )
    model.learn(total_timesteps=args.timesteps, callback=SaveEvery(args.output))
    model.save(args.output / "model_final.zip")
    env.save(str(args.output / "vecnorm_final.pkl"))
    env.close()


def evaluate_one(
    args: argparse.Namespace, checkpoint: Path, policy: str
) -> tuple[list[dict], dict]:
    base = DummyVecEnv([lambda: Monitor(make_env([checkpoint], args))])
    env = VecNormalize.load(str(args.output / "vecnorm_final.pkl"), base)
    env.training = False
    env.norm_reward = False
    model = (
        PPO.load(args.output / "model_final.zip", env=env, device="cpu")
        if policy == "ppo"
        else None
    )
    env.seed(args.seed)
    obs = env.reset()
    rows = []
    for step in range(args.eval_steps):
        action = (
            model.predict(obs, deterministic=True)[0]
            if model is not None
            else np.zeros((1, 1), dtype=np.float32)
        )
        obs, _, done, infos = env.step(action)
        row = {
            "checkpoint": checkpoint.name,
            "policy": policy,
            "step": step + 1,
            **infos[0],
        }
        rows.append(
            {
                key: value
                for key, value in row.items()
                if isinstance(value, (str, int, float, np.number))
            }
        )
        if bool(done[0]):
            break
    env.close()
    values = rows[args.warmup :]
    if not values:
        raise RuntimeError("Evaluation ended before warmup completed")
    cl = np.array([row["cl"] for row in values], dtype=float)
    cd = np.array([row["cd"] for row in values], dtype=float)
    act = np.array([row["applied_action"] for row in values], dtype=float)
    reward = np.array([row["physical_reward"] for row in values], dtype=float)
    result = {
        "checkpoint": checkpoint.name,
        "policy": policy,
        "steps": len(rows),
        "analysis_steps": len(values),
        "cd_mean": float(cd.mean()),
        "cl_rms": float(np.sqrt(np.mean(cl**2))),
        "action_rms": float(np.sqrt(np.mean(act**2))),
        "physical_reward_mean": float(reward.mean()),
        "finite": bool(
            np.isfinite(cl).all()
            and np.isfinite(cd).all()
            and np.isfinite(reward).all()
        ),
    }
    return rows, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timesteps", type=int, default=8192)
    parser.add_argument("--eval-steps", type=int, default=600)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lift-weight", type=float, default=0.2)
    parser.add_argument("--action-weight", type=float, default=0.01)
    parser.add_argument("--rate-weight", type=float, default=0.001)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if args.timesteps < 1 or args.eval_steps <= args.warmup or args.warmup < 0:
        parser.error("Invalid training/evaluation lengths")
    if min(args.lift_weight, args.action_weight, args.rate_weight) < 0:
        parser.error("Reward weights must be non-negative")
    paths = checkpoints(args.cache)
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.evaluate_only:
        if (args.output / "model_final.zip").exists():
            raise FileExistsError("Refusing to overwrite a completed model")
        train(args, paths)
    records = []
    summaries = []
    for checkpoint in paths[-5:]:
        for policy in ("zero", "ppo"):
            rows, summary = evaluate_one(args, checkpoint, policy)
            records.extend(rows)
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
    with (args.output / "heldout_rollouts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    paired = []
    for index in range(0, len(summaries), 2):
        zero, ppo = summaries[index : index + 2]
        paired.append(
            {
                "checkpoint": zero["checkpoint"],
                "cd_change_pct": 100 * (ppo["cd_mean"] / zero["cd_mean"] - 1),
                "cl_rms_change_pct": 100 * (ppo["cl_rms"] / zero["cl_rms"] - 1),
                "reward_change": ppo["physical_reward_mean"]
                - zero["physical_reward_mean"],
            }
        )
    report = {
        "hydrogym_commit": "4ab9854dea3d84e38a59c25e0f5835a00cf8225f",
        "checkpoint_revision": HF_REVISION,
        "environment": "Firedrake RotaryCylinder Re100 medium 50 pressure probes",
        "training_checkpoints": [paths[0].name],
        "heldout_checkpoints": [path.name for path in paths[-5:]],
        "reward_weights": {
            "drag": 1.0,
            "lift_square": args.lift_weight,
            "action_square": args.action_weight,
            "action_delta_square": args.rate_weight,
        },
        "training_timesteps": args.timesteps,
        "eval_steps": args.eval_steps,
        "warmup_steps": args.warmup,
        "summaries": summaries,
        "paired": paired,
    }
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print("HYDROGYM_ROTARY_PHYSICAL_REWARD_AUDIT_OK", flush=True)


if __name__ == "__main__":
    main()
