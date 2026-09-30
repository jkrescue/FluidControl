#!/usr/bin/env python3
"""Audit HydroGym rotary-cylinder policies with physical force metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def build_env():
    import hydrogym.firedrake as hgym
    from hydrogym import FlowEnv

    probes = [
        (float(x), float(y))
        for x in np.linspace(1.0, 8.0, 10)
        for y in np.linspace(-1.0, 1.0, 5)
    ]
    return FlowEnv(
        {
            "flow": hgym.RotaryCylinder,
            "flow_config": {
                "Re": 100.0,
                "mesh": "medium",
                "observation_type": "pressure_probes",
                "probes": probes,
            },
            "solver": hgym.SemiImplicitBDF,
            "solver_config": {"dt": 0.01, "order": 3, "stabilization": "none"},
            "actuation_config": {"num_substeps": 1, "reward_aggregation": "mean"},
            "callbacks": [],
            "max_steps": 1_000_000,
        }
    )


def row(step, reward, requested_action, env):
    cl, cd = env.flow.compute_forces()
    return {
        "step": int(step),
        "time": float(env.flow.t),
        "reward": float(np.asarray(reward).item()),
        "requested_action": float(np.asarray(requested_action).reshape(-1)[0]),
        "applied_action": float(np.asarray(env.flow.control_state[0]).item()),
        "cl": float(cl),
        "cd": float(cd),
    }


def evaluate_plain(name, steps, seed):
    env = build_env()
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    rows = []
    for step in range(steps):
        if name == "zero":
            action = np.zeros(env.action_space.shape, dtype=np.float64)
        elif name == "random":
            action = rng.uniform(env.action_space.low, env.action_space.high)
        else:
            raise ValueError(name)
        obs, reward, terminated, truncated, _ = env.step(action)
        rows.append(row(step + 1, reward, action, env))
        if terminated or truncated:
            break
    env.close()
    return rows


def evaluate_ppo(steps, seed, model_path, stats_path):
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    raw_vec = DummyVecEnv([lambda: Monitor(build_env())])
    raw_vec.seed(seed)
    env = VecNormalize.load(str(stats_path), raw_vec)
    env.training = False
    env.norm_reward = False
    model = PPO.load(str(model_path), env=env, device="cpu")
    obs = env.reset()
    base_env = raw_vec.envs[0].unwrapped
    rows = []
    for step in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, _ = env.step(action)
        rows.append(row(step + 1, rewards[0], action, base_env))
        if bool(dones[0]):
            break
    env.close()
    return rows


def summarize(rows, warmup):
    selected = rows[warmup:]
    if not selected:
        raise ValueError("warmup must be smaller than the number of completed steps")
    cl = np.asarray([r["cl"] for r in selected])
    cd = np.asarray([r["cd"] for r in selected])
    action = np.asarray([r["applied_action"] for r in selected])
    reward = np.asarray([r["reward"] for r in selected])
    return {
        "steps": len(rows),
        "analysis_steps": len(selected),
        "warmup_steps": warmup,
        "reward_sum": float(reward.sum()),
        "reward_mean": float(reward.mean()),
        "cd_mean": float(cd.mean()),
        "cd_std": float(cd.std()),
        "cl_mean": float(cl.mean()),
        "cl_rms": float(np.sqrt(np.mean(cl**2))),
        "cl_std": float(cl.std()),
        "action_rms": float(np.sqrt(np.mean(action**2))),
        "action_energy_mean": float(np.mean(action**2)),
        "max_abs_action": float(np.max(np.abs(action))),
        "finite": bool(np.isfinite(cl).all() and np.isfinite(cd).all()),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vec-normalize", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "environment": {
            "flow": "RotaryCylinder",
            "reynolds": 100.0,
            "mesh": "medium",
            "dt": 0.01,
            "observation": "50 pressure probes",
            "reward": "-dt * Cd",
        },
        "policies": {},
    }

    for name in ("zero", "random"):
        rows = evaluate_plain(name, args.steps, args.seed)
        write_rows(args.output / f"{name}.csv", rows)
        report["policies"][name] = summarize(rows, args.warmup)

    rows = evaluate_ppo(args.steps, args.seed, args.model, args.vec_normalize)
    write_rows(args.output / "ppo_smoke.csv", rows)
    report["policies"]["ppo_smoke"] = summarize(rows, args.warmup)

    zero = report["policies"]["zero"]
    for name in ("random", "ppo_smoke"):
        values = report["policies"][name]
        values["cd_change_vs_zero_percent"] = 100.0 * (values["cd_mean"] / zero["cd_mean"] - 1.0)
        values["cl_rms_change_vs_zero_percent"] = 100.0 * (values["cl_rms"] / zero["cl_rms"] - 1.0)

    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("HYDROGYM_POLICY_AUDIT_OK")


def write_rows(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
