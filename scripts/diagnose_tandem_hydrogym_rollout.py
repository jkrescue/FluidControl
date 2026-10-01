#!/usr/bin/env python3
"""Record every step of one zero-action and PPO tandem-surrogate rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from physicsnemo.utils import load_checkpoint
from stable_baselines3 import PPO

from evaluate_tandem_fno import load_composed_config
from train_tandem_fno import build_model
from train_tandem_hydrogym_ppo_smoke import make_env


def rollout(env, policy: PPO | None, horizon: int) -> dict:
    observation, _ = env.reset()
    trajectory: list[dict] = []
    status = "completed"
    error = None
    for step in range(1, horizon + 1):
        action = (
            np.zeros((1,), dtype=np.float32)
            if policy is None
            else policy.predict(observation, deterministic=True)[0]
        )
        try:
            observation, reward, terminated, truncated, info = env.step(action)
        except (FloatingPointError, RuntimeError, ValueError) as exc:
            status = "exception"
            error = f"{type(exc).__name__}: {exc}"
            break
        trajectory.append({
            "step": step,
            "reward": float(reward),
            "predicted_cd": float(info["predicted_cd"]),
            "predicted_cl": float(info["predicted_cl"]),
            "requested_omega": float(info["requested_omega"]),
            "applied_omega": float(info["applied_omega"]),
            "max_abs_normalized_state": float(info["max_abs_normalized_state"]),
            "max_abs_normalized_state_guard": float(
                info["max_abs_normalized_state_guard"]
            ),
            "training_state_bound": float(info["training_state_bound"]),
            "initial_state_bound": float(info["initial_state_bound"]),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "termination_reason": info.get("termination_reason"),
        })
        if terminated:
            status = "terminated"
            break
        if truncated and step < horizon:
            status = "truncated_early"
            break
    return {
        "status": status,
        "error": error,
        "steps_completed": len(trajectory),
        "maximum_state_bound": max(
            (row["max_abs_normalized_state"] for row in trajectory), default=None
        ),
        "trajectory": trajectory,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if not output.is_relative_to(project / "artifacts" / "hydrogym"):
        parser.error("output must be under artifacts/hydrogym")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    if args.horizon < 1:
        parser.error("horizon must be positive")

    torch.set_num_threads(2)
    network = build_model(load_composed_config(args.config)).cpu().eval().requires_grad_(False)
    epoch = load_checkpoint(args.checkpoint_dir, models=network, device=torch.device("cpu"))
    policy = PPO.load(args.policy, device="cpu")
    results = {}
    for name, controller in (("zero", None), ("ppo", policy)):
        env = make_env(
            data=args.data,
            split=args.split,
            case=args.case,
            frame=args.frame,
            network=network,
            epoch=epoch,
            episode_steps=args.horizon,
        )
        try:
            results[name] = rollout(env, controller, args.horizon)
        finally:
            env.close()
    report = {
        "status": "ROLLOUT_DIAGNOSTIC_COMPLETE",
        "scientific_scope": "frozen_physicsnemo_fno_surrogate_not_real_cfd",
        "checkpoint_epoch": epoch,
        "split": args.split,
        "case": args.case,
        "frame": args.frame,
        "horizon": args.horizon,
        "controllers": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "zero": {key: value for key, value in results["zero"].items() if key != "trajectory"},
        "ppo": {key: value for key, value in results["ppo"].items() if key != "trajectory"},
    }, indent=2))


if __name__ == "__main__":
    main()
