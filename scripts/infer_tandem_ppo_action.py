#!/usr/bin/env python3
"""Return one deterministic PPO action from a real-CFD 67-channel observation."""
from __future__ import annotations

import argparse
import json

import numpy as np
from stable_baselines3 import PPO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--observation-json", required=True)
    args = parser.parse_args()
    observation = np.asarray(json.loads(args.observation_json), dtype=np.float32)
    if observation.shape != (67,) or not np.isfinite(observation).all():
        raise ValueError("expected 67 finite CFD observation channels")
    model = PPO.load(args.policy, device="cpu")
    action, _ = model.predict(observation, deterministic=True)
    requested = np.asarray(action, dtype=np.float64).reshape(-1)
    if requested.shape != (1,) or not np.isfinite(requested).all():
        raise ValueError("PPO produced invalid one-dimensional action")
    if abs(float(requested[0])) > 5 + 1e-6:
        raise ValueError("PPO requested rotation outside declared action support")
    print("POLICY_ACTION_JSON=" + json.dumps({
        "requested_omega": float(requested[0]),
        "observation_channels": 67,
        "deterministic": True,
        "policy_backend": "stable_baselines3_ppo_cpu",
    }), flush=True)


if __name__ == "__main__":
    main()
