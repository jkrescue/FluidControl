#!/usr/bin/env python3
"""Infer one deterministic full40 canonical PPO action from a 69D observation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

ACTION_LIMIT = 0.75


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--observation-json", required=True)
    args = parser.parse_args()
    if sha256(args.policy) != args.policy_sha256:
        raise ValueError("PPO checkpoint SHA differs")
    observation = np.asarray(json.loads(args.observation_json), dtype=np.float32)
    if observation.shape != (69,) or not np.isfinite(observation).all():
        raise ValueError("expected 69 finite canonical CFD observation channels")
    model = PPO.load(args.policy, device="cpu")
    action, _ = model.predict(observation, deterministic=True)
    requested = np.asarray(action, dtype=np.float64).reshape(-1)
    if requested.shape != (1,) or not np.isfinite(requested).all():
        raise ValueError("PPO produced an invalid one-dimensional action")
    value = float(requested[0])
    if abs(value) > ACTION_LIMIT + 1e-6:
        raise ValueError("PPO action exceeds full40 support")
    print(
        "CANONICAL_POLICY_ACTION_JSON="
        + json.dumps(
            {
                "requested_omega": value,
                "observation_channels": 69,
                "deterministic": True,
                "policy_sha256": args.policy_sha256,
                "policy_backend": "stable_baselines3_ppo_cpu",
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
