#!/usr/bin/env python3
"""Serve deterministic full40 canonical PPO actions for 69D observations."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

ACTION_LIMIT = 0.75


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predict(model: PPO, observation_payload, policy_sha256: str) -> dict:
    observation = np.asarray(observation_payload, dtype=np.float32)
    if observation.shape != (69,) or not np.isfinite(observation).all():
        raise ValueError("expected 69 finite canonical CFD observation channels")
    action, _ = model.predict(observation, deterministic=True)
    requested = np.asarray(action, dtype=np.float64).reshape(-1)
    if requested.shape != (1,) or not np.isfinite(requested).all():
        raise ValueError("PPO produced an invalid one-dimensional action")
    value = float(requested[0])
    if abs(value) > ACTION_LIMIT + 1e-6:
        raise ValueError("PPO action exceeds full40 support")
    return {
        "requested_omega": value,
        "observation_channels": 69,
        "deterministic": True,
        "policy_sha256": policy_sha256,
        "policy_backend": "stable_baselines3_ppo_cpu_persistent_process",
    }


def emit(payload: dict) -> None:
    print(
        "CANONICAL_POLICY_ACTION_JSON=" + json.dumps(payload, sort_keys=True),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--policy-sha256", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--observation-json")
    mode.add_argument("--serve-jsonl", action="store_true")
    args = parser.parse_args()
    if sha256(args.policy) != args.policy_sha256:
        raise ValueError("PPO checkpoint SHA differs")
    model = PPO.load(args.policy, device="cpu")
    if args.observation_json is not None:
        emit(predict(model, json.loads(args.observation_json), args.policy_sha256))
        return
    for line in sys.stdin:
        if line.strip():
            emit(predict(model, json.loads(line), args.policy_sha256))


if __name__ == "__main__":
    main()
