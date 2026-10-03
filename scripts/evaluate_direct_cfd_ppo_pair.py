#!/usr/bin/env python3
"""Container-side frozen PPO versus zero paired rollout on real OpenFOAM."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from functools import partial
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from fluid_control.direct_cfd_hydrogym import UnixJSONClient, make_direct_cfd_env

EVALUATION_STEPS = 800


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def policy_tensor_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.policy.named_parameters()):
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode() + b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def observation_statistics_fingerprint(env: VecNormalize) -> dict:
    rms = env.obs_rms
    if isinstance(rms, dict):
        raise TypeError("canonical 69D Box observation must use one RunningMeanStd")
    digest = hashlib.sha256()
    for value in (rms.mean, rms.var, np.asarray([rms.count])):
        digest.update(np.asarray(value, dtype=np.float64).tobytes())
    return {"sha256": digest.hexdigest(), "count": float(rms.count)}


def write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(payload, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def configure_frozen_normalization(env: VecNormalize) -> VecNormalize:
    env.training = False
    env.norm_reward = False
    return env


def deterministic_policy_action(model, observation: np.ndarray) -> np.ndarray:
    action, _ = model.predict(observation, deterministic=True)
    result = np.asarray(action, dtype=np.float32).reshape(1, 1)
    if not np.isfinite(result).all():
        raise FloatingPointError("frozen PPO returned a non-finite action")
    return result


def make_one(socket_path: str, baseline: dict):
    return Monitor(
        make_direct_cfd_env(
            transport_factory=partial(UnixJSONClient, socket_path, timeout=180.0),
            baseline=baseline,
            episode_steps=EVALUATION_STEPS,
        )
    )


def execute(args) -> dict:
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))["b00"]
    raw = SubprocVecEnv(
        [partial(make_one, path, baseline) for path in args.socket],
        start_method="spawn",
    )
    env = configure_frozen_normalization(VecNormalize.load(str(args.vecnormalize), raw))
    model = PPO.load(args.policy, env=env, device="cpu")
    policy_before = policy_tensor_sha256(model)
    obs_rms_before = observation_statistics_fingerprint(env)
    observations = env.reset()
    rows = []
    cases = {}
    try:
        for step in range(1, EVALUATION_STEPS + 1):
            ppo_action = deterministic_policy_action(model, observations[0:1])
            actions = np.concatenate((ppo_action, np.zeros((1, 1), dtype=np.float32)))
            observations, rewards, dones, infos = env.step(actions)
            if step < EVALUATION_STEPS and np.any(dones):
                raise RuntimeError("paired evaluation ended before 800 decisions")
            for role, info, reward in zip(("ppo", "zero"), infos, rewards, strict=True):
                worker = info["worker_info"]
                cases[role] = worker["case"]
                rows.append(
                    {
                        "step": step,
                        "role": role,
                        "requested_omega": float(info["requested_omega"]),
                        "applied_omega": float(info["applied_omega"]),
                        "applied_delta_omega": float(info["applied_delta_omega"]),
                        "raw_reward": float(reward),
                        "cfd_time": float(info["cfd_time"]),
                        "case": worker["case"],
                    }
                )
            if step % 100 == 0:
                write_atomic(
                    args.output / "progress.json",
                    {
                        "status": "DIRECT_CFD_FROZEN_PPO_PAIR_RUNNING",
                        "completed_steps_per_branch": step,
                        "cases": cases,
                    },
                )
        policy_after = policy_tensor_sha256(model)
        obs_rms_after = observation_statistics_fingerprint(env)
        if policy_after != policy_before:
            raise AssertionError("policy parameter tensors changed during evaluation")
        if obs_rms_after != obs_rms_before:
            raise AssertionError("observation normalization statistics changed during evaluation")
        result = {
            "status": "DIRECT_CFD_FROZEN_PPO_PAIR_ROLLOUT_COMPLETE",
            "steps_per_branch": EVALUATION_STEPS,
            "branches": cases,
            "policy_sha256": sha256(args.policy),
            "vecnormalize_sha256": sha256(args.vecnormalize),
            "vecnormalize_training": env.training,
            "vecnormalize_norm_reward": env.norm_reward,
            "policy_deterministic": True,
            "policy_tensor_sha256_before": policy_before,
            "policy_tensor_sha256_after": policy_after,
            "policy_parameters_unchanged": True,
            "observation_rms_before": obs_rms_before,
            "observation_rms_after": obs_rms_after,
            "observation_rms_unchanged": True,
            "phase": "b00_train",
            "rows": rows,
            "scientific_scope": (
                "training-phase preliminary paired physical validation; not independent "
                "generalization, frozen-test evidence, or final paper conclusion"
            ),
        }
        write_atomic(args.output / "rollout_result.json", result)
        return result
    finally:
        try:
            env.close()
        except (EOFError, BrokenPipeError):
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", action="append", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--vecnormalize", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.socket) != 2:
        parser.error("exactly two b00 sockets are required: PPO then zero")
    args.output.mkdir(parents=True, exist_ok=False)
    print(execute(args)["status"], flush=True)


if __name__ == "__main__":
    main()
