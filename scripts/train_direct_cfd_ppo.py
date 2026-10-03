#!/usr/bin/env python3
"""Container-side CPU PPO training on two real-OpenFOAM socket environments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from functools import partial
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from fluid_control.direct_cfd_hydrogym import UnixJSONClient, make_direct_cfd_env

ENV_COUNT = 2
EPISODE_STEPS = 128
TOTAL_TIMESTEPS = 2048
CHECKPOINT_INTERVAL = 256
SEED = 20261003


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_fingerprint(model, initial_parameters: dict[str, torch.Tensor]) -> dict:
    digest = hashlib.sha256()
    squared_delta = 0.0
    squared_initial = 0.0
    current = dict(model.policy.named_parameters())
    if set(current) != set(initial_parameters):
        raise ValueError("policy parameter schema changed during PPO training")
    for name in sorted(current):
        value = current[name].detach().cpu().contiguous()
        initial = initial_parameters[name]
        digest.update(name.encode() + b"\0")
        digest.update(value.numpy().tobytes())
        squared_delta += float(torch.sum((value - initial) ** 2))
        squared_initial += float(torch.sum(initial**2))
    delta_l2 = math.sqrt(squared_delta)
    initial_l2 = math.sqrt(squared_initial)
    return {
        "parameter_tensor_sha256": digest.hexdigest(),
        "parameter_delta_l2_from_initial": delta_l2,
        "parameter_relative_delta_l2_from_initial": delta_l2 / max(initial_l2, 1e-30),
    }


def reset_resume_exploration_std(model: PPO, target_std: float) -> dict:
    """Reset only PPO's Gaussian log_std while preserving optimizer state."""
    if not math.isfinite(target_std) or target_std <= 0.0:
        raise ValueError("resume exploration std must be positive and finite")
    parameter = model.policy.log_std
    before = {
        name: value.detach().cpu().clone()
        for name, value in model.policy.named_parameters()
    }
    optimizer_state = model.policy.optimizer.state.get(parameter, {})
    if not optimizer_state:
        raise ValueError("resume log_std lacks saved optimizer state")
    optimizer_before = {
        key: value.detach().cpu().clone() if hasattr(value, "detach") else value
        for key, value in optimizer_state.items()
    }
    old_std = parameter.detach().exp().cpu().tolist()
    shape = tuple(model.observation_space.shape)
    diagnostic_observations = np.stack(
        [np.zeros(shape, dtype=np.float32), np.ones(shape, dtype=np.float32)]
    )
    deterministic_before, _ = model.predict(
        diagnostic_observations, deterministic=True
    )
    with torch.no_grad():
        parameter.fill_(math.log(target_std))
    after = dict(model.policy.named_parameters())
    deterministic_after, _ = model.predict(
        diagnostic_observations, deterministic=True
    )
    deterministic_unchanged = np.array_equal(
        deterministic_before, deterministic_after
    )
    changed = [
        name
        for name, value in before.items()
        if not torch.equal(value, after[name].detach().cpu())
    ]
    optimizer_after = model.policy.optimizer.state.get(parameter, {})
    optimizer_preserved = optimizer_before.keys() == optimizer_after.keys() and all(
        torch.equal(value, optimizer_after[key].detach().cpu())
        if hasattr(value, "detach")
        else value == optimizer_after[key]
        for key, value in optimizer_before.items()
    )
    if changed != ["log_std"] or not optimizer_preserved or not deterministic_unchanged:
        raise RuntimeError("exploration reset changed more than log_std or optimizer state")
    return {
        "old_std": old_std,
        "new_std": parameter.detach().exp().cpu().tolist(),
        "target_std": target_std,
        "changed_parameters": changed,
        "deterministic_mean_parameters_unchanged": True,
        "deterministic_action_probe_unchanged": True,
        "deterministic_action_probe_sha256": hashlib.sha256(
            np.asarray(deterministic_after).tobytes()
        ).hexdigest(),
        "optimizer_state_preserved": True,
        "log_std_remains_trainable": bool(parameter.requires_grad),
    }


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


def make_one(socket_path: str, baseline: dict, seed: int):
    env = make_direct_cfd_env(
        transport_factory=partial(UnixJSONClient, socket_path, timeout=180.0),
        baseline=baseline,
        episode_steps=EPISODE_STEPS,
    )
    env.action_space.seed(seed)
    return Monitor(env)


def make_raw_vector_env(sockets: list[str], baselines: list[dict], seed: int):
    return SubprocVecEnv(
        [
            partial(make_one, socket_path, baseline, seed + index)
            for index, (socket_path, baseline) in enumerate(
                zip(sockets, baselines, strict=True)
            )
        ],
        start_method="spawn",
    )


def make_vector_env(sockets: list[str], baselines: list[dict], seed: int):
    return VecNormalize(
        make_raw_vector_env(sockets, baselines, seed),
        training=True,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )


class PhysicalJournalCallback(BaseCallback):
    """Persist unnormalized physics rewards separately from VecNormalize."""

    def __init__(self, output: Path) -> None:
        super().__init__(verbose=0)
        self.output = output
        self.rows: list[dict] = []
        self.flushed_rows = 0
        self.checkpoints: list[dict] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", ()):
            components = {
                key: float(value)
                for key, value in info.items()
                if key.startswith("reward_")
            }
            if not components or not all(math.isfinite(value) for value in components.values()):
                raise FloatingPointError("missing or non-finite physical reward components")
            ledger = info["canonical_joint_ledger"]
            worker = info["worker_info"]
            self.rows.append(
                {
                    "environment_step": len(self.rows) + 1,
                    "raw_physical_reward": float(sum(components.values())),
                    "reward_components": components,
                    "applied_omega": float(info["applied_omega"]),
                    "applied_delta_omega": float(info["applied_delta_omega"]),
                    "rate_limited": bool(info["rate_limited"]),
                    "cfd_time": float(info["cfd_time"]),
                    "canonical_short_window": {
                        "total_drag_reduction": float(ledger["total_drag_reduction"]),
                        "rear_cl_fluctuation_ratio": float(
                            ledger["rear_cl_fluctuation_ratio"]
                        ),
                        "abs_mean_rear_cl_over_baseline_clprime_rms": float(
                            ledger["abs_mean_rear_cl_over_baseline_clprime_rms"]
                        ),
                        "joint_pass_diagnostic_only": bool(
                            ledger["canonical_joint_gate_pass"]
                        ),
                    },
                    "worker": worker,
                    "backend": info["backend"],
                }
            )
        return True

    def checkpoint_after_update(
        self,
        timestep: int,
        update_count: int,
        initial_parameters: dict[str, torch.Tensor],
        training_metrics: dict[str, float | int],
    ) -> None:
        """Save only after ``PPO.learn`` returns from its optimizer update."""
        checkpoint = self.output / "checkpoints" / f"ppo_{timestep:08d}"
        self.model.save(checkpoint)
        policy_path = checkpoint.with_suffix(".zip")
        vec_path = self.output / "checkpoints" / f"vecnormalize_{timestep:08d}.pkl"
        self.training_env.save(str(vec_path))
        recent = self.rows[self.flushed_rows :]
        if len(recent) != CHECKPOINT_INTERVAL:
            raise AssertionError(
                f"checkpoint expected {CHECKPOINT_INTERVAL} new transitions, got {len(recent)}"
            )
        raw = np.asarray([row["raw_physical_reward"] for row in recent], dtype=float)
        component_keys = sorted(recent[0]["reward_components"])
        summary = {
            "timesteps": timestep,
            "ppo_update_count": update_count,
            "checkpoint_timing": "after PPO optimizer update returned",
            "sb3_training_metrics": training_metrics,
            "policy_parameter_evidence": parameter_fingerprint(
                self.model, initial_parameters
            ),
            "policy": str(policy_path),
            "policy_sha256": sha256(policy_path),
            "vecnormalize": str(vec_path),
            "vecnormalize_sha256": sha256(vec_path),
            "raw_physical_reward": {
                "mean": float(np.mean(raw)),
                "minimum": float(np.min(raw)),
                "maximum": float(np.max(raw)),
                "std": float(np.std(raw)),
            },
            "raw_reward_component_mean": {
                key: float(np.mean([row["reward_components"][key] for row in recent]))
                for key in component_keys
            },
            "applied_omega_rms": float(
                np.sqrt(np.mean([row["applied_omega"] ** 2 for row in recent]))
            ),
            "short_window_joint_pass_fraction_diagnostic_only": float(
                np.mean(
                    [
                        row["canonical_short_window"]["joint_pass_diagnostic_only"]
                        for row in recent
                    ]
                )
            ),
            "scientific_scope": (
                "train-only 6.15-D/U diagnostic; not final 80-D/U paired-CFD gate"
            ),
        }
        self.checkpoints.append(summary)
        with (self.output / "physical_steps.jsonl").open("a", encoding="utf-8") as stream:
            for row in recent:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.flushed_rows = len(self.rows)
        write_atomic(
            self.output / "progress.json",
            {
                "status": "DIRECT_REAL_CFD_PPO_RUNNING",
                "completed_transitions": timestep,
                "checkpoints": self.checkpoints,
                "physical_reward_is_unnormalized": True,
                "policy_training_uses_vecnormalize": True,
            },
        )


def run_probe(env, output: Path, transitions: int) -> dict:
    if transitions <= 0 or transitions % ENV_COUNT:
        raise ValueError("probe transitions must be positive and divisible by two")
    observations = env.reset()
    del observations
    rows = []
    for vector_step in range(transitions // ENV_COUNT):
        actions = np.zeros((ENV_COUNT, 1), dtype=np.float32)
        _, _, _, infos = env.step(actions)
        for info in infos:
            rows.append(
                {
                    "raw_physical_reward": float(
                        sum(
                            float(value)
                            for key, value in info.items()
                            if key.startswith("reward_")
                        )
                    ),
                    "worker": info["worker_info"],
                    "cfd_time": info["cfd_time"],
                }
            )
    result = {
        "status": "DIRECT_REAL_CFD_RUNTIME_PROBE_COMPLETE",
        "transitions": len(rows),
        "policy": "zero action runtime/transport probe; PPO not trained",
        "rows": rows,
        "scientific_scope": "runtime and integration probe only; no benefit claim",
    }
    write_atomic(output / "probe_result.json", result)
    return result


def execute(args) -> dict:
    if len(args.socket) != ENV_COUNT:
        raise ValueError("exactly two worker sockets are required")
    baseline_payload = json.loads(args.baselines.read_text(encoding="utf-8"))
    baselines = [baseline_payload[phase] for phase in ("b00", "b02")]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "checkpoints").mkdir()
    resume = args.resume_policy is not None
    if resume != (args.resume_vecnormalize is not None):
        raise ValueError("resume policy and VecNormalize must be supplied together")
    if resume:
        if args.prior_transitions <= 0:
            raise ValueError("resume requires a positive prior transition count")
        raw = make_raw_vector_env(args.socket, baselines, args.seed)
        env = VecNormalize.load(str(args.resume_vecnormalize), raw)
        env.training = True
        env.norm_reward = True
    else:
        if args.prior_transitions != 0:
            raise ValueError("a fresh run must have zero prior transitions")
        env = make_vector_env(args.socket, baselines, args.seed)
    try:
        if args.probe_transitions:
            return run_probe(env, args.output, args.probe_transitions)
        if resume:
            model = PPO.load(args.resume_policy, env=env, device="cpu")
            if model.num_timesteps != args.prior_transitions:
                raise ValueError(
                    f"resume policy has {model.num_timesteps} transitions, "
                    f"expected {args.prior_transitions}"
                )
            exploration_reset = (
                reset_resume_exploration_std(model, args.resume_exploration_std)
                if args.resume_exploration_std is not None
                else None
            )
        else:
            if args.resume_exploration_std is not None:
                raise ValueError("exploration std reset is allowed only for continuation")
            exploration_reset = None
            model = PPO(
                "MlpPolicy",
                env,
                seed=args.seed,
                device="cpu",
                n_steps=128,
                batch_size=64,
                n_epochs=4,
                learning_rate=3e-4,
                gamma=0.99,
                verbose=1,
            )
        callback = PhysicalJournalCallback(args.output)
        initial_parameters = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.policy.named_parameters()
        }
        initial_fingerprint = parameter_fingerprint(model, initial_parameters)
        completed = 0
        update_count = 0
        while completed < args.additional_timesteps:
            model.learn(
                total_timesteps=CHECKPOINT_INTERVAL,
                callback=callback,
                reset_num_timesteps=(not resume and completed == 0),
            )
            completed += CHECKPOINT_INTERVAL
            update_count += 1
            logger_values = model.logger.name_to_value
            metric_names = (
                "train/approx_kl",
                "train/policy_gradient_loss",
                "train/value_loss",
                "train/entropy_loss",
                "train/clip_fraction",
                "train/loss",
            )
            training_metrics = {
                name.removeprefix("train/"): float(logger_values[name])
                for name in metric_names
                if name in logger_values
            }
            training_metrics["n_updates"] = int(model._n_updates)
            callback.checkpoint_after_update(
                args.prior_transitions + completed,
                update_count,
                initial_parameters,
                training_metrics,
            )
        final_policy = args.output / "ppo_policy_final"
        model.save(final_policy)
        final_policy = final_policy.with_suffix(".zip")
        final_vec = args.output / "vecnormalize_final.pkl"
        env.save(str(final_vec))
        result = {
            "status": (
                "DIRECT_REAL_CFD_PPO_CONTINUATION_COMPLETE"
                if resume
                else "DIRECT_REAL_CFD_PPO_TRAINING_COMPLETE"
            ),
            "backend": "official_hydrogym_flowenv_with_project_openfoam_adapter",
            "surrogate_used": False,
            "prior_transitions": args.prior_transitions,
            "new_transitions": args.additional_timesteps,
            "cumulative_transitions": args.prior_transitions + args.additional_timesteps,
            "resume_policy_sha256": sha256(args.resume_policy) if resume else None,
            "resume_vecnormalize_sha256": (
                sha256(args.resume_vecnormalize) if resume else None
            ),
            "resume_exploration_std_reset": exploration_reset,
            "environment_count": ENV_COUNT,
            "episode_steps": EPISODE_STEPS,
            "observation_dimension": 69,
            "policy": str(final_policy),
            "policy_sha256": sha256(final_policy),
            "vecnormalize": str(final_vec),
            "vecnormalize_sha256": sha256(final_vec),
            "checkpoint_summaries": callback.checkpoints,
            "initial_policy_parameter_evidence": initial_fingerprint,
            "reward_contract": "canonical_joint_v1 with actual causal prehistory",
            "scientific_scope": (
                "genuine train-only real-CFD RL/basic online closure; not evidence of "
                "the final paired 80-D/U physical acceptance gate"
            ),
        }
        write_atomic(args.output / "result.json", result)
        return result
    finally:
        try:
            env.close()
        except (EOFError, BrokenPipeError):
            # A worker-side exception already carries the primary failure.
            # SubprocVecEnv close must not mask it with a secondary EOF.
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", action="append", required=True)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--probe-transitions", type=int, default=0)
    parser.add_argument("--resume-policy", type=Path)
    parser.add_argument("--resume-vecnormalize", type=Path)
    parser.add_argument("--prior-transitions", type=int, default=0)
    parser.add_argument("--resume-exploration-std", type=float)
    parser.add_argument("--additional-timesteps", type=int, default=TOTAL_TIMESTEPS)
    args = parser.parse_args()
    if args.additional_timesteps <= 0 or args.additional_timesteps % CHECKPOINT_INTERVAL:
        parser.error("additional timesteps must be positive and divisible by 256")
    result = execute(args)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
