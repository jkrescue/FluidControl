#!/usr/bin/env python3
"""Check identity-VecNormalize versus persistent raw-69D PPO inference.

This is an interface fixture.  It does not validate a policy scientifically and it
does not execute an environment step, CFD solver, FNO, or PPO training.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from fluid_control.openfoam_observation import total_drag_observation_at


CASES = tuple(
    f"matched_start_acquisition_train_b{phase}_zero"
    for phase in ("00", "02", "04", "06")
)
PARITY_STATUS = "FULL40_TRAIN_FRAME0_69D_PARITY_PASS"
RESULT_STATUS = "IDENTITY_VEC_PERSISTENT_POLICY_INTERFACE_FIXTURE_PASS"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


class ObservationFixture(gym.Env):
    observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(69,), dtype=np.float32
    )
    action_space = gym.spaces.Box(low=-0.75, high=0.75, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(69, dtype=np.float32), {}

    def step(self, action):  # pragma: no cover - forbidden by this audit
        raise RuntimeError("interface fixture must not execute an environment step")


def dummy_vec() -> DummyVecEnv:
    return DummyVecEnv([ObservationFixture])


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("canonical_policy_inference", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load inference helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def historical_vec_contract(path: Path) -> dict:
    raw = dummy_vec()
    try:
        vec = VecNormalize.load(str(path), raw)
        obs_rms = getattr(vec, "obs_rms", None)
        return {
            "norm_obs": bool(vec.norm_obs),
            "norm_reward": bool(vec.norm_reward),
            "training": bool(vec.training),
            "observation_shape": list(vec.observation_space.shape),
            "observation_count": float(obs_rms.count) if obs_rms is not None else None,
            "canonical_identity_compatible": bool(
                vec.norm_obs is False and vec.norm_reward is False
            ),
            "canonical_compatibility_status": (
                "IDENTITY_VEC_COMPATIBLE"
                if vec.norm_obs is False and vec.norm_reward is False
                else "BLOCKED_HISTORICAL_VECNORMALIZE_IS_NOT_IDENTITY"
            ),
        }
    finally:
        raw.close()


def identity_observations(observations: list[np.ndarray]) -> list[np.ndarray]:
    raw = dummy_vec()
    try:
        vec = VecNormalize(raw, training=False, norm_obs=False, norm_reward=False)
        transformed = [
            np.asarray(vec.normalize_obs(value[None, :])[0], dtype=np.float32)
            for value in observations
        ]
    finally:
        raw.close()
    if not all(np.array_equal(before, after) for before, after in zip(observations, transformed, strict=True)):
        raise AssertionError("identity VecNormalize changed a raw 69D observation")
    return transformed


def reconstruct_observations(repo: Path, parity: dict) -> tuple[list[np.ndarray], list[dict]]:
    if parity.get("status") != PARITY_STATUS or parity.get("ppo_executed") is not False:
        raise ValueError("frame-0 parity evidence status/scope differs")
    rows = parity.get("cases")
    if not isinstance(rows, list) or [row.get("case") for row in rows] != list(CASES):
        raise ValueError("frame-0 parity cases/order differs")
    cases_root = repo / "cfd/tandem_cylinders/cases"
    observations = []
    evidence = []
    for row in rows:
        config_path = cases_root / row["case"] / "case_config.json"
        config = read_object(config_path)
        source = (cases_root / str(config["source_restart_case"])).resolve()
        if not source.is_relative_to(cases_root.resolve()):
            raise ValueError("source restart escapes cases root")
        if sha256(config_path) != row.get("case_config_sha256"):
            raise ValueError("case config SHA differs from parity evidence")
        time = float(config["source_restart_time"])
        if time != float(row["source_restart_time"]):
            raise ValueError("restart time differs from parity evidence")
        recorded_raw64 = np.asarray(row.get("raw_observation_69d"), dtype=np.float64)
        if (
            recorded_raw64.shape != (69,)
            or not np.isfinite(recorded_raw64).all()
            or hashlib.sha256(recorded_raw64.tobytes()).hexdigest()
            != row.get("raw_observation_float64_sha256")
        ):
            raise ValueError("recorded raw69D vector/SHA differs")
        reconstructed, sources = total_drag_observation_at(source, time, 0.0)
        reconstructed = np.asarray(reconstructed, dtype=np.float64)
        if not np.array_equal(reconstructed, recorded_raw64):
            raise ValueError("recorded raw69D vector differs from its bound source")
        value = recorded_raw64.astype(np.float32)
        if value.shape != (69,) or not np.isfinite(value).all():
            raise ValueError("reconstructed observation is not finite raw69D")
        observations.append(value)
        evidence.append(
            {
                "case": row["case"],
                "source_restart_case": source.name,
                "source_restart_time": time,
                "raw_float64_sha256": row["raw_observation_float64_sha256"],
                "inference_float32_sha256": array_sha256(value),
                "front_force_sources": sources["front_force_sources"],
            }
        )
    return observations, evidence


def persistent_actions(
    command: list[str], observations: list[np.ndarray], policy_sha: str
) -> list[dict]:
    process = subprocess.Popen(
        command,
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
    )
    prefix = "CANONICAL_POLICY_ACTION_JSON="
    request_text = "".join(json.dumps(value.tolist()) + "\n" for value in observations)
    try:
        stdout, stderr = process.communicate(input=request_text, timeout=30)
    except subprocess.TimeoutExpired as error:
        process.kill()
        stdout, stderr = process.communicate()
        raise TimeoutError("persistent inference exceeded 30 seconds") from error
    if process.returncode != 0:
        raise RuntimeError(stderr[-2000:] or "persistent inference failed")
    lines = [line for line in stdout.splitlines() if line.strip()]
    if len(lines) != len(observations) or any(not line.startswith(prefix) for line in lines):
        raise RuntimeError("persistent inference response count/marker differs")
    results = []
    for line in lines:
        payload = json.loads(line.removeprefix(prefix))
        if (
            payload.get("policy_sha256") != policy_sha
            or payload.get("observation_channels") != 69
            or payload.get("deterministic") is not True
        ):
            raise ValueError("persistent inference metadata differs")
        results.append(payload)
    return results


def negative_contract_checks(helper, model, inference: Path, policy: Path, policy_sha: str) -> dict:
    rejected = []
    for label, payload in (
        ("68_channels", np.zeros(68, dtype=np.float32)),
        ("70_channels", np.zeros(70, dtype=np.float32)),
        ("nonfinite", np.full(69, np.nan, dtype=np.float32)),
    ):
        try:
            helper.predict(model, payload, policy_sha)
        except ValueError:
            rejected.append(label)
        else:  # pragma: no cover - fail closed
            raise AssertionError(f"inference helper accepted {label}")
    wrong_sha = "0" * 64 if policy_sha != "0" * 64 else "1" * 64
    completed = subprocess.run(
        [
            sys.executable,
            str(inference),
            "--policy",
            str(policy),
            "--policy-sha256",
            wrong_sha,
            "--observation-json",
            json.dumps(np.zeros(69, dtype=np.float32).tolist()),
        ],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if completed.returncode == 0 or "PPO checkpoint SHA differs" not in completed.stderr:
        raise AssertionError("persistent inference did not reject a wrong policy SHA")
    rejected.append("wrong_policy_sha256")
    return {"all_rejected": True, "cases": rejected}


def audit(args) -> dict:
    repo = args.repo.resolve()
    policy = args.policy.resolve()
    historical_vec = args.vecnormalize.resolve()
    inference = args.inference.resolve()
    parity_path = args.parity.resolve()
    parity = read_object(parity_path)
    policy_sha = sha256(policy)
    observations, observation_evidence = reconstruct_observations(repo, parity)
    identity = identity_observations(observations)
    model = PPO.load(policy, device="cpu")
    if tuple(model.observation_space.shape) != (69,):
        raise ValueError("historical software-fixture policy is not 69D")
    helper = load_module(inference)
    in_process = [helper.predict(model, value, policy_sha) for value in identity]
    order = (0, 1, 2, 3, 3, 1, 0, 2)
    ordered = [observations[index] for index in order]
    command = [
        sys.executable,
        str(inference),
        "--policy",
        str(policy),
        "--policy-sha256",
        policy_sha,
        "--serve-jsonl",
    ]
    persistent = persistent_actions(command, ordered, policy_sha)
    expected = [in_process[index]["requested_omega"] for index in order]
    observed = [row["requested_omega"] for row in persistent]
    if expected != observed:
        raise AssertionError("persistent JSONL and in-process deterministic actions differ")
    if any(abs(value) > 0.75 + 1e-6 for value in observed):
        raise ValueError("fixture action exceeds support")
    historical = historical_vec_contract(historical_vec)
    if historical["canonical_identity_compatible"] is not False:
        raise AssertionError("historical VecNormalize unexpectedly satisfies identity contract")
    negative = negative_contract_checks(helper, model, inference, policy, policy_sha)
    return {
        "status": RESULT_STATUS,
        "scope": "CPU-only software-interface fixture; not policy or closed-loop validation",
        "scientific_admission_gate": False,
        "policy_training_performed": False,
        "environment_step_executed": False,
        "cfd_solver_executed": False,
        "fno_loaded": False,
        "validation_or_frozen_accessed": False,
        "historical_policy_original_contract_compatible": False,
        "historical_vecnormalize": historical,
        "identity_fixture": {
            "constructed_in_memory": True,
            "norm_obs": False,
            "norm_reward": False,
            "raw_float32_unchanged": True,
            "actual_observation_count": len(observations),
            "persistent_request_count": len(ordered),
            "request_order": list(order),
            "actions_exactly_equal": True,
            "actions": observed,
        },
        "negative_contract_checks": negative,
        "observations": observation_evidence,
        "input_sha256": {
            "policy": policy_sha,
            "historical_vecnormalize": sha256(historical_vec),
            "parity_evidence": sha256(parity_path),
            "persistent_inference": sha256(inference),
            "audit": sha256(Path(__file__)),
        },
        "future_candidate_requirement": (
            "repeat with the new policy's bound saved identity VecNormalize; this historical "
            "fixture does not establish candidate-policy compatibility or control benefit"
        ),
    }


def write_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise ValueError(f"refusing to overwrite {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--vecnormalize", type=Path, required=True)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args)
    write_exclusive(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
