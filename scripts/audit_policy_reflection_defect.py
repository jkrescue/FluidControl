"""Read-only reflection-defect audit for fixed physical-69 PPO observations.

This module never advances CFD and never mutates a policy.  ``snapshot`` reads an
active atomic progress file exactly once and writes a byte-bound completed prefix.
``audit`` accepts only that fixed snapshot and evaluates deterministic CPU policy
inference on each observation and its prescribed reflection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Callable

import numpy as np


SNAPSHOT_STATUS = "EXPLORATORY_POLICY_OBSERVATION_PREFIX_SNAPSHOT_NOT_ADMISSION"
APPROVAL_STATUS = "EXPLORATORY_POLICY_REFLECTION_DEFECT_AUDIT_APPROVED"
RESULT_STATUS = "EXPLORATORY_POLICY_REFLECTION_DEFECT_AUDIT_COMPLETE_NOT_ADMISSION"
ACTION_LIMIT = 0.75
DELTA_LIMIT = 0.1
GIB = 2**30
STARTUP_MEMAVAILABLE_BYTES = 50 * GIB
RUNTIME_MEMAVAILABLE_BYTES = 22 * GIB


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha(path: Path) -> str:
    return sha_bytes(Path(path).read_bytes())


def mem_available_bytes() -> int:
    rows = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        rows[key] = value.strip().split()
    require(rows.get("MemAvailable", [None, None])[1] == "kB", "MemAvailable unavailable")
    value = int(rows["MemAvailable"][0]) * 1024
    require(value > 0, "invalid MemAvailable")
    return value


def host_guard(*, startup: bool) -> int:
    available = mem_available_bytes()
    floor = STARTUP_MEMAVAILABLE_BYTES if startup else RUNTIME_MEMAVAILABLE_BYTES
    require(available >= floor, f"physical MemAvailable below {floor} bytes")
    return available


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def exclusive_json(path: Path, value: Any) -> None:
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), f"refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def physical69(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    require(array.shape == (69,) and np.isfinite(array).all(), "finite physical69 required")
    require(abs(float(array[68])) <= ACTION_LIMIT + 1e-7, "omega outside support")
    return array


def reflect_physical69(value: Any) -> np.ndarray:
    """Reflect about y=7.5: probes reverse; v/Cl/omega are odd."""
    source = physical69(value)
    result = source.copy()
    probes = source[:64].reshape(32, 2)
    result[:64] = (probes[::-1] * np.asarray([1.0, -1.0], np.float32)).reshape(-1)
    result[64] = source[64]   # front Cd: even
    result[65] = -source[65]  # front Cl: odd
    result[66] = source[66]   # rear Cd: even
    result[67] = -source[67]  # rear Cl: odd
    result[68] = -source[68]  # applied rear-cylinder omega: odd
    require(np.array_equal(reflect_physical69_once(result), source), "reflection not involutive")
    return result


def reflect_physical69_once(source: np.ndarray) -> np.ndarray:
    """Internal non-recursive reflection used by the involution check."""
    result = np.asarray(source, dtype=np.float32).copy()
    probes = result[:64].reshape(32, 2).copy()
    result[:64] = (probes[::-1] * np.asarray([1.0, -1.0], np.float32)).reshape(-1)
    result[65] = -result[65]
    result[67] = -result[67]
    result[68] = -result[68]
    return result


def constrain(requested: float, previous: float) -> dict[str, Any]:
    values = (requested, previous)
    require(all(math.isfinite(value) for value in values), "nonfinite action")
    bounded = float(np.clip(requested, -ACTION_LIMIT, ACTION_LIMIT))
    applied = float(np.clip(bounded, previous - DELTA_LIMIT, previous + DELTA_LIMIT))
    return {
        "requested_omega": float(requested),
        "bounded_requested_omega": bounded,
        "applied_omega": applied,
        "applied_delta_omega": applied - previous,
        "action_clipped": not np.isclose(bounded, requested, rtol=0, atol=1e-12),
        "rate_limited": not np.isclose(applied, bounded, rtol=0, atol=1e-12),
    }


def validate_rows(rows: Any, cycles: int) -> list[dict[str, Any]]:
    require(isinstance(rows, list) and len(rows) >= cycles > 0, "invalid completed prefix")
    selected = rows[:cycles]
    previous = 0.0
    for index, row in enumerate(selected, 1):
        require(isinstance(row, dict) and row.get("step") == index, "nonsequential row")
        observation = physical69(row.get("input_observation"))
        require(np.isclose(float(observation[68]), previous, rtol=0, atol=2e-7),
                "observation omega differs from recorded prior")
        for key in ("requested_omega", "applied_omega", "applied_delta_omega"):
            require(isinstance(row.get(key), (int, float)) and math.isfinite(row[key]),
                    f"invalid {key}")
        require(abs(float(row["applied_omega"])) <= ACTION_LIMIT + 1e-12,
                "recorded action outside support")
        require(abs(float(row["applied_delta_omega"])) <= DELTA_LIMIT + 1e-12,
                "recorded rate outside support")
        require(np.isclose(float(row["applied_omega"]) - previous,
                           float(row["applied_delta_omega"]), rtol=0, atol=1e-12),
                "recorded delta mismatch")
        previous = float(row["applied_omega"])
    return selected


def snapshot_progress(source: Path, output: Path, cycles: int | None = None) -> dict[str, Any]:
    """Read ``source`` once; parse and hash those same bytes; write a fixed prefix."""
    source = Path(source)
    payload = source.read_bytes()  # Deliberately the only read of the moving file.
    parsed = json.loads(payload)
    completed = parsed.get("completed_cycles")
    require(isinstance(completed, int) and not isinstance(completed, bool) and completed > 0,
            "invalid completed_cycles")
    chosen = completed if cycles is None else cycles
    require(isinstance(chosen, int) and not isinstance(chosen, bool) and 0 < chosen <= completed,
            "requested prefix exceeds completed cycles")
    rows = validate_rows(parsed.get("rows"), chosen)
    result = {
        "status": SNAPSHOT_STATUS,
        "source_path": str(source),
        "source_read_sha256": sha_bytes(payload),
        "source_completed_cycles": completed,
        "prefix_cycles": chosen,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "first_start_time": float(rows[0]["start_time"]),
        "last_end_time": float(rows[-1]["end_time"]),
        "channel_order": "32*(u,v),front_cd,front_cl,rear_cd,rear_cl,applied_omega",
        "rows": rows,
        "scientific_admission": False,
    }
    exclusive_json(output, result)
    return result


def identity_vec(vec: Any) -> None:
    require(vec.norm_obs is False and vec.norm_reward is False, "identity VecNormalize required")
    require(vec.observation_space.shape == (69,) and vec.action_space.shape == (1,), "spaces")
    probe = np.linspace(-20, 20, 69, dtype=np.float32)[None]
    require(np.array_equal(vec.normalize_obs(probe), probe), "nonidentity observation transform")
    vec.training = False


def predict(model: Any, vec: Any, observation: np.ndarray) -> float:
    physical = physical69(observation)
    normalized = vec.normalize_obs(physical.copy())
    require(np.array_equal(normalized, physical), "physical observation altered")
    action, _ = model.predict(normalized, deterministic=True)
    action = np.asarray(action)
    require(action.shape == (1,) and np.isfinite(action).all(), "invalid deterministic action")
    require(abs(float(action[0])) <= ACTION_LIMIT + 1e-7, "policy output outside support")
    return float(action[0])


def distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    require(array.ndim == 1 and len(array) > 0 and np.isfinite(array).all(), "distribution input")
    return {
        "min": float(array.min()), "max": float(array.max()), "mean": float(array.mean()),
        "rms": float(np.sqrt(np.mean(array * array))),
        "q05": float(np.quantile(array, .05)), "median": float(np.quantile(array, .5)),
        "q95": float(np.quantile(array, .95)),
    }


def audit_policy(rows: list[dict[str, Any]], inference: Callable[[np.ndarray], float],
                 *, require_recorded_reproduction: bool,
                 filter_action: Callable[[float, float], dict[str, Any]] = constrain,
                 ) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    previous = 0.0
    for row in rows:
        source = physical69(row["input_observation"])
        mirrored = reflect_physical69(source)
        raw = inference(source)
        reflected = inference(mirrored)
        antisymmetric = 0.5 * (raw - reflected)
        # Re-evaluating the projected law on Ro gives exactly the opposite scalar.
        antisymmetric_reflected = 0.5 * (reflected - raw)
        require(antisymmetric_reflected == -antisymmetric, "projected law not exactly odd")
        raw_filtered = filter_action(raw, previous)
        sym_filtered = filter_action(antisymmetric, previous)
        if require_recorded_reproduction:
            require(np.isclose(raw, float(row["requested_omega"]), rtol=0, atol=2e-7),
                    "policy does not reproduce recorded request")
            require(np.isclose(raw_filtered["applied_omega"], float(row["applied_omega"]),
                               rtol=0, atol=2e-7),
                    "single filter does not reproduce recorded action")
        records.append({
            "step": int(row["step"]), "prior_recorded_omega": previous,
            "raw_request": raw, "reflected_request": reflected,
            "raw_symmetry_defect": raw + reflected,
            "antisymmetrized_request": antisymmetric,
            "raw_filtered": raw_filtered, "antisymmetrized_filtered": sym_filtered,
        })
        previous = float(row["applied_omega"])
    summary = {
        "cycles": len(records),
        "raw_request": distribution([row["raw_request"] for row in records]),
        "reflected_request": distribution([row["reflected_request"] for row in records]),
        "raw_symmetry_defect": distribution([row["raw_symmetry_defect"] for row in records]),
        "absolute_raw_symmetry_defect": distribution(
            [abs(row["raw_symmetry_defect"]) for row in records]),
        "antisymmetrized_request": distribution(
            [row["antisymmetrized_request"] for row in records]),
        "raw_filter": {
            "action_clipped": sum(row["raw_filtered"]["action_clipped"] for row in records),
            "rate_limited": sum(row["raw_filtered"]["rate_limited"] for row in records),
            "saturated": sum(abs(row["raw_filtered"]["applied_omega"]) >= ACTION_LIMIT - 1e-12
                             for row in records),
        },
        "antisymmetrized_filter": {
            "action_clipped": sum(row["antisymmetrized_filtered"]["action_clipped"] for row in records),
            "rate_limited": sum(row["antisymmetrized_filtered"]["rate_limited"] for row in records),
            "saturated": sum(abs(row["antisymmetrized_filtered"]["applied_omega"]) >= ACTION_LIMIT - 1e-12
                             for row in records),
        },
        "recorded_reproduction_required": require_recorded_reproduction,
    }
    return {"summary": summary, "records": records}


def cgroup_limits() -> tuple[int, int]:
    relative = next(line.split("::", 1)[1] for line in Path("/proc/self/cgroup").read_text().splitlines()
                    if line.startswith("0::"))
    root = Path("/sys/fs/cgroup") / relative.lstrip("/")
    return int((root / "memory.max").read_text()), int((root / "memory.swap.max").read_text())


def load_hashed_module(name: str, path: Path, expected_sha256: str) -> Any:
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and sha(path) == expected_sha256,
            f"module identity: {path}")
    module_spec = importlib.util.spec_from_file_location(name, path)
    require(module_spec is not None and module_spec.loader is not None, "module spec")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    module_spec.loader.exec_module(module)
    return module


def execute_audit(spec: dict[str, Any], spec_path: Path, output: Path) -> dict[str, Any]:
    require(spec.get("status") == APPROVAL_STATUS and spec.get("execution_authorized") is True,
            "explicit audit approval required")
    require(spec.get("scientific_admission") is False and spec.get("cfd_execution") is False
            and spec.get("training_performed") is False, "read-only exploratory scope")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only audit")
    require(spec.get("deadline_seconds") == 300 and spec.get("memory_max_bytes") == 4 * 2**30,
            "fixed bounded resources")
    require(spec.get("startup_memavailable_bytes") == STARTUP_MEMAVAILABLE_BYTES
            and spec.get("runtime_memavailable_bytes") == RUNTIME_MEMAVAILABLE_BYTES
            and spec.get("cpu_quota_percent") == 100
            and spec.get("outer_timeout_seconds") == 330
            and spec.get("cleanup_grace_seconds") == 30, "fixed host resource contract")
    memory, swap = cgroup_limits()
    require(0 < memory <= spec["memory_max_bytes"] and swap == 0, "4GiB/no-swap cgroup required")
    minimum_available = host_guard(startup=True)
    require(sha(Path(__file__)) == spec.get("audit_source_sha256"), "audit source identity")
    repo = Path(spec["repo"]).resolve()
    require(repo.is_dir() and not repo.is_symlink(), "repo identity")
    filter_row = spec["action_filter_source"]
    filter_path = (repo / filter_row["path"]).resolve()
    require(filter_row["path"] == "src/fluid_control/canonical_joint_v1.py"
            and filter_path.is_relative_to(repo), "canonical action filter path")
    filter_module = load_hashed_module("reflection_audit_canonical_joint_v1", filter_path,
                                       filter_row["sha256"])
    snapshot_path = Path(spec["snapshot"]["path"])
    require(sha(snapshot_path) == spec["snapshot"]["sha256"], "snapshot identity")
    snapshot = json.loads(snapshot_path.read_text())
    require(snapshot.get("status") == SNAPSHOT_STATUS, "snapshot status")
    rows = validate_rows(snapshot.get("rows"), snapshot.get("prefix_cycles"))
    started = time.monotonic()

    def guarded_inference(model: Any, vec: Any, observation: np.ndarray) -> float:
        nonlocal minimum_available
        available = host_guard(startup=False)
        minimum_available = min(minimum_available, available)
        require(time.monotonic() - started < 300, "audit deadline")
        return predict(model, vec, observation)

    import stable_baselines3
    import gymnasium
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    expected_versions = {"stable_baselines3": "2.7.1", "gymnasium": "1.2.3",
                         "torch": "2.14.1", "numpy": "2.5.3"}
    require({name: importlib.metadata.version(name) for name in expected_versions} == expected_versions,
            "runtime package versions")
    overlay = Path(spec["runtime_root"]).resolve()
    require(all(Path(module.__file__).resolve().is_relative_to(overlay)
                for module in (stable_baselines3, gymnasium)), "runtime import origin")

    policy_results = {}
    for item in spec["policies"]:
        require(time.monotonic() - started < 300, "audit deadline")
        label = item["label"]
        require(isinstance(label, str) and label and label not in policy_results, "policy label")
        policy_path, vec_path, result_path = map(Path, (item["policy"], item["vecnormalize"],
                                                        item["training_result"]))
        for key, path in (("policy_sha256", policy_path), ("vecnormalize_sha256", vec_path),
                          ("training_result_sha256", result_path)):
            require(sha(path) == item[key], f"{label} {key}")
        training_result = json.loads(result_path.read_text())
        require(training_result.get("scientific_admission") is False
                and training_result.get("status") == item["expected_training_status"]
                and training_result.get("timesteps") == item["expected_timesteps"],
                "exploratory policy training identity")
        vec = pickle.loads(vec_path.read_bytes())
        require(type(vec) is VecNormalize, "exact VecNormalize type")
        identity_vec(vec)
        model = PPO.load(policy_path, device="cpu")
        require(model.observation_space == vec.observation_space
                and model.action_space == vec.action_space, "policy/normalizer spaces")
        model.policy.set_training_mode(False)
        require(all(parameter.device.type == "cpu" for parameter in model.policy.parameters()),
                "CPU policy required")
        audited = audit_policy(rows, lambda obs: guarded_inference(model, vec, obs),
                               require_recorded_reproduction=item["reproduces_snapshot_policy"],
                               filter_action=filter_module.apply_action_rate_limit)
        audited["identities"] = {key: item[key] for key in
                                 ("policy_sha256", "vecnormalize_sha256", "training_result_sha256")}
        policy_results[label] = audited
    result = {
        "status": RESULT_STATUS, "snapshot_sha256": spec["snapshot"]["sha256"],
        "snapshot_cycles": len(rows), "policies": policy_results,
        "approval_sha256": sha(spec_path), "audit_source_sha256": sha(Path(__file__)),
        "wall_seconds": time.monotonic() - started,
        "minimum_physical_memavailable_bytes": minimum_available,
        "interpretation_limit": (
            "policy symmetry/action-filter diagnostic only; projected filtered actions are conditioned "
            "on prior omega recorded by the original trajectory, not a projected counterfactual rollout "
            "or closed loop; no CFD or physical benefit claim"
        ),
        "optimizer_steps": 0,
        "policy_optimizer_reconstructed_by_load": True,
        "training_performed": False, "model_saved": False,
        "cfd_execution": False, "scientific_admission": False,
    }
    exclusive_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    snap = commands.add_parser("snapshot")
    snap.add_argument("--source", type=Path, required=True)
    snap.add_argument("--output", type=Path, required=True)
    snap.add_argument("--cycles", type=int)
    audit = commands.add_parser("audit")
    audit.add_argument("--spec", type=Path, required=True)
    audit.add_argument("--spec-sha256", required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.command == "snapshot":
        snapshot_progress(args.source, args.output, args.cycles)
        return
    require(sha(args.spec) == args.spec_sha256, "approval digest")
    require(args.execute, "audit execution not authorized")
    execute_audit(json.loads(args.spec.read_text()), args.spec, args.output)


if __name__ == "__main__":
    main()
