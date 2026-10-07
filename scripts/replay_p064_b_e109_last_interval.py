"""Disposable E109 327.9->328 recovery replay; never extends beyond known data."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import re
import shutil
import signal
import subprocess
import sys
import time

import numpy as np


STATUS = "P064_B_E109_LAST_INTERVAL_RECOVERY_REPLAY_EXECUTION_APPROVED"
PREP_STATUS = "P064_B_E109_LAST_INTERVAL_RECOVERY_REPLAY_PREPARATION"
FLOAT = re.compile(r"(?<![A-Za-z_])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def require(ok, why):
    if not ok:
        raise ValueError(why)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def internal_values(path):
    """Numeric internalField only; ignores path/header/boundary formatting."""
    text = Path(path).read_text()
    begin = text.find("internalField")
    end = text.find("boundaryField", begin)
    require(begin >= 0 and end > begin, "OpenFOAM internalField block")
    values = np.asarray([float(x) for x in FLOAT.findall(text[begin:end])], np.float64)
    require(values.size and np.isfinite(values).all(), "finite internalField values")
    return values


def compare_restart_numeric(actual, expected):
    """Compare saved internalField payloads, not boundary text/header bytes."""
    rows = {}
    for name in ("U", "U_0", "p", "phi", "phi_0"):
        left, right = internal_values(Path(actual) / name), internal_values(Path(expected) / name)
        require(left.shape == right.shape, "restart numeric shape: " + name)
        delta = float(np.max(np.abs(left - right)))
        # Same image, mesh, time step and input should be deterministic.  Keep
        # a tiny textual round-off allowance but do not accept physical drift.
        require(delta <= 1e-12, "restart numeric replay mismatch: " + name)
        rows[name] = {"count": int(left.size), "max_abs_difference": delta}
    return rows


def compare_uniform_time(actual, expected):
    left = np.asarray([float(x) for x in FLOAT.findall(Path(actual).read_text())], np.float64)
    right = np.asarray([float(x) for x in FLOAT.findall(Path(expected).read_text())], np.float64)
    require(left.shape == right.shape and np.array_equal(left, right), "uniform/time replay mismatch")
    return {"count": int(left.size), "max_abs_difference": 0.0}


def bound_source_tree(base, source, dirname, role, inventory):
    trees = {part: base.tree(Path(source) / dirname / part)
             for part in ("327.9", "328", "constant", "system")}
    require(trees == inventory[role], "bound E109 source tree: " + role)
    return trees


def copy_bound_cases(base, transport, source, output, inventory):
    """Execution seam: bind first, then copy only the required restart trees."""
    cases, source_trees = {}, {}
    for role, dirname in (("ppo", "case_mpc"), ("zero", "case_zero")):
        src, dst = Path(source) / dirname, Path(output) / dirname
        source_trees[role] = bound_source_tree(base, source, dirname, role, inventory)
        dst.mkdir()
        for part in ("327.9", "constant", "system"):
            shutil.copytree(src / part, dst / part)
        transport.substitute(dst / "system/controlDict", "writeInterval", .1)
        cases[role] = dst
    return cases, source_trees


def validate_spec(spec, spec_path):
    authorized = spec["status"] == STATUS and spec["execution_authorized"] is True
    preparation = spec["status"] == PREP_STATUS and spec["execution_authorized"] is False
    require(authorized or preparation, "approval/preparation state")
    require(spec["interval"] == [327.9, 328.0] and spec["steps"] == 1, "single known interval")
    require(spec["resources"] == {"memory_gib": 8, "swap_gib": 0, "cpu_quota_percent": 400,
                                  "runtime_max_seconds": 300, "startup_mem_available_gib": 50,
                                  "runtime_mem_available_gib": 22, "disk_reserve_gib": 20,
                                  "inner_deadline_seconds": 240}, "resources")
    require(sha(__file__) == spec["driver_sha256"], "driver identity")
    for value in spec["inputs"].values():
        require(sha(value["path"]) == value["sha256"], "input identity")
    require(not Path(spec["output"]).exists(), "exclusive output")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only inference")
    return spec


def preflight(spec, spec_path):
    spec = validate_spec(spec, spec_path)
    old = json.loads(Path(spec["inputs"]["e109_result"]["path"]).read_text())
    require(old["cycles"] == 800 and len(old["rows"]) == 800, "complete E109")
    require(old["rows"][-2]["applied_omega"] == 0.2990577340126038, "row799 limiter")
    require(old["rows"][-1]["applied_omega"] == 0.19905773401260382, "row800 limiter")
    require(np.asarray(old["rows"][-1]["input_observation"], np.float32).shape == (69,), "input69")
    require(np.asarray(old["rows"][-1]["output_observation"], np.float32).shape == (69,), "output69")
    source = Path(spec["source_e109_output"])
    inventory = json.loads(Path(spec["inputs"]["source_tree_inventory"]["path"]).read_text())
    needed = {"U", "U_0", "p", "phi", "phi_0", "uniform/time"}
    for role in ("case_mpc", "case_zero"):
        for when in ("327.9", "328"):
            root = source / role / when
            found = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
            require(needed.issubset(found), f"{role}/{when} backward restart fields")
    return spec


def execute(spec, spec_path):
    spec = preflight(spec, spec_path)
    require(spec["status"] == STATUS and spec["execution_authorized"] is True, "execution approval")
    old = json.loads(Path(spec["inputs"]["e109_result"]["path"]).read_text())
    require(old["cycles"] == 800 and len(old["rows"]) == 800, "complete E109")
    prior, target = old["rows"][-2], old["rows"][-1]
    require(prior["step"] == 799 and target["step"] == 800, "terminal rows")
    previous = float(prior["applied_omega"])
    require(previous == 0.2990577340126038, "row799 double limiter state")
    input_obs = np.asarray(target["input_observation"], np.float32)
    require(input_obs.shape == (69,) and float(input_obs[-1]) == 0.2990577220916748,
            "row800 saved input observation")

    original = load_module("e109_original_driver", spec["inputs"]["e109_driver"]["path"])
    base = load_module("e109_pair_runtime", spec["inputs"]["pair_driver"]["path"])
    transport = load_module("e109_transport", spec["inputs"]["transport"]["path"])
    symmetry = load_module("e109_symmetry", spec["inputs"]["symmetry_adapter"]["path"])
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize
    vec = pickle.loads(Path(spec["inputs"]["vecnormalize"]["path"]).read_bytes())
    original.validate_vec(vec, VecNormalize)
    model = PPO.load(spec["inputs"]["policy"]["path"], device="cpu")
    model.policy.set_training_mode(False)

    base.host_guard(startup=True)
    repo = Path(spec["repo"])
    require(shutil.disk_usage(repo).free >= spec["resources"]["disk_reserve_gib"] * 2**30,
            "startup disk reserve")
    started = time.monotonic()
    def guard():
        base.host_guard()
        require(shutil.disk_usage(repo).free >= spec["resources"]["disk_reserve_gib"] * 2**30,
                "runtime disk reserve")
        require(time.monotonic() - started < spec["resources"]["inner_deadline_seconds"],
                "replay inner deadline")
    def terminated(*_):
        raise RuntimeError("termination requested; PairSolvers context will clean owned containers")
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, terminated)

    canonical = symmetry.canonicalize_physical69(input_obs)
    canonical_action = original.predict(model, vec, canonical.value)
    physical_request = float(symmetry.restore_physical_action(
        np.asarray([canonical_action], np.float32), canonical.orientation)[0])
    action = transport.apply_action_rate_limit(physical_request, previous)
    for key in ("requested_omega", "bounded_requested_omega", "applied_omega",
                "applied_delta_omega", "action_clipped", "rate_limited"):
        require(float(action[key]) == float(target[key]), "saved action replay: " + key)

    source = Path(spec["source_e109_output"])
    inventory = json.loads(Path(spec["inputs"]["source_tree_inventory"]["path"]).read_text())
    output = Path(spec["output"])
    output.mkdir(parents=True)
    cases, source_trees = copy_bound_cases(base, transport, source, output, inventory)

    transport.configure_interval(cases["ppo"], 327.9, 328.0, previous, float(action["applied_omega"]))
    transport.configure_interval(cases["zero"], 327.9, 328.0, 0.0, 0.0)
    image = subprocess.check_output(
        ["docker", "image", "inspect", base.IMAGE, "--format", "{{.Id}}"],
        text=True, timeout=10).strip()
    require(image == spec["openfoam_image_id"], "OpenFOAM image identity")
    with base.PairSolvers(cases, output, image) as solvers:
        health = solvers.pair(800, 328.0, transport.check_segment, guard)
    obs, sources = transport.total_drag_observation_at(cases["ppo"], 328.0, float(action["applied_omega"]))
    zero, zero_sources = transport.total_drag_observation_at(cases["zero"], 328.0, 0.0)
    obs, zero = np.asarray(obs, np.float32), np.asarray(zero, np.float32)
    expected_obs = np.asarray(target["output_observation"], np.float32)
    expected_zero = np.asarray(target["zero_observation"], np.float32)
    require(np.array_equal(obs, expected_obs), "controlled physical69 replay")
    require(np.array_equal(zero, expected_zero), "zero physical69 replay")
    numeric = {role: compare_restart_numeric(case / "328", source / ("case_mpc" if role == "ppo" else "case_zero") / "328")
               for role, case in cases.items()}
    replay_time = {role: compare_uniform_time(case / "328/uniform/time",
                  source / ("case_mpc" if role == "ppo" else "case_zero") / "328/uniform/time")
                   for role, case in cases.items()}
    require(all({part: base.tree(source / ("case_mpc" if role == "ppo" else "case_zero") / part)
                 for part in ("327.9", "328", "constant", "system")} == source_trees[role]
                for role in cases), "E109 source changed")
    result = {"status": "P064_B_E109_LAST_INTERVAL_RECOVERY_REPLAY_COMPLETE",
              "approval_sha256": sha(spec_path), "source_e109_result_sha256": sha(spec["inputs"]["e109_result"]["path"]),
              "previous_applied_omega": previous, "input_observation_sha256": hashlib.sha256(input_obs.tobytes()).hexdigest(),
              "action": action, "controlled_observation_max_abs_difference": 0.0,
              "zero_observation_max_abs_difference": 0.0, "restart_numeric": numeric,
              "uniform_time_numeric": replay_time,
              "observation_sources": sources, "zero_sources": zero_sources,
              "solver_health": health, "source_unchanged": True, "extension_executed": False}
    base.atomic_json(output / "result.json", result, exclusive=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    require(sha(args.spec) == args.spec_sha256, "spec identity")
    value = json.loads(args.spec.read_text())
    if args.execute:
        execute(value, args.spec)
    else:
        preflight(value, args.spec)
        print("P064_B_E109_RECOVERY_REPLAY_PREFLIGHT_PASS_NOT_RUNNING")
