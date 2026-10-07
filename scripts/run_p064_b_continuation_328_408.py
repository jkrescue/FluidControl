"""Continue the retained B paired CFD from E109 t=328 to t=408."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import pickle
import shutil
import signal
import subprocess
import sys
import time

import numpy as np


PREP_STATUS = "P064_B_E109_CONTINUATION_328_408_PREPARATION"
APPROVED_STATUS = "P064_B_E109_CONTINUATION_328_408_EXECUTION_APPROVED"
COMPLETE_STATUS = "P064_B_E109_CONTINUATION_328_408_COMPLETE_NOT_INDEPENDENT_CONDITION"
POLICY_SHA = "5c05699e0851787d85d40c407647f80c19d3aebeb7dff82e019336cde77c6c6e"
VEC_SHA = "1d25005144b6436c3e2641ee89d1585e3c9f8b9fdb1f26b9cd39c7d83610c145"
START, END, STEPS = 328.0, 408.0, 800
WINDOWS = (
    ("tail_block_1", 328.0, 348.0), ("tail_block_2", 348.0, 368.0),
    ("tail_block_3", 368.0, 388.0), ("tail_block_4", 388.0, 408.0),
    ("tail_full_80", 328.0, 408.0),
    ("joined_full_160", 248.0, 408.0),
    ("joined_post_transition_140", 268.0, 408.0),
)


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


def validate_spec(spec):
    state = ((spec["status"] == PREP_STATUS and spec["execution_authorized"] is False)
             or (spec["status"] == APPROVED_STATUS and spec["execution_authorized"] is True))
    require(state, "preparation/approval state")
    require(spec["start_time"] == START and spec["end_time"] == END and spec["steps"] == STEPS,
            "fixed continuation interval")
    require(spec["windows"] == [{"name": n, "interval": [a, b], "left_endpoint_included": False}
                                 for n, a, b in WINDOWS], "fixed prospective windows")
    require(spec["physical_thresholds"] == {"paired_drag_reduction_min": .02,
            "rear_cl_rms_ratio_max": 1.05, "mean_bias_fraction_max": .10}, "unchanged thresholds")
    require(spec["resources"] == {"controller_memory_gib": 8, "solver_memory_gib_each": 8,
            "swap_gib": 0, "controller_cpu_quota_percent": 400, "solver_cpus_each": 2,
            "runtime_max_seconds": 2400, "inner_deadline_seconds": 2280,
            "startup_mem_available_gib": 50, "runtime_mem_available_gib": 22,
            "physical_mem_reserve_gib": 20, "disk_reserve_gib": 20}, "resources")
    require(sha(__file__) == spec["driver_sha256"], "driver identity")
    require(spec["inputs"]["policy"]["sha256"] == POLICY_SHA
            and spec["inputs"]["vecnormalize"]["sha256"] == VEC_SHA, "retained B policy")
    for row in spec["inputs"].values():
        require(sha(row["path"]) == row["sha256"], "input identity: " + row["path"])
    require(spec["source_files"] == {spec["inputs"][key]["path"]: spec["inputs"][key]["sha256"]
            for key in ("e109_driver", "pair_driver", "transport", "symmetry_adapter")},
            "minimal inherited source closure")
    require(not Path(spec["output"]).exists(), "exclusive output")
    return spec


def terminal_seed(e109):
    require(e109["cycles"] == 800 and len(e109["rows"]) == 800, "complete E109")
    row = e109["rows"][-1]
    require(row["step"] == 800 and row["start_time"] == 327.9 and row["end_time"] == START,
            "E109 endpoint")
    obs = np.asarray(row["output_observation"], np.float32)
    previous = float(row["applied_omega"])
    require(obs.shape == (69,) and np.isfinite(obs).all(), "terminal physical69")
    require(previous == 0.19905773401260382
            and float(obs[-1]) == 0.19905772805213928, "double limiter and float32 observation")
    return obs, previous


def bound_copy_pair(base, transport, source, output, inventory):
    cases, original = {}, {}
    for role, dirname in (("ppo", "case_mpc"), ("zero", "case_zero")):
        src, dst = Path(source) / dirname, Path(output) / dirname
        actual = {part: base.tree(src / part) for part in ("328", "constant", "system")}
        require(actual == {part: inventory[role][part] for part in actual},
                "bound t328 source tree: " + role)
        dst.mkdir()
        for part in ("328", "constant", "system"):
            shutil.copytree(src / part, dst / part)
        transport.substitute(dst / "system/controlDict", "writeInterval", .1)
        cases[role], original[role] = dst, actual
    return cases, original


def raw_force(case, body):
    paths = sorted((Path(case) / "postProcessing" / body).glob("*/coefficient.dat"))
    require(paths, "missing raw force stream")
    arrays, hashes = [], {}
    for path in paths:
        digest = sha(path)
        data = np.loadtxt(path, ndmin=2)
        require(data.shape[1] >= 5 and np.isfinite(data).all(), "invalid raw force")
        require(sha(path) == digest, "raw force changed")
        arrays.append(data[:, [0, 1, 4]])
        hashes[str(path)] = digest
    data = np.concatenate(arrays)
    order = np.argsort(data[:, 0], kind="stable")
    return data[order], hashes


def fixed_window(data, begin, end):
    selected = data[(data[:, 0] > begin) & (data[:, 0] <= end)]
    expected = begin + .005 * np.arange(1, round((end - begin) / .005) + 1)
    require(len(selected) == len(expected)
            and np.allclose(selected[:, 0], expected, rtol=0, atol=1e-8), "exact force grid")
    return selected


def summarize(old_source, cases, metric, old_windows, expected_old_hashes):
    streams, hashes, seen_old = {}, {}, set()
    for role, dirname in (("ppo", "case_mpc"), ("zero", "case_zero")):
        streams[role] = []
        for body in ("forceFront", "forceRear"):
            old, old_hash = raw_force(Path(old_source) / dirname, body)
            new, new_hash = raw_force(cases[role], body)
            require(all(expected_old_hashes.get(path) == digest for path, digest in old_hash.items()),
                    "E109 raw force identity")
            seen_old.update(old_hash)
            require(old[:, 0].max() <= START + 1e-8 and new[:, 0].min() >= START - 1e-8,
                    "old/new force segment boundary")
            # Exclude the copied junction from the new stream.  The original
            # stream owns t=328; the continuation owns (328,408].
            old = fixed_window(old, 248., 328.)
            new = fixed_window(new, 328., 408.)
            joined = np.concatenate((old, new))
            require(joined.shape[0] == 32000 and np.all(np.diff(joined[:, 0]) > 0),
                    "nonoverlapping old16000+tail16000 force grid")
            streams[role].append(joined)
            hashes.update(old_hash); hashes.update(new_hash)
    require(seen_old == set(expected_old_hashes), "complete E109 raw force closure")
    windows = {}
    for name, begin, end in WINDOWS:
        branches = {role: metric(*(fixed_window(a, begin, end) for a in arrays))
                    for role, arrays in streams.items()}
        for role, arrays in streams.items():
            branches[role]["rear_cl_abs_peak"] = float(
                np.max(np.abs(fixed_window(arrays[1], begin, end)[:, 2])))
        p, z = branches["ppo"], branches["zero"]
        windows[name] = {"interval": [begin, end], "left_endpoint_included": False,
            "branches": branches, "paired_drag_reduction": 1-p["total_cd_mean"]/z["total_cd_mean"],
            "paired_rear_cl_fluctuation_rms_ratio": p["rear_cl_fluctuation_rms"]/z["rear_cl_fluctuation_rms"],
            "absolute_mean_rear_cl_over_paired_zero_rms": abs(p["rear_cl_mean"])/z["rear_cl_fluctuation_rms"],
            "original_mean_bias_reference": .10, "mean_bias_sensitivity_reference": .20}
    return {"preserved_e109_windows": old_windows, "prospective_continuation_windows": windows}, hashes


def values_summary(values):
    a = np.asarray(values, np.float64)
    require(a.shape == (STEPS,) and np.isfinite(a).all(), "action summary")
    return {"min": float(a.min()), "max": float(a.max()), "mean": float(a.mean()),
            "rms": float(np.sqrt(np.mean(a*a)))}


def preflight(spec):
    spec = validate_spec(spec)
    e109 = json.loads(Path(spec["inputs"]["e109_result"]["path"]).read_text())
    terminal_seed(e109)
    inventory = json.loads(Path(spec["inputs"]["source_tree_inventory"]["path"]).read_text())
    require(set(inventory) == {"ppo", "zero"}
            and all(set(row).issuperset({"328", "constant", "system"}) for row in inventory.values()),
            "continuation inventory")
    replay = json.loads(Path(spec["inputs"]["replay_result"]["path"]).read_text())
    require(replay["status"] == "P064_B_E109_LAST_INTERVAL_RECOVERY_REPLAY_COMPLETE"
            and replay["controlled_observation_max_abs_difference"] == 0
            and replay["zero_observation_max_abs_difference"] == 0
            and replay["source_unchanged"] is True and replay["extension_executed"] is False,
            "accepted exact recovery replay")
    return spec


def execute(spec, spec_path):
    spec = preflight(spec)
    require(spec["status"] == APPROVED_STATUS and spec["execution_authorized"] is True, "execution approval")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only policy")
    repo = Path(spec["repo"])
    require(Path(sys.prefix).resolve() == repo / ".venv-curator-py312", "Python environment")
    require({k: importlib.metadata.version(k) for k in ("stable_baselines3", "gymnasium", "torch", "numpy")}
            == {"stable_baselines3": "2.7.1", "gymnasium": "1.2.3", "torch": "2.14.1", "numpy": "2.5.3"},
            "runtime packages")
    old = load_module("e109_driver_bound", spec["inputs"]["e109_driver"]["path"])
    base = load_module("e109_pair_bound", spec["inputs"]["pair_driver"]["path"])
    transport = load_module("e109_transport_bound", spec["inputs"]["transport"]["path"])
    symmetry = load_module("e109_symmetry_bound", spec["inputs"]["symmetry_adapter"]["path"])
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize
    vec = pickle.loads(Path(spec["inputs"]["vecnormalize"]["path"]).read_bytes())
    old.validate_vec(vec, VecNormalize)
    model = PPO.load(spec["inputs"]["policy"]["path"], device="cpu")
    model.policy.set_training_mode(False)
    require(all(p.device.type == "cpu" for p in model.policy.parameters()), "CPU policy")

    e109 = json.loads(Path(spec["inputs"]["e109_result"]["path"]).read_text())
    obs, previous = terminal_seed(e109)
    inventory = json.loads(Path(spec["inputs"]["source_tree_inventory"]["path"]).read_text())
    output, source = Path(spec["output"]), Path(spec["source_e109_output"])
    base.host_guard(startup=True)
    cgroup = Path("/sys/fs/cgroup") / next(line.split("::", 1)[1]
        for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::")).lstrip("/")
    require(0 < int((cgroup / "memory.max").read_text()) <= 8*2**30
            and int((cgroup / "memory.swap.max").read_text()) == 0,
            "controller 8GiB/noSwap cgroup")
    require(shutil.disk_usage(repo).free >= 20*2**30, "startup disk reserve")
    output.mkdir()
    cases, original = bound_copy_pair(base, transport, source, output, inventory)
    started = time.monotonic()
    def guard():
        mem = base.host_guard()
        require(shutil.disk_usage(repo).free >= 20*2**30, "runtime disk reserve")
        require(time.monotonic()-started < spec["resources"]["inner_deadline_seconds"], "deadline")
        with (output / "resources.jsonl").open("a") as stream:
            stream.write(json.dumps({"elapsed": time.monotonic()-started, **mem}) + "\n")
    def terminated(*_):
        raise RuntimeError("termination requested; cleaning owned solver containers")
    for sig in (signal.SIGTERM, signal.SIGINT): signal.signal(sig, terminated)
    image = subprocess.check_output(["docker", "image", "inspect", base.IMAGE, "--format", "{{.Id}}"],
                                    text=True, timeout=10).strip()
    require(image == spec["openfoam_image_id"], "OpenFOAM image")
    rows = []
    with base.PairSolvers(cases, output, image) as solvers:
        for step in range(1, STEPS+1):
            guard()
            begin, end = round(START+.1*(step-1), 10), round(START+.1*step, 10)
            require(all(abs(transport.latest_time(case)-begin) < 2e-6 for case in cases.values()), "stale state")
            input_obs = obs.copy()
            canonical = symmetry.canonicalize_physical69(input_obs)
            canonical_action = old.predict(model, vec, canonical.value)
            physical_request = float(symmetry.restore_physical_action(
                np.asarray([canonical_action], np.float32), canonical.orientation)[0])
            action = transport.apply_action_rate_limit(physical_request, previous)
            applied = float(action["applied_omega"])
            transport.configure_interval(cases["ppo"], begin, end, previous, applied)
            transport.configure_interval(cases["zero"], begin, end, 0., 0.)
            health = solvers.pair(step, end, transport.check_segment, guard)
            obs, sources = transport.total_drag_observation_at(cases["ppo"], end, applied)
            zero, zero_sources = transport.total_drag_observation_at(cases["zero"], end, 0.)
            obs, zero = old.observation(obs), old.observation(zero)
            rows.append({"step": step, "start_time": begin, "end_time": end,
                "canonical_policy_request": canonical_action,
                "canonical_orientation_applied": canonical.orientation,
                "physical_requested_omega_before_filter": physical_request, **action,
                "input_observation": input_obs.tolist(), "output_observation": obs.tolist(),
                "zero_observation": zero.tolist(), "observation_sources": sources,
                "zero_sources": zero_sources, "solver_health": health})
            base.atomic_json(output / "progress.json", {"completed_cycles": step, "rows": rows})
            previous = applied
    windows, raw_sha = summarize(source, cases, transport.force_metrics, e109["windows"],
                                 e109["raw_file_sha256"])
    for role, dirname in (("ppo", "case_mpc"), ("zero", "case_zero")):
        require({part: base.tree(source/dirname/part) for part in ("328", "constant", "system")} == original[role],
                "E109 source changed")
    require(sha(spec["inputs"]["policy"]["path"]) == POLICY_SHA
            and sha(spec["inputs"]["vecnormalize"]["path"]) == VEC_SHA, "policy changed")
    result = {"status": COMPLETE_STATUS, "approval_sha256": sha(spec_path),
        "source_e109_result_sha256": sha(spec["inputs"]["e109_result"]["path"]),
        "recovery_replay_result_sha256": sha(spec["inputs"]["replay_result"]["path"]),
        "cycles": STEPS, "start_time": START, "end_time": END, "rows": rows,
        "windows": windows, "raw_file_sha256": raw_sha,
        "initial_output_observation": terminal_seed(e109)[0].tolist(),
        "initial_previous_applied_omega": terminal_seed(e109)[1],
        "action_summary": {"applied": values_summary([r["applied_omega"] for r in rows]),
            "max_abs_applied_delta": max(abs(r["applied_omega"]-(terminal_seed(e109)[1] if i == 0 else rows[i-1]["applied_omega"]))
                                         for i, r in enumerate(rows))},
        "source_unchanged": True, "owned_containers_cleaned": True,
        "policy_recurrent_state": False, "fno_inference": False, "scientific_admission": False,
        "interpretation": "same retained B policy and physical condition; longer continuous evidence, not independent validation",
        "wall_seconds": time.monotonic()-started}
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
        preflight(value)
        print("P064_B_E109_CONTINUATION_PREFLIGHT_PASS_NOT_RUNNING")
