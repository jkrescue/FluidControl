"""CPU-only matched-observation action decomposition for two frozen PPO seeds."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np

STATUS = "P064_SEED_MATCHED_OBSERVATION_CPU_REPLAY_APPROVED"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
    temporary.replace(path)


def mem_available():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.strip().split()[0]) * 1024
    return values["MemAvailable"]


def cgroup_limits():
    relative = next(
        row.split("::", 1)[1]
        for row in Path("/proc/self/cgroup").read_text().splitlines()
        if row.startswith("0::")
    )
    root = Path("/sys/fs/cgroup") / relative.lstrip("/")
    return int((root / "memory.max").read_text()), int((root / "memory.swap.max").read_text())


def load_module(path):
    spec = importlib.util.spec_from_file_location("reviewed_p064_cfd_driver", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


WINDOWS = {
    "early_12p4": (148.0, 160.4, False),
    "early_first_6p2": (148.0, 154.2, False),
    "early_trailing_6p2": (154.2, 160.4, False),
    "primary_final_60": (168.0, 228.0, False),
    "historical_inclusive_final_60": (168.0, 228.0, True),
    "full_80": (148.0, 228.0, False),
}


def select_window(rows, window):
    lower, upper, inclusive = window
    return [
        row
        for row in rows
        if (row["end_time"] >= lower if inclusive else row["end_time"] > lower)
        and row["end_time"] <= upper
    ]


def values_summary(values):
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 1 and len(values) > 0 and np.isfinite(values).all(), "finite summary values")
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "mean_abs": float(np.abs(values).mean()),
        "rms": float(np.sqrt(np.mean(values * values))),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def summarize(rows):
    result = {}
    for policy in ("seed20261006", "seed20261007"):
        result[policy] = {
            component: values_summary([row[policy][component] for row in rows])
            for component in ("raw", "reflected", "odd", "even")
        }
    result["paired"] = {
        component + "_mean_abs_difference": float(
            np.mean(
                [
                    abs(row["seed20261007"][component] - row["seed20261006"][component])
                    for row in rows
                ]
            )
        )
        for component in ("raw", "reflected", "odd", "even")
    }
    return result


def validate_spec(spec):
    require(spec.get("status") == STATUS and spec.get("execution_authorized") is True, "execution approval required")
    require(spec.get("purpose") == "matched_observation_action_decomposition_no_filter", "purpose differs")
    require(spec.get("rows") == 800 and spec.get("device") == "cpu", "fixed replay contract")
    require(spec.get("old_b00_result_sha256") == sha(spec["old_b00_result"]), "old b00 result changed")
    require(spec.get("driver_sha256") == sha(spec["driver"]), "reviewed driver changed")
    require(spec["source_files"].get(str(Path(__file__).resolve())) == sha(__file__), "worker source not bound")
    for row in spec["training"] .values():
        for key in ("result", "policy", "vecnormalize"):
            require(sha(row[key]["path"]) == row[key]["sha256"], f"{key} changed")


def execute(spec, output, approval_sha256):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "GPU must be hidden")
    memory_max, swap_max = cgroup_limits()
    require(memory_max == 4 * 2**30 and swap_max == 0, "4GiB/no-swap cgroup required")
    minimum_available = mem_available()
    require(minimum_available >= 22 * 2**30, "22GiB MemAvailable reserve")
    start = time.monotonic()
    driver = load_module(spec["driver"])
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize
    old_result = json.load(open(spec["training"]["seed20261006"]["result"]["path"]))
    new_result = json.load(open(spec["training"]["seed20261007"]["result"]["path"]))
    for result, row in ((old_result, spec["training"]["seed20261006"]), (new_result, spec["training"]["seed20261007"])):
        driver.validate_training(result, row["policy"]["sha256"], row["vecnormalize"]["sha256"], "B", spec["candidate_manifest_sha256"])
    models = {}
    for name, row in spec["training"].items():
        vec = pickle.loads(Path(row["vecnormalize"]["path"]).read_bytes())
        driver.validate_vec(vec, VecNormalize)
        model = PPO.load(row["policy"]["path"], device="cpu")
        require(model.observation_space == vec.observation_space and model.action_space == vec.action_space, "policy/Vec spaces differ")
        model.policy.set_training_mode(False)
        models[name] = (model, vec)
    source = json.load(open(spec["old_b00_result"]))
    require(len(source["rows"]) == 800 and source["policy_sha256"] == spec["training"]["seed20261006"]["policy"]["sha256"], "fixed old b00 rows")
    records = []
    maximum_old_reproduction_error = 0.0
    for index, source_row in enumerate(source["rows"], 1):
        require(time.monotonic() - start < 180, "180-second replay deadline")
        if index % 50 == 0:
            minimum_available = min(minimum_available, mem_available())
            require(minimum_available >= 22 * 2**30, "22GiB runtime reserve")
        record = {"step": index, "end_time": source_row["end_time"], "source_observation_sha256": hashlib.sha256(np.asarray(source_row["input_observation"], dtype=np.float32).tobytes()).hexdigest()}
        for name, (model, vec) in models.items():
            requests = driver.projected_request(lambda value, m=model, v=vec: driver.predict(m, v, value), source_row["input_observation"])
            raw, reflected = requests["raw_policy_request"], requests["reflected_policy_request"]
            record[name] = {"raw": raw, "reflected": reflected, "odd": requests["projected_policy_request"], "even": 0.5 * (raw + reflected)}
        maximum_old_reproduction_error = max(maximum_old_reproduction_error, *(abs(record["seed20261006"][key] - source_row[saved]) for key, saved in (("raw", "raw_policy_request"), ("reflected", "reflected_policy_request"), ("odd", "projected_policy_request"))))
        records.append(record)
    require(maximum_old_reproduction_error <= 1e-7, "old deterministic requests not reproduced")
    atomic_json(output / "raw_records.json", {"rows": records})
    windows = {name: summarize(select_window(records, definition)) for name, definition in WINDOWS.items()}
    result = {
        "status": "P064_SEED_MATCHED_OBSERVATION_CPU_REPLAY_COMPLETE",
        "approval_sha256": approval_sha256,
        "rows": 800,
        "maximum_old_reproduction_error": maximum_old_reproduction_error,
        "windows": windows,
        "minimum_available_bytes": minimum_available,
        "wall_seconds": time.monotonic() - start,
        "gpu_used": False,
        "fno_loaded": False,
        "cfd_executed": False,
        "filter_state_propagated": False,
        "scientific_admission": False,
        "raw_records_sha256": sha(output / "raw_records.json"),
    }
    atomic_json(output / "result.json", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    require(sha(args.spec) == args.spec_sha256, "spec SHA differs")
    spec = json.load(args.spec.open())
    validate_spec(spec)
    require(args.output == Path(spec["output"]) and not args.output.exists(), "exclusive output required")
    if not args.execute:
        print(json.dumps({"status": "PREPARATION_ONLY", "rows": 800}))
        return
    args.output.mkdir()
    execute(spec, args.output, args.spec_sha256)


if __name__ == "__main__":
    main()
