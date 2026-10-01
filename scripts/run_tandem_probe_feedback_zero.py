#!/usr/bin/env python3
"""Three-step real OpenFOAM zero-action observation-loop plumbing test.

This is not RL feedback or evidence of drag/lift improvement.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
CFD = PROJECT / "cfd" / "tandem_cylinders"
sys.path.insert(0, str(CFD))
from analyze_baseline import log_health  # noqa: E402
from make_probe_feedback_case import substitute  # noqa: E402
sys.path.insert(0, str(PROJECT / "src"))
from fluid_control.openfoam_observation import observation_at  # noqa: E402


def available_memory_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    raise RuntimeError("MemAvailable unavailable")


def latest_time(case: Path) -> float:
    times = []
    for path in case.iterdir():
        if path.is_dir():
            try:
                times.append(float(path.name))
            except ValueError:
                pass
    if not times:
        raise ValueError(f"no OpenFOAM time directories: {case}")
    return max(times)


def run_zero(case_name: str, output: Path) -> dict:
    if not case_name.startswith("probe_feedback_zero_"):
        raise ValueError("zero smoke requires probe_feedback_zero_* case name")
    case = CFD / "cases" / case_name
    config = json.loads((case / "case_config.json").read_text(encoding="utf-8"))
    if config["case"] != case_name or config["status"] != "initialized_not_solved":
        raise ValueError("case is not a fresh initialized zero-smoke case")
    if config["source_restart_time"] != 80 or config["initial_omega"] != 0:
        raise ValueError("unexpected physical restart or initial action")
    steps = int(config["steps"])
    if steps < 2 or steps > 20:
        raise ValueError("zero integration smoke must have 2..20 steps")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite results: {output}")
    if not math.isclose(latest_time(case), 80, abs_tol=1e-8):
        raise ValueError("case already has computed time directories")
    if available_memory_gib() < 40:
        raise RuntimeError("insufficient MemAvailable for extra CFD work")
    output.mkdir(parents=True)
    rows = []
    for step in range(1, steps + 1):
        if available_memory_gib() < 40:
            raise RuntimeError("MemAvailable fell below 40 GiB")
        start = 80 + 0.1 * (step - 1)
        end = 80 + 0.1 * step
        if not math.isclose(latest_time(case), start, abs_tol=2e-6):
            raise ValueError(f"restart time mismatch before step {step}")
        control = case / "system" / "controlDict"
        substitute(control, "startTime", start)
        substitute(control, "endTime", end)
        log_path = case / f"log.pimpleFoam.segment_{step:03d}"
        if log_path.exists():
            raise FileExistsError(f"refusing to overwrite solver log: {log_path}")
        command = [
            "bash", str(CFD / "run_openfoam.sh"),
            "pimpleFoam", "-case", f"/case/cases/{case_name}",
        ]
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise subprocess.CalledProcessError(completed.returncode, command)
        if not math.isclose(latest_time(case), end, abs_tol=2e-6):
            raise ValueError(f"CFD did not reach t={end} at step {step}")
        health = log_health(log_path)
        if not health["solver_ended_cleanly"] or health["steps"] != 20:
            raise ValueError(f"bad solver segment {step}: {health}")
        if health["max_courant"] >= 1 or health["max_abs_global_continuity_per_step"] >= 1e-5:
            raise ValueError(f"unstable CFD segment {step}: {health}")
        observation, sources = observation_at(case, end, 0.0)
        if observation.shape != (67,) or not np.isfinite(observation).all():
            raise ValueError(f"invalid feedback observation after step {step}")
        row = {
            "step": step,
            "start_time": start,
            "end_time": end,
            "requested_omega": 0.0,
            "applied_omega": 0.0,
            "rear_cd": float(observation[64]),
            "rear_cl": float(observation[65]),
            "probe_abs_max": float(np.abs(observation[:64]).max()),
            "solver": health,
            "observation_sources": sources,
        }
        rows.append(row)
        (output / "progress.json").write_text(
            json.dumps({"completed_steps": step, "rows": rows}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"event": "zero_cfd_segment_complete", **row}), flush=True)
    result = {
        "status": "REAL_OPENFOAM_SEGMENTED_ZERO_OBSERVATION_SMOKE_OK",
        "scientific_status": "not_rl_not_control_benefit",
        "case": case_name,
        "source_restart_case": config["source_restart_case"],
        "source_restart_time": 80,
        "steps": steps,
        "rows": rows,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    config["status"] = "segmented_zero_observation_smoke_completed"
    (case / "case_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(result["status"], flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_name")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT / "artifacts"):
        raise ValueError("output must be under the project artifacts directory")
    run_zero(args.case_name, output)


if __name__ == "__main__":
    main()
