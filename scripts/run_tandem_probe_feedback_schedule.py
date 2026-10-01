#!/usr/bin/env python3
"""Run an isolated real-CFD diagnostic action schedule; not an RL policy."""
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
from make_expanded_control_dataset import replace_rear_patch  # noqa: E402
from make_probe_feedback_case import substitute  # noqa: E402
sys.path.insert(0, str(PROJECT / "src"))
from fluid_control.openfoam_observation import observation_at  # noqa: E402
from run_tandem_probe_feedback_zero import available_memory_gib, latest_time  # noqa: E402


def run_schedule(case_name: str, output: Path, actions: list[float]) -> dict:
    if not case_name.startswith("probe_feedback_") or "/" in case_name:
        raise ValueError("case name must begin probe_feedback_")
    case = CFD / "cases" / case_name
    config = json.loads((case / "case_config.json").read_text(encoding="utf-8"))
    if config["case"] != case_name or config["status"] != "initialized_not_solved":
        raise ValueError("case is not fresh and initialized")
    if config["source_restart_time"] != 80 or config["initial_omega"] != 0:
        raise ValueError("unexpected source restart")
    if len(actions) != int(config["steps"]) or not 2 <= len(actions) <= 20:
        raise ValueError("action sequence length must equal configured steps (2..20)")
    if actions[0] != 0:
        raise ValueError("first interval must be a zero-action observation warm-up")
    previous = 0.0
    for action in actions:
        if not math.isfinite(action) or abs(action) > config["omega_limit"]:
            raise ValueError("nonfinite or out-of-range action")
        if abs(action - previous) > config["delta_omega_limit"] + 1e-9:
            raise ValueError("action rate exceeds configured limit")
        previous = action
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    if not math.isclose(latest_time(case), 80, abs_tol=1e-8):
        raise ValueError("case is not at its initial restart")
    if available_memory_gib() < 40:
        raise RuntimeError("MemAvailable below 40 GiB")
    output.mkdir(parents=True)
    rows = []
    previous = 0.0
    for step, action in enumerate(actions, 1):
        if available_memory_gib() < 40:
            raise RuntimeError("MemAvailable below 40 GiB")
        start = round(80 + 0.1 * (step - 1), 10)
        end = round(start + 0.1, 10)
        if not math.isclose(latest_time(case), start, abs_tol=2e-6):
            raise ValueError(f"restart mismatch before step {step}")
        velocity = case / f"{start:g}" / "U"
        if not velocity.is_file():
            raise FileNotFoundError(velocity)
        velocity.write_text(
            replace_rear_patch(
                velocity.read_text(encoding="utf-8"),
                [(start, previous), (end, action)],
            ),
            encoding="utf-8",
        )
        control = case / "system" / "controlDict"
        substitute(control, "startTime", start)
        substitute(control, "endTime", end)
        log_path = case / f"log.pimpleFoam.segment_{step:03d}"
        if log_path.exists():
            raise FileExistsError(log_path)
        command = [
            "bash", str(CFD / "run_openfoam.sh"),
            "pimpleFoam", "-case", f"/case/cases/{case_name}",
        ]
        with log_path.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(command, cwd=PROJECT, stdout=handle, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise subprocess.CalledProcessError(completed.returncode, command)
        if not math.isclose(latest_time(case), end, abs_tol=2e-6):
            raise ValueError(f"solver did not reach t={end}")
        health = log_health(log_path)
        if not health["solver_ended_cleanly"] or health["steps"] != 20:
            raise ValueError(f"bad solver segment: {health}")
        if health["max_courant"] >= 1 or health["max_abs_global_continuity_per_step"] >= 1e-5:
            raise ValueError(f"unstable segment: {health}")
        observation, sources = observation_at(case, end, action)
        if observation.shape != (67,) or not np.isfinite(observation).all():
            raise ValueError("bad CFD observation")
        row = {
            "step": step,
            "start_time": start,
            "end_time": end,
            "previous_omega": previous,
            "requested_omega": action,
            "applied_omega": action,
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
        print(json.dumps({"event": "cfd_diagnostic_segment_complete", **row}), flush=True)
        previous = action
    result = {
        "status": "REAL_OPENFOAM_DIAGNOSTIC_ACTION_SCHEDULE_OK",
        "scientific_status": "fixed_diagnostic_actions_not_rl_not_control_benefit",
        "case": case_name,
        "source_restart_case": config["source_restart_case"],
        "source_restart_time": 80,
        "actions": actions,
        "rows": rows,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    config["status"] = "diagnostic_action_schedule_completed"
    config["action_points"] = [[80.0, 0.0]] + [[row["end_time"], row["applied_omega"]] for row in rows]
    config["policy_status"] = "fixed diagnostic schedule; not RL control"
    (case / "case_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(result["status"], flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_name")
    parser.add_argument("--actions", required=True, help="comma-separated endpoint omega per 0.1 interval")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT / "artifacts"):
        raise ValueError("output must be under project artifacts")
    run_schedule(args.case_name, output, [float(token) for token in args.actions.split(",")])


if __name__ == "__main__":
    main()
