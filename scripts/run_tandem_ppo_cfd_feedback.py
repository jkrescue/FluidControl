#!/usr/bin/env python3
"""Short real-OpenFOAM PPO feedback pilot, gated and explicitly not a benefit claim."""
from __future__ import annotations

import argparse
import json
import math
import os
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

IMAGE = "fluid-control-physicsnemo-hydrogym:2.2.2-4ab9854"
HYDROGYM_COMMIT = "4ab9854dea3d84e38a59c25e0f5835a00cf8225f"


def infer_action(policy: Path, observation: np.ndarray) -> tuple[float, str]:
    if observation.shape != (67,) or not np.isfinite(observation).all():
        raise ValueError("expected 67 finite real-CFD channels before policy inference")
    command = [
        "docker", "run", "--rm", "--network", "none", "--read-only",
        "--tmpfs", "/tmp:rw,nosuid,nodev,size=1g",
        "--cpus", "2", "--memory", "8g", "--pids-limit", "128",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--env", "XDG_CACHE_HOME=/tmp/cache",
        "--env", "MPLCONFIGDIR=/tmp/matplotlib",
        "--env", "PYTHONDONTWRITEBYTECODE=1",
        "--mount", f"type=bind,src={PROJECT},dst=/workspace,readonly",
        "--workdir", "/workspace", IMAGE,
        "python", "scripts/infer_tandem_ppo_action.py",
        "--policy", str(Path("/workspace") / policy.relative_to(PROJECT)),
        "--observation-json", json.dumps(observation.tolist()),
    ]
    completed = subprocess.run(command, cwd=PROJECT, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(f"policy inference failed: {completed.stdout[-1200:]} {completed.stderr[-1200:]}")
    records = [
        line.partition("POLICY_ACTION_JSON=")[2]
        for line in completed.stdout.splitlines()
        if line.startswith("POLICY_ACTION_JSON=")
    ]
    if len(records) != 1:
        raise ValueError("expected exactly one PPO action marker")
    payload = json.loads(records[0])
    if payload.get("observation_channels") != 67 or payload.get("deterministic") is not True:
        raise ValueError("invalid PPO inference metadata")
    requested = float(payload["requested_omega"])
    if not math.isfinite(requested) or abs(requested) > 5 + 1e-6:
        raise ValueError("out-of-support PPO action")
    return requested, completed.stdout


def objective(cd: float, cl: float, omega: float, delta: float) -> float:
    return cd + 0.2 * cl * cl + 0.01 * omega * omega + 0.001 * delta * delta


def run(case_name: str, policy_run: Path, readiness: Path, output: Path) -> dict:
    if not case_name.startswith("probe_feedback_ppo_") or "/" in case_name:
        raise ValueError("case name must begin probe_feedback_ppo_")
    case = CFD / "cases" / case_name
    config = json.loads((case / "case_config.json").read_text(encoding="utf-8"))
    if config["case"] != case_name or config["status"] != "initialized_not_solved":
        raise ValueError("fresh initialized CFD case required")
    if config["source_restart_case"] != "tandem_backward_dt005" or config["source_restart_time"] != 80:
        raise ValueError("unexpected CFD baseline")
    steps = int(config["steps"])
    if not 2 <= steps <= 32:
        raise ValueError("real-CFD PPO pilot limited to 2..32 intervals")
    if not math.isclose(latest_time(case), 80, abs_tol=1e-8):
        raise ValueError("CFD case no longer at its t=80 restart")
    if output.exists():
        raise FileExistsError(output)
    policy_run = policy_run.resolve()
    if not policy_run.is_relative_to(PROJECT / "artifacts" / "hydrogym"):
        raise ValueError("policy must be under project artifacts/hydrogym")
    audit = json.loads((policy_run / "audit.json").read_text(encoding="utf-8"))
    if audit.get("status") != "SURROGATE_RL_PILOT_EVALUATED" or audit.get("physicsnemo_checkpoint_epoch") != 20:
        raise ValueError("PPO pilot audit is absent or failed")
    if audit.get("hydrogym_commit") != HYDROGYM_COMMIT:
        raise ValueError("PPO pilot HydroGym version mismatch")
    gate = json.loads(readiness.read_text(encoding="utf-8"))
    if gate.get("status") != "CANDIDATE_SURROGATE_SCREEN_PASS" or gate.get("checkpoint_epoch") != 20:
        raise ValueError("independent surrogate gate has not passed")
    policy = policy_run / "ppo_policy.zip"
    if not policy.is_file():
        raise FileNotFoundError(policy)
    commit = subprocess.check_output(
        ["git", "-C", str(PROJECT / ".tools" / "hydrogym"), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != HYDROGYM_COMMIT:
        raise ValueError("HydroGym checkout mismatch")
    subprocess.run(["docker", "image", "inspect", IMAGE], check=True, capture_output=True)
    if available_memory_gib() < 40:
        raise RuntimeError("MemAvailable below 40 GiB")
    reference = CFD / "cases" / "tandem_backward_dt005"
    observation, initial_sources = observation_at(reference, 80.0, 0.0)
    output.mkdir(parents=True)
    rows = []
    previous = 0.0
    for step in range(1, steps + 1):
        if available_memory_gib() < 40:
            raise RuntimeError("MemAvailable below 40 GiB")
        start = round(80.0 + 0.1 * (step - 1), 10)
        end = round(start + 0.1, 10)
        if not math.isclose(latest_time(case), start, abs_tol=2e-6):
            raise ValueError(f"CFD restart mismatch at interval {step}")
        requested, policy_stdout = infer_action(policy, observation)
        applied = float(np.clip(requested, previous - 0.5, previous + 0.5))
        applied = float(np.clip(applied, -5.0, 5.0))
        velocity = case / f"{start:g}" / "U"
        if not velocity.is_file():
            raise FileNotFoundError(velocity)
        velocity.write_text(
            replace_rear_patch(
                velocity.read_text(encoding="utf-8"), [(start, previous), (end, applied)]
            ),
            encoding="utf-8",
        )
        control = case / "system" / "controlDict"
        substitute(control, "startTime", start)
        substitute(control, "endTime", end)
        log_path = case / f"log.pimpleFoam.segment_{step:03d}"
        if log_path.exists():
            raise FileExistsError(log_path)
        command = ["bash", str(CFD / "run_openfoam.sh"), "pimpleFoam", "-case", f"/case/cases/{case_name}"]
        with log_path.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(command, cwd=PROJECT, stdout=handle, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise subprocess.CalledProcessError(completed.returncode, command)
        if not math.isclose(latest_time(case), end, abs_tol=2e-6):
            raise ValueError(f"solver did not reach t={end}")
        health = log_health(log_path)
        if not health["solver_ended_cleanly"] or health["steps"] != 20:
            raise ValueError(f"bad solver segment {step}: {health}")
        if health["max_courant"] >= 1 or health["max_abs_global_continuity_per_step"] >= 1e-5:
            raise ValueError(f"unstable solver segment {step}: {health}")
        observation, sources = observation_at(case, end, applied)
        zero_observation, _ = observation_at(reference, end, 0.0)
        cd, cl = float(observation[64]), float(observation[65])
        zcd, zcl = float(zero_observation[64]), float(zero_observation[65])
        row = {
            "step": step, "start_time": start, "end_time": end,
            "requested_omega": requested, "applied_omega": applied,
            "previous_omega": previous, "rate_limited": not math.isclose(requested, applied, abs_tol=1e-8),
            "rear_cd": cd, "rear_cl": cl,
            "zero_reference_rear_cd": zcd, "zero_reference_rear_cl": zcl,
            "instantaneous_objective": objective(cd, cl, applied, applied - previous),
            "zero_reference_instantaneous_objective": objective(zcd, zcl, 0.0, 0.0),
            "max_abs_probe_difference_from_zero": float(np.max(np.abs(observation[:64] - zero_observation[:64]))),
            "solver": health, "observation_sources": sources,
            "policy_marker": next(line for line in policy_stdout.splitlines() if line.startswith("POLICY_ACTION_JSON=")),
        }
        rows.append(row)
        (output / "progress.json").write_text(
            json.dumps({"completed_steps": step, "rows": rows}, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"event": "real_cfd_ppo_feedback_step", **row}), flush=True)
        previous = applied
    result = {
        "status": "REAL_CFD_PPO_FEEDBACK_PILOT_COMPLETED",
        "scientific_status": "short_feedback_diagnostic_not_long_horizon_control_benefit",
        "case": case_name,
        "policy_run": str(policy_run),
        "physicsnemo_checkpoint_epoch": 20,
        "hydrogym_commit": HYDROGYM_COMMIT,
        "initial_observation_sources": initial_sources,
        "steps": steps,
        "mean_objective_difference_from_zero": float(np.mean([
            row["instantaneous_objective"] - row["zero_reference_instantaneous_objective"]
            for row in rows
        ])),
        "rows": rows,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    config["status"] = "real_cfd_ppo_feedback_pilot_completed"
    config["policy_status"] = "frozen HydroGym PPO surrogate policy evaluated on real CFD"
    config["action_points"] = [[80.0, 0.0]] + [[row["end_time"], row["applied_omega"]] for row in rows]
    (case / "case_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(result["status"], flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_name")
    parser.add_argument("--policy-run", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT / "artifacts" / "hydrogym"):
        parser.error("output must be under project artifacts/hydrogym")
    readiness = args.readiness.resolve()
    if not readiness.is_relative_to(PROJECT / "artifacts" / "tandem_fno_expanded_spark_20epoch"):
        parser.error("readiness must be from project 20-epoch model artifacts")
    run(args.case_name, args.policy_run, readiness, output)


if __name__ == "__main__":
    main()
