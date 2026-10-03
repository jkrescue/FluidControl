#!/usr/bin/env python3
"""Run a paired real-OpenFOAM proportional-lift feedback pilot.

This is a predeclared, physics-motivated controller baseline.  It is neither a
learned policy nor evidence that the Stage-C closed-loop objective has passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
CFD = PROJECT / "cfd" / "tandem_cylinders"
CASES = CFD / "cases"
sys.path.insert(0, str(CFD))
from analyze_baseline import load_coefficients, log_health  # noqa: E402
from make_expanded_control_dataset import replace_rear_patch  # noqa: E402
from make_probe_feedback_case import initialize, substitute  # noqa: E402

sys.path.insert(0, str(PROJECT / "src"))
from fluid_control.openfoam_observation import total_drag_observation_at  # noqa: E402

START_TIME = 80.0
CONTROL_INTERVAL = 0.1
SHEDDING_PERIOD = 6.2
DEFAULT_STEPS = 124
GAIN = 0.75
OMEGA_LIMIT = 1.0
MAX_DELTA_OMEGA = 0.1
MEMORY_FLOOR_GIB = 40.0
MAX_COURANT = 0.8
MAX_CONTINUITY = 1.0e-5
HARD_FORCE_LIMIT = 10.0


def available_memory_gib() -> float:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    raise RuntimeError("MemAvailable unavailable")


def latest_time(case: Path) -> float:
    times: list[float] = []
    for path in case.iterdir():
        if not path.is_dir():
            continue
        try:
            times.append(float(path.name))
        except ValueError:
            continue
    if not times:
        raise ValueError(f"no OpenFOAM time directories: {case}")
    return max(times)


def feedback_action(
    rear_cl: float, previous_omega: float, gain: float = GAIN
) -> tuple[float, float]:
    """Oppose rear lift via the expected Magnus-force sign, with hard bounds."""
    if not all(math.isfinite(value) for value in (rear_cl, previous_omega)):
        raise ValueError("feedback inputs must be finite")
    if not math.isfinite(gain) or not 0.0 < gain <= 1.0:
        raise ValueError("feedback gain must lie in (0, 1]")
    target = float(np.clip(gain * rear_cl, -OMEGA_LIMIT, OMEGA_LIMIT))
    applied = float(
        np.clip(
            target,
            previous_omega - MAX_DELTA_OMEGA,
            previous_omega + MAX_DELTA_OMEGA,
        )
    )
    return target, applied


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_interval(case: Path, start: float, end: float, before: float, after: float) -> None:
    velocity = case / f"{start:g}" / "U"
    if not velocity.is_file():
        raise FileNotFoundError(velocity)
    velocity.write_text(
        replace_rear_patch(
            velocity.read_text(encoding="utf-8"), [(start, before), (end, after)]
        ),
        encoding="utf-8",
    )
    control = case / "system" / "controlDict"
    substitute(control, "startTime", start)
    substitute(control, "endTime", end)


def launch_segment(case_name: str, step: int) -> tuple[subprocess.Popen, object]:
    case = CASES / case_name
    log_path = case / f"log.pimpleFoam.phase_feedback_{step:03d}"
    if log_path.exists():
        raise FileExistsError(log_path)
    handle = log_path.open("w", encoding="utf-8")
    command = [
        "bash",
        str(CFD / "run_openfoam.sh"),
        "pimpleFoam",
        "-case",
        f"/case/cases/{case_name}",
    ]
    process = subprocess.Popen(
        command,
        cwd=PROJECT,
        stdout=handle,
        stderr=subprocess.STDOUT,
    )
    return process, handle


def check_segment(case: Path, step: int, expected_end: float) -> dict:
    log_path = case / f"log.pimpleFoam.phase_feedback_{step:03d}"
    if not math.isclose(latest_time(case), expected_end, abs_tol=2.0e-6):
        raise ValueError(f"{case.name} did not reach t={expected_end}")
    health = log_health(log_path)
    if not health["solver_ended_cleanly"] or health["steps"] != 20:
        raise ValueError(f"bad solver segment: {case.name}: {health}")
    if (
        health["max_courant"] >= MAX_COURANT
        or health["max_abs_global_continuity_per_step"] >= MAX_CONTINUITY
    ):
        raise ValueError(f"unsafe solver segment: {case.name}: {health}")
    return health


def read_force_window(case: Path, object_name: str, begin: float, end: float) -> np.ndarray:
    samples: dict[float, tuple[float, float]] = {}
    paths = sorted(case.glob(f"postProcessing/{object_name}/*/coefficient.dat"))
    if not paths:
        raise FileNotFoundError(f"missing {object_name} coefficients: {case}")
    for path in paths:
        for time, cd, cl in load_coefficients(path):
            if not begin - 1.0e-8 <= time <= end + 1.0e-8:
                continue
            key = round(time, 8)
            value = (cd, cl)
            if key in samples and not np.allclose(samples[key], value, rtol=0, atol=1e-8):
                raise ValueError(f"conflicting restart force at t={time}: {path}")
            samples[key] = value
    if len(samples) < 100:
        raise ValueError(f"insufficient force samples for {case.name}: {len(samples)}")
    return np.asarray([[time, *samples[time]] for time in sorted(samples)], dtype=float)


def force_metrics(front: np.ndarray, rear: np.ndarray) -> dict:
    if front.shape != rear.shape or not np.allclose(front[:, 0], rear[:, 0]):
        raise ValueError("front/rear force time grids differ")
    if not np.isfinite(front).all() or not np.isfinite(rear).all():
        raise ValueError("non-finite force window")
    front_mean = float(np.mean(front[:, 2]))
    rear_mean = float(np.mean(rear[:, 2]))
    return {
        "samples": len(front),
        "total_cd_mean": float(np.mean(front[:, 1] + rear[:, 1])),
        "front_cd_mean": float(np.mean(front[:, 1])),
        "rear_cd_mean": float(np.mean(rear[:, 1])),
        "front_cl_mean": front_mean,
        "rear_cl_mean": rear_mean,
        "front_cl_fluctuation_rms": float(
            np.sqrt(np.mean(np.square(front[:, 2] - front_mean)))
        ),
        "rear_cl_fluctuation_rms": float(
            np.sqrt(np.mean(np.square(rear[:, 2] - rear_mean)))
        ),
        "front_cl_total_rms": float(np.sqrt(np.mean(np.square(front[:, 2])))),
        "rear_cl_total_rms": float(np.sqrt(np.mean(np.square(rear[:, 2])))),
    }


def compare_metrics(control: dict, zero: dict) -> dict:
    reduction = 1.0 - control["total_cd_mean"] / zero["total_cd_mean"]
    result = {
        "total_drag_reduction": reduction,
        "rear_cl_total_rms_ratio": (
            control["rear_cl_total_rms"] / zero["rear_cl_total_rms"]
        ),
        "rear_cl_fluctuation_rms_ratio": (
            control["rear_cl_fluctuation_rms"]
            / zero["rear_cl_fluctuation_rms"]
        ),
        "front_cl_total_rms_ratio": (
            control["front_cl_total_rms"] / zero["front_cl_total_rms"]
        ),
        "abs_rear_cl_mean_over_zero_fluctuation_rms": (
            abs(control["rear_cl_mean"]) / zero["rear_cl_fluctuation_rms"]
        ),
    }
    result["predeclared_checks"] = {
        "total_drag_decreased": reduction > 0.0,
        "rear_cl_total_rms_increase_at_most_5pct": (
            result["rear_cl_total_rms_ratio"] <= 1.05
        ),
        "rear_mean_lift_at_most_10pct_zero_fluctuation_rms": (
            result["abs_rear_cl_mean_over_zero_fluctuation_rms"] <= 0.10
        ),
    }
    result["short_pilot_joint_check"] = all(result["predeclared_checks"].values())
    result["canonical_physical_checks"] = {
        "total_drag_reduction_at_least_2pct": reduction >= 0.02,
        "rear_cl_fluctuation_rms_increase_at_most_5pct": (
            result["rear_cl_fluctuation_rms_ratio"] <= 1.05
        ),
        "rear_mean_lift_at_most_10pct_zero_fluctuation_rms": (
            result["abs_rear_cl_mean_over_zero_fluctuation_rms"] <= 0.10
        ),
    }
    result["canonical_physical_joint_check"] = all(
        result["canonical_physical_checks"].values()
    )
    return result


def update_case_metadata(
    case: Path, role: str, steps: int, gain: float, lag_steps: int
) -> None:
    path = case / "case_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config.update(
        purpose="paired real-CFD proportional-lift feedback pilot",
        pair_role=role,
        controller=(
            f"omega_target=clip({gain:g}*rear_Cl(t-{lag_steps * CONTROL_INTERVAL:g}),-1,1); "
            "|delta_omega|<=0.1 per 0.1 D/U"
            if role == "feedback"
            else "same-phase segmented omega=0 control"
        ),
        controller_predeclared_before_run=True,
        feedback_model="none; no surrogate and no learned policy",
        feedback_lag_steps=lag_steps,
        feedback_lag_time=lag_steps * CONTROL_INTERVAL,
        steps=steps,
        end_time=START_TIME + steps * CONTROL_INTERVAL,
        omega_limit=OMEGA_LIMIT,
        delta_omega_limit=MAX_DELTA_OMEGA,
        warmup_first_interval="none",
    )
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def run_pair(
    control_name: str,
    zero_name: str,
    output: Path,
    steps: int,
    gain: float,
    lag_steps: int,
) -> dict:
    if not control_name.startswith("probe_feedback_phase_control_"):
        raise ValueError("control case must begin probe_feedback_phase_control_")
    if not zero_name.startswith("probe_feedback_phase_zero_"):
        raise ValueError("zero case must begin probe_feedback_phase_zero_")
    if control_name == zero_name:
        raise ValueError("paired cases must be distinct")
    if steps < 62 or steps > 200:
        raise ValueError("paired pilot requires 62..200 intervals")
    if not math.isfinite(gain) or not 0.0 < gain <= 1.0:
        raise ValueError("feedback gain must lie in (0, 1]")
    if not 0 <= lag_steps <= 31:
        raise ValueError("feedback lag_steps must lie in 0..31")
    if output.exists():
        raise FileExistsError(output)
    if (CASES / control_name).exists() or (CASES / zero_name).exists():
        raise FileExistsError("refusing to reuse an existing paired case")
    if available_memory_gib() < MEMORY_FLOOR_GIB:
        raise RuntimeError("insufficient MemAvailable")

    control_case = initialize(control_name, steps)
    zero_case = initialize(zero_name, steps)
    update_case_metadata(control_case, "feedback", steps, gain, lag_steps)
    update_case_metadata(zero_case, "zero", steps, gain, lag_steps)
    source_hashes = {
        field: {
            "feedback": file_sha256(control_case / "80" / field),
            "zero": file_sha256(zero_case / "80" / field),
        }
        for field in ("U", "p")
    }
    if any(values["feedback"] != values["zero"] for values in source_hashes.values()):
        raise ValueError("paired initial fields differ")

    output.mkdir(parents=True)
    reference = CASES / "tandem_backward_dt005"
    observation, initial_sources = total_drag_observation_at(reference, START_TIME, 0.0)
    previous_omega = 0.0
    lift_history: list[float] = []
    rows: list[dict] = []
    peak_courant = 0.0
    for step in range(1, steps + 1):
        if available_memory_gib() < MEMORY_FLOOR_GIB:
            raise RuntimeError("MemAvailable fell below the safety floor")
        start = round(START_TIME + CONTROL_INTERVAL * (step - 1), 10)
        end = round(start + CONTROL_INTERVAL, 10)
        if not all(
            math.isclose(latest_time(case), start, abs_tol=2.0e-6)
            for case in (control_case, zero_case)
        ):
            raise ValueError(f"paired restart mismatch before interval {step}")
        measured_rear_cl = float(observation[67])
        lift_history.append(measured_rear_cl)
        history_ready = len(lift_history) > lag_steps
        feedback_rear_cl = lift_history[-1 - lag_steps] if history_ready else 0.0
        target, applied = feedback_action(feedback_rear_cl, previous_omega, gain)
        configure_interval(control_case, start, end, previous_omega, applied)
        configure_interval(zero_case, start, end, 0.0, 0.0)

        launched = [
            (*launch_segment(control_name, step), control_case),
            (*launch_segment(zero_name, step), zero_case),
        ]
        try:
            return_codes = [process.wait() for process, _, _ in launched]
        finally:
            for _, handle, _ in launched:
                handle.close()
        if any(return_codes):
            raise RuntimeError(f"paired OpenFOAM segment failed: {return_codes}")
        health_control = check_segment(control_case, step, end)
        health_zero = check_segment(zero_case, step, end)
        peak_courant = max(
            peak_courant,
            health_control["max_courant"],
            health_zero["max_courant"],
        )
        observation, control_sources = total_drag_observation_at(
            control_case, end, applied
        )
        zero_observation, zero_sources = total_drag_observation_at(zero_case, end, 0.0)
        if max(np.max(np.abs(observation[64:68])), np.max(np.abs(zero_observation[64:68]))) > HARD_FORCE_LIMIT:
            raise RuntimeError("force safety limit exceeded")
        row = {
            "step": step,
            "start_time": start,
            "end_time": end,
            "measured_rear_cl_at_action_time": measured_rear_cl,
            "feedback_signal_rear_cl": feedback_rear_cl,
            "feedback_history_ready": history_ready,
            "feedback_signal_time": start - lag_steps * CONTROL_INTERVAL,
            "target_omega": target,
            "applied_omega": applied,
            "delta_omega": applied - previous_omega,
            "feedback_forces": {
                "front_cd": float(observation[64]),
                "front_cl": float(observation[65]),
                "rear_cd": float(observation[66]),
                "rear_cl": float(observation[67]),
            },
            "zero_forces": {
                "front_cd": float(zero_observation[64]),
                "front_cl": float(zero_observation[65]),
                "rear_cd": float(zero_observation[66]),
                "rear_cl": float(zero_observation[67]),
            },
            "solver": {"feedback": health_control, "zero": health_zero},
            "observation_sources": {
                "feedback": control_sources,
                "zero": zero_sources,
            },
        }
        rows.append(row)
        previous_omega = applied
        (output / "progress.json").write_text(
            json.dumps({"completed_steps": step, "rows": rows}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "event": "paired_real_cfd_phase_feedback_step",
                    "step": step,
                    "end_time": end,
                    "rear_cl_measurement": measured_rear_cl,
                    "applied_omega": applied,
                    "feedback_total_cd": float(observation[64] + observation[66]),
                    "zero_total_cd": float(zero_observation[64] + zero_observation[66]),
                    "mem_available_gib": available_memory_gib(),
                }
            ),
            flush=True,
        )

    end_time = START_TIME + steps * CONTROL_INTERVAL
    evaluation_start = end_time - SHEDDING_PERIOD
    metrics: dict[str, dict] = {}
    for label, case in (("feedback", control_case), ("zero", zero_case)):
        front = read_force_window(case, "forceFront", evaluation_start, end_time)
        rear = read_force_window(case, "forceRear", evaluation_start, end_time)
        metrics[label] = force_metrics(front, rear)
    comparison = compare_metrics(metrics["feedback"], metrics["zero"])
    omega = np.asarray([row["applied_omega"] for row in rows], dtype=float)
    result = {
        "status": "REAL_OPENFOAM_PHASE_FEEDBACK_PAIR_COMPLETED",
        "scientific_status": (
            "predeclared_physics_feedback_short_pilot_not_learned_policy_"
            "not_final_closed_loop_claim"
        ),
        "solver": "OpenFOAM v2512 pimpleFoam",
        "geometry": "tandem circular cylinders Re=100, L/D=5, rear rotation only",
        "source_restart": "tandem_backward_dt005 at t=80",
        "initial_observation_sources": initial_sources,
        "source_field_hashes": source_hashes,
        "controller": {
            "law": (
                f"omega_target=clip({gain:g}*rear_Cl"
                f"(t-{lag_steps * CONTROL_INTERVAL:g}),-1,1)"
            ),
            "gain": gain,
            "lag_steps": lag_steps,
            "lag_time": lag_steps * CONTROL_INTERVAL,
            "warmup": "omega=0 until the causal lag history is available",
            "physical_sign_rationale": (
                "positive rear lift requests positive rotation, whose Magnus side-force "
                "is expected to oppose the measured lift"
            ),
            "control_interval": CONTROL_INTERVAL,
            "omega_limit": OMEGA_LIMIT,
            "max_delta_omega": MAX_DELTA_OMEGA,
            "max_abs_domega_dt": MAX_DELTA_OMEGA / CONTROL_INTERVAL,
            "surrogate_or_learned_model_used": False,
        },
        "pair": {"feedback": control_name, "zero": zero_name},
        "steps": steps,
        "simulation_window": [START_TIME, end_time],
        "predeclared_evaluation_window": [evaluation_start, end_time],
        "metrics": metrics,
        "comparison": comparison,
        "effort_proxy": {
            "omega_rms": float(np.sqrt(np.mean(np.square(omega)))),
            "omega_abs_max": float(np.max(np.abs(omega))),
            "delta_omega_abs_max": float(
                np.max(np.abs(np.diff(np.concatenate(([0.0], omega)))))
            ),
            "note": "kinematic proxy only; torque and actuator power were not measured",
        },
        "numerical_health": {
            "peak_courant_across_pair": peak_courant,
            "memory_floor_gib": MEMORY_FLOOR_GIB,
        },
        "limitations": [
            "one initial phase and one coarse mesh",
            "only one shedding period is used for the predeclared comparison",
            "no torque or actuator power measurement",
            "not PPO, not PhysicsNeMo control, and not a final closed-loop result",
        ],
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    for role, case in (("feedback", control_case), ("zero", zero_case)):
        path = case / "case_config.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        config["status"] = "paired_real_cfd_phase_feedback_pilot_completed"
        config["result"] = str(output / "result.json")
        config["pair_role"] = role
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(result["status"], flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-case", required=True)
    parser.add_argument("--zero-case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--gain", type=float, default=GAIN)
    parser.add_argument("--lag-steps", type=int, default=0)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT / "artifacts" / "tandem_cylinders"):
        parser.error("output must stay under artifacts/tandem_cylinders")
    run_pair(
        args.control_case,
        args.zero_case,
        output,
        args.steps,
        args.gain,
        args.lag_steps,
    )


if __name__ == "__main__":
    main()
