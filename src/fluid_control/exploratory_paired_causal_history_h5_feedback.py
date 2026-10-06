"""Project glue for paired canonical-history-H5/real-CFD exploratory trial.

The caller owns the reviewed OpenFOAM and official-K1 transports.  This module
only enforces ordering: observe both current CFD states before choosing or
configuring an action, execute one ramped interval, then record actual forces.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np


START = 148.0
CONTROL_INTERVAL = 0.1
STEPS = 10
ACTION_LIMIT = 0.75
RATE_LIMIT = 0.10


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def array_sha256(value) -> str:
    array = np.asarray(value, dtype="<f4")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    """Publish progress without allowing a half-written JSON document."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    _require(not temporary.exists() and not temporary.is_symlink(), "stale temporary output")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
    temporary.replace(path)


def _force_row(value) -> dict[str, float]:
    values = np.asarray(value, dtype=np.float64)
    _require(values.shape == (4,) and np.isfinite(values).all(), "four finite CFD forces required")
    return {
        key: float(item)
        for key, item in zip(("front_cd", "front_cl", "rear_cd", "rear_cl"), values, strict=True)
    }


def run_paired_ten_cycle(
    *,
    cases: dict[str, Path],
    output: Path,
    latest_time: Callable[[Path], float],
    observe_current: Callable[[str, Path, float, float], dict],
    plan_action: Callable[[dict, float], dict],
    configure_interval: Callable[[Path, float, float, float, float], None],
    solve_pair: Callable[[int, float], dict],
    observe_forces: Callable[[str, Path, float, float], np.ndarray],
    summarize_actual: Callable[[str, Path, float, float], dict],
    guard: Callable[[], None],
    identity: dict,
) -> dict:
    """Run exactly ten reobserve/replan/execute cycles for MPC and zero cases."""
    _require(set(cases) == {"mpc", "zero"}, "exact paired branches required")
    output = Path(output)
    _require(output.is_dir() and not output.is_symlink(), "existing exclusive output required")
    _require(isinstance(identity, dict) and identity, "bound identity required")
    previous = 0.0
    rows: list[dict] = []
    for step in range(1, STEPS + 1):
        guard()
        begin = round(START + (step - 1) * CONTROL_INTERVAL, 10)
        end = round(begin + CONTROL_INTERVAL, 10)
        for case in cases.values():
            _require(math.isclose(latest_time(case), begin, abs_tol=2e-6), "paired current time differs")

        # Both packets are acquired before plan_action and before configure_interval.
        current = {
            role: observe_current(role, case, begin, previous if role == "mpc" else 0.0)
            for role, case in cases.items()
        }
        for role, packet in current.items():
            _require(math.isclose(float(packet["time"]), begin, abs_tol=1e-5), "stale current packet")
            _require(packet.get("sampled_before_action") is True, "current packet ordering proof missing")
            _require(packet.get("role") == role, "current packet role differs")

        decision = plan_action(current["mpc"], previous)
        _require(decision.get("status")
                 == "EXPLORATORY_CANONICAL_HISTORY_H5_SELECTION_NOT_ADMISSION",
                 "selector status differs")
        _require(decision.get("execute_only_first_action") is True, "must execute only first action")
        applied = float(decision["selected_action"])
        _require(math.isfinite(applied) and abs(applied) <= ACTION_LIMIT + 1e-12,
                 "selected action exceeds magnitude limit")
        _require(abs(applied - previous) <= RATE_LIMIT + 1e-12,
                 "selected action exceeds rate limit")

        # Existing project transport applies a linear ramp previous->applied over [begin,end].
        configure_interval(cases["mpc"], begin, end, previous, applied)
        configure_interval(cases["zero"], begin, end, 0.0, 0.0)
        health = solve_pair(step, end)
        _require(set(health) == {"mpc", "zero"}, "paired solver health missing")
        for case in cases.values():
            _require(math.isclose(latest_time(case), end, abs_tol=2e-6), "paired CFD endpoint differs")
        actual = {
            role: _force_row(observe_forces(role, case, end, applied if role == "mpc" else 0.0))
            for role, case in cases.items()
        }
        selected = int(decision["selected_index"])
        predicted = np.asarray(decision["predicted_forces_h5"], dtype=np.float64)
        _require(predicted.shape == (5, 5, 4) and np.isfinite(predicted).all(),
                 "fixed five-by-H5 force predictions required")
        predicted_endpoint = _force_row(predicted[selected, 0])
        force_keys = tuple(predicted_endpoint)
        endpoint_error = {
            key: predicted_endpoint[key] - actual["mpc"][key] for key in force_keys
        }
        row = {
            "step": step,
            "start_time": begin,
            "end_time": end,
            "current_sample_sha256": {
                role: str(packet["sample_sha256"]) for role, packet in current.items()
            },
            "previous_omega": previous,
            "selected_omega": applied,
            "boundary_semantics": "linear_ramp_previous_to_selected_over_0.1_D_over_U",
            "zero_omega": 0.0,
            "decision": decision,
            "selected_predicted_next_forces": predicted_endpoint,
            "selected_prediction_minus_actual_next_force": endpoint_error,
            "selected_prediction_vs_actual_zero_direction_matches_actual_pair_effect": {
                key: bool(
                    np.sign(predicted_endpoint[key] - actual["zero"][key])
                    == np.sign(actual["mpc"][key] - actual["zero"][key])
                )
                for key in force_keys
            },
            "actual_endpoint_forces": actual,
            "solver_health": health,
        }
        rows.append(row)
        previous = applied
        atomic_json(output / "progress.json", {
            "status": "EXPLORATORY_REAL_CFD_CANONICAL_HISTORY_H5_RUNNING_NOT_ADMISSION",
            "completed_cycles": step,
            "identity": identity,
            "rows": rows,
        })
        guard()

    metrics = {role: summarize_actual(role, case, START, START + STEPS * CONTROL_INTERVAL)
               for role, case in cases.items()}
    omega = np.asarray([row["selected_omega"] for row in rows], dtype=np.float64)
    before = np.concatenate(([0.0], omega[:-1]))
    delta = np.diff(np.concatenate(([0.0], omega)))
    return {
        "status": "EXPLORATORY_REAL_CFD_CANONICAL_HISTORY_H5_COMPLETE_NOT_ADMISSION",
        "scientific_admission": False,
        "original_long_ar_gate_passed": False,
        "ppo_executed": False,
        "hydrogym_solver_used": False,
        "project_openfoam_transport_used": True,
        "cycles": STEPS,
        "control_interval": CONTROL_INTERVAL,
        "identity": identity,
        "rows": rows,
        "actual_force_metrics": metrics,
        "action_audit": {
            "max_abs_omega": float(np.max(np.abs(omega))),
            "max_abs_delta_omega": float(np.max(np.abs(delta))),
            "omega_rms": float(np.sqrt(np.mean(np.square(omega)))),
            "sum_squared_endpoint_omega_proxy": float(np.sum(np.square(omega))),
            "integral_omega_squared_linear_ramp_D_over_U": float(
                CONTROL_INTERVAL * np.sum((before**2 + before * omega + omega**2) / 3.0)
            ),
            "not_motor_energy": True,
        },
        "physical_duration_D_over_U": STEPS * CONTROL_INTERVAL,
        "scope": (
            "one paired ten-cycle exploratory real-CFD trial; not robust control proof, "
            "not surrogate admission, and not a replacement for real-CFD PPO"
        ),
    }
