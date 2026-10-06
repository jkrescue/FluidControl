"""Alternative exploratory H2 selector using the existing canonical force ledger.

This explicit mode leaves the accepted instantaneous-H2 implementation intact.
Candidate predictions are temporary.  Only measured real-CFD endpoints may
advance the persistent 62-sample causal history.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import numpy as np


MODE = "canonical_causal_history_h2_v1"
STATUS = "EXPLORATORY_CANONICAL_HISTORY_H2_SELECTION_NOT_ADMISSION"
SAMPLES = 62
DT = 0.1
HORIZON = 2
CANDIDATES = 5
ACTION_LIMIT = 0.75
INCREMENTS = np.asarray((-0.10, -0.05, 0.0, 0.05, 0.10), dtype=np.float64)


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def validate_history(times, forces, *, expected_end: float) -> tuple[np.ndarray, np.ndarray]:
    time = np.asarray(times, dtype=np.float64)
    values = np.asarray(forces, dtype=np.float64)
    _require(time.shape == (SAMPLES,) and values.shape == (SAMPLES, 4),
             "exact 62-by-four causal history required")
    _require(np.isfinite(time).all() and np.isfinite(values).all(), "finite history required")
    expected = expected_end - DT * (SAMPLES - 1) + DT * np.arange(SAMPLES)
    _require(np.allclose(time, expected, rtol=0.0, atol=1e-8),
             "exact causal 0.1 grid ending at current time required")
    return time.copy(), values.copy()


def load_bound_actual_history(
    reference: Path,
    restart_time: float,
    *,
    provenance_root: Path,
    expected_sources: dict,
    actual_causal_prehistory: Callable,
) -> dict:
    """Load the existing raw OpenFOAM history and bind every provenance row."""
    _require(math.isfinite(restart_time), "finite restart time required")
    times, forces, sources = actual_causal_prehistory(
        reference, restart_time, provenance_root=provenance_root,
        control_dt=DT, sample_count=SAMPLES,
    )
    time, values = validate_history(times, forces, expected_end=restart_time)
    _require(sources == expected_sources and set(sources) == {"forceFront", "forceRear"},
             "causal OpenFOAM force provenance differs")
    return {"mode": MODE, "times": time, "forces": values, "sources": sources}


def append_actual_endpoint(history: dict, *, endpoint: float, actual_force) -> dict:
    """Advance persistent history with measured CFD only; never predicted force."""
    times, forces = validate_history(
        history["times"], history["forces"], expected_end=float(history["times"][-1]))
    force = np.asarray(actual_force, dtype=np.float64)
    _require(force.shape == (4,) and np.isfinite(force).all(), "four finite actual forces required")
    _require(math.isclose(endpoint, times[-1] + DT, rel_tol=0.0, abs_tol=1e-8),
             "actual endpoint must be the next control time")
    return {
        "mode": MODE,
        "times": np.concatenate((times[1:], [endpoint])),
        "forces": np.concatenate((forces[1:], force[None]), axis=0),
        "sources": history["sources"],
        "persistent_update_source": "actual_real_cfd_endpoint_only",
    }


def score_five_candidates(
    predicted_forces,
    actions,
    history: dict,
    *,
    current_omega: float,
    baseline: dict,
    canonical_force_ledger: Callable,
    canonical_joint_cost_components: Callable,
) -> dict:
    """Roll each candidate locally and average existing canonical cost over H2."""
    _, persistent = validate_history(
        history["times"], history["forces"], expected_end=float(history["times"][-1]))
    predictions = np.asarray(predicted_forces, dtype=np.float64)
    commands = np.asarray(actions, dtype=np.float64)
    _require(predictions.shape == (CANDIDATES, HORIZON, 4)
             and commands.shape == (CANDIDATES, HORIZON),
             "fixed five-by-H2 candidate contract required")
    _require(np.isfinite(predictions).all() and np.isfinite(commands).all(),
             "finite candidate data required")
    expected_actions = np.repeat(
        np.clip(float(current_omega) + INCREMENTS, -ACTION_LIMIT, ACTION_LIMIT)[:, None],
        HORIZON, axis=1)
    _require(np.array_equal(commands, expected_actions),
             "exact fixed five held-action candidates required")
    component_names = (
        "drag_screen", "drag_gate_violation", "rear_cl_fluctuation_gate_violation",
        "rear_cl_mean_bias_gate_violation", "actuation", "rate",
    )
    totals, reports = [], []
    for candidate in range(CANDIDATES):
        local = persistent.copy()
        previous = float(current_omega)
        stages = []
        for step in range(HORIZON):
            local = np.concatenate((local[1:], predictions[candidate, step][None]), axis=0)
            ledger = canonical_force_ledger(local, baseline, window_ready=True)
            omega = float(commands[candidate, step])
            components = canonical_joint_cost_components(
                ledger, omega=omega, delta_omega=omega - previous)
            _require(tuple(components) == component_names
                     and all(math.isfinite(float(value)) for value in components.values()),
                     "canonical component contract differs")
            stages.append({"ledger": ledger, "components": components})
            previous = omega
        mean_components = {
            name: float(np.mean([stage["components"][name] for stage in stages]))
            for name in component_names
        }
        totals.append(float(sum(mean_components.values())))
        reports.append({"stages": stages, "mean_components": mean_components})
    # Candidate-local prediction rolls must never change persistent real history.
    _require(np.array_equal(persistent, history["forces"]), "predictions mutated real history")
    return {
        "mode": MODE,
        "status": STATUS,
        "h2_cost": totals,
        "candidate_reports": reports,
        "persistent_history_mutated": False,
        "scientific_admission": False,
    }


def select_canonical_history_action(
    predicted_forces,
    state_bounds,
    actions,
    history,
    *,
    current_omega,
    state_abs_limit,
    baseline,
    canonical_force_ledger,
    canonical_joint_cost_components,
) -> dict:
    """Select one feasible first action; exact ties prefer the smallest rate."""
    report = score_five_candidates(
        predicted_forces, actions, history, current_omega=current_omega,
        baseline=baseline, canonical_force_ledger=canonical_force_ledger,
        canonical_joint_cost_components=canonical_joint_cost_components,
    )
    bounds = np.asarray(state_bounds, dtype=np.float64)
    commands = np.asarray(actions, dtype=np.float64)
    _require(bounds.shape == (CANDIDATES,) and np.isfinite(bounds).all(),
             "five finite state bounds required")
    feasible = bounds <= float(state_abs_limit)
    ranked = np.where(feasible, report["h2_cost"], np.inf)
    _require(np.isfinite(ranked).any(), "no feasible canonical-history candidate")
    order = np.lexsort((np.arange(CANDIDATES),
                        np.abs(commands[:, 0] - current_omega), ranked))
    selected = int(order[0])
    return {
        **report,
        "selected_index": selected,
        "selected_action": float(commands[selected, 0]),
        "actions": commands.tolist(),
        "predicted_forces_h2": np.asarray(predicted_forces, dtype=np.float64).tolist(),
        "state_bounds": bounds.tolist(),
        "execute_only_first_action": True,
        "state_feasible": feasible.tolist(),
        "original_long_ar_gate_passed": False,
    }
