"""Explicit H5 extension of the reviewed causal-history H2 selector.

This is project glue around the same frozen official K1 models and canonical
force ledger.  Horizon two delegates to the reviewed implementation exactly;
horizon five is the only new profile prepared here.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch

from fluid_control import exploratory_causal_history_mpc as h2_history
from fluid_control import exploratory_short_mpc as h2_rollout


H5_MODE = "canonical_causal_history_h5_v1"
H5_STATUS = "EXPLORATORY_CANONICAL_HISTORY_H5_SELECTION_NOT_ADMISSION"
H5 = 5


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def five_held_sequences(current_omega: float, *, horizon: int) -> np.ndarray:
    """Return the same five held candidates for the explicit H2 or H5 profile."""
    _require(type(horizon) is int and horizon in (2, H5), "only reviewed H2/H5 supported")
    if horizon == 2:
        return h2_rollout.five_hold_sequences(current_omega)
    current = float(current_omega)
    _require(math.isfinite(current) and abs(current) <= h2_rollout.ACTION_SCALE,
             "current action outside exact bound")
    following = np.clip(
        current + np.asarray(h2_rollout.INCREMENTS),
        -h2_rollout.ACTION_SCALE, h2_rollout.ACTION_SCALE,
    )
    _require(bool(np.all(np.abs(following - current) <= h2_rollout.MAX_DELTA + 1e-12)),
             "candidate rate bound differs")
    return np.repeat(following[:, None], horizon, axis=1)


def rollout_five_held_horizon(
    flow,
    aerodynamic,
    current_state: torch.Tensor,
    mask: torch.Tensor,
    current_omega: float,
    force_mean: torch.Tensor,
    force_std: torch.Tensor,
    *,
    build_input: Callable,
    state_abs_limit: float,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Stream five frozen-model rolls; retain only forces and maximum state bounds."""
    if horizon == 2:
        return h2_rollout.rollout_five_h2(
            flow, aerodynamic, current_state, mask, current_omega,
            force_mean, force_std, build_input=build_input,
            state_abs_limit=state_abs_limit,
        )
    _require(horizon == H5, "only reviewed H2/H5 supported")
    _require(current_state.ndim == 4 and current_state.shape[:2] == (1, 3),
             "one normalized current field required")
    _require(mask.shape == (1, 1, *current_state.shape[-2:]), "mask shape differs")
    _require(current_state.dtype == mask.dtype and current_state.device == mask.device,
             "state/mask device or dtype differs")
    _require(force_mean.shape == force_std.shape == (4,), "four-force normalization required")
    _require(force_mean.device == force_std.device == current_state.device,
             "force normalization device differs")
    _require(not current_state.requires_grad and not mask.requires_grad,
             "observed current field must not carry gradients")
    _require(bool(torch.isfinite(current_state).all() and torch.isfinite(mask).all()
                  and torch.isfinite(force_mean).all() and torch.isfinite(force_std).all()),
             "nonfinite current inference input")
    _require(bool(((mask == 0) | (mask == 1)).all() and mask.any()),
             "binary nonempty mask required")
    _require(bool((current_state * (1 - mask) == 0).all()),
             "current field must be masked")
    _require(type(state_abs_limit) in (int, float) and math.isfinite(state_abs_limit)
             and state_abs_limit > 0, "positive finite state bound required")
    for model in (flow, aerodynamic):
        _require(not model.training and all(not parameter.requires_grad and parameter.grad is None
                                            for parameter in model.parameters()),
                 "official models must be frozen eval")

    forces, bounds = [], []
    with torch.inference_mode():
        for sequence in five_held_sequences(current_omega, horizon=horizon):
            state = current_state.clone()
            previous = float(current_omega)
            rows = []
            maximum = float(state.abs().amax().cpu())
            for following in sequence:
                packed = h2_rollout._packed(
                    state, mask, previous, float(following), build_input)
                flow_raw = flow(packed)
                aero_raw = aerodynamic(packed)
                _require(flow_raw.shape == (1, 7, *state.shape[-2:]),
                         "official flow raw seven-channel output required")
                if not bool(torch.isfinite(flow_raw).all()):
                    raise FloatingPointError("nonfinite flow output")
                rows.append(h2_rollout._physical_force(
                    aero_raw, mask, force_mean, force_std).cpu().numpy())
                state = (state + flow_raw[:, :3]) * mask
                if not bool(torch.isfinite(state).all()):
                    raise FloatingPointError("nonfinite predicted state")
                maximum = max(maximum, float(state.abs().amax().cpu()))
                previous = float(following)
            forces.append(np.stack(rows))
            bounds.append(maximum)
    return np.stack(forces), np.asarray(bounds, dtype=np.float64)


def score_five_candidates_horizon(
    predicted_forces,
    actions,
    history: dict,
    *,
    current_omega: float,
    baseline: dict,
    canonical_force_ledger: Callable,
    canonical_joint_cost_components: Callable,
    horizon: int,
) -> dict:
    """Average unchanged canonical components over H5; H2 delegates exactly."""
    if horizon == 2:
        return h2_history.score_five_candidates(
            predicted_forces, actions, history, current_omega=current_omega,
            baseline=baseline, canonical_force_ledger=canonical_force_ledger,
            canonical_joint_cost_components=canonical_joint_cost_components,
        )
    _require(horizon == H5, "only reviewed H2/H5 supported")
    _, persistent = h2_history.validate_history(
        history["times"], history["forces"], expected_end=float(history["times"][-1]))
    predictions = np.asarray(predicted_forces, dtype=np.float64)
    commands = np.asarray(actions, dtype=np.float64)
    _require(predictions.shape == (h2_history.CANDIDATES, horizon, 4)
             and commands.shape == (h2_history.CANDIDATES, horizon),
             "fixed five-by-H5 candidate contract required")
    _require(np.isfinite(predictions).all() and np.isfinite(commands).all(),
             "finite candidate data required")
    expected = five_held_sequences(current_omega, horizon=horizon)
    _require(np.array_equal(commands, expected),
             "exact fixed five held-action candidates required")
    names = (
        "drag_screen", "drag_gate_violation", "rear_cl_fluctuation_gate_violation",
        "rear_cl_mean_bias_gate_violation", "actuation", "rate",
    )
    totals, reports = [], []
    for candidate in range(h2_history.CANDIDATES):
        local = persistent.copy()
        previous = float(current_omega)
        stages = []
        for step in range(horizon):
            local = np.concatenate((local[1:], predictions[candidate, step][None]), axis=0)
            ledger = canonical_force_ledger(local, baseline, window_ready=True)
            omega = float(commands[candidate, step])
            components = canonical_joint_cost_components(
                ledger, omega=omega, delta_omega=omega - previous)
            _require(tuple(components) == names
                     and all(math.isfinite(float(value)) for value in components.values()),
                     "canonical component contract differs")
            stages.append({"ledger": ledger, "components": components})
            previous = omega
        mean_components = {
            name: float(np.mean([stage["components"][name] for stage in stages]))
            for name in names
        }
        totals.append(float(sum(mean_components.values())))
        reports.append({"stages": stages, "mean_components": mean_components})
    _require(np.array_equal(persistent, history["forces"]), "predictions mutated real history")
    return {
        "mode": H5_MODE,
        "status": H5_STATUS,
        "h5_cost": totals,
        "candidate_reports": reports,
        "persistent_history_mutated": False,
        "scientific_admission": False,
    }


def select_canonical_history_horizon(
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
    horizon: int,
) -> dict:
    """Select the first action; retain exact H2 behavior through direct delegation."""
    if horizon == 2:
        return h2_history.select_canonical_history_action(
            predicted_forces, state_bounds, actions, history,
            current_omega=current_omega, state_abs_limit=state_abs_limit,
            baseline=baseline, canonical_force_ledger=canonical_force_ledger,
            canonical_joint_cost_components=canonical_joint_cost_components,
        )
    report = score_five_candidates_horizon(
        predicted_forces, actions, history, current_omega=current_omega,
        baseline=baseline, canonical_force_ledger=canonical_force_ledger,
        canonical_joint_cost_components=canonical_joint_cost_components,
        horizon=horizon,
    )
    bounds = np.asarray(state_bounds, dtype=np.float64)
    commands = np.asarray(actions, dtype=np.float64)
    _require(bounds.shape == (h2_history.CANDIDATES,) and np.isfinite(bounds).all(),
             "five finite state bounds required")
    feasible = bounds <= float(state_abs_limit)
    ranked = np.where(feasible, report["h5_cost"], np.inf)
    _require(np.isfinite(ranked).any(), "no feasible canonical-history candidate")
    order = np.lexsort((np.arange(h2_history.CANDIDATES),
                        np.abs(commands[:, 0] - current_omega), ranked))
    selected = int(order[0])
    return {
        **report,
        "selected_index": selected,
        "selected_action": float(commands[selected, 0]),
        "actions": commands.tolist(),
        "predicted_forces_h5": np.asarray(predicted_forces, dtype=np.float64).tolist(),
        "state_bounds": bounds.tolist(),
        "execute_only_first_action": True,
        "state_feasible": feasible.tolist(),
        "original_long_ar_gate_passed": False,
    }
