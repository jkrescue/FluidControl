"""Canonical train/validation control objective for the full40 surrogate.

This module is isolated from the historical ``legacy_rear`` and
``stage_c_total_drag`` objectives.  Its scalar cost is a surrogate-screening
signal; only the separately reported three-part gate can claim joint passage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

ACTION_LIMIT = 0.75
MODEL_ACTION_SCALE = 0.75
CONTROL_DT = 0.1
MAX_ABS_ACTION_RATE = 1.0
MAX_DELTA_OMEGA = CONTROL_DT * MAX_ABS_ACTION_RATE
MIN_EPISODE_STEPS = 100
DEFAULT_WINDOW_SECONDS = 6.15
DRAG_REDUCTION_THRESHOLD = 0.02
REAR_CL_FLUCTUATION_RATIO_LIMIT = 1.05
REAR_CL_MEAN_BIAS_RATIO_LIMIT = 0.10
BASELINE_KEYS = ("total_drag", "rear_cl_fluctuation_rms", "source")
FORCE_KEYS = (
    "predicted_front_cd",
    "predicted_front_cl",
    "predicted_rear_cd",
    "predicted_rear_cl",
)


def validate_full40_action_contract(
    manifest: Mapping[str, object], *, dt: float = CONTROL_DT
) -> dict[str, float]:
    """Bind the new model and policy to the full40 ±0.75 action contract."""
    profile = manifest.get("profile")
    scale = float(manifest.get("max_abs_omega", np.nan))
    if profile != "matched_start_full40_v1":
        raise ValueError("canonical_joint_v1 requires matched_start_full40_v1")
    if not np.isclose(scale, ACTION_LIMIT, rtol=0.0, atol=1e-12):
        raise ValueError("full40 max_abs_omega must be exactly 0.75")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("control dt must be positive and finite")
    delta = float(dt) * MAX_ABS_ACTION_RATE
    return {
        "model_action_scale": scale,
        "policy_action_limit": ACTION_LIMIT,
        "control_dt": float(dt),
        "max_abs_action_rate": MAX_ABS_ACTION_RATE,
        "max_delta_omega": delta,
    }


def validate_baseline(
    baseline: Mapping[str, float | str],
) -> dict[str, float | str]:
    missing = [key for key in BASELINE_KEYS if key not in baseline]
    if missing:
        raise ValueError(f"canonical baseline missing keys: {missing}")
    source = str(baseline["source"])
    if not source:
        raise ValueError("canonical baseline source must be nonempty")
    result: dict[str, float | str] = {"source": source}
    for key in BASELINE_KEYS[:2]:
        value = float(baseline[key])
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"canonical baseline {key} must be positive and finite")
        result[key] = value
    return result


def apply_action_rate_limit(
    requested_omega: float,
    previous_omega: float,
    *,
    action_limit: float = ACTION_LIMIT,
    dt: float = CONTROL_DT,
    max_abs_rate: float = MAX_ABS_ACTION_RATE,
) -> dict[str, float | bool]:
    """Apply both the policy action bound and the physical slew-rate bound."""
    values = (requested_omega, previous_omega, action_limit, dt, max_abs_rate)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("action-rate inputs must be finite")
    if action_limit <= 0.0 or dt <= 0.0 or max_abs_rate <= 0.0:
        raise ValueError("action-rate limits must be positive")
    bounded_request = float(np.clip(requested_omega, -action_limit, action_limit))
    max_delta = dt * max_abs_rate
    applied = float(
        np.clip(bounded_request, previous_omega - max_delta, previous_omega + max_delta)
    )
    applied = float(np.clip(applied, -action_limit, action_limit))
    delta = applied - previous_omega
    return {
        "requested_omega": float(requested_omega),
        "bounded_requested_omega": bounded_request,
        "applied_omega": applied,
        "applied_delta_omega": delta,
        "applied_abs_rate": abs(delta) / dt,
        "action_clipped": not np.isclose(bounded_request, requested_omega),
        "rate_limited": not np.isclose(applied, bounded_request),
    }


def canonical_force_ledger(
    forces: np.ndarray,
    baseline: Mapping[str, float | str],
    *,
    window_ready: bool,
) -> dict[str, float | bool | str]:
    checked = validate_baseline(baseline)
    values = np.asarray(forces, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4 or values.shape[0] < 1:
        raise ValueError("forces must have shape (samples, 4)")
    if not np.isfinite(values).all():
        raise FloatingPointError("forces contain non-finite values")
    total_drag = float(np.mean(values[:, 0] + values[:, 2]))
    rear_cl = values[:, 3]
    rear_cl_mean = float(np.mean(rear_cl))
    rear_cl_fluctuation = float(
        np.sqrt(np.mean(np.square(rear_cl - rear_cl_mean)))
    )
    baseline_drag = float(checked["total_drag"])
    baseline_fluctuation = float(checked["rear_cl_fluctuation_rms"])
    drag_reduction = 1.0 - total_drag / baseline_drag
    fluctuation_ratio = rear_cl_fluctuation / baseline_fluctuation
    mean_bias_ratio = abs(rear_cl_mean) / baseline_fluctuation
    drag_pass = drag_reduction >= DRAG_REDUCTION_THRESHOLD
    fluctuation_pass = fluctuation_ratio <= REAR_CL_FLUCTUATION_RATIO_LIMIT
    mean_bias_pass = mean_bias_ratio <= REAR_CL_MEAN_BIAS_RATIO_LIMIT
    return {
        "baseline_source": str(checked["source"]),
        "window_ready": bool(window_ready),
        "sample_count": len(values),
        "mean_total_drag": total_drag,
        "mean_rear_cl": rear_cl_mean,
        "rear_cl_fluctuation_rms": rear_cl_fluctuation,
        "baseline_total_drag": baseline_drag,
        "baseline_rear_cl_fluctuation_rms": baseline_fluctuation,
        "total_drag_reduction": drag_reduction,
        "rear_cl_fluctuation_ratio": fluctuation_ratio,
        "abs_mean_rear_cl_over_baseline_clprime_rms": mean_bias_ratio,
        "drag_gate_pass": bool(window_ready and drag_pass),
        "rear_cl_fluctuation_gate_pass": bool(window_ready and fluctuation_pass),
        "rear_cl_mean_bias_gate_pass": bool(window_ready and mean_bias_pass),
        "canonical_joint_gate_pass": bool(
            window_ready and drag_pass and fluctuation_pass and mean_bias_pass
        ),
    }


def causal_window_ledger(
    times: np.ndarray,
    forces: np.ndarray,
    baseline: Mapping[str, float | str],
    *,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
) -> dict[str, float | int | bool | str]:
    """Use only the trailing causal force window ending at the latest sample."""
    time = np.asarray(times, dtype=np.float64).reshape(-1)
    values = np.asarray(forces, dtype=np.float64)
    if len(time) != len(values) or len(time) < 1:
        raise ValueError("time and force sample counts must match and be nonempty")
    if not np.isfinite(time).all() or np.any(np.diff(time) <= 0.0):
        raise ValueError("times must be finite and strictly increasing")
    if not np.isfinite(window_seconds) or window_seconds <= 0.0:
        raise ValueError("window_seconds must be positive and finite")
    begin = time[-1] - window_seconds
    selected = time >= begin - 1e-10
    selected_time = time[selected]
    coverage = float(selected_time[-1] - selected_time[0])
    sample_interval = (
        float(np.median(np.diff(selected_time))) if len(selected_time) > 1 else 0.0
    )
    # Point samples represent intervals on the causal grid.  For a 6.15-D/U
    # target and dt=0.1, 62 samples span 6.1 by timestamp but cover 6.2 D/U.
    ready = coverage + sample_interval >= window_seconds - 1e-8
    ledger = canonical_force_ledger(values[selected], baseline, window_ready=ready)
    return {
        **ledger,
        "window_seconds": float(window_seconds),
        "window_start": float(selected_time[0]),
        "window_end": float(selected_time[-1]),
        "window_coverage": coverage,
        "sample_interval": sample_interval,
    }


def canonical_joint_cost_components(
    ledger: Mapping[str, float | bool | str],
    *,
    omega: float,
    delta_omega: float,
    action_scale: float = MODEL_ACTION_SCALE,
    max_delta_omega: float = MAX_DELTA_OMEGA,
) -> dict[str, float]:
    """Return threshold-normalized feasibility costs plus action regularizers."""
    if not np.isclose(action_scale, MODEL_ACTION_SCALE, rtol=0.0, atol=1e-12):
        raise ValueError("canonical_joint_v1 action_scale must be 0.75")
    if not np.isclose(max_delta_omega, MAX_DELTA_OMEGA, rtol=0.0, atol=1e-12):
        raise ValueError("canonical_joint_v1 max_delta_omega must be 0.1")
    if not np.isfinite(omega) or not np.isfinite(delta_omega):
        raise ValueError("actions must be finite")
    if abs(omega) > ACTION_LIMIT + 1e-12 or abs(delta_omega) > MAX_DELTA_OMEGA + 1e-12:
        raise ValueError("applied action violates canonical action/rate contract")
    ready = bool(ledger["window_ready"])
    if ready:
        drag = float(ledger["total_drag_reduction"])
        fluctuation = float(ledger["rear_cl_fluctuation_ratio"])
        mean_bias = float(ledger["abs_mean_rear_cl_over_baseline_clprime_rms"])
        drag_gate = max(
            0.0, (DRAG_REDUCTION_THRESHOLD - drag) / DRAG_REDUCTION_THRESHOLD
        ) ** 2
        fluctuation_gate = max(
            0.0,
            (fluctuation - REAR_CL_FLUCTUATION_RATIO_LIMIT)
            / (REAR_CL_FLUCTUATION_RATIO_LIMIT - 1.0),
        ) ** 2
        mean_bias_gate = max(
            0.0,
            (mean_bias - REAR_CL_MEAN_BIAS_RATIO_LIMIT)
            / REAR_CL_MEAN_BIAS_RATIO_LIMIT,
        ) ** 2
        drag_cost = -float(np.clip(drag, -1.0, 1.0))
    else:
        drag_gate = fluctuation_gate = mean_bias_gate = drag_cost = 0.0
    return {
        "drag_screen": drag_cost,
        "drag_gate_violation": drag_gate,
        "rear_cl_fluctuation_gate_violation": fluctuation_gate,
        "rear_cl_mean_bias_gate_violation": mean_bias_gate,
        "actuation": 0.01 * (omega / action_scale) ** 2,
        "rate": 0.01 * (delta_omega / max_delta_omega) ** 2,
    }


def evaluate_canonical_episode(
    rows: Sequence[Mapping[str, float | int | bool | str]],
    baseline: Mapping[str, float | str],
    *,
    initial_omega: float,
    dt: float = CONTROL_DT,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    minimum_steps: int = MIN_EPISODE_STEPS,
) -> dict:
    """Evaluate one four-force surrogate episode without touching any dataset split."""
    if len(rows) < minimum_steps:
        raise ValueError(f"canonical episode requires at least {minimum_steps} steps")
    times = []
    forces = []
    actions = []
    previous = float(initial_omega)
    max_rate = 0.0
    for index, row in enumerate(rows):
        try:
            time = float(row["time"])
            force = [float(row[key]) for key in FORCE_KEYS]
            applied = float(row["applied_omega"])
            reported_delta = float(row["applied_delta_omega"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid four-force episode row {index}") from error
        actual_delta = applied - previous
        if not np.isclose(actual_delta, reported_delta, rtol=0.0, atol=1e-8):
            raise ValueError(f"applied delta mismatch at row {index}")
        rate = abs(actual_delta) / dt
        if abs(applied) > ACTION_LIMIT + 1e-8 or rate > MAX_ABS_ACTION_RATE + 1e-8:
            raise ValueError(f"action/rate contract violated at row {index}")
        times.append(time)
        forces.append(force)
        actions.append(applied)
        max_rate = max(max_rate, rate)
        previous = applied
    ledger = causal_window_ledger(
        np.asarray(times), np.asarray(forces), baseline, window_seconds=window_seconds
    )
    action = np.asarray(actions)
    return {
        "status": "CANONICAL_JOINT_V1_SURROGATE_EPISODE_EVALUATED",
        "steps": len(rows),
        "action_limit": ACTION_LIMIT,
        "max_abs_action": float(np.max(np.abs(action))),
        "max_abs_action_rate": max_rate,
        "action_rms": float(np.sqrt(np.mean(np.square(action)))),
        "ledger": ledger,
        "canonical_joint_gate_pass": ledger["canonical_joint_gate_pass"],
        "scientific_scope": (
            "surrogate-only canonical-aligned screen; not real-CFD or closed-loop proof"
        ),
    }
