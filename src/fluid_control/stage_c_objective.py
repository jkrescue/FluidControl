"""Single source of truth for the Stage-C tandem-cylinder objective."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


BASELINE_KEYS = ("total_drag", "front_lift_rms", "rear_lift_rms", "source")


def validate_stage_c_baseline(
    baseline: Mapping[str, float | str] | None,
) -> dict[str, float | str] | None:
    """Validate an audited phase-matched zero-action baseline."""
    if baseline is None:
        return None
    missing = [key for key in BASELINE_KEYS if key not in baseline]
    if missing:
        raise ValueError(f"phase_baseline missing keys: {missing}")
    result: dict[str, float | str] = {"source": str(baseline["source"])}
    if not result["source"]:
        raise ValueError("phase_baseline source must be nonempty")
    for key in BASELINE_KEYS[:3]:
        value = float(baseline[key])
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"phase_baseline {key} must be positive and finite")
        result[key] = value
    return result


def stage_c_force_ledger(
    forces: np.ndarray,
    baseline: Mapping[str, float | str],
) -> dict[str, float | str]:
    """Aggregate four-force predictions over a causal reward window."""
    checked = validate_stage_c_baseline(baseline)
    if checked is None:
        raise ValueError("stage_c_force_ledger requires a baseline")
    values = np.asarray(forces, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4 or values.shape[0] < 1:
        raise ValueError("forces must have shape (samples, 4)")
    if not np.isfinite(values).all():
        raise FloatingPointError("forces contain non-finite values")

    total_drag = float(np.mean(values[:, 0] + values[:, 2]))
    front_lift_rms = float(np.sqrt(np.mean(np.square(values[:, 1]))))
    rear_lift_rms = float(np.sqrt(np.mean(np.square(values[:, 3]))))
    baseline_drag = float(checked["total_drag"])
    baseline_front = float(checked["front_lift_rms"])
    baseline_rear = float(checked["rear_lift_rms"])
    return {
        "baseline_source": str(checked["source"]),
        "total_drag": total_drag,
        "front_lift_rms": front_lift_rms,
        "rear_lift_rms": rear_lift_rms,
        "baseline_total_drag": baseline_drag,
        "baseline_front_lift_rms": baseline_front,
        "baseline_rear_lift_rms": baseline_rear,
        "drag_improvement": 1.0 - total_drag / baseline_drag,
        "front_lift_ratio": front_lift_rms / baseline_front,
        "rear_lift_ratio": rear_lift_rms / baseline_rear,
    }


def stage_c_cost_components(
    ledger: Mapping[str, float | int | bool | str],
    *,
    omega: float,
    delta_omega: float,
    action_scale: float,
    max_delta_omega: float,
) -> dict[str, float]:
    """Return the exact cost terms shared by HydroGym and CEM-MPC."""
    if not np.isfinite(action_scale) or action_scale <= 0.0:
        raise ValueError("action_scale must be positive and finite")
    if not np.isfinite(max_delta_omega) or max_delta_omega <= 0.0:
        raise ValueError("max_delta_omega must be positive and finite")
    if not np.isfinite(omega) or not np.isfinite(delta_omega):
        raise ValueError("actions must be finite")
    if bool(ledger["window_ready"]):
        drag_cost = -float(np.clip(float(ledger["drag_improvement"]), -1.0, 1.0))
        rear_lift_cost = 0.10 * max(0.0, float(ledger["rear_lift_ratio"]) - 1.0) ** 2
        front_lift_cost = 0.05 * max(0.0, float(ledger["front_lift_ratio"]) - 1.0) ** 2
    else:
        drag_cost = rear_lift_cost = front_lift_cost = 0.0
    return {
        "total_drag": drag_cost,
        "rear_lift_excess": rear_lift_cost,
        "front_lift_excess": front_lift_cost,
        "actuation": 0.01 * (omega / action_scale) ** 2,
        "rate": 0.01 * (delta_omega / max_delta_omega) ** 2,
    }


def stage_c_sequence_costs(
    forces: np.ndarray,
    actions: np.ndarray,
    *,
    current_omega: float,
    baseline: Mapping[str, float | str],
    action_scale: float,
    max_delta_omega: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Vectorize the locked full-window objective for CEM action sequences.

    ``forces`` must follow ``front_cd, front_cl, rear_cd, rear_cl`` and have
    shape ``(population, horizon, 4)``. The physical terms are computed once
    over the complete planning window; actuation and slew penalties are means
    over that same window.
    """
    checked = validate_stage_c_baseline(baseline)
    if checked is None:
        raise ValueError("stage_c_sequence_costs requires a baseline")
    predicted = np.asarray(forces, dtype=np.float64)
    control = np.asarray(actions, dtype=np.float64)
    if predicted.ndim != 3 or predicted.shape[2] != 4:
        raise ValueError("forces must have shape (population, horizon, 4)")
    if control.shape != predicted.shape[:2] or control.shape[1] < 1:
        raise ValueError("actions must match the force population and horizon")
    if not np.isfinite(predicted).all() or not np.isfinite(control).all():
        raise FloatingPointError("sequence objective inputs must be finite")
    if not np.isfinite(current_omega):
        raise ValueError("current_omega must be finite")
    if not np.isfinite(action_scale) or action_scale <= 0.0:
        raise ValueError("action_scale must be positive and finite")
    if not np.isfinite(max_delta_omega) or max_delta_omega <= 0.0:
        raise ValueError("max_delta_omega must be positive and finite")

    total_drag = np.mean(predicted[:, :, 0] + predicted[:, :, 2], axis=1)
    front_lift_rms = np.sqrt(np.mean(np.square(predicted[:, :, 1]), axis=1))
    rear_lift_rms = np.sqrt(np.mean(np.square(predicted[:, :, 3]), axis=1))
    drag_improvement = 1.0 - total_drag / float(checked["total_drag"])
    front_lift_ratio = front_lift_rms / float(checked["front_lift_rms"])
    rear_lift_ratio = rear_lift_rms / float(checked["rear_lift_rms"])
    deltas = np.diff(
        np.concatenate(
            (
                np.full((control.shape[0], 1), float(current_omega)),
                control,
            ),
            axis=1,
        ),
        axis=1,
    )
    components = {
        "total_drag": -np.clip(drag_improvement, -1.0, 1.0),
        "rear_lift_excess": 0.10 * np.square(np.maximum(rear_lift_ratio - 1.0, 0.0)),
        "front_lift_excess": 0.05 * np.square(np.maximum(front_lift_ratio - 1.0, 0.0)),
        "actuation": 0.01 * np.mean(np.square(control / action_scale), axis=1),
        "rate": 0.01 * np.mean(np.square(deltas / max_delta_omega), axis=1),
    }
    total = np.sum(np.stack(tuple(components.values()), axis=1), axis=1)
    return total, components
