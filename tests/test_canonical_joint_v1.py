import numpy as np
import pytest

from fluid_control.canonical_joint_v1 import (
    ACTION_LIMIT,
    MAX_DELTA_OMEGA,
    apply_action_rate_limit,
    canonical_force_ledger,
    canonical_joint_cost_components,
    causal_window_ledger,
    evaluate_canonical_episode,
    validate_full40_action_contract,
)

BASELINE = {
    "total_drag": 2.0,
    "rear_cl_fluctuation_rms": 0.4,
    "source": "same_phase_zero_fixture",
}


def test_full40_action_scale_is_0p75_and_old_v4_scale_is_rejected():
    result = validate_full40_action_contract(
        {"profile": "matched_start_full40_v1", "max_abs_omega": 0.75}
    )
    assert result["model_action_scale"] == 0.75
    assert result["policy_action_limit"] == 0.75
    assert result["max_delta_omega"] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="exactly 0.75"):
        validate_full40_action_contract(
            {"profile": "matched_start_full40_v1", "max_abs_omega": 5.0}
        )


def test_action_limiter_separates_action_bound_and_physical_rate():
    result = apply_action_rate_limit(2.0, 0.0)
    assert result["bounded_requested_omega"] == ACTION_LIMIT
    assert result["applied_omega"] == MAX_DELTA_OMEGA
    assert result["action_clipped"] is True
    assert result["rate_limited"] is True
    assert result["applied_abs_rate"] == pytest.approx(1.0)


def test_ledger_separates_rear_mean_bias_from_fluctuation():
    phase = np.linspace(0.0, 4.0 * np.pi, 101)
    forces = np.zeros((len(phase), 4))
    forces[:, 0] = 0.9
    forces[:, 2] = 1.0
    forces[:, 3] = 0.3 + 0.4 * np.sqrt(2.0) * np.sin(phase)
    ledger = canonical_force_ledger(forces, BASELINE, window_ready=True)
    assert ledger["drag_gate_pass"] is True
    assert ledger["rear_cl_fluctuation_ratio"] == pytest.approx(1.0, rel=2e-2)
    assert ledger["rear_cl_fluctuation_gate_pass"] is True
    assert ledger["abs_mean_rear_cl_over_baseline_clprime_rms"] == pytest.approx(0.75)
    assert ledger["rear_cl_mean_bias_gate_pass"] is False
    assert ledger["canonical_joint_gate_pass"] is False
    terms = canonical_joint_cost_components(
        ledger, omega=0.5, delta_omega=0.1
    )
    assert terms["rear_cl_mean_bias_gate_violation"] > 0.0


def test_constant_0p75_like_drag_gain_cannot_be_mislabeled_joint_pass():
    forces = np.zeros((101, 4))
    forces[:, 0] = 0.9
    forces[:, 2] = 1.0
    forces[:, 3] = 0.25
    ledger = canonical_force_ledger(forces, BASELINE, window_ready=True)
    assert ledger["drag_gate_pass"] is True
    assert ledger["rear_cl_fluctuation_gate_pass"] is True
    assert ledger["rear_cl_mean_bias_gate_pass"] is False
    assert ledger["canonical_joint_gate_pass"] is False
    terms = canonical_joint_cost_components(
        ledger, omega=0.75, delta_omega=0.1
    )
    assert terms["rear_cl_mean_bias_gate_violation"] > 0.0


def test_causal_window_ignores_older_force_samples():
    times = np.arange(0.0, 10.1, 0.1)
    forces = np.zeros((len(times), 4))
    forces[:, 0] = 10.0
    forces[:, 2] = 10.0
    forces[:, 3] = np.sin(times)
    forces[times >= 4.0, 0] = 0.9
    forces[times >= 4.0, 2] = 1.0
    ledger = causal_window_ledger(times, forces, BASELINE, window_seconds=6.0)
    assert ledger["window_ready"] is True
    assert ledger["window_start"] == pytest.approx(4.0)
    assert ledger["mean_total_drag"] == pytest.approx(1.9)


def _episode_rows(count=100):
    rows = []
    previous = 0.0
    for index in range(count):
        applied = min(0.5, previous + 0.1)
        rows.append(
            {
                "time": 0.1 * (index + 1),
                "predicted_front_cd": 0.9,
                "predicted_front_cl": 0.0,
                "predicted_rear_cd": 1.0,
                "predicted_rear_cl": 0.4 * np.sin(index / 5),
                "applied_omega": applied,
                "applied_delta_omega": applied - previous,
            }
        )
        previous = applied
    return rows


def test_four_force_episode_evaluator_requires_100_steps_and_rate_contract():
    with pytest.raises(ValueError, match="at least 100"):
        evaluate_canonical_episode(_episode_rows(99), BASELINE, initial_omega=0.0)
    result = evaluate_canonical_episode(
        _episode_rows(), BASELINE, initial_omega=0.0
    )
    assert result["steps"] == 100
    assert result["max_abs_action_rate"] <= 1.0 + 1e-8
    assert result["ledger"]["window_ready"] is True
    assert "surrogate-only" in result["scientific_scope"]


def test_episode_evaluator_rejects_legacy_two_force_keys():
    rows = _episode_rows()
    rows[0].pop("predicted_front_cd")
    rows[0]["predicted_cd"] = 1.9
    with pytest.raises(ValueError, match="four-force"):
        evaluate_canonical_episode(rows, BASELINE, initial_omega=0.0)
