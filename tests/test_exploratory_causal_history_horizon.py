import copy
import importlib.util
from pathlib import Path

import numpy as np
import torch

from fluid_control import exploratory_causal_history_mpc as h2_history
from fluid_control import exploratory_short_mpc as h2_rollout
from fluid_control.canonical_joint_v1 import (
    canonical_force_ledger, canonical_joint_cost_components,
)


SOURCE = Path(__file__).parents[1] / "src/fluid_control/exploratory_causal_history_horizon.py"
SPEC = importlib.util.spec_from_file_location("exploratory_causal_history_horizon", SOURCE)
horizon = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(horizon)


def history():
    return {
        "times": 141.9 + .1 * np.arange(62),
        "forces": np.arange(248, dtype=np.float64).reshape(62, 4) / 100,
        "sources": {"forceFront": [], "forceRear": []},
    }


def ledger(values, baseline, *, window_ready):
    return {"value": float(values[:, 0].mean()), "window_ready": window_ready}


def components(row, *, omega, delta_omega):
    return {
        "drag_screen": row["value"],
        "drag_gate_violation": 0.0,
        "rear_cl_fluctuation_gate_violation": 0.0,
        "rear_cl_mean_bias_gate_violation": 0.0,
        "actuation": omega * omega,
        "rate": delta_omega * delta_omega,
    }


def test_h2_actions_and_score_delegate_exactly():
    old_actions = h2_rollout.five_hold_sequences(.1)
    new_actions = horizon.five_held_sequences(.1, horizon=2)
    assert np.array_equal(new_actions, old_actions)
    predictions = np.arange(40, dtype=np.float64).reshape(5, 2, 4) / 100
    old = h2_history.score_five_candidates(
        predictions, old_actions, history(), current_omega=.1, baseline={},
        canonical_force_ledger=ledger, canonical_joint_cost_components=components)
    new = horizon.score_five_candidates_horizon(
        predictions, new_actions, history(), current_omega=.1, baseline={},
        canonical_force_ledger=ledger, canonical_joint_cost_components=components,
        horizon=2)
    assert new == old
    bounds = np.ones(5)
    old_selected = h2_history.select_canonical_history_action(
        predictions, bounds, old_actions, history(), current_omega=.1,
        state_abs_limit=2.0, baseline={}, canonical_force_ledger=ledger,
        canonical_joint_cost_components=components)
    new_selected = horizon.select_canonical_history_horizon(
        predictions, bounds, new_actions, history(), current_omega=.1,
        state_abs_limit=2.0, baseline={}, canonical_force_ledger=ledger,
        canonical_joint_cost_components=components, horizon=2)
    assert new_selected == old_selected


def test_h5_rolls_history_locally_and_rate_only_first_stage():
    source = history(); before = copy.deepcopy(source)
    actions = horizon.five_held_sequences(0.0, horizon=5)
    predictions = np.arange(100, dtype=np.float64).reshape(5, 5, 4) / 1000
    report = horizon.score_five_candidates_horizon(
        predictions, actions, source, current_omega=0.0, baseline={},
        canonical_force_ledger=ledger, canonical_joint_cost_components=components,
        horizon=5)
    assert report["mode"] == horizon.H5_MODE and len(report["h5_cost"]) == 5
    assert all(len(row["stages"]) == 5 for row in report["candidate_reports"])
    for candidate, row in enumerate(report["candidate_reports"]):
        rates = [stage["components"]["rate"] for stage in row["stages"]]
        assert rates[1:] == [0.0] * 4
        assert rates[0] == actions[candidate, 0] ** 2
    assert np.array_equal(source["times"], before["times"])
    assert np.array_equal(source["forces"], before["forces"])


def test_h5_each_candidate_starts_from_same_history_and_keeps_62_rows():
    seen = []

    def capture(values, baseline, *, window_ready):
        seen.append(values.copy())
        return {"value": float(values[:, 0].mean()), "window_ready": window_ready}

    actions = horizon.five_held_sequences(0.0, horizon=5)
    predictions = np.arange(100, dtype=np.float64).reshape(5, 5, 4) / 1000
    horizon.score_five_candidates_horizon(
        predictions, actions, history(), current_omega=0.0, baseline={},
        canonical_force_ledger=capture, canonical_joint_cost_components=components,
        horizon=5)
    assert len(seen) == 25 and all(row.shape == (62, 4) for row in seen)
    original = history()["forces"]
    assert np.array_equal(seen[0][:-1], original[1:])
    assert np.array_equal(seen[4][:-5], original[5:])
    assert np.array_equal(seen[5][:-1], original[1:])
    assert np.array_equal(seen[4][-5:], predictions[0])
    assert np.array_equal(seen[9][-5:], predictions[1])


def test_h5_calls_actual_canonical_components_for_all_25_stages():
    actions = horizon.five_held_sequences(0.0, horizon=5)
    predictions = np.arange(100, dtype=np.float64).reshape(5, 5, 4) / 1000
    report = horizon.score_five_candidates_horizon(
        predictions, actions, history(), current_omega=0.0,
        baseline={"total_drag": 3.0, "rear_cl_fluctuation_rms": 1.0,
                  "source": "synthetic-cpu-fixture"},
        canonical_force_ledger=canonical_force_ledger,
        canonical_joint_cost_components=canonical_joint_cost_components,
        horizon=5)
    stages = [stage for candidate in report["candidate_reports"]
              for stage in candidate["stages"]]
    assert len(stages) == 25
    assert all(stage["ledger"]["sample_count"] == 62 for stage in stages)
    assert all(tuple(stage["components"]) == (
        "drag_screen", "drag_gate_violation", "rear_cl_fluctuation_gate_violation",
        "rear_cl_mean_bias_gate_violation", "actuation", "rate") for stage in stages)


class Toy(torch.nn.Module):
    def __init__(self, force=False):
        super().__init__();self.force=force;self.calls=0

    def forward(self, packed):
        self.calls += 1
        raw = torch.zeros((1, 7, *packed.shape[-2:]), dtype=packed.dtype)
        if self.force:
            raw[:, 3:7] = self.calls / 100
        else:
            raw[:, :3] = .01
        return raw


def build_input(state, mask, now, following):
    now = torch.as_tensor(now).reshape(1)
    following = torch.as_tensor(following).reshape(1)
    return torch.cat((state[0], mask, now[:, None, None].expand(1, *state.shape[-2:]),
                      following[:, None, None].expand(1, *state.shape[-2:])), 0)


def test_h5_rollout_is_full_five_by_five_streaming():
    flow, aero = Toy().eval(), Toy(force=True).eval()
    state = torch.zeros((1, 3, 2, 2));mask = torch.ones((1, 1, 2, 2))
    forces, bounds = horizon.rollout_five_held_horizon(
        flow, aero, state, mask, 0.0, torch.zeros(4), torch.ones(4),
        build_input=build_input, state_abs_limit=12.0, horizon=5)
    assert forces.shape == (5, 5, 4) and bounds.shape == (5,)
    assert flow.calls == aero.calls == 25
    assert np.all(bounds > 0)


def test_h2_rollout_calls_reviewed_function_exactly(monkeypatch):
    sentinel = (np.ones((5, 2, 4)), np.ones(5))
    monkeypatch.setattr(h2_rollout, "rollout_five_h2", lambda *a, **k: sentinel)
    got = horizon.rollout_five_held_horizon(
        object(), object(), torch.zeros((1, 3, 1, 1)), torch.ones((1, 1, 1, 1)),
        0.0, torch.zeros(4), torch.ones(4), build_input=object(),
        state_abs_limit=1.0, horizon=2)
    assert got is sentinel


def test_rejects_unreviewed_horizon():
    for value in (1, 3, 10, True):
        try:
            horizon.five_held_sequences(0.0, horizon=value)
        except ValueError:
            pass
        else:
            raise AssertionError(value)
