"""Synthetic CPU tests only: no official model, HDF, CFD, or GPU access."""

from types import SimpleNamespace
import hashlib
import json

import numpy as np
import pytest
import torch

from fluid_control import exploratory_short_mpc as mpc


class Toy(torch.nn.Module):
    def __init__(self, *, force=False):
        super().__init__()
        self.force = force
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.eval()

    def forward(self, value):
        raw = torch.zeros(value.shape[0], 7, *value.shape[-2:], dtype=value.dtype)
        if self.force:
            # Candidate action plane is last; direct response is observable at H1.
            raw[:, 3] = 1.0 + value[:, 5]
            raw[:, 6] = value[:, 5]
        else:
            raw[:, 0] = 0.01 + 0.02 * value[:, 5]
        return raw


def build_input(states, mask, actions, selected):
    planes = torch.cat((actions.reshape(-1), torch.as_tensor(selected).reshape(1)))
    return torch.cat((states.reshape(-1, *states.shape[-2:]), mask,
                      planes[:, None, None].expand(-1, *states.shape[-2:])), 0)


def test_exact_five_hold_candidates_and_bounds():
    expected = np.array([[-.1, -.1], [-.05, -.05], [0, 0], [.05, .05], [.1, .1]])
    np.testing.assert_array_equal(mpc.five_hold_sequences(0), expected)
    edge = mpc.five_hold_sequences(.75)
    assert np.max(np.abs(edge)) <= .75 and np.max(np.abs(edge - .75)) <= .1


@pytest.mark.parametrize(("now", "nxt"), [(.01, .02), (.1, .05), (-.1, -.05), (.05, .1)])
def test_packed_action_normalization_matches_canonical_float32_order(now, nxt):
    q = torch.zeros(1, 3, 2, 3, dtype=torch.float32)
    mask = torch.ones(1, 1, 2, 3, dtype=torch.float32)
    captured = {}

    def capture(state, current_mask, current, following):
        captured.update(state=state, mask=current_mask, current=current, following=following)
        return torch.zeros(6, 2, 3, dtype=torch.float32)

    mpc._packed(q, mask, now, nxt, capture)
    assert torch.equal(captured["current"], torch.tensor([now], dtype=torch.float32) / .75)
    assert torch.equal(captured["following"], torch.tensor(nxt, dtype=torch.float32) / .75)


def test_wrong_flow_channel_shape_is_rejected():
    class BadFlow(Toy):
        def forward(self, value):
            return torch.zeros(value.shape[0], 3, *value.shape[-2:])

    q = torch.zeros(1, 3, 2, 3)
    mask = torch.ones(1, 1, 2, 3)
    mean = torch.zeros(4)
    std = torch.ones(4)
    with pytest.raises(ValueError, match="flow raw"):
        mpc.rollout_five_h2(BadFlow(), Toy(force=True), q, mask, 0.0, mean, std,
                            build_input=build_input, state_abs_limit=2.0)


def test_h2_rollout_is_causal_and_direct_force_response():
    q = torch.zeros(1, 3, 2, 3)
    mask = torch.ones(1, 1, 2, 3)
    forces, bounds = mpc.rollout_five_h2(
        Toy(), Toy(force=True), q, mask, 0.0, torch.zeros(4), torch.ones(4),
        build_input=build_input, state_abs_limit=2.0,
    )
    assert forces.shape == (5, 2, 4) and bounds.shape == (5,)
    np.testing.assert_allclose(forces[:, 0, 3], np.array(mpc.INCREMENTS) / .75)
    # The second force consumes q1/current action, never a future truth field.
    np.testing.assert_allclose(forces[:, 1, 3], forces[:, 0, 3])


def test_replanning_is_stateless_across_new_cfd_observations():
    flow, aero = Toy(), Toy(force=True)
    mask = torch.ones(1, 1, 2, 3)
    kwargs = dict(
        flow=flow, aerodynamic=aero, mask=mask,
        force_mean=torch.zeros(4), force_std=torch.ones(4),
        build_input=build_input, state_abs_limit=2,
        baseline_total_drag=2, baseline_rear_cl_rms=1,
    )
    first = mpc.plan_from_current_observation(
        current_state=torch.zeros(1, 3, 2, 3), current_omega=0, **kwargs,
    )
    second = mpc.plan_from_current_observation(
        current_state=torch.ones(1, 3, 2, 3) * mask, current_omega=.05, **kwargs,
    )
    again = mpc.plan_from_current_observation(
        current_state=torch.zeros(1, 3, 2, 3), current_omega=0, **kwargs,
    )
    assert first == again
    assert second["actions"] != first["actions"]


def test_cost_identity_and_h2_selection_not_h1_admission():
    actions = mpc.five_hold_sequences(0)
    forces = np.zeros((5, 2, 4))
    forces[:, :, 0] = 1.0
    forces[:, :, 2] = 1.0
    forces[:, :, 3] = np.array([[1, -1], [.5, -.5], [0, 0], [.5, .5], [1, 1]])
    total, parts = mpc.cost_components(
        forces, actions, current_omega=0, baseline_total_drag=2,
        baseline_rear_cl_rms=1,
    )
    np.testing.assert_allclose(
        parts["rear_cl_variance"] + parts["rear_cl_mean_squared"],
        np.mean(forces[:, :, 3] ** 2, axis=1),
    )
    result = mpc.select_exploratory_action(
        forces, np.zeros(5), current_omega=0, state_abs_limit=1,
        baseline_total_drag=2, baseline_rear_cl_rms=1,
    )
    assert result["selected_index"] == 2 and result["selected_action"] == 0
    assert result["execute_only_first_action"] is True
    assert result["original_long_ar_gate_passed"] is False
    assert result["scientific_admission"] is False


def test_tie_break_prefers_hold_and_state_guard_rejects():
    forces = np.zeros((5, 2, 4))
    forces[:, :, (0, 2)] = 1
    result = mpc.select_exploratory_action(
        forces, np.zeros(5), current_omega=0, state_abs_limit=1,
        baseline_total_drag=2, baseline_rear_cl_rms=1,
    )
    assert result["selected_action"] == 0
    with pytest.raises(ValueError, match="no finite"):
        mpc.select_exploratory_action(
            forces, np.ones(5) * 2, current_omega=0, state_abs_limit=1,
            baseline_total_drag=2, baseline_rear_cl_rms=1,
        )


def test_exact_k1_loader_binding_and_wrong_identity_rejected():
    identity = SimpleNamespace(
        payload={"kind": mpc.K1_KIND},
        flow=SimpleNamespace(model_sha256=mpc.K1_FLOW_MODEL_SHA256,
                             state_sha256=mpc.K1_FLOW_STATE_SHA256),
        aerodynamic=SimpleNamespace(model_sha256=mpc.K1_AERO_MODEL_SHA256,
                                    state_sha256=mpc.K1_AERO_STATE_SHA256),
    )
    seen = {}
    def loader(path, cfg, device, **kwargs):
        seen.update(path=path, cfg=cfg, device=device, kwargs=kwargs)
        return SimpleNamespace(flow_model="flow", aerodynamic_model="aero"), identity
    assert mpc.load_bound_k1("manifest", "cfg", "cpu", load_dual_fno=loader,
                             build_model="builder")[:2] == ("flow", "aero")
    assert seen["kwargs"]["expected_manifest_sha256"] == mpc.K1_MANIFEST_SHA256
    identity.payload["kind"] = "FC_P029_CONTROL_AWARE_FLOW_REPAIR"
    with pytest.raises(ValueError, match="K1 kind"):
        mpc.validate_k1_identity(identity)


def test_pinned_baseline_and_failed_formal_receipt(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({
        "status": "FULL40_TRAIN20_OPEN_LOOP_PHYSICS_SUMMARY",
        "phases": {"b00": {"same_phase_zero": {"metrics": {
            "mean_cd_total": 2.3, "rms_cl_rear_fluctuation": 1.2,
        }}}},
    }))
    digest = hashlib.sha256(baseline.read_bytes()).hexdigest()
    monkeypatch.setattr(mpc, "BASELINE_SHA256", digest)
    assert mpc.load_bound_b00_baseline(baseline) == (2.3, 1.2)

    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({
        "status": "FC_P026_ORIGINAL_FORMAL_COMPLETE_NOT_ADMISSION",
        "history_k": 1, "scientific_admission": False,
        "ppo_auto_launched": False,
    }))
    monkeypatch.setattr(
        mpc, "K1_FORMAL_RECEIPT_SHA256",
        hashlib.sha256(receipt.read_bytes()).hexdigest(),
    )
    assert mpc.validate_recorded_k1_failure(receipt)["scientific_admission"] is False
    damaged = json.loads(receipt.read_text())
    damaged["scientific_admission"] = True
    receipt.write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match="formal failure"):
        mpc.validate_recorded_k1_failure(receipt)


@pytest.mark.parametrize("damage", ["training", "gradient", "nonfinite", "shape"])
def test_fail_closed_model_and_input_contracts(damage):
    flow, aero = Toy(), Toy(force=True)
    q, mask = torch.zeros(1, 3, 2, 3), torch.ones(1, 1, 2, 3)
    if damage == "training":
        flow.train()
    elif damage == "gradient":
        next(aero.parameters()).grad = torch.ones(())
    elif damage == "nonfinite":
        q[0, 0, 0, 0] = float("nan")
    else:
        mask = torch.ones(1, 1, 1, 3)
    with pytest.raises((ValueError, FloatingPointError)):
        mpc.rollout_five_h2(
            flow, aero, q, mask, 0, torch.zeros(4), torch.ones(4),
            build_input=build_input, state_abs_limit=2,
        )
