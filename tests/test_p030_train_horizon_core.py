"""Synthetic CPU engineering tests only; no CFD/model/HDF evidence."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

MODULE = Path(__file__).resolve().parents[1] / "scripts" / "p030_train_horizon_core.py"
SPEC = importlib.util.spec_from_file_location("p030_train_horizon_core_tested", MODULE)
core = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(core)


def field_error_sums(error, target, mask):
    weight = mask.double()
    return torch.stack([
        (error.double().square() * weight).sum((0, 2, 3)),
        (target.double().square() * weight).sum((0, 2, 3)),
    ])


def relative_field_metrics(sums):
    error, reference = np.asarray(sums, dtype=np.float64)
    ratio = lambda e, r: float(np.sqrt(e / r)) if r > 0 else None
    return {
        "field_squared_error_sums_u_v_p": error.tolist(),
        "field_reference_squared_sums_u_v_p": reference.tolist(),
        "field_relative_l2_u_v_p": [ratio(e, r) for e, r in zip(error, reference)],
        "velocity_relative_l2": ratio(error[:2].sum(), reference[:2].sum()),
    }


def selection():
    rows = []
    families = ["base"] * 20 + ["train8"] * 8 + ["train16"] * 16
    phases = (
        [phase for phase in ("b00", "b02", "b04", "b06") for _ in range(5)]
        + [phase for phase in ("b00", "b02", "b04", "b06") for _ in range(2)]
        + ["b00"] * 8 + ["b02"] * 8
    )
    dataset_ids = {"base": 0, "train8": 1, "train16": 2}
    for index, (family, phase) in enumerate(zip(families, phases, strict=True)):
        rows.append({
            "family": family,
            "dataset_index": dataset_ids[family],
            "case": f"synthetic_{family}_{index:02d}",
            "start": 0,
            "rollout_steps": 100,
            "canonical_phase": phase,
            "action_profile": "synthetic_profile",
            "source_manifest_sha256": "a" * 64,
            "hdf_sha256": f"{index:064x}",
            "split": "train",
            "frames": core.EXPECTED_FRAMES[family],
        })
    membership = {(row["case"], 0, row["dataset_index"]) for row in rows}
    return rows, membership


class Flow(torch.nn.Module):
    def __init__(self, rate):
        super().__init__()
        self.register_buffer("rate", torch.tensor(float(rate)))
        self.calls = 0


class Aero(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0


def make_inputs(current, mask, now, nxt):
    return current, mask, now, nxt


def predict(model, inputs, mask):
    current, _, now, nxt = inputs
    model.calls += 1
    if isinstance(model, Aero):
        mean = (current * mask).sum((1, 2, 3)) / (mask.sum((1, 2, 3)) * 3)
        force = torch.stack((mean, now[:, 0], nxt[:, 0], mean + nxt[:, 0]), dim=1)
        return torch.zeros_like(current), force
    delta = current * model.rate + nxt[:, :, None, None] * mask * 0.01
    return delta, torch.zeros((current.shape[0], 4), dtype=current.dtype)


def sample():
    dtype = torch.float32
    state = torch.full((1, 3, 2, 2), 0.2, dtype=dtype)
    targets = torch.stack(
        [state[0] * (1.03 ** (step + 1)) for step in range(100)], dim=0
    )[None]
    mask = torch.ones((1, 1, 2, 2), dtype=dtype)
    omega = torch.linspace(-0.5, 0.5, 101, dtype=dtype)[None, :, None]
    force = torch.zeros((1, 100, 4), dtype=dtype)
    time = (torch.tensor(148.0, dtype=dtype) + 0.1 * torch.arange(101, dtype=dtype))[None, :, None]
    return dict(state=state, target_state=targets, mask=mask, omega=omega,
                target_force=force, time=time)


def rollout(flow, aero, data=None, counter=None, case="synthetic_case", dataset_index=0):
    def sums(*args):
        if counter is not None:
            counter[0] += 1
        return field_error_sums(*args)

    return core.rollout_window(
        flow.eval(), aero.eval(), sample() if data is None else data,
        predict_fn=predict, make_inputs_fn=make_inputs, field_error_sums_fn=sums,
        state_mean=torch.zeros((3, 1, 1)), state_std=torch.ones((3, 1, 1)),
        force_mean=torch.zeros(4), force_std=torch.tensor([2.0, 3.0, 5.0, 7.0]),
        identity={"case": case, "start": 0, "dataset_index": dataset_index},
    )


def fake_record(error_scale=1.0, reference_scale=1.0, force_shift=0.0,
                case="synthetic_case", dataset_index=0):
    return {
        "identity": {"case": case, "start": 0, "dataset_index": dataset_index},
        "rollout_steps": 100,
        "field_sums_by_lead": [
            [[error_scale * (step + 1) ** 2] * 3, [reference_scale] * 3]
            for step in range(100)
        ],
        "predicted_force_physical_by_lead": [
            [2.0 + force_shift, 0.0, -1.0, float(step + 1) + force_shift]
            for step in range(100)
        ],
        "target_force_physical_by_lead": [[0.0, 0.0, 0.0, float(step + 1)] for step in range(100)],
    }


def test_selection_exact44_counts_membership_and_start0():
    rows, membership = selection()
    result = core.validate_selection(rows, membership)
    assert result["family_counts"] == {"base": 20, "train8": 8, "train16": 16}
    assert result["phase_counts"] == {"b00": 15, "b02": 15, "b04": 7, "b06": 7}
    broken = copy.deepcopy(rows)
    broken[-1]["start"] = 2
    with pytest.raises(ValueError):
        core.validate_selection(broken, membership)
    missing = set(membership)
    missing.remove((rows[0]["case"], 0, rows[0]["dataset_index"]))
    with pytest.raises(ValueError):
        core.validate_selection(rows, missing)
    wrong_phase = copy.deepcopy(rows)
    wrong_phase[28]["canonical_phase"] = "b04"  # train16 must be b00/b02 only
    wrong_phase[10]["canonical_phase"] = "b00"  # preserve global counts, break family×phase
    with pytest.raises(ValueError):
        core.validate_selection(wrong_phase, membership)
    bool_index = copy.deepcopy(rows)
    bool_index[0]["dataset_index"] = False
    with pytest.raises(ValueError):
        core.validate_selection(bool_index, membership)


def test_time_grid_is_whole_grid_two_ulp_not_per_diff():
    times = np.float32(148.0) + np.float32(0.1) * np.arange(101, dtype=np.float32)
    tolerance = core.validate_time_grid(times)
    assert tolerance >= 1e-7
    drift = times.astype(np.float64)
    drift[50:] += tolerance * 2
    with pytest.raises(ValueError):
        core.validate_time_grid(drift)
    nonmonotonic = times.copy()
    nonmonotonic[10] = nonmonotonic[9]
    with pytest.raises(ValueError):
        core.validate_time_grid(nonmonotonic)


def test_one_uninterrupted_rollout_future_truth_poison_and_action_timing():
    left_flow, left_aero = Flow(0.1), Aero()
    counter = [0]
    baseline = rollout(left_flow, left_aero, counter=counter)
    poisoned_data = sample()
    poisoned_data["target_state"][:, 1:] = 999.0
    poisoned = rollout(Flow(0.1), Aero(), poisoned_data)
    assert baseline["predicted_state_sha256_by_lead"] == poisoned["predicted_state_sha256_by_lead"]
    assert baseline["predicted_force_physical_by_lead"] == poisoned["predicted_force_physical_by_lead"]
    assert baseline["field_sums_by_lead"] != poisoned["field_sums_by_lead"]
    assert left_flow.calls == left_aero.calls == 100
    assert counter == [100]
    assert baseline["flow_transition_count"] == baseline["aerodynamic_evaluation_count"] == 100
    assert baseline["full_field_history_retained"] is False
    sums = np.asarray(baseline["predicted_state_channel_sum_by_lead"])
    assert np.unique(sums[:, 0]).size == 100  # no truth reset at report leads
    # Aero sees qhat_j and omega_j/omega_j+1.  Lead1 is based on q0 and omega1.
    expected_lead1_rear_cl_normalized = 0.2 + float(sample()["omega"][0, 1, 0])
    assert baseline["predicted_force_physical_by_lead"][0][3] == pytest.approx(
        expected_lead1_rear_cl_normalized * 7.0
    )


def test_h1_force_identity_but_flow_field_may_change():
    aero = Aero()
    k1 = rollout(Flow(0.01), aero)
    p029 = rollout(Flow(0.2), Aero())
    core.assert_h1_force_identity(k1, p029)
    assert k1["predicted_state_sha256_by_lead"][0] != p029["predicted_state_sha256_by_lead"][0]
    changed = copy.deepcopy(p029)
    changed["predicted_force_physical_by_lead"][0][0] += 1e-7
    with pytest.raises(ValueError):
        core.assert_h1_force_identity(k1, changed)


@pytest.mark.parametrize("mutation", ["empty_mask", "nonbinary_mask", "nan_state", "wrong_actions"])
def test_window_rejects_bad_mask_nonfinite_or_shape(mutation):
    data = sample()
    if mutation == "empty_mask":
        data["mask"].zero_()
    elif mutation == "nonbinary_mask":
        data["mask"][0, 0, 0, 0] = 0.5
    elif mutation == "nan_state":
        data["state"][0, 0, 0, 0] = float("nan")
    else:
        data["omega"] = data["omega"][:, :-1]
    with pytest.raises((ValueError, FloatingPointError)):
        rollout(Flow(0.1), Aero(), data)


def test_at_lead_is_not_prefix_and_field_ratio_is_pooled():
    first = fake_record(error_scale=1.0, reference_scale=1.0)
    second = fake_record(error_scale=9.0, reference_scale=100.0)
    summary = core.summarize_records(
        [first, second], relative_field_metrics_fn=relative_field_metrics, leads=(1, 10)
    )
    expected_lead10 = np.sqrt((100.0 + 900.0) / (1.0 + 100.0))
    assert summary["at_lead"]["10"]["velocity_relative_l2"] == pytest.approx(expected_lead10)
    assert summary["at_lead"]["10"]["velocity_relative_l2"] != pytest.approx(
        (10.0 + 3.0) / 2.0
    )
    assert summary["at_lead"]["10"]["velocity_relative_l2"] != summary[
        "cumulative_prefix_1_to_H"
    ]["10"]["velocity_relative_l2"]
    # Total Cd error is abs((2)+(-1)), not abs(2)+abs(-1).
    assert summary["at_lead"]["1"]["total_cd_mae"] == pytest.approx(1.0)
    assert summary["at_lead"]["10"]["count"] == 2
    assert summary["cumulative_prefix_1_to_H"]["10"]["count"] == 20


def test_zero_reference_is_none_without_epsilon():
    record = fake_record(reference_scale=0.0)
    summary = core.summarize_records(
        [record], relative_field_metrics_fn=relative_field_metrics, leads=(1,)
    )
    assert summary["at_lead"]["1"]["velocity_relative_l2"] is None
    assert summary["at_lead"]["1"]["field_relative_l2_u_v_p"] == [None, None, None]


def test_groups_counts_and_paired_signs_cover_all44():
    metadata, _ = selection()
    k1 = [
        fake_record(error_scale=1.0, force_shift=0.0, case=row["case"],
                    dataset_index=row["dataset_index"])
        for row in metadata
    ]
    p029 = [
        fake_record(error_scale=0.81, force_shift=(-0.1 if i % 2 else 0.1),
                    case=row["case"], dataset_index=row["dataset_index"])
        for i, row in enumerate(metadata)
    ]
    result = core.grouped_and_paired(
        k1, p029, metadata, relative_field_metrics_fn=relative_field_metrics
    )
    assert {key: value["count"] for key, value in result["groups"]["family"].items()} == {
        "base": 20, "train16": 16, "train8": 8
    }
    assert {key: value["count"] for key, value in result["groups"]["canonical_phase"].items()} == {
        "b00": 15, "b02": 15, "b04": 7, "b06": 7
    }
    paired = result["paired_at_lead"]["100"]["summary"]
    assert paired["velocity_relative_l2"] == {
        "count": 44, "negative": 44, "zero": 0, "positive": 0,
        "mean_delta": pytest.approx(-10.0),
    }
    assert paired["rear_cl_absolute_error"]["count"] == 44
    assert len(result["paired_at_lead"]["100"]["cases"]) == 44
    assert result["groups"]["family"]["train16"]["paired_at_lead"]["100"][
        "summary"
    ]["velocity_relative_l2"]["count"] == 16


def test_pairing_rejects_reordered_identity_or_different_truth():
    metadata, _ = selection()
    k1 = [
        fake_record(case=row["case"], dataset_index=row["dataset_index"])
        for row in metadata
    ]
    p029 = copy.deepcopy(k1)
    p029[0], p029[1] = p029[1], p029[0]
    with pytest.raises(ValueError, match="identity/order"):
        core.grouped_and_paired(
            k1, p029, metadata, relative_field_metrics_fn=relative_field_metrics
        )
    p029 = copy.deepcopy(k1)
    p029[0]["target_force_physical_by_lead"][50][3] += 1e-6
    with pytest.raises(ValueError, match="truth/reference"):
        core.grouped_and_paired(
            k1, p029, metadata, relative_field_metrics_fn=relative_field_metrics
        )
    p029 = copy.deepcopy(k1)
    p029[0]["field_sums_by_lead"][50][1][0] += 1e-6
    with pytest.raises(ValueError, match="truth/reference"):
        core.grouped_and_paired(
            k1, p029, metadata, relative_field_metrics_fn=relative_field_metrics
        )


def test_record_requires_exact100_nonnegative_sufficient_statistics():
    record = fake_record()
    record["field_sums_by_lead"].pop()
    with pytest.raises(ValueError, match="shape"):
        core.summarize_records(
            [record], relative_field_metrics_fn=relative_field_metrics, leads=(1,)
        )
    record = fake_record()
    record["field_sums_by_lead"][0][0][0] = -1.0
    with pytest.raises(FloatingPointError):
        core.summarize_records(
            [record], relative_field_metrics_fn=relative_field_metrics, leads=(1,)
        )


def test_paired_undefined_velocity_fails_intentionally():
    metadata, _ = selection()
    k1 = [
        fake_record(case=row["case"], dataset_index=row["dataset_index"])
        for row in metadata
    ]
    p029 = copy.deepcopy(k1)
    for rows in (k1, p029):
        rows[0]["field_sums_by_lead"][0][1][0] = 0.0
        rows[0]["field_sums_by_lead"][0][1][1] = 0.0
    with pytest.raises(FloatingPointError, match="undefined paired"):
        core.grouped_and_paired(
            k1, p029, metadata, relative_field_metrics_fn=relative_field_metrics
        )


def test_eval_and_gradient_contract_rejects_train_mode_or_gradients():
    flow, aero = Flow(0.1), Aero()
    flow.train()
    with pytest.raises(ValueError):
        core.rollout_window(
            flow, aero.eval(), sample(), predict_fn=predict, make_inputs_fn=make_inputs,
            field_error_sums_fn=field_error_sums,
            state_mean=torch.zeros((3, 1, 1)), state_std=torch.ones((3, 1, 1)),
            force_mean=torch.zeros(4), force_std=torch.ones(4),
            identity={"case": "synthetic_case", "start": 0, "dataset_index": 0},
        )
    flow.eval()
    flow.extra = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
    with pytest.raises(ValueError):
        core.rollout_window(
            flow, aero.eval(), sample(), predict_fn=predict, make_inputs_fn=make_inputs,
            field_error_sums_fn=field_error_sums,
            state_mean=torch.zeros((3, 1, 1)), state_std=torch.ones((3, 1, 1)),
            force_mean=torch.zeros(4), force_std=torch.ones(4),
            identity={"case": "synthetic_case", "start": 0, "dataset_index": 0},
        )
