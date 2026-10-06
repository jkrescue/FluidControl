import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from fluid_control.canonical_joint_v1 import (
    canonical_force_ledger,
    canonical_joint_cost_components,
)


SOURCE = Path(__file__).parents[1] / "src/fluid_control/exploratory_causal_history_mpc.py"
SPEC = importlib.util.spec_from_file_location("causal_h2", SOURCE)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
BASELINE = {"total_drag": 2.3, "rear_cl_fluctuation_rms": 1.2, "source": "train-b00"}


def history():
    times = 141.9 + .1 * np.arange(62)
    forces = np.zeros((62, 4))
    forces[:, 0] = 1.4
    forces[:, 2] = .9
    forces[:, 3] = np.sin(np.linspace(0, 2 * np.pi, 62))
    return {"mode": M.MODE, "times": times, "forces": forces,
            "sources": {"forceFront": [{"path": "front", "sha256": "a" * 64}],
                        "forceRear": [{"path": "rear", "sha256": "b" * 64}]}}


def candidates():
    forces = np.zeros((5, 2, 4))
    forces[:, :, 0] = 1.4
    forces[:, :, 2] = .9
    forces[:, :, 3] = np.arange(5)[:, None] * .1
    actions = np.repeat(np.array([-.1, -.05, 0, .05, .1])[:, None], 2, axis=1)
    return forces, actions


def test_loads_exact_62_samples_and_binds_raw_provenance(tmp_path):
    expected = history()["sources"]
    def loader(reference, restart_time, **kwargs):
        assert reference == tmp_path and restart_time == 148.0
        assert kwargs == {"provenance_root": tmp_path.parent,
                          "control_dt": .1, "sample_count": 62}
        row = history()
        return row["times"], row["forces"], copy.deepcopy(expected)
    result = M.load_bound_actual_history(
        tmp_path, 148.0, provenance_root=tmp_path.parent,
        expected_sources=expected, actual_causal_prehistory=loader)
    assert result["times"][0] == pytest.approx(141.9)
    assert result["times"][-1] == pytest.approx(148.0)
    with pytest.raises(ValueError, match="provenance"):
        M.load_bound_actual_history(
            tmp_path, 148.0, provenance_root=tmp_path.parent,
            expected_sources={}, actual_causal_prehistory=loader)


def test_actual_endpoint_only_advances_persistent_history():
    original = history()
    predicted = np.full(4, 99.0)
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    advanced = M.append_actual_endpoint(original, endpoint=148.1, actual_force=actual)
    np.testing.assert_array_equal(advanced["forces"][-1], actual)
    assert not np.array_equal(advanced["forces"][-1], predicted)
    np.testing.assert_array_equal(original["forces"], history()["forces"])
    assert advanced["persistent_update_source"] == "actual_real_cfd_endpoint_only"


def test_candidate_rolls_are_local_and_use_existing_canonical_functions():
    original = history()
    before = original["forces"].copy()
    forces, actions = candidates()
    result = M.score_five_candidates(
        forces, actions, original, current_omega=0.0, baseline=BASELINE,
        canonical_force_ledger=canonical_force_ledger,
        canonical_joint_cost_components=canonical_joint_cost_components)
    np.testing.assert_array_equal(original["forces"], before)
    assert result["persistent_history_mutated"] is False
    assert len(result["candidate_reports"]) == 5
    assert all(len(row["stages"]) == 2 for row in result["candidate_reports"])


def test_h2_mean_uses_first_rate_then_zero_rate_for_held_action():
    forces, actions = candidates()
    result = M.score_five_candidates(
        forces, actions, history(), current_omega=0.0, baseline=BASELINE,
        canonical_force_ledger=canonical_force_ledger,
        canonical_joint_cost_components=canonical_joint_cost_components)
    candidate = result["candidate_reports"][3]  # +0.05 held twice
    assert candidate["stages"][0]["components"]["rate"] == pytest.approx(.0025)
    assert candidate["stages"][1]["components"]["rate"] == 0.0
    assert candidate["mean_components"]["rate"] == pytest.approx(.00125)
    assert candidate["mean_components"]["actuation"] == pytest.approx(
        .01 * (.05 / .75) ** 2)


def test_report_is_exact_mean_of_two_existing_canonical_stage_costs():
    forces, actions = candidates()
    result = M.score_five_candidates(
        forces, actions, history(), current_omega=0.0, baseline=BASELINE,
        canonical_force_ledger=canonical_force_ledger,
        canonical_joint_cost_components=canonical_joint_cost_components)
    row = result["candidate_reports"][4]
    for name, value in row["mean_components"].items():
        expected = np.mean([stage["components"][name] for stage in row["stages"]])
        assert value == pytest.approx(expected)
    assert result["h2_cost"][4] == pytest.approx(sum(row["mean_components"].values()))


def test_selects_only_feasible_first_action_and_retains_failure_label():
    forces, actions = candidates()
    # Make all costs equal to test the exact hold tie preference.
    forces[:] = forces[0]
    report = M.select_canonical_history_action(
        forces, np.zeros(5), actions, history(), current_omega=0.0,
        state_abs_limit=1.0, baseline=BASELINE,
        canonical_force_ledger=canonical_force_ledger,
        canonical_joint_cost_components=canonical_joint_cost_components)
    assert report["status"].endswith("NOT_ADMISSION")
    assert report["selected_index"] == 2
    assert report["selected_action"] == 0.0
    assert report["original_long_ar_gate_passed"] is False


def test_rejects_bad_grid_shape_nonfinite_and_wrong_endpoint():
    row = history()
    with pytest.raises(ValueError):
        M.validate_history(row["times"][:-1], row["forces"][:-1], expected_end=148.0)
    bad = row["times"].copy()
    bad[4] += .01
    with pytest.raises(ValueError, match="grid"):
        M.validate_history(bad, row["forces"], expected_end=148.0)
    with pytest.raises(ValueError, match="next control time"):
        M.append_actual_endpoint(row, endpoint=148.2, actual_force=np.zeros(4))


def test_rejects_changed_second_action_or_candidate_set():
    forces, actions = candidates()
    actions[1, 1] = 0.0
    with pytest.raises(ValueError, match="fixed five"):
        M.score_five_candidates(
            forces, actions, history(), current_omega=0.0, baseline=BASELINE,
            canonical_force_ledger=canonical_force_ledger,
            canonical_joint_cost_components=canonical_joint_cost_components)
