import importlib.util
from pathlib import Path

import numpy as np
import pytest


SOURCE = Path(__file__).parents[1] / "src/fluid_control/exploratory_paired_causal_history_feedback.py"
SPEC = importlib.util.spec_from_file_location("paired_h2", SOURCE)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def harness(tmp_path, *, bad_order=False):
    cases = {name: tmp_path / name for name in ("mpc", "zero")}
    for case in cases.values():
        case.mkdir()
        (case / "time").write_text("148.0")
    out = tmp_path / "output"
    out.mkdir()
    events = []

    def latest(case):
        return float((case / "time").read_text())

    def observe(role, case, now, omega):
        events.append(("observe", role, now))
        return {"time": now, "role": role, "sampled_before_action": not bad_order,
                "sample_sha256": f"{role}-{now}", "state": np.zeros((1, 3, 2, 3))}

    def plan(packet, previous):
        events.append(("plan", previous))
        return {"status": "EXPLORATORY_CANONICAL_HISTORY_H2_SELECTION_NOT_ADMISSION",
                "execute_only_first_action": True, "selected_action": min(previous + .05, .5),
                "selected_index": 3,
                "actions": [[previous + delta] * 2 for delta in (-.1, -.05, 0, .05, .1)],
                "predicted_forces_h2": np.tile(np.array([2.0, .1, .3, -.2]), (5, 2, 1)).tolist()}

    def configure(case, begin, end, before, after):
        events.append(("configure", case.name, begin, before, after))

    def solve(step, end):
        events.append(("solve", step))
        for case in cases.values():
            (case / "time").write_text(str(end))
        return {"mpc": {"clean": True}, "zero": {"clean": True}}

    def forces(role, case, end, omega):
        return np.array([2.0 + omega, .1, .3, -.2])

    def summarize(role, case, begin, end):
        return {"real_cfd": True, "samples": 201, "mean_total_cd": 2.3}

    def run():
        return M.run_paired_ten_cycle(
            cases=cases, output=out, latest_time=latest, observe_current=observe,
            plan_action=plan, configure_interval=configure, solve_pair=solve,
            observe_forces=forces, summarize_actual=summarize, guard=lambda: None,
            identity={"k1": "failed-long-ar-reference"})
    return run, events, out


def test_ten_cycles_observe_before_plan_and_configuration(tmp_path):
    run, events, out = harness(tmp_path)
    result = run()
    assert result["cycles"] == 10
    assert len(result["rows"]) == 10
    assert result["status"].endswith("NOT_ADMISSION")
    assert result["original_long_ar_gate_passed"] is False
    for step in range(10):
        chunk = events[step * 6:(step + 1) * 6]
        assert [item[0] for item in chunk] == ["observe", "observe", "plan", "configure", "configure", "solve"]
        assert {chunk[0][1], chunk[1][1]} == {"mpc", "zero"}
    assert result["rows"][0]["boundary_semantics"].startswith("linear_ramp")
    assert result["rows"][-1]["end_time"] == 149.0
    assert result["physical_duration_D_over_U"] == 1.0
    assert result["action_audit"]["not_motor_energy"] is True
    assert "selected_prediction_minus_actual_next_force" in result["rows"][0]
    assert (out / "progress.json").is_file()


def test_pair_starts_from_same_restart_and_zero_stays_zero(tmp_path):
    run, _, _ = harness(tmp_path)
    result = run()
    assert all(row["zero_omega"] == 0.0 for row in result["rows"])
    assert result["rows"][0]["previous_omega"] == 0.0
    assert result["action_audit"]["max_abs_delta_omega"] <= .1
    assert set(result["actual_force_metrics"]) == {"mpc", "zero"}


def test_rejects_missing_pre_action_evidence(tmp_path):
    run, _, _ = harness(tmp_path, bad_order=True)
    with pytest.raises(ValueError, match="ordering"):
        run()


def test_rejects_illegal_selected_rate(tmp_path):
    run, _, _ = harness(tmp_path)
    original = M.RATE_LIMIT
    M.RATE_LIMIT = .01
    try:
        with pytest.raises(ValueError, match="rate"):
            run()
    finally:
        M.RATE_LIMIT = original


def test_actual_force_must_be_four_finite_channels(tmp_path):
    values = M._force_row(np.array([1.0, 2.0, 3.0, 4.0]))
    assert values["rear_cl"] == 4.0
    with pytest.raises(ValueError):
        M._force_row(np.array([1.0, np.nan, 3.0, 4.0]))
