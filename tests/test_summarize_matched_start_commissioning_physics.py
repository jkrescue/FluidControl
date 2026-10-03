import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/summarize_matched_start_commissioning_physics.py"
)
SPEC = importlib.util.spec_from_file_location("commissioning_physics", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def force_rows(cd, cl, cm=0.0):
    time = np.linspace(20.0, 80.0, MODULE.EXPECTED_ANALYSIS_SAMPLES)
    rows = np.zeros((len(time), 13))
    rows[:, 0] = time
    rows[:, 1] = cd
    rows[:, 4] = cl
    rows[:, 7] = cm
    return rows


def test_branch_metrics_total_drag_and_mean_removed_lift_rms():
    phase = np.linspace(0.0, 8.0 * np.pi, MODULE.EXPECTED_ANALYSIS_SAMPLES)
    front = force_rows(1.0, np.zeros_like(phase))
    rear = force_rows(0.5, 3.0 + 2.0 * np.sin(phase), cm=0.2)
    omega = np.full(len(front), -0.75)
    result = MODULE.branch_metrics(front, rear, omega, 1.0)
    assert result["mean_cd_total"] == pytest.approx(1.5)
    assert result["mean_cl_rear"] == pytest.approx(3.0)
    assert result["rms_cl_rear_fluctuation"] == pytest.approx(
        np.sqrt(np.mean((2.0 * np.sin(phase)) ** 2))
    )
    assert result["mean_omega_times_cm_pitch_rear"] == pytest.approx(-0.15)


def test_same_phase_comparison_uses_canonical_three_part_diagnostic():
    zero = {
        "mean_cd_total": 2.0,
        "rms_cl_rear_fluctuation": 0.4,
        "mean_cl_rear": 0.0,
    }
    passing = {
        "mean_cd_total": 1.95,
        "rms_cl_rear_fluctuation": 0.41,
        "mean_cl_rear": 0.03,
    }
    result = MODULE.comparison(passing, zero)
    assert result["total_drag_reduction_fraction_positive_is_better"] == pytest.approx(0.025)
    assert result["rear_cl_fluctuation_rms_ratio_to_zero"] == pytest.approx(1.025)
    assert result["abs_mean_rear_cl_over_zero_fluctuation_rms"] == pytest.approx(0.075)
    assert result["canonical_joint_diagnostic_pass"] is True
    failing = dict(passing, mean_cl_rear=0.05)
    assert MODULE.comparison(failing, zero)["canonical_joint_diagnostic_pass"] is False


def test_exact_window_rejects_missing_timestamp():
    rows = force_rows(1.0, 0.0)
    with pytest.raises(ValueError, match="requires 12001 samples"):
        MODULE.select_exact_window(np.delete(rows, 123, axis=0), 20.0, 80.0)


def test_incomplete_receipts_fail_before_output_is_created(tmp_path):
    cases = tmp_path / "cases"
    receipts = tmp_path / "receipts"
    aggregate = tmp_path / "aggregate.json"
    names = [
        MODULE.case_name(phase_bin, action)
        for phase_bin in MODULE.PHASES
        for action in MODULE.ACTIONS
    ]
    aggregate.write_text(
        json.dumps(
            {
                "status": "MATCHED_START_9_CASE_COMMISSIONING_QC_PASS",
                "complete_nine_case_panel": True,
                "phase_manifest_sha256": MODULE.PHASE_MANIFEST_SHA256,
                "cases": [{"case": name} for name in names],
                "transfer": {name: {} for name in names},
            }
        )
    )
    output = tmp_path / "not-created" / "result.json"
    with pytest.raises(FileNotFoundError, match="missing raw-transfer receipt"):
        report = MODULE.build_report(cases, receipts, aggregate)
        MODULE.write_json_exclusive_atomic(output, report)
    assert not output.exists()
    assert not output.parent.exists()
