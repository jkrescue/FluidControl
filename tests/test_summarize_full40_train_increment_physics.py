import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/summarize_full40_train_increment_physics.py"
)
SPEC = importlib.util.spec_from_file_location("full40_train_increment", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def force_rows(cd, cl):
    time = np.linspace(20.0, 80.0, MODULE.EXPECTED_ANALYSIS_SAMPLES)
    rows = np.zeros((len(time), 13))
    rows[:, 0] = time
    rows[:, 1] = cd
    rows[:, 4] = cl
    return rows


def test_exact_scope_is_four_train_branches_and_two_train_zero_references():
    assert MODULE.EXACT_INCREMENT_CASES == (
        "matched_start_acquisition_train_b00_m0375",
        "matched_start_acquisition_train_b00_p0375",
        "matched_start_acquisition_train_b02_m0375",
        "matched_start_acquisition_train_b02_p0375",
    )
    assert MODULE.EXACT_ZERO_CASES == (
        "matched_start_acquisition_train_b00_zero",
        "matched_start_acquisition_train_b02_zero",
    )
    assert all(
        "validation" not in name and "frozen" not in name
        for name in MODULE.EXACT_INCREMENT_CASES + MODULE.EXACT_ZERO_CASES
    )


def test_metrics_and_canonical_joint_gate_are_same_phase_zero_relative():
    phase = np.linspace(0.0, 8.0 * np.pi, MODULE.EXPECTED_ANALYSIS_SAMPLES)
    zero = MODULE.branch_metrics(
        force_rows(1.0, 0.0), force_rows(1.0, 0.4 * np.sin(phase))
    )
    control = MODULE.branch_metrics(
        force_rows(0.98, 0.0),
        force_rows(0.97, 0.02 + 0.41 * np.sin(phase)),
    )
    result = MODULE.comparison(control, zero)
    assert result["total_drag_reduction_fraction_positive_is_better"] == pytest.approx(
        0.025
    )
    assert result["canonical_joint_diagnostic_pass"] is True
    control["mean_cl_rear"] = 0.05
    assert MODULE.comparison(control, zero)["canonical_joint_diagnostic_pass"] is False


def test_config_guard_rejects_non_train_identity_and_nonfixed_window():
    valid = {
        "case": "matched_start_acquisition_train_b00_m0375",
        "split": "train",
        "phase_bin": 0,
        "action_target": -0.375,
        "phase_manifest_sha256": MODULE.PHASE_MANIFEST_SHA256,
        "start_time": 148.0,
        "end_time": 228.0,
        "analysis_window": [168.0, 228.0],
    }
    MODULE._validate_config(valid, valid["case"], 0, -0.375)
    with pytest.raises(ValueError, match="non-train identity"):
        MODULE._validate_config(
            dict(valid, case="matched_start_acquisition_validation_b00_m0375"),
            "matched_start_acquisition_validation_b00_m0375",
            0,
            -0.375,
        )
    with pytest.raises(ValueError, match="fixed train/window"):
        MODULE._validate_config(dict(valid, analysis_window=[180.0, 228.0]), valid["case"], 0, -0.375)


def test_exact_window_rejects_missing_force_sample():
    rows = force_rows(1.0, 0.0)
    with pytest.raises(ValueError, match="requires 12001 samples"):
        MODULE.select_exact_window(np.delete(rows, 12, axis=0), 20.0, 80.0)


def test_atomic_writer_refuses_overwrite(tmp_path):
    output = tmp_path / "nested" / "result.json"
    MODULE.write_json_exclusive_atomic(output, {"value": 1})
    assert json.loads(output.read_text()) == {"value": 1}
    with pytest.raises(ValueError, match="refusing to overwrite"):
        MODULE.write_json_exclusive_atomic(output, {"value": 2})
    assert json.loads(output.read_text()) == {"value": 1}
