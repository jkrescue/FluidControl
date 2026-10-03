import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "summarize_full40_train20_physics.py"
SPEC = importlib.util.spec_from_file_location("train20_physics", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def matrix_fixture():
    cases = {}
    for phase in MODULE.PHASES:
        for action in MODULE.ACTIONS:
            name = MODULE.case_name(phase, action)
            cases[name] = {
                "split": "train",
                "phase_bin": phase,
                "action_target": action,
                "disposition": (
                    "existing_nine_case_commissioning"
                    if name in MODULE.EXISTING_NINE
                    else "planned_new_remainder_case"
                ),
            }
    cases["matched_start_acquisition_validation_b01_zero"] = {
        "split": "validation"
    }
    cases["matched_start_acquisition_frozen_test_b03_zero"] = {
        "split": "frozen_test"
    }
    return {"cases": cases}


def test_exact_train20_matrix_excludes_validation_and_frozen():
    train = MODULE.select_train_matrix(matrix_fixture())
    assert len(train) == 20
    assert set(train) == set(MODULE.EXACT_TRAIN_CASES)
    assert len(MODULE.EXISTING_NINE) == 9
    assert len(MODULE.NEW_TRAIN_ELEVEN) == 11
    assert all("validation" not in name and "frozen" not in name for name in train)


def test_train_matrix_fails_closed_on_missing_or_relabelled_case():
    fixture = matrix_fixture()
    fixture["cases"].pop(MODULE.EXACT_TRAIN_CASES[0])
    with pytest.raises(ValueError, match="exact 4x5 panel"):
        MODULE.select_train_matrix(fixture)
    fixture = matrix_fixture()
    fixture["cases"][MODULE.EXACT_TRAIN_CASES[0]]["disposition"] = "wrong"
    with pytest.raises(ValueError, match="identity differs"):
        MODULE.select_train_matrix(fixture)


def test_macro_and_worst_preserves_case_identity_and_joint_counts():
    rows = [
        {
            "case": "a",
            "total_drag_reduction_fraction_positive_is_better": 0.03,
            "rear_cl_fluctuation_rms_ratio_to_zero": 1.0,
            "abs_mean_rear_cl_over_zero_fluctuation_rms": 0.05,
            "canonical_joint_diagnostic_pass": True,
        },
        {
            "case": "b",
            "total_drag_reduction_fraction_positive_is_better": -0.01,
            "rear_cl_fluctuation_rms_ratio_to_zero": 1.2,
            "abs_mean_rear_cl_over_zero_fluctuation_rms": 0.3,
            "canonical_joint_diagnostic_pass": False,
        },
    ]
    result = MODULE._macro_and_worst(rows)
    assert result["macro_mean_total_drag_reduction_fraction"] == pytest.approx(0.01)
    assert result["joint_pass_count"] == 1
    assert result["worst_drag_reduction"]["case"] == "b"
    assert result["worst_rear_cl_fluctuation_ratio"]["case"] == "b"
    assert result["worst_abs_rear_cl_bias_ratio"]["case"] == "b"


def test_atomic_output_refuses_overwrite(tmp_path):
    output = tmp_path / "result.json"
    MODULE.write_json_exclusive_atomic(output, {"status": "one"})
    with pytest.raises(ValueError, match="refusing to overwrite"):
        MODULE.write_json_exclusive_atomic(output, {"status": "two"})
    assert json.loads(output.read_text()) == {"status": "one"}
