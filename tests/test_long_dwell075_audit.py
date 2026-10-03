"""Pure unit tests for the long-dwell real-CFD audit."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load_module():
    path = Path(__file__).parents[1] / "scripts" / "audit_long_dwell075_panel.py"
    spec = importlib.util.spec_from_file_location("long_dwell075_audit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = load_module()


def test_fluctuation_rms_removes_nonzero_mean() -> None:
    values = np.asarray([4.0, 6.0, 4.0, 6.0])
    assert MODULE.fluctuation_rms(values) == pytest.approx(1.0)


def test_canonical_joint_gate_uses_all_three_unchanged_thresholds() -> None:
    zero = {"total_cd_mean": 2.0, "rear_cl_fluctuation_rms": 1.0}
    passing = {
        "total_cd_mean": 1.96,
        "rear_cl_fluctuation_rms": 1.05,
        "rear_cl_mean": -0.10,
    }
    assert MODULE.comparison(passing, zero)["canonical_joint_check"]

    for field, value in (
        ("total_cd_mean", 1.961),
        ("rear_cl_fluctuation_rms", 1.051),
        ("rear_cl_mean", -0.101),
    ):
        failing = dict(passing)
        failing[field] = value
        assert not MODULE.comparison(failing, zero)["canonical_joint_check"]


def test_phase_windows_cover_two_complete_predeclared_periods() -> None:
    assert MODULE.PHASES["t90"]["window"] == (130.0, 210.0)
    assert MODULE.PHASES["t94"]["window"] == (134.0, 214.0)
    assert all(end - begin == 80.0 for begin, end in (
        phase["window"] for phase in MODULE.PHASES.values()
    ))


def test_analysis_time_rejects_a_missing_sample() -> None:
    time = np.linspace(130.0, 210.0, MODULE.EXPECTED_ANALYSIS_SAMPLES)
    with pytest.raises(ValueError, match="expected 16001"):
        MODULE.validate_analysis_time(np.delete(time, 8000), 130.0, 210.0)


def test_analysis_time_rejects_wrong_timestamp() -> None:
    time = np.linspace(130.0, 210.0, MODULE.EXPECTED_ANALYSIS_SAMPLES)
    time[8000] += 1e-4
    with pytest.raises(ValueError, match="complete 0.005 grid"):
        MODULE.validate_analysis_time(time, 130.0, 210.0)
