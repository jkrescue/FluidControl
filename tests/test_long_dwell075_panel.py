"""Unit tests for the predeclared long-dwell physical-control panel."""

from __future__ import annotations

import make_long_dwell075_panel as panel
import pytest


def test_panel_has_matched_zero_and_control_at_two_phases() -> None:
    assert {spec.start for spec in panel.PANEL} == {90.0, 94.0}
    for start in (90.0, 94.0):
        phase_cases = [spec for spec in panel.PANEL if spec.start == start]
        assert len(phase_cases) == 2
        assert sorted(spec.controlled for spec in phase_cases) == [False, True]


def test_control_action_is_zero_mean_bounded_and_rate_limited() -> None:
    for spec in panel.PANEL:
        metrics = panel.action_metrics(panel.action_points(spec))
        if spec.controlled:
            assert metrics["signed_time_mean_omega"] == pytest.approx(0.0)
            assert metrics["max_abs_omega"] == pytest.approx(0.75)
            assert metrics["max_abs_domega_dt"] == pytest.approx(0.375)
        else:
            assert metrics["max_abs_omega"] == 0.0


def test_analysis_window_is_fixed_two_complete_action_periods() -> None:
    assert panel.DURATION - panel.ANALYSIS_OFFSET == 80.0
    assert (panel.DURATION - panel.ANALYSIS_OFFSET) / panel.ACTION_PERIOD == 2.0
    for spec in panel.PANEL:
        schedule = panel.predeclared_audit()["schedules"][spec.name]
        assert schedule["analysis_window"] == [spec.start + 40.0, spec.start + 120.0]
