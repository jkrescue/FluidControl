"""Pure pre-solve tests for matched-start commissioning generation."""

from __future__ import annotations

import make_matched_start_acquisition_commissioning as commissioning
import pytest


def test_reviewed_v3_manifest_yields_exact_nine_case_contract() -> None:
    _, specs = commissioning.load_panel(commissioning.APPROVED_MANIFEST)
    assert len(specs) == 9
    assert {spec.phase_bin for spec in specs} == {0, 2, 4}
    assert {spec.action for spec in specs} == {-0.75, 0.0, 0.75}
    assert len({spec.name for spec in specs}) == 9


def test_manifest_and_raw_force_are_hard_bound() -> None:
    assert (
        commissioning.sha256(commissioning.APPROVED_MANIFEST)
        == commissioning.APPROVED_MANIFEST_SHA256
    )
    manifest, _ = commissioning.load_panel(commissioning.APPROVED_MANIFEST)
    force = commissioning.REPO / manifest["signal_qc"]["force_path"]
    assert commissioning.sha256(force) == manifest["signal_qc"]["force_sha256"]


@pytest.mark.parametrize("action", [-0.75, 0.0, 0.75])
def test_action_is_bounded_rate_limited_and_spans_full_run(action: float) -> None:
    points = commissioning.action_points(106.0, action)
    metrics = commissioning.validate_action(points)
    assert points[0] == (106.0, 0.0)
    assert points[-1][0] == 186.0
    assert metrics["max_abs_omega"] == abs(action)
    assert metrics["max_abs_domega_dt"] <= 1.0


def test_case_naming_is_stable() -> None:
    assert commissioning.case_name(0, -0.75).endswith("b00_m075")
    assert commissioning.case_name(2, 0.0).endswith("b02_zero")
    assert commissioning.case_name(4, 0.75).endswith("b04_p075")
