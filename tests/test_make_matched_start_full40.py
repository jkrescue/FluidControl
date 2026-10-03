"""Pure pre-execution tests for the frozen full40 acquisition design."""

from __future__ import annotations

import json

import make_matched_start_full40 as full40
import pytest


def test_plan_is_exact_40_with_expected_split_counts_and_remainder() -> None:
    _, specs = full40.load_plan()
    assert len(specs) == 40
    assert len({spec.name for spec in specs}) == 40
    assert sum(spec.split == "train" for spec in specs) == 20
    assert sum(spec.split == "validation" for spec in specs) == 10
    assert sum(spec.split == "frozen_test" for spec in specs) == 10
    assert sum(spec.disposition == "existing_nine_case_commissioning" for spec in specs) == 9
    assert sum(spec.disposition == "planned_new_remainder_case" for spec in specs) == 31


def test_existing_nine_are_exactly_old_names_and_never_generation_targets() -> None:
    _, specs = full40.load_plan()
    existing = {
        spec.name for spec in specs if spec.disposition == "existing_nine_case_commissioning"
    }
    expected = {
        f"matched_start_acquisition_train_b{phase_bin:02d}_{label}"
        for phase_bin in (0, 2, 4)
        for label in ("m075", "zero", "p075")
    }
    assert existing == expected
    for spec in specs:
        if spec.name in existing:
            with pytest.raises(ValueError, match="validate-only"):
                full40.generate(spec)


@pytest.mark.parametrize("action", full40.ACTIONS)
def test_all_actions_are_bounded_rate_limited_and_span_80(action: float) -> None:
    points = full40.action_points(102.0, action)
    metrics = full40.validate_action(points)
    assert points[0] == [102.0, 0.0]
    assert points[-1][0] == 182.0
    assert metrics["max_abs_omega"] == abs(action)
    assert metrics["max_abs_domega_dt"] <= 1.0


def test_predeclaration_has_fixed_windows_and_interpretation_guards() -> None:
    manifest, specs = full40.load_plan()
    artifact = full40.predeclaration(manifest, specs)
    assert artifact["status"] == "MATCHED_START_FULL40_PREDECLARED_NO_NEW_CASES_GENERATED"
    assert artifact["matrix"]["planned_new_remainder_cases"] == 31
    assert artifact["matrix"]["split_case_counts"] == {
        "train": 20,
        "validation": 10,
        "frozen_test": 10,
    }
    assert all(
        row["analysis_window"][1] - row["analysis_window"][0] == 60.0
        for row in artifact["cases"].values()
    )
    assert "not closed-loop control" in " ".join(artifact["interpretation_guards"])


def test_predeclaration_is_bound_but_extension_authorization_is_disabled() -> None:
    manifest, specs = full40.load_plan()
    assert full40.sha256(full40.PREDECLARATION) == full40.APPROVED_PREDECLARATION_SHA256
    full40.validate_approved_predeclaration(manifest, specs)
    assert len(full40.APPROVED_EXTENSION_AUTHORIZATION_SHA256) != 64
    with pytest.raises(ValueError, match="not reviewed and hard-bound"):
        full40.validate_extension_authorization(specs)


def test_predeclaration_serialization_is_deterministic() -> None:
    manifest, specs = full40.load_plan()
    first = json.dumps(full40.predeclaration(manifest, specs), indent=2) + "\n"
    second = json.dumps(full40.predeclaration(manifest, specs), indent=2) + "\n"
    assert first == second
