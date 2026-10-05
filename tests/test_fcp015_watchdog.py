"""Staged CPU identity fixtures; no services or recovery actions are executed."""
from datetime import UTC, datetime

import pytest
import watch_training_evaluation_state as watch
import reconcile_training_evaluation_state as reconcile


@pytest.fixture
def authority(tmp_path, monkeypatch):
    for path in (watch.P013_APPROVAL, watch.P015_APPROVAL):
        p = tmp_path / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("fixture")
    monkeypatch.setattr(watch, "file_sha256", lambda p:
        watch.P015_APPROVAL_SHA256 if p.name == watch.P015_APPROVAL.name else watch.P013_APPROVAL_SHA256)
    state = {"active_state": "active", "sub_state": "running", "result": "success",
             "main_pid": 123, "exec_main_status": "0", "invocation_id": watch.P015_INVOCATION,
             "exec_start": "{ argv[]=/usr/bin/env FCP015_APPROVAL_SHA256=" + watch.P015_APPROVAL_SHA256
                + " /bin/bash " + str(tmp_path / watch.P015_LAUNCHER) + " --execute ; }"}
    return tmp_path, state


def test_exact_running(authority):
    repo, state = authority
    task = watch.p015_authority(repo, state)
    assert task["state"] == "RUNNING"
    assert not task["stage_complete"] and not task["scientific_admission"]
    assert task["approved_action_id"] is None


@pytest.mark.parametrize("change,expected", [
    ({"sub_state": "exited", "main_pid": 0}, "TERMINAL_REQUIRES_INDEPENDENT_REVIEW"),
    ({"active_state": "failed", "result": "exit-code"}, "NEEDS_AGENT_ANALYSIS"),
    ({"exec_main_status": "75"}, "NEEDS_AGENT_ANALYSIS"),
    ({"invocation_id": watch.P013_INVOCATION}, "NEEDS_AGENT_ANALYSIS"),
    ({"exec_start": ""}, "NEEDS_AGENT_ANALYSIS"),
])
def test_terminal_failure_identity_never_restarts(authority, change, expected):
    repo, state = authority
    state.update(change)
    task = watch.p015_authority(repo, state)
    assert task["state"] == expected
    result = reconcile.plan_recovery({"current_authority": "p015_training", "stage_complete": True,
        "authority_tasks": {"p015_training": task}}, {}, {"anything": {"command": ["forbidden"]}})
    assert result["decision"] == "NEEDS_AGENT_ANALYSIS"
    assert "command" not in result


@pytest.mark.parametrize("old,new", [("--execute", "--dry-run"),
    ("FCP015_APPROVAL_SHA256=", "WRONG_APPROVAL_SHA256="),
    ("fcp015_window_accumulation_source", "unapproved_source"),
    (watch.P015_APPROVAL_SHA256, "0" * 64)])
def test_command_binding(authority, old, new):
    repo, state = authority
    state["exec_start"] = state["exec_start"].replace(old, new)
    assert watch.p015_authority(repo, state)["state"] == "NEEDS_AGENT_ANALYSIS"


def test_approval_bytes_fail_closed(authority, monkeypatch):
    repo, state = authority
    monkeypatch.setattr(watch, "file_sha256", lambda _: "bad")
    assert watch.p015_authority(repo, state)["state"] == "NEEDS_AGENT_ANALYSIS"


def test_p015_priority_preserves_alerts_not_recursive_history(authority, monkeypatch):
    repo, state = authority
    legacy = {"status": "OLD", "stage_complete": True,
        "alerts": ["SPARK_MEMORY_BELOW_20_GIB", "UNRELATED_ERROR",
                   "D015_COMPLETE_CALIBRATION_ENGINEERING_WITH_NO_RUNNING_TASK_FOR_300_SECONDS"],
        "resources": {"mem_available_gib": 19}, "blocker_reasons": []}
    monkeypatch.setattr(watch, "_build_legacy_sample", lambda *a: dict(legacy))
    previous = {"historical_legacy_sample": {"obsolete_nested_history": True},
                "alerts": ["FC_P013_FORMAL_TERMINAL_REQUIRES_INDEPENDENT_REVIEW"]}
    sample = watch.build_sample(repo, previous, {watch.P015_UNIT: state,
        watch.P013_UNIT: {"active_state": "active", "sub_state": "exited"}}, {}, datetime.now(UTC))
    assert sample["current_authority"] == "p015_training"
    assert set(sample["authority_tasks"]) == {"p015_training"}
    assert sample["historical_legacy_sample"] == legacy
    assert "historical_legacy_sample" not in sample["historical_legacy_sample"]
    assert sample["alerts"] == ["SPARK_MEMORY_BELOW_20_GIB", "UNRELATED_ERROR"]
    assert sample["stage_complete"] is False
    decision = reconcile.plan_recovery(sample, {}, {})
    assert decision["decision"] == "NO_ACTION_AUTHORITY_RUNNING"
    assert decision["alerts"] == sample["alerts"]
    assert decision["resources"] == sample["resources"]


def test_missing_p015_unit_does_not_fall_back_to_historical_p013(authority, monkeypatch):
    repo, _ = authority
    monkeypatch.setattr(watch, "_build_legacy_sample", lambda *a: {"alerts": [], "blocker_reasons": []})
    sample = watch.build_sample(repo, None, {}, {}, datetime.now(UTC))
    assert sample["current_authority"] == "p015_training"
    assert sample["authority_tasks"]["p015_training"]["state"] == "NEEDS_AGENT_ANALYSIS"
