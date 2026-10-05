"""Software fixtures only: no service execution or scientific admission."""
from datetime import UTC, datetime

import pytest
import watch_training_evaluation_state as watch
import reconcile_training_evaluation_state as reconcile


@pytest.fixture
def authority(tmp_path, monkeypatch):
    approval = tmp_path / watch.P013_APPROVAL
    approval.parent.mkdir(parents=True)
    approval.write_text("fixture")
    monkeypatch.setattr(watch, "file_sha256", lambda _: watch.P013_APPROVAL_SHA256)
    state = {"active_state": "active", "sub_state": "running", "result": "success",
             "main_pid": 123, "invocation_id": watch.P013_INVOCATION,
             "exec_start": "{ argv[]=/usr/bin/env FCP_POSTEVAL_PROFILE=p013 /bin/bash "
                           + str(tmp_path / watch.P013_LAUNCHER) + " --execute ; }"}
    return tmp_path, state


def test_running(authority):
    repo, state = authority
    task = watch.p013_authority(repo, state)
    assert task["state"] == "RUNNING"
    assert task["stage_complete"] is False
    assert task["scientific_admission"] is False
    assert task["approved_action_id"] is None


@pytest.mark.parametrize("change,expected", [
    ({"sub_state": "exited", "main_pid": 0}, "TERMINAL_REQUIRES_INDEPENDENT_REVIEW"),
    ({"active_state": "inactive", "sub_state": "dead", "main_pid": 0}, "TERMINAL_REQUIRES_INDEPENDENT_REVIEW"),
    ({"active_state": "failed", "result": "exit-code"}, "NEEDS_AGENT_ANALYSIS"),
    ({"invocation_id": "other"}, "NEEDS_AGENT_ANALYSIS"),
    ({"exec_start": ""}, "NEEDS_AGENT_ANALYSIS"),
])
def test_nonrunning_is_never_retry(authority, change, expected):
    repo, state = authority
    state.update(change)
    task = watch.p013_authority(repo, state)
    assert task["state"] == expected
    result = reconcile.plan_recovery({"current_authority": "p013_formal",
        "stage_complete": True, "authority_tasks": {"p013_formal": task}}, {}, {})
    assert result["decision"] == "NEEDS_AGENT_ANALYSIS"


@pytest.mark.parametrize("old,new", [("p013", "p009"), ("--execute", "--dry-run"),
                                     ("f95048c3a786", "other")])
def test_command_binding(authority, old, new):
    repo, state = authority
    state["exec_start"] = state["exec_start"].replace(old, new)
    assert watch.p013_authority(repo, state)["state"] == "NEEDS_AGENT_ANALYSIS"


def test_hash_mismatch(authority, monkeypatch):
    repo, state = authority
    monkeypatch.setattr(watch, "file_sha256", lambda _: "bad")
    assert watch.p013_authority(repo, state)["state"] == "NEEDS_AGENT_ANALYSIS"


def test_overlay_preserves_errors_resources_without_recursive_history(authority, monkeypatch):
    repo, state = authority
    legacy = {"status": "OLD", "stage_complete": True,
              "alerts": ["D015_COMPLETE_CALIBRATION_ENGINEERING_WITH_NO_RUNNING_TASK_FOR_300_SECONDS", "SPARK_MEMORY_BELOW_20_GIB", "FUTURE_NO_RUNNING_ERROR"],
              "blocker_reasons": ["existing error"], "resources": {"mem_available_gib": 19}}
    monkeypatch.setattr(watch, "_build_legacy_sample", lambda *a: dict(legacy))
    previous = {"historical_legacy_sample": {"arbitrarily_deep": "history"}}
    result = watch.build_sample(repo, previous, {watch.P013_UNIT: state}, {}, datetime.now(UTC))
    assert result["historical_legacy_sample"] == legacy
    assert result["stage_complete"] is False
    assert result["active_units"] == [watch.P013_UNIT]
    assert result["alerts"] == ["SPARK_MEMORY_BELOW_20_GIB", "FUTURE_NO_RUNNING_ERROR"]
    decision = reconcile.plan_recovery(result, {}, {})
    assert decision["decision"] == "NO_ACTION_AUTHORITY_RUNNING"
    assert decision["alerts"] == result["alerts"]
    assert decision["resources"]["mem_available_gib"] == 19


def test_legacy_unchanged_without_approval(tmp_path, monkeypatch):
    legacy = {"stage_complete": True}
    monkeypatch.setattr(watch, "_build_legacy_sample", lambda *a: legacy)
    assert watch.build_sample(tmp_path, None, {}, {}, datetime.now(UTC)) is legacy
    assert reconcile.plan_recovery(legacy, {}, {}) == {"decision": "NO_ACTION_STAGE_COMPLETE"}
