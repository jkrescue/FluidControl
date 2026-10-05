"""Synthetic CPU terminal-evidence fixtures; no training or scientific claims."""
import copy
import json

import pytest

import finalize_fcp013_collected_training as recovery
from audit_fcp013_dual_candidate import validate_guard


def events():
    actor = {"ID": recovery.CONTAINER_ID, "Attributes": {
        "image": recovery.IMAGE, "name": recovery.CONTAINER_NAME, "exitCode": "0"}}
    return [{"Type": "container", "Action": action, "Actor": copy.deepcopy(actor),
             "time": 200, "timeNano": 200_000_000_000 + index}
            for index, action in enumerate(("die", "destroy"))]


def validate(rows):
    return recovery.validate_docker_events("\n".join(map(json.dumps, rows)),
                                           since_seconds=100, until_seconds=201)


def test_exact_successful_die_then_destroy_required():
    result = validate(events())
    assert result["exit_code"] == 0
    assert result["destroy_time_nano"] > result["die_time_nano"]


@pytest.mark.parametrize("fault", ["id", "image", "name", "exit", "duplicate_die", "missing_destroy", "order", "future"])
def test_docker_rejects_wrong_identity_failure_and_ambiguous_terminal(fault):
    rows = events()
    if fault == "id":
        rows[0]["Actor"]["ID"] = "other"
    elif fault in ("image", "name"):
        rows[0]["Actor"]["Attributes"][fault] = "other"
    elif fault == "exit":
        rows[0]["Actor"]["Attributes"]["exitCode"] = "1"
    elif fault == "duplicate_die":
        rows.append(copy.deepcopy(rows[0]))
    elif fault == "missing_destroy":
        rows.pop()
    elif fault == "order":
        rows[1]["timeNano"] = rows[0]["timeNano"]
    else:
        rows[0].update(time=202, timeNano=202_000_000_000)
    with pytest.raises(ValueError):
        validate(rows)


def guard():
    return {"event": "gpu_guard_complete", "exit_code": 0, "memory_samples": 100,
            "min_required_mem_available_gib": 20, "min_observed_mem_available_gib": 100}


def journal():
    return {"_SYSTEMD_USER_UNIT": recovery.UNIT, "_SYSTEMD_INVOCATION_ID": recovery.INVOCATION,
            "MESSAGE": json.dumps(guard()), "__REALTIME_TIMESTAMP": "199000000", "__CURSOR": "fixture"}


def check_journal(rows, log=None):
    return recovery.validate_journal("\n".join(map(json.dumps, rows)), log or json.dumps(guard()),
                                     validate_guard=validate_guard, since_seconds=100, die_time=200)


def test_journal_success_is_bound_to_original_log():
    assert check_journal([journal()])["invocation_id"] == recovery.INVOCATION


@pytest.mark.parametrize("fault", ["unit", "invocation", "duplicate", "time", "log", "failure"])
def test_journal_rejects_wrong_generation_duplicate_guard_or_log_difference(fault):
    rows = [journal()]
    log = json.dumps(guard())
    if fault == "unit":
        rows[0]["_SYSTEMD_USER_UNIT"] = "other"
    elif fault == "invocation":
        rows[0]["_SYSTEMD_INVOCATION_ID"] = "other"
    elif fault == "duplicate":
        rows.append(journal())
    elif fault == "time":
        rows[0]["__REALTIME_TIMESTAMP"] = "99000000"
    elif fault == "log":
        value = guard(); value["memory_samples"] = 101; log = json.dumps(value)
    else:
        value = guard(); value["exit_code"] = 1; rows[0]["MESSAGE"] = json.dumps(value)
    with pytest.raises(ValueError):
        check_journal(rows, log)


def test_collected_state_does_not_fabricate_loaded_success():
    fields = {"LoadState": "not-found", "ActiveState": "inactive", "SubState": "dead", "InvocationID": ""}
    recovery.collected_unit(fields)
    for key, value in (("LoadState", "loaded"), ("ActiveState", "active"),
                       ("SubState", "running"), ("InvocationID", recovery.INVOCATION)):
        with pytest.raises(ValueError):
            recovery.collected_unit({**fields, key: value})


def test_raw_evidence_is_exclusive_and_identical_retry_only(tmp_path):
    path = tmp_path / "evidence.jsonl"
    recovery.persist_text(path, "original\n")
    recovery.persist_text(path, "original\n")
    with pytest.raises(ValueError, match="differs"):
        recovery.persist_text(path, "new\n")
    assert path.read_text() == "original\n"
