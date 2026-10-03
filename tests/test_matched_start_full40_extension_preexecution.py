"""Tests for the disabled-by-default full40 extension execution stack."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCHEDULER = load("full40_scheduler", "scripts/schedule_matched_start_full40_extension.py")
AUDIT = load("full40_audit", "scripts/audit_matched_start_full40_extension.py")
AUTH = load(
    "full40_authorization",
    "cfd/tandem_cylinders/prepare_matched_start_full40_extension_authorization.py",
)


def snapshot(default: str = "PENDING", **overrides: str) -> dict:
    cases = {name: {"status": default} for name in SCHEDULER.QUEUE}
    for name, status in overrides.items():
        cases[name] = {"status": status}
    return {
        "cases": cases,
        "mem_available_kib": 80 * 1024 * 1024,
        "disk_free_bytes": 200 * 1024**3,
    }


def test_queue_is_exact_31_and_authorized_but_review_blocked() -> None:
    assert len(SCHEDULER.QUEUE) == 31
    assert len(SCHEDULER.AUTHORIZATION_SHA256) == 64
    report = SCHEDULER.dry_run_report()
    assert report["execution_enabled"] is False
    assert report["authorization"] == "BOUND_BUT_BLOCKED_PENDING_IMPLEMENTATION_REVIEW"
    assert report["maximum_parallel_cases"] == 4
    assert SCHEDULER.IMPLEMENTATION_REVIEWED is False
    assert all("_train_" in name for name in SCHEDULER.QUEUE[:11])
    assert all("_validation_" in name for name in SCHEDULER.QUEUE[11:21])
    assert all("_frozen_test_" in name for name in SCHEDULER.QUEUE[21:])


def test_authorization_binding_is_identical_across_execution_stack() -> None:
    runner = (REPO / "cfd/tandem_cylinders/run_matched_start_full40_case.sh").read_text()
    match = re.search(r"^authorization_sha='([^']+)'$", runner, re.MULTILINE)
    assert match is not None
    assert match.group(1) == SCHEDULER.AUTHORIZATION_SHA256
    assert AUDIT.AUTHORIZATION_SHA256 == SCHEDULER.AUTHORIZATION_SHA256
    assert AUTH.full40.APPROVED_EXTENSION_AUTHORIZATION_SHA256 == SCHEDULER.AUTHORIZATION_SHA256


def test_authorized_cases_are_set_semantics_not_scheduler_order() -> None:
    lexical_builder_order = sorted(SCHEDULER.QUEUE)
    assert lexical_builder_order != list(SCHEDULER.QUEUE)
    assert SCHEDULER.authorized_case_set_matches(lexical_builder_order)
    assert SCHEDULER.authorized_case_set_matches(list(reversed(SCHEDULER.QUEUE)))
    assert not SCHEDULER.authorized_case_set_matches(lexical_builder_order[:-1])
    assert not SCHEDULER.authorized_case_set_matches(
        lexical_builder_order[:-1] + [lexical_builder_order[0]]
    )


def test_external_runner_and_solver_are_counted_once_per_case() -> None:
    external = "matched_start_acquisition_train_b00_zero"
    processes = [
        ["bash", f"bash run_matched_start_acquisition_case.sh {external}"],
        ["pimpleFoam", f"pimpleFoam -case /case/cases/{external}"],
        ["pimpleFoam", f"pimpleFoam -case /case/cases/{SCHEDULER.QUEUE[0]}"],
    ]
    assert SCHEDULER.count_external_matched_cases(processes) == 1


def test_worker_runner_sync_and_solver_dependencies_are_sha_guarded(monkeypatch) -> None:
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(SCHEDULER, "run", fake_run)
    SCHEDULER.sync_worker_runner()
    SCHEDULER.verify_worker_solver_dependencies()
    assert len(calls) == 2
    runner_call, dependency_call = calls
    assert "worker_script_updates" in runner_call[0][-1]
    assert "authorization_sha=" in runner_call[1]["input"]
    dependency_command = dependency_call[0][-1]
    assert "docker image inspect" in dependency_command
    assert SCHEDULER.OPENFOAM_IMAGE in dependency_command
    assert all(sha in dependency_command for sha in SCHEDULER.WORKER_DEPENDENCIES.values())


def test_baseline_forces_are_synced_from_authorized_sha_only(monkeypatch) -> None:
    source = SCHEDULER.CASES / "tandem_backward_dt005"
    expected = {
        force: SCHEDULER.sha256(
            source / f"postProcessing/{force}/0/coefficient.dat"
        )
        for force in ("forceFront", "forceRear")
    }
    calls = []
    monkeypatch.setattr(
        SCHEDULER,
        "sync_worker_artifact",
        lambda path, digest: calls.append((path, digest)),
    )
    SCHEDULER.sync_worker_baseline_forces(
        {"baseline_force_source_sha256": expected}
    )
    assert [path.name for path, _ in calls] == ["coefficient.dat", "coefficient.dat"]
    assert [digest for _, digest in calls] == [expected["forceFront"], expected["forceRear"]]
    bad = dict(expected, forceFront="0" * 64)
    with pytest.raises(ValueError, match="Spark baseline force SHA differs"):
        SCHEDULER.sync_worker_baseline_forces(
            {"baseline_force_source_sha256": bad}
        )


def test_scheduler_never_plans_more_than_four_and_fail_stops() -> None:
    result = SCHEDULER.plan(snapshot(), set())
    assert len(result["start"]) == 4
    assert all("_train_" in name for name in result["start"])
    assert [SCHEDULER.PREDECLARED[name]["phase_bin"] for name in result["start"]] == [
        0, 0, 2, 2,
    ]
    failed = SCHEDULER.QUEUE[0]
    result = SCHEDULER.plan(snapshot(**{failed: "FAILED"}), set())
    assert result["stop"] is True
    assert result["start"] == []


def test_current_split_cross_phase_fill_preserves_predeclared_order() -> None:
    result = SCHEDULER.plan(snapshot(default="NOT_GENERATED"), set())
    assert result["start"] == []
    assert result["prepare"] == list(SCHEDULER.QUEUE[:4])
    assert [SCHEDULER.PREDECLARED[name]["phase_bin"] for name in result["prepare"]] == [
        0, 0, 2, 2,
    ]
    assert all(SCHEDULER.PREDECLARED[name]["split"] == "train" for name in result["prepare"])


def test_group_barriers_keep_validation_and_frozen_sealed() -> None:
    train = {name for name in SCHEDULER.QUEUE if "_train_" in name}
    validation = {name for name in SCHEDULER.QUEUE if "_validation_" in name}
    result = SCHEDULER.plan(snapshot(), train)
    assert result["current_group"]
    assert all("_validation_b01_" in name for name in result["current_group"])
    result = SCHEDULER.plan(snapshot(), train | validation)
    assert result["current_group"]
    assert all("_frozen_test_b03_" in name for name in result["current_group"])


def test_worker_and_spark_states_are_distinct(tmp_path, monkeypatch) -> None:
    first, second = SCHEDULER.QUEUE[:2]
    monkeypatch.setattr(SCHEDULER, "CASES", tmp_path)
    state = snapshot(default="NOT_STAGED")
    (tmp_path / second).mkdir()
    (tmp_path / second / "case_config.json").write_text("{}")
    enriched = SCHEDULER.enrich_snapshot(state)
    assert enriched["cases"][first]["status"] == "NOT_GENERATED"
    assert enriched["cases"][second]["status"] == "SPARK_GENERATED"


def test_transfer_precedes_start_and_process_filter_excludes_python() -> None:
    complete, active = SCHEDULER.QUEUE[:2]
    result = SCHEDULER.plan(
        snapshot(**{complete: "COMPLETED", active: "ACQUIRED"}), set()
    )
    assert result["transfer"] == [complete]
    assert result["start"] == []
    assert not SCHEDULER.is_case_process(
        "python3", f"python3 -c pimpleFoam {complete}", complete
    )
    assert SCHEDULER.is_case_process(
        "bash", f"bash run_matched_start_full40_case.sh {complete}", complete
    )


@pytest.mark.parametrize(
    ("mem_gib", "worker_disk_gib", "spark_disk_gib", "expected"),
    [
        (63, 200, 300, "worker_mem_available"),
        (80, 99, 300, "worker_free_disk"),
        (80, 200, 249, "spark_free_disk"),
    ],
)
def test_start_resource_guards(mem_gib, worker_disk_gib, spark_disk_gib, expected):
    state = snapshot()
    state["mem_available_kib"] = mem_gib * 1024 * 1024
    state["disk_free_bytes"] = worker_disk_gib * 1024**3
    failures = SCHEDULER.resource_gate(
        state, spark_disk_gib * 1024**3, starting=True
    )
    assert expected in failures


def test_authorization_builder_refuses_absent_aggregate(tmp_path):
    with pytest.raises(FileNotFoundError, match="aggregate QC is absent"):
        AUTH.build_authorization(tmp_path / "missing.json")


def test_qc_config_matches_frozen_predeclaration_and_rejects_old_nine() -> None:
    planned = AUDIT.planned_cases()
    name, row = next(iter(planned.items()))
    config = {
        "case": name,
        "panel": "matched_start_acquisition_full40_v1",
        "split": row["split"],
        "phase_bin": row["phase_bin"],
        "source_restart_time": row["source_restart_time"],
        "source_state_sha256": row["source_state_sha256"],
        "action_target": row["action_target"],
        "action_points": row["action_points"],
        "start_time": row["run_window"][0],
        "end_time": row["run_window"][1],
        "analysis_window": row["analysis_window"],
        "full40_predeclaration_sha256": AUDIT.PREDECLARATION_SHA256,
        "full40_extension_authorization_sha256": "a" * 64,
        "expected_solver_steps": 16000,
    }
    assert AUDIT.validate_case_config(name, config, "a" * 64) == row
    with pytest.raises(ValueError, match="not in the 31-case remainder"):
        AUDIT.validate_case_config(
            "matched_start_acquisition_train_b00_zero", config, "a" * 64
        )


def test_qc_exact_grid_rejects_one_missing_raw_force_sample() -> None:
    time = np.linspace(100.005, 180.0, 16000)
    AUDIT.exact_grid(time, 100.005, 180.0, 0.005)
    with pytest.raises(ValueError, match="incomplete fixed grid"):
        AUDIT.exact_grid(np.delete(time, 100), 100.005, 180.0, 0.005)


def test_mock_one_step_is_transfer_first_and_never_starts(monkeypatch) -> None:
    events = []
    monkeypatch.setattr(SCHEDULER, "transfer_case", lambda name: events.append(("transfer", name)))
    monkeypatch.setattr(SCHEDULER, "generate_case", lambda name: events.append(("generate", name)))
    monkeypatch.setattr(SCHEDULER, "stage_case", lambda name: events.append(("stage", name)))
    monkeypatch.setattr(SCHEDULER, "start_case", lambda name: events.append(("start", name)))
    completed = SCHEDULER.QUEUE[0]
    actions = {
        "stop": False, "failures": [], "transfer": [completed],
        "prepare": [], "start": [SCHEDULER.QUEUE[1]],
    }
    SCHEDULER.execute_actions(actions, set())
    assert events == [("transfer", completed)]


def test_watch_rechecks_until_complete_and_sleeps_between_rounds(monkeypatch) -> None:
    outcomes = iter((False, False, True))
    sleeps = []
    monkeypatch.setattr(SCHEDULER, "iteration", lambda: next(outcomes))
    monkeypatch.setattr(SCHEDULER.time, "sleep", lambda seconds: sleeps.append(seconds))
    assert SCHEDULER.run_iterations(watch=True, interval=15.0) is True
    assert sleeps == [15.0, 15.0]


def test_one_step_never_sleeps_or_repeats(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(SCHEDULER, "iteration", lambda: calls.append("iteration") or False)
    monkeypatch.setattr(
        SCHEDULER.time,
        "sleep",
        lambda seconds: pytest.fail(f"unexpected sleep: {seconds}"),
    )
    assert SCHEDULER.run_iterations(watch=False, interval=15.0) is False
    assert calls == ["iteration"]


def test_action_table_parser_rejects_tampered_u(tmp_path) -> None:
    path = tmp_path / "U"
    path.write_text(
        "rearCylinder\n{\n type rotatingWallVelocity;\n omega table\n"
        "(\n (100 0)\n (100.75 0.75)\n (180 0.75)\n);\n}\n"
    )
    assert AUDIT.parse_rear_omega_table(path)[1] == [100.75, 0.75]
    path.write_text(path.read_text().replace("0.75 0.75", "0.75 -0.75"))
    assert AUDIT.parse_rear_omega_table(path)[1] != [100.75, 0.75]


def test_authorization_template_is_explicitly_non_authorizing() -> None:
    path = (
        REPO
        / "cfd/tandem_cylinders/templates/"
        "matched_start_full40_extension_authorization.template.json"
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["status"] == "TEMPLATE_ONLY_NOT_AUTHORIZED"
    assert data["maximum_parallel_cases"] == 4
