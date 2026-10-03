"""Tests for the disabled-by-default full40 extension execution stack."""

from __future__ import annotations

import importlib.util
import json
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


def test_queue_is_exact_31_and_execution_is_unbound() -> None:
    assert len(SCHEDULER.QUEUE) == 31
    assert len(SCHEDULER.AUTHORIZATION_SHA256) != 64
    report = SCHEDULER.dry_run_report()
    assert report["execution_enabled"] is False
    assert report["maximum_parallel_cases"] == 4


def test_scheduler_never_plans_more_than_four_and_fail_stops() -> None:
    result = SCHEDULER.plan(snapshot(), set())
    assert len(result["start"]) == 4
    failed = SCHEDULER.QUEUE[0]
    result = SCHEDULER.plan(snapshot(**{failed: "FAILED"}), set())
    assert result["stop"] is True
    assert result["start"] == []


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


def test_authorization_template_is_explicitly_non_authorizing() -> None:
    path = (
        REPO
        / "cfd/tandem_cylinders/templates/"
        "matched_start_full40_extension_authorization.template.json"
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["status"] == "TEMPLATE_ONLY_NOT_AUTHORIZED"
    assert data["maximum_parallel_cases"] == 4
