"""Pure scheduling-state tests for the matched-start worker queue."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def load_module():
    path = Path(__file__).parents[1] / "scripts/schedule_matched_start_worker.py"
    spec = importlib.util.spec_from_file_location("matched_start_scheduler", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCHEDULER = load_module()


def snapshot(default: str = "PENDING", **overrides: str) -> dict:
    states = {name: {"status": default} for name in SCHEDULER.QUEUE}
    for name, status in overrides.items():
        states[name] = {"status": status}
    return {"cases": states, "mem_available_kib": 100 * 1024 * 1024, "disk_free_bytes": 10**12}


def test_adopts_two_running_cases_without_restarting() -> None:
    first, second = SCHEDULER.QUEUE[:2]
    result = SCHEDULER.plan(snapshot(**{first: "ACQUIRED", second: "ACQUIRED"}), set())
    assert result["active"] == 2
    assert result["start"] == []


def test_completed_case_is_transferred_before_filling_slot() -> None:
    first, second = SCHEDULER.QUEUE[:2]
    result = SCHEDULER.plan(snapshot(**{first: "COMPLETED", second: "ACQUIRED"}), set())
    assert result["transfer"] == [first]
    assert result["start"] == []


def test_failure_stops_every_new_launch() -> None:
    first = SCHEDULER.QUEUE[0]
    result = SCHEDULER.plan(snapshot(**{first: "FAILED"}), set())
    assert result["stop"]
    assert result["start"] == []


def test_orphaned_lock_stops_every_new_launch() -> None:
    first = SCHEDULER.QUEUE[0]
    result = SCHEDULER.plan(snapshot(**{first: "ORPHANED"}), set())
    assert result["stop"]
    assert result["failures"] == [first]


def test_process_filter_excludes_snapshot_python_and_accepts_real_solver() -> None:
    name = SCHEDULER.QUEUE[0]
    python_command = f"python3 -c code containing pimpleFoam {name}"
    assert not SCHEDULER.is_case_process("python3", python_command, name)
    assert SCHEDULER.is_case_process("pimpleFoam", f"pimpleFoam -case /case/{name}", name)
    assert SCHEDULER.is_case_process(
        "bash", f"bash run_matched_start_acquisition_case.sh {name}", name
    )


def test_never_plans_more_than_two_slots() -> None:
    result = SCHEDULER.plan(snapshot(), set())
    assert result["start"] == list(SCHEDULER.QUEUE[:2])
