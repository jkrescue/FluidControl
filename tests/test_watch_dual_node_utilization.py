import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/watch_dual_node_utilization.py"
SPEC = importlib.util.spec_from_file_location("dual_watchdog", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_cpu_utilization_uses_counter_delta():
    previous = [100, 0, 100, 700, 100, 0, 0, 0, 0, 0]
    current = [150, 0, 150, 750, 150, 0, 0, 0, 0, 0]
    assert MODULE.cpu_utilization(previous, current) == pytest.approx(50.0)
    assert MODULE.cpu_utilization(None, current) is None


def test_compute_process_filter_does_not_count_scheduler_itself():
    text = """1 100 0.1 python schedule_matched_start_full40_extension.py
2 30 99.0 pimpleFoam -case /case/a
3 20 250.0 python curate_low_action_phase94_validation.py --cases b
"""
    rows = MODULE.parse_compute_processes(text)
    assert [row["pid"] for row in rows] == [2, 3]


def test_idle_alert_state_persists_across_samples():
    now = datetime(2026, 10, 3, 0, 10, tzinfo=UTC)
    node = {"reachable": True, "useful_compute_active": False}
    previous = {"idle_since_utc": "2026-10-03T00:00:00+00:00"}
    result = MODULE.update_idle_state(node, True, previous, now)
    assert result["idle_duration_seconds"] == 600
    assert result["idle_since_utc"] == previous["idle_since_utc"]


def test_active_compute_or_completed_task_resets_idle():
    now = datetime(2026, 10, 3, tzinfo=UTC)
    active = {"reachable": True, "useful_compute_active": True}
    result = MODULE.update_idle_state(active, True, {"idle_since_utc": "old"}, now)
    assert result["idle_since_utc"] is None
    idle = {"reachable": True, "useful_compute_active": False}
    result = MODULE.update_idle_state(idle, False, {"idle_since_utc": "old"}, now)
    assert result["idle_since_utc"] is None


def test_project_progress_counts_receipts_without_reading_case_data(tmp_path):
    receipt_dir = tmp_path / "artifacts/matched_start_full40_extension/transfer_verified"
    receipt_dir.mkdir(parents=True)
    for index in range(3):
        (receipt_dir / f"{index}.json").write_text("{}")
    aggregate = tmp_path / "artifacts/matched_start_full40_extension/aggregate_qc"
    aggregate.mkdir(parents=True)
    (aggregate / "result.json").write_text(
        json.dumps({"status": MODULE.RAW_AGGREGATE_STATUS})
    )
    result = MODULE.project_progress(tmp_path)
    assert result["strict_full40_receipt_count"] == 3
    assert result["raw_31_case_aggregate_complete"] is True
    assert result["project_complete"] is False


def test_full40_data_completion_alone_does_not_complete_research(tmp_path):
    final = tmp_path / "data/curated/tandem_cylinders_matched_start_full40_v1"
    final.mkdir(parents=True)
    (final / "manifest.json").write_text(
        json.dumps(
            {
                "profile": "matched_start_full40_v1",
                "trajectory_counts": {
                    "train": 20,
                    "validation": 10,
                    "frozen_test": 10,
                },
            }
        )
    )
    result = MODULE.project_progress(tmp_path)
    assert result["full40_final_manifest_complete"] is True
    assert result["project_complete"] is False
    assert all(
        row["passed"] is False for row in result["formal_research_gates"].values()
    )

    for relative, status in MODULE.FORMAL_COMPLETION_GATES.values():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"status": status}))
    assert MODULE.project_progress(tmp_path)["project_complete"] is True


def test_worker_assignment_continues_after_raw_aggregate_until_full40_release():
    progress = {
        "raw_31_case_aggregate_complete": True,
        "full40_remainder_staging_hdf5_count": 0,
        "full40_final_manifest_complete": False,
    }
    assert MODULE.worker_assignment_incomplete(progress) is True

    progress["full40_remainder_staging_hdf5_count"] = 31
    assert MODULE.worker_assignment_incomplete(progress) is True

    progress["full40_final_manifest_complete"] = True
    assert MODULE.worker_assignment_incomplete(progress) is False
