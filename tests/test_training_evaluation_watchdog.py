from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/watch_training_evaluation_state.py"
if not SCRIPT.exists():
    SCRIPT = Path("/tmp/watch_training_evaluation_state.py")
SPEC = importlib.util.spec_from_file_location("training_evaluation_watchdog", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def unit(active="inactive", result="success", error=None):
    return {
        "active_state": active,
        "sub_state": "running" if active == "active" else "dead",
        "result": result,
        "exec_main_status": "1" if result == "exit-code" else "0",
        "last_error_line": error,
    }


RESOURCES = {
    "cpu_utilization_pct": 3.0,
    "cpu_counters": [1, 2, 3, 4, 5],
    "gpu_utilization_pct": 0.0,
    "gpu_temperature_c": 40.0,
    "gpu_power_w": 10.0,
    "mem_available_gib": 100.0,
    "mem_total_gib": 120.0,
}


class TrainingEvaluationWatchdogTests(unittest.TestCase):
    def test_specific_exception_wins_over_later_systemd_failure_line(self) -> None:
        lines = [
            "runner: ValueError: best files differ from their checkpoint generation",
            "systemd: service failed with result exit-code",
        ]
        self.assertEqual(MODULE.most_specific_error(lines), lines[0])

    def make_repo(self, directory: str, *, receipts=False) -> Path:
        repo = Path(directory)
        for run in (MODULE.MAIN_RUN, MODULE.BALANCED_RUN):
            root = repo / run
            root.mkdir(parents=True)
            (root / "training_history.json").write_text(
                json.dumps([{"epoch": 1}, {"epoch": 2}]), encoding="utf-8"
            )
            if receipts:
                target = root / "posteval_complete/receipt.json"
                target.parent.mkdir()
                target.write_text(
                    json.dumps({"status": "CONTROL_TRAIN16_POSTEVAL_COMPLETE"}),
                    encoding="utf-8",
                )
        return repo

    def test_failed_posteval_is_immediate_explicit_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = self.make_repo(directory)
            result = MODULE.build_sample(
                repo,
                None,
                {
                    MODULE.TRAINING_UNIT: unit(),
                    MODULE.POSTEVAL_UNIT: unit(
                        "failed", "exit-code", "ValueError: lineage differs"
                    ),
                },
                RESOURCES,
                datetime(2026, 10, 4, 1, 0, tzinfo=UTC),
            )
        self.assertEqual(result["status"], "ALERT")
        self.assertIn("TRAIN16_POSTEVAL_FAILED_WITH_PENDING_WORK", result["alerts"])
        self.assertIn("ValueError: lineage differs", result["blocker_reasons"][0])
        self.assertFalse(result["policy"]["automatic_restart_or_repair"])

    def test_pending_without_running_alerts_after_five_minutes(self) -> None:
        now = datetime(2026, 10, 4, 1, 10, tzinfo=UTC)
        with tempfile.TemporaryDirectory() as directory:
            repo = self.make_repo(directory)
            result = MODULE.build_sample(
                repo,
                {"no_running_since_utc": (now - timedelta(seconds=301)).isoformat()},
                {MODULE.TRAINING_UNIT: unit(), MODULE.POSTEVAL_UNIT: unit()},
                RESOURCES,
                now,
            )
        self.assertIn(
            "TRAIN16_PENDING_WITH_NO_RUNNING_UNIT_FOR_300_SECONDS", result["alerts"]
        )
        self.assertEqual(result["no_running_duration_seconds"], 301)

    def test_active_recovery_unit_suppresses_operational_idle_alert(self) -> None:
        now = datetime(2026, 10, 4, 1, 10, tzinfo=UTC)
        with tempfile.TemporaryDirectory() as directory:
            repo = self.make_repo(directory)
            result = MODULE.build_sample(
                repo,
                {"no_running_since_utc": (now - timedelta(hours=1)).isoformat()},
                {
                    MODULE.TRAINING_UNIT: unit(),
                    MODULE.POSTEVAL_UNIT: unit("failed", "exit-code"),
                    "fluid-control-train16-posteval-recovery.service": unit("active"),
                },
                RESOURCES,
                now,
            )
        self.assertEqual(result["status"], "MONITORING")
        self.assertEqual(result["active_units"], ["fluid-control-train16-posteval-recovery.service"])
        self.assertEqual(result["no_running_duration_seconds"], 0)

    def test_complete_receipts_end_operational_pending_without_science_claim(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = self.make_repo(directory, receipts=True)
            result = MODULE.build_sample(
                repo,
                None,
                {MODULE.POSTEVAL_UNIT: unit("failed", "exit-code")},
                RESOURCES,
                datetime(2026, 10, 4, 1, 0, tzinfo=UTC),
            )
        self.assertFalse(result["workflow_pending"])
        self.assertEqual(result["alerts"], [])
        self.assertFalse(result["progress"]["scientific_gate_bypassed"])

    def test_low_memory_is_separate_alert(self) -> None:
        resources = dict(RESOURCES, mem_available_gib=19.9)
        with tempfile.TemporaryDirectory() as directory:
            repo = self.make_repo(directory)
            result = MODULE.build_sample(
                repo,
                None,
                {MODULE.POSTEVAL_UNIT: unit("active")},
                resources,
                datetime(2026, 10, 4, 1, 0, tzinfo=UTC),
            )
        self.assertIn("SPARK_MEMORY_BELOW_20_GIB", result["alerts"])


if __name__ == "__main__":
    unittest.main()
