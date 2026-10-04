from __future__ import annotations

import importlib.util
import hashlib
import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/watch_training_evaluation_state.py"
assert SCRIPT.exists(), "watchdog implementation must be tracked in the repository"
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
    def test_reviewed_main_transient_is_retry_eligible(self) -> None:
        with mock.patch.object(MODULE, "PRODUCTION_AUTO_RECOVERY_ENABLED", True):
            state, action = MODULE.classify_authority_task(
                unit(
                    "failed",
                    "exit-code",
                    "invalid choice: 'control_train16_development'",
                ),
                False,
                allow_resume=True,
            )
        self.assertEqual(state, "RETRY_ELIGIBLE")
        self.assertEqual(action, MODULE.REVIEWED_MAIN_RESUME_ACTION)

    def test_lineage_failure_is_never_auto_retried(self) -> None:
        state, action = MODULE.classify_authority_task(
            unit("failed", "exit-code", "ValueError: lineage SHA mismatch"),
            False,
            allow_resume=True,
        )
        self.assertEqual(state, "NEEDS_AGENT_ANALYSIS")
        self.assertIsNone(action)

    def test_production_recovery_uses_only_reviewed_action(self) -> None:
        state, action = MODULE.classify_authority_task(
            unit(
                "failed",
                "exit-code",
                "invalid choice: 'control_train16_development'",
            ),
            False,
            allow_resume=True,
        )
        self.assertEqual(state, "RETRY_ELIGIBLE")
        self.assertEqual(action, MODULE.REVIEWED_MAIN_RESUME_ACTION)

    def test_specific_exception_wins_over_later_systemd_failure_line(self) -> None:
        lines = [
            "runner: ValueError: best files differ from their checkpoint generation",
            "systemd: service failed with result exit-code",
        ]
        self.assertEqual(MODULE.most_specific_error(lines), lines[0])

    def test_argparse_contract_error_wins_over_systemd_failure_line(self) -> None:
        lines = [
            "audit.py: error: argument --candidate-kind: invalid choice: control",
            "systemd: service failed with result exit-code",
        ]
        self.assertEqual(MODULE.most_specific_error(lines), lines[0])

    def make_repo(
        self, directory: str, *, receipts=False, paired=False, paired_posteval=False
    ) -> Path:
        repo = Path(directory)
        for run in (MODULE.MAIN_RUN, MODULE.BALANCED_RUN):
            root = repo / run
            root.mkdir(parents=True)
            (root / "training_history.json").write_text(
                json.dumps([{"epoch": 1}, {"epoch": 2}]), encoding="utf-8"
            )
        if receipts:
            targets = (
                (
                    repo / MODULE.MAIN_RECEIPT,
                    "CONTROL_TRAIN16_POSTEVAL_COMPLETE",
                ),
                (
                    repo / MODULE.WORKER_RECEIPT,
                    "CONTROL_TRAIN16_H100_LIFT_BALANCED_WORKER_POSTEVAL_COMPLETE",
                ),
            )
            for target, status in targets:
                target.parent.mkdir(parents=True, exist_ok=True)
                evidence = target.parent / "evidence.json"
                evidence.write_text('{"ok":true}\n', encoding="utf-8")
                target.write_text(
                    json.dumps(
                        {
                            "status": status,
                            "sha256": {
                                "evidence.json": hashlib.sha256(
                                    evidence.read_bytes()
                                ).hexdigest()
                            },
                        }
                    ),
                    encoding="utf-8",
                )
        if paired:
            for root in (MODULE.PAIRED_LAMBDA0_ROOT, MODULE.PAIRED_LAMBDA10_ROOT):
                target = repo / root / "completion_receipt.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                evidence = target.parent / "training_history.json"
                evidence.write_text('[{"epoch":1},{"epoch":2}]\n')
                target.write_text(
                    json.dumps(
                        {
                            "status": MODULE.PAIRED_TRAINING_STATUS,
                            "sha256": {
                                "training_history.json": hashlib.sha256(
                                    evidence.read_bytes()
                                ).hexdigest()
                            },
                        }
                    )
                )
            transfer = repo / MODULE.PAIRED_LAMBDA10_ROOT / "worker_transfer_complete.json"
            completion = transfer.parent / "completion_receipt.json"
            transfer.write_text(
                json.dumps(
                    {
                        "status": MODULE.PAIRED_LAMBDA10_TRANSFER_STATUS,
                        "sha256": {
                            "completion_receipt.json": hashlib.sha256(
                                completion.read_bytes()
                            ).hexdigest()
                        },
                    }
                )
            )
        if paired_posteval:
            for receipt, gate in (
                (
                    MODULE.PAIRED_LAMBDA0_POSTEVAL_RECEIPT,
                    MODULE.PAIRED_LAMBDA0_DEVELOPMENT_GATE,
                ),
                (
                    MODULE.PAIRED_LAMBDA10_POSTEVAL_RECEIPT,
                    MODULE.PAIRED_LAMBDA10_DEVELOPMENT_GATE,
                ),
            ):
                target = repo / receipt
                target.parent.mkdir(parents=True, exist_ok=True)
                evidence = target.parent / "evaluation.json"
                evidence.write_text('{"finite":true}\n')
                gate_path = repo / gate
                gate_path.write_text(
                    json.dumps({"status": MODULE.PAIRED_DEVELOPMENT_FAIL_STATUS})
                )
                target.write_text(
                    json.dumps(
                        {
                            "status": MODULE.PAIRED_POSTEVAL_STATUS,
                            "sha256": {
                                "evaluation.json": hashlib.sha256(
                                    evidence.read_bytes()
                                ).hexdigest(),
                                "development_gate.json": hashlib.sha256(
                                    gate_path.read_bytes()
                                ).hexdigest(),
                            },
                        }
                    )
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
                    MODULE.MAIN_AUTHORITY_UNIT: unit(
                        "failed", "exit-code", "ValueError: lineage differs"
                    ),
                    MODULE.WORKER_AUTHORITY_UNIT: unit("active"),
                },
                RESOURCES,
                datetime(2026, 10, 4, 1, 0, tzinfo=UTC),
            )
        self.assertEqual(result["status"], "ALERT")
        self.assertIn("TRAIN16_POSTEVAL_NEEDS_AGENT_ANALYSIS", result["alerts"])
        self.assertIn("ValueError: lineage differs", result["blocker_reasons"][0])
        self.assertFalse(result["policy"]["automatic_restart_or_repair"])

    def test_pending_without_running_alerts_after_five_minutes(self) -> None:
        now = datetime(2026, 10, 4, 1, 10, tzinfo=UTC)
        with tempfile.TemporaryDirectory() as directory:
            repo = self.make_repo(directory)
            result = MODULE.build_sample(
                repo,
                {"no_running_since_utc": (now - timedelta(seconds=301)).isoformat()},
                {
                    MODULE.TRAINING_UNIT: unit(),
                    MODULE.MAIN_AUTHORITY_UNIT: unit(),
                    MODULE.WORKER_AUTHORITY_UNIT: unit(),
                },
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
                    MODULE.MAIN_AUTHORITY_UNIT: unit("active"),
                    MODULE.WORKER_AUTHORITY_UNIT: unit("active"),
                },
                RESOURCES,
                now,
            )
        self.assertEqual(result["status"], "MONITORING")
        self.assertEqual(
            result["active_units"],
            [MODULE.WORKER_AUTHORITY_UNIT, MODULE.MAIN_AUTHORITY_UNIT],
        )
        self.assertEqual(result["no_running_duration_seconds"], 0)

    def test_complete_receipts_end_operational_pending_without_science_claim(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = self.make_repo(directory, receipts=True, paired=True)
            result = MODULE.build_sample(
                repo,
                None,
                {
                    MODULE.MAIN_AUTHORITY_UNIT: unit("failed", "exit-code"),
                    MODULE.WORKER_AUTHORITY_UNIT: unit("failed", "exit-code"),
                },
                RESOURCES,
                datetime(2026, 10, 4, 1, 0, tzinfo=UTC),
            )
        self.assertTrue(result["workflow_pending"])
        self.assertFalse(result["stage_complete"])
        self.assertTrue(result["prior_posteval_stage_complete"])
        self.assertFalse(result["project_goal_complete"])
        self.assertEqual(result["project_status"], "NEEDS_MODEL_IMPROVEMENT")
        self.assertEqual(
            result["scientific_next_stage"]["status"],
            "PAIRED_POSTEVAL_APPROVED_PREFLIGHT",
        )
        self.assertFalse(
            result["scientific_next_stage"]["automatic_restart_allowed"]
        )
        self.assertTrue(result["progress"]["paired_training_complete"])
        self.assertFalse(result["progress"]["paired_posteval_complete"])
        self.assertEqual(
            result["progress"]["next_owner"], "Surrogate + Physics/Data"
        )
        self.assertIsNone(result["progress"]["approval_required"])
        self.assertEqual(result["progress"]["approval_state"], "LEAD_APPROVED")
        self.assertEqual(
            result["progress"]["approval_reference"], "docs/FC-P001_APPROVAL.md"
        )
        self.assertEqual(
            result["scientific_next_stage"]["active_work"],
            "paired_posteval_approved_preflight",
        )

    def test_paired_posteval_idle_alert_names_owner_and_experiment(self) -> None:
        now = datetime(2026, 10, 4, 14, 0, tzinfo=UTC)
        with tempfile.TemporaryDirectory() as directory:
            repo = self.make_repo(directory, receipts=True, paired=True)
            result = MODULE.build_sample(
                repo,
                {"no_running_since_utc": (now - timedelta(seconds=301)).isoformat()},
                {
                    MODULE.MAIN_AUTHORITY_UNIT: unit(),
                    MODULE.WORKER_AUTHORITY_UNIT: unit(),
                    MODULE.PAIRED_LAMBDA0_UNIT: unit(),
                },
                RESOURCES,
                now,
            )
        self.assertIn(
            "PAIRED_POSTEVAL_APPROVED_WITH_NO_RUNNING_UNIT_FOR_300_SECONDS",
            result["alerts"],
        )
        self.assertIn("FC-P001", result["blocker_reasons"][-1])
        self.assertIn("Lead-approved", result["blocker_reasons"][-1])
        self.assertNotIn("pending Lead approval", result["blocker_reasons"][-1])

    def test_active_paired_units_are_authoritative_running_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = self.make_repo(directory, receipts=True, paired=True)
            result = MODULE.build_sample(
                repo,
                None,
                {
                    MODULE.MAIN_AUTHORITY_UNIT: unit(),
                    MODULE.WORKER_AUTHORITY_UNIT: unit(),
                    MODULE.PAIRED_LAMBDA0_POSTEVAL_UNIT: unit("active"),
                    MODULE.PAIRED_LAMBDA10_POSTEVAL_UNIT: unit("active"),
                },
                RESOURCES,
                datetime(2026, 10, 4, 15, 30, tzinfo=UTC),
            )
        self.assertEqual(
            result["scientific_next_stage"]["status"], "PAIRED_POSTEVAL_RUNNING"
        )
        self.assertEqual(
            result["scientific_next_stage"]["active_work"],
            "paired_posteval_running",
        )
        self.assertIn(MODULE.PAIRED_LAMBDA0_POSTEVAL_UNIT, result["active_units"])
        self.assertIn(MODULE.PAIRED_LAMBDA10_POSTEVAL_UNIT, result["active_units"])
        self.assertEqual(result["alerts"], [])
        review = result["scientific_next_stage"]["protocol_review"]
        self.assertEqual(
            review["status"], "LAMBDA0_LAMBDA10_PROTOCOL_MATCH_VERIFIED"
        )
        self.assertEqual(review["validation10_expected_segments"]["H100"], 290)
        self.assertEqual(review["dynamic6_expected_segments"]["all"], 3858)
        self.assertFalse(review["training_epoch_metrics_used_for_verdict"])
        self.assertFalse(review["frozen_test_accessed"])

    def test_both_strict_paired_receipts_complete_stage_not_project(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = self.make_repo(
                directory, receipts=True, paired=True, paired_posteval=True
            )
            result = MODULE.build_sample(
                repo,
                None,
                {
                    MODULE.MAIN_AUTHORITY_UNIT: unit(),
                    MODULE.WORKER_AUTHORITY_UNIT: unit(),
                    MODULE.PAIRED_LAMBDA0_POSTEVAL_UNIT: unit(),
                    MODULE.PAIRED_LAMBDA10_POSTEVAL_UNIT: unit(),
                },
                RESOURCES,
                datetime(2026, 10, 4, 16, 30, tzinfo=UTC),
            )
        self.assertTrue(result["stage_complete"])
        self.assertFalse(result["workflow_pending"])
        self.assertFalse(result["project_goal_complete"])
        self.assertEqual(
            result["scientific_next_stage"]["status"],
            "PAIRED_POSTEVAL_COMPLETE_SCIENTIFIC_REJECTED",
        )
        self.assertEqual(
            result["scientific_next_stage"]["active_work"], "fc_p002_failure_map"
        )
        self.assertEqual(
            result["scientific_next_stage"]["fc_p001_verdict"]["status"],
            "FC_P001_SCIENTIFIC_REJECTED",
        )
        self.assertEqual(
            result["scientific_next_stage"]["lambda0"]["state"],
            "TRAINING_AND_POSTEVAL_STAGE_COMPLETE",
        )
        self.assertEqual(
            result["scientific_next_stage"]["lambda10"]["state"],
            "TRAINING_AND_POSTEVAL_STAGE_COMPLETE",
        )

    def test_latest_active_lambda0_generation_becomes_authority(self) -> None:
        units = {
            "fluid-control-paired-lambda0-posteval-fcp001-20261004.service": unit(
                "failed", "exit-code"
            ),
            MODULE.PAIRED_LAMBDA0_POSTEVAL_UNIT: unit("active"),
        }
        self.assertEqual(
            MODULE.select_versioned_authority(
                units,
                MODULE.PAIRED_LAMBDA0_POSTEVAL_PREFIX,
                MODULE.PAIRED_LAMBDA0_POSTEVAL_UNIT,
            ),
            MODULE.PAIRED_LAMBDA0_POSTEVAL_UNIT,
        )

    def test_paired_lambda0_active_is_current_scientific_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = self.make_repo(directory, receipts=True)
            result = MODULE.build_sample(
                repo,
                None,
                {
                    MODULE.MAIN_AUTHORITY_UNIT: unit(),
                    MODULE.WORKER_AUTHORITY_UNIT: unit(),
                    MODULE.PAIRED_LAMBDA0_UNIT: unit("active"),
                },
                RESOURCES,
                datetime(2026, 10, 4, 2, 0, tzinfo=UTC),
            )
        next_stage = result["scientific_next_stage"]
        self.assertEqual(next_stage["status"], "PAIRED_STATS_CONTROLLED_TRAINING_RUNNING")
        self.assertEqual(next_stage["lambda0"]["state"], "RUNNING")
        self.assertFalse(result["project_goal_complete"])
        self.assertEqual(result["alerts"], [])
        self.assertFalse(result["progress"]["scientific_gate_bypassed"])

    def test_low_memory_is_separate_alert(self) -> None:
        resources = dict(RESOURCES, mem_available_gib=19.9)
        with tempfile.TemporaryDirectory() as directory:
            repo = self.make_repo(directory)
            result = MODULE.build_sample(
                repo,
                None,
                {
                    MODULE.MAIN_AUTHORITY_UNIT: unit("active"),
                    MODULE.WORKER_AUTHORITY_UNIT: unit("active"),
                },
                resources,
                datetime(2026, 10, 4, 1, 0, tzinfo=UTC),
            )
        self.assertIn("SPARK_MEMORY_BELOW_20_GIB", result["alerts"])

    def test_superseded_failure_is_audit_only_while_v3_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = self.make_repo(directory)
            result = MODULE.build_sample(
                repo,
                None,
                {
                    MODULE.MAIN_AUTHORITY_UNIT: unit("active"),
                    MODULE.WORKER_AUTHORITY_UNIT: unit("active"),
                    MODULE.SUPERSEDED_MAIN_UNITS[0]: unit("failed", "exit-code"),
                },
                RESOURCES,
                datetime(2026, 10, 4, 1, 0, tzinfo=UTC),
            )
        self.assertEqual(result["status"], "MONITORING")
        self.assertEqual(result["failed_units"], [])
        self.assertEqual(
            result["superseded_failures"], [MODULE.SUPERSEDED_MAIN_UNITS[0]]
        )

    def test_new_active_generation_becomes_authority(self) -> None:
        units = {
            MODULE.MAIN_AUTHORITY_UNIT: unit("failed", "exit-code"),
            "fluid-control-train16-posteval-main-v4-20261004.service": unit("active"),
        }
        self.assertEqual(
            MODULE.select_main_authority(units),
            "fluid-control-train16-posteval-main-v4-20261004.service",
        )

    def test_receipt_payload_tamper_prevents_stage_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = self.make_repo(directory, receipts=True)
            (repo / MODULE.MAIN_RECEIPT.parent / "evidence.json").write_text("tampered")
            progress = MODULE.workflow_progress(repo)
        self.assertFalse(progress["main_posteval_complete"])
        self.assertIn("payload SHA differs: evidence.json", progress["main_receipt_issues"])


if __name__ == "__main__":
    unittest.main()
