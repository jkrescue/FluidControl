from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/reconcile_training_evaluation_state.py"
)
SPEC = importlib.util.spec_from_file_location("training_evaluation_reconciler", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def transient_sample(memory=100.0):
    return {
        "stage_complete": False,
        "active_units": [],
        "resources": {"mem_available_gib": memory},
        "authority_tasks": {
            "main_posteval": {
                "state": "RETRY_ELIGIBLE",
                "approved_action_id": "reviewed-resume-v1",
            }
        },
    }


class TrainingEvaluationReconcilerTests(unittest.TestCase):
    def test_transient_resume_then_success_is_finite(self) -> None:
        ledger = {"attempts": {}, "executions": []}
        actions = {
            "reviewed-resume-v1": {"command": ["/bin/true"], "max_attempts": 1}
        }
        decision = MODULE.plan_recovery(transient_sample(), ledger, actions)
        self.assertEqual(decision["decision"], "EXECUTE_REVIEWED_RESUME")
        with tempfile.TemporaryDirectory() as directory:
            result = MODULE.execute_reviewed_resume(
                decision, ledger, actions, Path(directory)
            )
        self.assertEqual(result["decision"], "REVIEWED_RESUME_DISPATCHED")
        self.assertEqual(ledger["attempts"]["reviewed-resume-v1"], 1)
        complete = dict(transient_sample(), stage_complete=True)
        self.assertEqual(
            MODULE.plan_recovery(complete, ledger, actions)["decision"],
            "NO_ACTION_STAGE_COMPLETE",
        )

    def test_duplicate_lock_refuses_second_dispatch(self) -> None:
        actions = {"reviewed-resume-v1": {"command": ["/bin/true"]}}
        decision = MODULE.plan_recovery(
            transient_sample(), {"attempts": {}, "executions": []}, actions
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "reviewed-resume-v1.lock").write_text("held\n")
            result = MODULE.execute_reviewed_resume(
                decision, {"attempts": {}, "executions": []}, actions, output
            )
        self.assertEqual(result["decision"], "NO_ACTION_DUPLICATE_LOCK")

    def test_memory_floor_blocks_dispatch(self) -> None:
        actions = {"reviewed-resume-v1": {"command": ["/bin/true"]}}
        result = MODULE.plan_recovery(
            transient_sample(memory=19.99),
            {"attempts": {}, "executions": []},
            actions,
        )
        self.assertEqual(result["decision"], "RESOURCE_BLOCKED")

    def test_unknown_failure_requires_agent_analysis(self) -> None:
        sample = transient_sample()
        sample["authority_tasks"]["main_posteval"]["state"] = (
            "NEEDS_AGENT_ANALYSIS"
        )
        result = MODULE.plan_recovery(
            sample, {"attempts": {}, "executions": []}, {}
        )
        self.assertEqual(result["decision"], "NEEDS_AGENT_ANALYSIS")

    def test_unknown_worker_failure_is_not_hidden_by_other_active_task(self) -> None:
        sample = transient_sample()
        sample["active_units"] = ["main-v3.service"]
        sample["authority_tasks"]["worker"] = {"state": "NEEDS_AGENT_ANALYSIS"}
        result = MODULE.plan_recovery(
            sample, {"attempts": {}, "executions": []}, {}
        )
        self.assertEqual(result["decision"], "NEEDS_AGENT_ANALYSIS")

    def test_retry_budget_prevents_infinite_repeat(self) -> None:
        actions = {
            "reviewed-resume-v1": {"command": ["/bin/true"], "max_attempts": 1}
        }
        result = MODULE.plan_recovery(
            transient_sample(),
            {"attempts": {"reviewed-resume-v1": 1}, "executions": []},
            actions,
        )
        self.assertEqual(result["decision"], "NEEDS_AGENT_ANALYSIS")

    def test_reviewed_script_sha_mismatch_refuses_dispatch(self) -> None:
        ledger = {"attempts": {}, "executions": []}
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "resume.sh"
            script.write_text("#!/bin/sh\nexit 0\n")
            actions = {
                "reviewed-resume-v1": {
                    "command": ["/bin/true"],
                    "script_path": str(script),
                    "script_sha256": "0" * 64,
                }
            }
            decision = MODULE.plan_recovery(transient_sample(), ledger, actions)
            result = MODULE.execute_reviewed_resume(
                decision, ledger, actions, Path(directory) / "state"
            )
        self.assertEqual(result["decision"], "NEEDS_AGENT_ANALYSIS")
        self.assertEqual(ledger["attempts"], {})


if __name__ == "__main__":
    unittest.main()
