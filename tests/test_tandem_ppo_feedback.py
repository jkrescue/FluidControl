"""Unit tests for the guarded real-CFD PPO policy inference boundary."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_tandem_ppo_cfd_feedback import (  # noqa: E402
    HYDROGYM_COMMIT,
    PROJECT,
    infer_action,
    objective,
    validate_policy_audit,
)


class PPOFeedbackBoundaryTests(unittest.TestCase):
    def test_policy_marker_and_isolated_cpu_container(self) -> None:
        policy = PROJECT / "artifacts" / "hydrogym" / "test" / "ppo_policy.zip"
        completed = type("Result", (), {
            "returncode": 0,
            "stdout": "banner\nPOLICY_ACTION_JSON=" + json.dumps({
                "requested_omega": 0.25,
                "observation_channels": 67,
                "deterministic": True,
            }) + "\n",
            "stderr": "",
        })()
        with patch("run_tandem_ppo_cfd_feedback.subprocess.run", return_value=completed) as run:
            requested, _ = infer_action(policy, np.zeros(67, dtype=np.float32))
        self.assertEqual(requested, 0.25)
        command = run.call_args.args[0]
        self.assertIn("--network", command)
        self.assertIn("none", command)
        self.assertIn("--read-only", command)
        self.assertNotIn("--gpus", command)

    def test_invalid_observation_fails_before_docker(self) -> None:
        policy = PROJECT / "artifacts" / "hydrogym" / "test" / "ppo_policy.zip"
        with patch("run_tandem_ppo_cfd_feedback.subprocess.run") as run:
            with self.assertRaises(ValueError):
                infer_action(policy, np.zeros(66, dtype=np.float32))
            with self.assertRaises(ValueError):
                infer_action(policy, np.full(67, np.nan, dtype=np.float32))
            run.assert_not_called()

    def test_objective_matches_hydrogym_weights(self) -> None:
        self.assertAlmostEqual(objective(1.0, 2.0, 0.5, -0.5), 1.80275)

    @staticmethod
    def passing_audit(t80_reward: float = 0.01) -> dict:
        return {
            "status": "SURROGATE_RL_PILOT_EVALUATED",
            "physicsnemo_checkpoint_epoch": 20,
            "hydrogym_commit": HYDROGYM_COMMIT,
            "summary_weighting": "common_t80_frame0_counted_once_per_split",
            "training_environments": 8,
            "summary": {
                split: {
                    "unique_initial_states": 9,
                    "reward_change_mean": 0.02,
                    "positive_reward_initial_states": 7,
                }
                for split in ("validation", "test")
            },
            "evaluations": [
                {"split": split, "case": f"expanded_{split}_00", "initial_frame": 0,
                 "reward_change": t80_reward}
                for split in ("validation", "test")
            ],
        }

    def test_policy_audit_requires_positive_shared_t80_start(self) -> None:
        with self.assertRaisesRegex(ValueError, "t=80-specific"):
            validate_policy_audit(self.passing_audit(t80_reward=-0.003))

    def test_policy_audit_accepts_broad_and_t80_gates(self) -> None:
        result = validate_policy_audit(self.passing_audit())
        self.assertEqual(result["shared_t80_rows"], 2)
        self.assertEqual(result["shared_t80_reward_change"], 0.01)


if __name__ == "__main__":
    unittest.main()
