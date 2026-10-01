"""Unit tests for the guarded real-CFD PPO policy inference boundary."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_tandem_ppo_cfd_feedback import PROJECT, infer_action, objective  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
