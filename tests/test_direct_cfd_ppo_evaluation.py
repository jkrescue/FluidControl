from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_direct_cfd_ppo_pair.py"
SPEC = importlib.util.spec_from_file_location("direct_cfd_eval", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

HOST_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_direct_cfd_ppo_evaluation.py"
)
HOST_SPEC = importlib.util.spec_from_file_location("direct_cfd_eval_host", HOST_SCRIPT)
assert HOST_SPEC is not None and HOST_SPEC.loader is not None
HOST_MODULE = importlib.util.module_from_spec(HOST_SPEC)
HOST_SPEC.loader.exec_module(HOST_MODULE)


class FakeNormalize:
    training = True
    norm_reward = True


class FakePolicy:
    def __init__(self) -> None:
        self.deterministic = None

    def predict(self, observation, *, deterministic):
        self.deterministic = deterministic
        return np.asarray([[0.25]], dtype=np.float32), None


class TensorPolicy:
    def __init__(self) -> None:
        self.policy = torch.nn.Linear(2, 1)


class FakeRMS:
    mean = np.zeros(69)
    var = np.ones(69)
    count = 256.0


class FakeVecNormalize:
    obs_rms = FakeRMS()


class DirectCFDEvaluationTest(unittest.TestCase):
    def test_normalization_is_frozen_and_reward_is_physical(self) -> None:
        env = FakeNormalize()
        self.assertIs(MODULE.configure_frozen_normalization(env), env)
        self.assertFalse(env.training)
        self.assertFalse(env.norm_reward)

    def test_policy_inference_is_deterministic(self) -> None:
        policy = FakePolicy()
        action = MODULE.deterministic_policy_action(policy, np.zeros((1, 69)))
        self.assertTrue(policy.deterministic)
        np.testing.assert_array_equal(action, [[0.25]])

    def test_parameter_and_observation_fingerprints_detect_mutation(self) -> None:
        policy = TensorPolicy()
        before = MODULE.policy_tensor_sha256(policy)
        with torch.no_grad():
            policy.policy.weight.add_(1.0)
        self.assertNotEqual(before, MODULE.policy_tensor_sha256(policy))
        fingerprint = MODULE.observation_statistics_fingerprint(FakeVecNormalize())
        self.assertEqual(fingerprint["count"], 256.0)

    def test_evaluation_horizon_is_full_80d_u(self) -> None:
        self.assertEqual(MODULE.EVALUATION_STEPS, 800)

    def test_paired_rollout_contract_checks_time_action_and_zero(self) -> None:
        rows = []
        for step in range(1, 801):
            for role in ("ppo", "zero"):
                rows.append(
                    {
                        "step": step,
                        "role": role,
                        "cfd_time": 148.0 + 0.1 * step,
                        "applied_omega": 0.0,
                        "applied_delta_omega": 0.0,
                    }
                )
        rollout = {"branches": {"ppo": "p", "zero": "z"}, "rows": rows}
        HOST_MODULE.validate_rollout_contract(rollout)
        rows[-1]["applied_omega"] = 0.01
        with self.assertRaisesRegex(ValueError, "zero branch"):
            HOST_MODULE.validate_rollout_contract(rollout)


if __name__ == "__main__":
    unittest.main()
