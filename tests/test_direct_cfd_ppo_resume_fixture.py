"""Small CPU-only SB3 persistence fixture; this is not CFD evidence."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_direct_cfd_ppo import reset_resume_exploration_std  # noqa: E402


class FixtureEnv(gym.Env):
    observation_space = gym.spaces.Box(-10.0, 10.0, shape=(2,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def __init__(self) -> None:
        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        self.step_count += 1
        value = float(np.asarray(action).reshape(-1)[0])
        observation = np.asarray([self.step_count / 10.0, value], dtype=np.float32)
        reward = 1.0 - value * value
        return observation, reward, False, self.step_count >= 8, {}


def make_normalized_env() -> VecNormalize:
    return VecNormalize(
        DummyVecEnv([FixtureEnv]), training=True, norm_obs=True, norm_reward=True
    )


class DirectCFDPPOResumeFixtureTest(unittest.TestCase):
    def test_log_std_reset_changes_only_exploration_and_preserves_optimizer(self) -> None:
        env = make_normalized_env()
        model = PPO(
            "MlpPolicy",
            env,
            seed=23,
            device="cpu",
            n_steps=8,
            batch_size=4,
            n_epochs=2,
            verbose=0,
        )
        model.learn(total_timesteps=16)
        observations = np.asarray([[0.0, 0.0], [0.25, -0.5]], dtype=np.float32)
        deterministic_before, _ = model.predict(observations, deterministic=True)
        parameters_before = {
            name: value.detach().cpu().clone()
            for name, value in model.policy.named_parameters()
        }
        log_std_state_before = {
            key: value.detach().cpu().clone() if hasattr(value, "detach") else value
            for key, value in model.policy.optimizer.state[model.policy.log_std].items()
        }

        receipt = reset_resume_exploration_std(model, 0.075)

        deterministic_after, _ = model.predict(observations, deterministic=True)
        np.testing.assert_array_equal(deterministic_before, deterministic_after)
        self.assertEqual(receipt["changed_parameters"], ["log_std"])
        self.assertTrue(receipt["optimizer_state_preserved"])
        self.assertTrue(receipt["log_std_remains_trainable"])
        np.testing.assert_allclose(
            model.policy.log_std.detach().exp().cpu().numpy(), 0.075, rtol=1e-6
        )
        for name, value in model.policy.named_parameters():
            if name != "log_std":
                self.assertTrue(torch.equal(parameters_before[name], value.detach().cpu()))
        log_std_state_after = model.policy.optimizer.state[model.policy.log_std]
        for key, value in log_std_state_before.items():
            if hasattr(value, "detach"):
                self.assertTrue(torch.equal(value, log_std_state_after[key].detach().cpu()))
            else:
                self.assertEqual(value, log_std_state_after[key])
        env.close()

    def test_archive_resume_preserves_and_advances_training_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = make_normalized_env()
            model = PPO(
                "MlpPolicy",
                env,
                seed=17,
                device="cpu",
                n_steps=8,
                batch_size=4,
                n_epochs=2,
                verbose=0,
            )
            model.learn(total_timesteps=16)
            model.save(root / "policy")
            env.save(root / "vec.pkl")
            prior_timesteps = model.num_timesteps
            prior_updates = model._n_updates
            prior_count = float(env.obs_rms.count)
            prior_optimizer_entries = len(model.policy.optimizer.state)
            self.assertGreater(prior_optimizer_entries, 0)
            env.close()

            resumed_env = VecNormalize.load(str(root / "vec.pkl"), DummyVecEnv([FixtureEnv]))
            resumed_env.training = True
            resumed_env.norm_reward = True
            resumed = PPO.load(root / "policy.zip", env=resumed_env, device="cpu")
            self.assertEqual(resumed.num_timesteps, prior_timesteps)
            self.assertEqual(resumed._n_updates, prior_updates)
            self.assertEqual(len(resumed.policy.optimizer.state), prior_optimizer_entries)
            self.assertAlmostEqual(float(resumed_env.obs_rms.count), prior_count)

            resumed.learn(total_timesteps=16, reset_num_timesteps=False)
            self.assertEqual(resumed.num_timesteps, prior_timesteps + 16)
            self.assertGreater(resumed._n_updates, prior_updates)
            self.assertGreater(float(resumed_env.obs_rms.count), prior_count)
            self.assertEqual(len(resumed.policy.optimizer.state), prior_optimizer_entries)
            resumed_env.close()


if __name__ == "__main__":
    unittest.main()
