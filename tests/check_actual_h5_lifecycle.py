"""Actual HydroGym+SB3 lifecycle, synthetic PDE/policy; no FNO/data/optimizer/CUDA.

Run only under separately approved CPU runtime smoke. This is not PPO training.
"""

import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from hydrogym import FlowEnv, PDEBase, TransientSolver
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv

from exploratory_h5_hydrogym import wrap_exploratory_h5


class SyntheticFlow(PDEBase):
    num_inputs = 1
    num_outputs = 69
    MAX_CONTROL = 0.75
    initial_cfd_time = 148.0
    frame = 0
    max_abs_normalized_state_guard = 10.0
    case_path = Path("synthetic_train_fixture")

    def load_mesh(self, name):
        return None

    def initialize_state(self):
        self.q = torch.zeros(1)

    def init_bcs(self):
        pass

    def copy_state(self, deepcopy=True):
        return self.q.clone()

    def reset(self, q0=None, t=0.0):
        self.q = torch.zeros(1)
        self.t = self.initial_cfd_time
        self.force = np.zeros(4)
        self.omega = self.requested_action = self.applied_delta = 0.0
        self.rate_limited = False
        self._reward_history = deque(
            (148.0 - 0.1 * (61 - i), np.zeros(4)) for i in range(62)
        )

    def save_checkpoint(self, filename):
        raise AssertionError("no save")

    def load_checkpoint(self, filename):
        raise AssertionError("no load")

    def render(self, **kwargs):
        raise AssertionError("no rendering")

    def get_observations(self):
        result = np.zeros(69, dtype=np.float32)
        result[0] = float(self.q[0])
        return result

    def objective_terms(self):
        return {"synthetic_constant_cost": 1.0}

    def evaluate_objective(self, q=None):
        return 1.0

    def canonical_ledger(self):
        return {"synthetic_fixture": True, "window_samples": 62}


class SyntheticStepper(TransientSolver):
    def step(self, iteration, control):
        self.flow.q += 1.0
        self.flow.t += self.dt
        self.flow._reward_history.popleft()
        self.flow._reward_history.append((self.flow.t, np.zeros(4)))
        return self.flow


def make_env():
    return wrap_exploratory_h5(
        FlowEnv(
            {
                "flow": SyntheticFlow,
                "flow_config": {},
                "solver": SyntheticStepper,
                "solver_config": {"dt": 0.1},
                "max_steps": 5,
            }
        )
    )


class SyntheticPolicy:
    squash_output = False

    def set_training_mode(self, mode):
        assert mode is False

    def __call__(self, obs):
        return torch.zeros((len(obs), 1)), obs[:, 0], torch.zeros(len(obs))

    def obs_to_tensor(self, obs):
        return torch.as_tensor(obs)[None], False

    def predict_values(self, obs):
        return obs[:, :1]


class LifecycleTests(unittest.TestCase):
    def test_actual_flowenv_dummyvec_timeout_terminal_obs_and_reset(self):
        env = DummyVecEnv([make_env])
        try:
            obs = env.reset()
            self.assertEqual(obs[0, 0], 0.0)
            for index in range(5):
                obs, _, dones, infos = env.step(np.zeros((1, 1)))
                self.assertEqual(bool(dones[0]), index == 4)
            self.assertTrue(infos[0]["TimeLimit.truncated"])
            self.assertEqual(infos[0]["terminal_observation"][0], 5.0)
            self.assertEqual(
                obs[0, 0], 0.0
            )  # reset observation is NOT terminal observation.
            flow = env.envs[0].unwrapped.flow
            self.assertEqual(flow.t, 148.0)
            self.assertEqual(len(flow._reward_history), 62)
        finally:
            env.close()

    def test_actual_sb3_collect_rollouts_bootstraps_terminal_not_reset(self):
        env = DummyVecEnv([make_env])
        try:
            state = SimpleNamespace(
                _last_obs=env.reset(),
                _last_episode_starts=np.ones(1, dtype=bool),
                policy=SyntheticPolicy(),
                use_sde=False,
                device=torch.device("cpu"),
                action_space=env.action_space,
                num_timesteps=0,
                gamma=0.99,
                _update_info_buffer=lambda infos, dones: None,
            )
            callback = SimpleNamespace(
                on_rollout_start=lambda: None,
                update_locals=lambda values: None,
                on_step=lambda: True,
                on_rollout_end=lambda: None,
            )
            buffer = RolloutBuffer(
                5,
                env.observation_space,
                env.action_space,
                device="cpu",
                gamma=0.99,
                n_envs=1,
            )
            self.assertTrue(
                OnPolicyAlgorithm.collect_rollouts(state, env, callback, buffer, 5)
            )
            np.testing.assert_allclose(buffer.rewards[:4, 0], -0.1, rtol=0, atol=1e-7)
            self.assertAlmostEqual(
                float(buffer.rewards[4, 0]), -0.1 + 0.99 * 5.0, places=6
            )
            self.assertEqual(state._last_obs[0, 0], 0.0)
            self.assertEqual(state.num_timesteps, 5)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
