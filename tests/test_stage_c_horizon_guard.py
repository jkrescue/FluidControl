"""Regression test: PPO cannot train on a drag-free Stage-C episode."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import gymnasium as gym

from fluid_control.tandem_hydrogym import TandemRewardAudit


class FakeEnv(gym.Env):
    def __init__(self, *, max_steps: int, reward_mode: str, period: float):
        self.max_steps = max_steps
        self.solver = SimpleNamespace(dt=0.1)
        self.flow = SimpleNamespace(
            reward_mode=reward_mode,
            shedding_period=period,
            max_abs_normalized_state_guard=40.0,
        )


class StageCHorizonGuardTests(unittest.TestCase):
    def test_short_stage_c_episode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "1.5 shedding periods"):
            TandemRewardAudit(
                FakeEnv(max_steps=32, reward_mode="stage_c_total_drag", period=6.15)
            )

    def test_long_stage_c_episode_is_allowed(self) -> None:
        wrapped = TandemRewardAudit(
            FakeEnv(max_steps=100, reward_mode="stage_c_total_drag", period=6.15)
        )
        self.assertEqual(wrapped.max_abs_normalized_state, 40.0)

    def test_legacy_reward_is_unaffected(self) -> None:
        wrapped = TandemRewardAudit(
            FakeEnv(max_steps=32, reward_mode="legacy_rear", period=6.15)
        )
        self.assertEqual(wrapped.max_abs_normalized_state, 40.0)


if __name__ == "__main__":
    unittest.main()
