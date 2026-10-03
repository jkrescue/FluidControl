from __future__ import annotations

import math
import unittest

import numpy as np

from fluid_control.direct_cfd_hydrogym import make_direct_cfd_env


class ScriptedTransport:
    def __init__(self, *, future_history: bool = False) -> None:
        self.future_history = future_history
        self.time = 100.0
        self.omega = 0.0
        self.episode = 0
        self.steps = 0
        self.closed = False

    @staticmethod
    def observation(time: float, omega: float) -> list[float]:
        del time
        result = np.zeros(69, dtype=np.float32)
        result[64:68] = [1.0, 0.0, 1.0 - 0.02 * abs(omega), -0.1 * omega]
        result[68] = omega
        return result.tolist()

    def request(self, payload) -> dict:
        if payload["command"] == "reset":
            self.time = 100.0
            self.omega = 0.0
            self.steps = 0
            self.episode += 1
            times = [round(93.9 + 0.1 * index, 8) for index in range(62)]
            if self.future_history:
                times[-1] = 100.1
            forces = [[1.0, 0.0, 1.0, 0.1 * math.sin(index)] for index in range(62)]
            return {
                "ok": True,
                "observation": self.observation(self.time, self.omega),
                "time": self.time,
                "prehistory_times": times,
                "prehistory_forces": forces,
                "worker_info": {"episode": self.episode, "case": f"mock_{self.episode}"},
            }
        if payload["command"] == "step":
            requested = float(payload["requested_omega"])
            applied = float(np.clip(requested, self.omega - 0.1, self.omega + 0.1))
            applied = float(np.clip(applied, -0.75, 0.75))
            limited = not np.isclose(requested, applied)
            self.omega = applied
            self.time = round(self.time + 0.1, 8)
            self.steps += 1
            return {
                "ok": True,
                "observation": self.observation(self.time, self.omega),
                "time": self.time,
                "applied_omega": applied,
                "rate_limited": bool(limited),
                "worker_info": {
                    "episode": self.episode,
                    "step": self.steps,
                    "case": f"mock_{self.episode}",
                },
            }
        raise AssertionError(payload)

    def close(self) -> None:
        self.closed = True


BASELINE = {
    "total_drag": 2.0,
    "rear_cl_fluctuation_rms": 0.1,
    "source": "unit-test train-only phase baseline",
}


class DirectCFDHydroGymTest(unittest.TestCase):
    def test_69d_causal_reward_and_action_rate_contract(self) -> None:
        transport = ScriptedTransport()
        env = make_direct_cfd_env(
            transport_factory=lambda: transport,
            baseline=BASELINE,
            episode_steps=128,
        )
        observation, reset_info = env.reset()
        self.assertEqual(observation.shape, (69,))
        self.assertEqual(reset_info, {})
        observation, reward, terminated, truncated, info = env.step(
            np.asarray([0.75], dtype=np.float32)
        )
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(observation.shape, (69,))
        self.assertAlmostEqual(info["applied_omega"], 0.1)
        self.assertAlmostEqual(info["applied_delta_omega"], 0.1)
        self.assertTrue(info["rate_limited"])
        self.assertTrue(info["canonical_joint_ledger"]["window_ready"])
        raw_components = [
            value for key, value in info.items() if key.startswith("reward_")
        ]
        self.assertAlmostEqual(reward, sum(raw_components), places=7)
        self.assertEqual(info["backend"], "real_openfoam_host_worker_not_surrogate")
        env.close()
        self.assertTrue(transport.closed)

    def test_exact_128_step_truncation(self) -> None:
        transport = ScriptedTransport()
        env = make_direct_cfd_env(
            transport_factory=lambda: transport,
            baseline=BASELINE,
            episode_steps=128,
        )
        env.reset()
        for step in range(1, 129):
            _, _, terminated, truncated, _ = env.step(np.asarray([0.0]))
            self.assertFalse(terminated)
            self.assertEqual(truncated, step == 128)
        self.assertEqual(transport.steps, 128)
        env.close()

    def test_future_prehistory_fails_closed(self) -> None:
        transport = ScriptedTransport(future_history=True)
        with self.assertRaisesRegex(ValueError, "future information"):
            make_direct_cfd_env(
                transport_factory=lambda: transport,
                baseline=BASELINE,
                episode_steps=128,
            )

    def test_episode_protocol_is_fixed(self) -> None:
        with self.assertRaisesRegex(ValueError, "128-step"):
            make_direct_cfd_env(
                transport_factory=ScriptedTransport,
                baseline=BASELINE,
                episode_steps=64,
            )


if __name__ == "__main__":
    unittest.main()
