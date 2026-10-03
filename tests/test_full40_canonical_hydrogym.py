from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from fluid_control.full40_canonical_hydrogym import make_full40_canonical_env


class ConstantFourForceFNO(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            inputs.shape[0],
            7,
            inputs.shape[-2],
            inputs.shape[-1],
            dtype=inputs.dtype,
            device=inputs.device,
        )


def make_fixture(root: Path) -> None:
    (root / "validation").mkdir(parents=True)
    with h5py.File(root / "validation/phase.h5", "w") as handle:
        handle.create_dataset("state", data=np.zeros((2, 3, 2, 2), np.float32))
        handle.create_dataset("mask", data=np.ones((2, 1, 2, 2), np.float32))
        handle.create_dataset("omega", data=np.zeros((2, 1), np.float32))
        handle.create_dataset(
            "force", data=np.asarray([[1.0, 0.2, 2.0, 0.4]] * 2, np.float32)
        )
        handle.create_dataset("time", data=np.asarray([[0.0], [0.1]]))
        handle.create_dataset("x", data=np.asarray([16.0, 18.0]))
        handle.create_dataset("y", data=np.asarray([5.0, 10.0]))
    (root / "manifest.json").write_text(
        json.dumps({"profile": "matched_start_full40_v1", "max_abs_omega": 0.75})
    )
    (root / "normalization.json").write_text(
        json.dumps(
            {
                "state_mean": [0.0, 0.0, 0.0],
                "state_std": [1.0, 1.0, 1.0],
                "state_abs_normalized_max_train": 1.0,
                "all_force_channels": [
                    "front_cd",
                    "front_cl",
                    "rear_cd",
                    "rear_cl",
                ],
                "all_force_mean": [1.0, 0.2, 2.0, 0.4],
                "all_force_std": [1.0, 1.0, 1.0, 1.0],
                "force_channels": ["rear_cd", "rear_cl"],
                "force_mean": [2.0, 0.4],
                "force_std": [1.0, 1.0],
            }
        )
    )


class Full40CanonicalHydroGymTests(unittest.TestCase):
    def test_official_flowenv_has_69d_and_enforces_point_one_slew(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_fixture(root)
            env = make_full40_canonical_env(
                data=root,
                split="validation",
                case="phase",
                frame=1,
                network=ConstantFourForceFNO(),
                checkpoint_epoch=10,
                baseline={
                    "total_drag": 3.0,
                    "rear_cl_fluctuation_rms": 0.5,
                    "source": "software_contract_fixture_not_cfd_benefit",
                },
                episode_steps=100,
                device="cpu",
            )
            observation, _ = env.reset()
            self.assertEqual(observation.shape, (69,))
            _, reward, _, _, info = env.step([0.75])
            self.assertAlmostEqual(info["applied_omega"], 0.1)
            self.assertAlmostEqual(info["applied_abs_rate"], 1.0)
            self.assertTrue(info["rate_limited"])
            self.assertEqual(info["observation_dimension"], 69)
            components = sum(
                value for key, value in info.items() if key.startswith("reward_")
            )
            self.assertAlmostEqual(components, reward)
            self.assertFalse(info["canonical_joint_ledger"]["window_ready"])


if __name__ == "__main__":
    unittest.main()
