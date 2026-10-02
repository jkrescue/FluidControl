from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch
from hydrogym import FlowEnv

from fluid_control.tandem_hydrogym import (
    TandemFNOStepper,
    TandemRewardAudit,
    TandemSurrogateFlow,
)


class ConstantFNO(torch.nn.Module):
    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.out_channels = out_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            inputs.shape[0],
            self.out_channels,
            inputs.shape[-2],
            inputs.shape[-1],
            dtype=inputs.dtype,
            device=inputs.device,
        )


class TandemHydroGymStageCTests(unittest.TestCase):
    def make_data(self, root: Path, *, all_forces: bool) -> None:
        (root / "validation").mkdir(parents=True)
        with h5py.File(root / "validation" / "phase.h5", "w") as handle:
            handle.create_dataset("state", data=np.zeros((2, 3, 2, 2), np.float32))
            handle.create_dataset("mask", data=np.ones((2, 1, 2, 2), np.float32))
            handle.create_dataset("omega", data=np.zeros((2, 1), np.float32))
            handle.create_dataset(
                "force",
                data=np.asarray([[1.0, 0.2, 2.0, 0.4]] * 2, np.float32),
            )
            handle.create_dataset("time", data=np.asarray([[0.0], [0.1]]))
            handle.create_dataset("x", data=np.asarray([16.0, 18.0]))
            handle.create_dataset("y", data=np.asarray([5.0, 10.0]))
        (root / "manifest.json").write_text(
            json.dumps({"max_abs_omega": 5.0}), encoding="utf-8"
        )
        stats: dict[str, object] = {
            "state_mean": [0.0, 0.0, 0.0],
            "state_std": [1.0, 1.0, 1.0],
            "state_abs_normalized_max_train": 1.0,
            "force_channels": ["rear_cd", "rear_cl"],
            "force_mean": [2.0, 0.4],
            "force_std": [1.0, 1.0],
        }
        if all_forces:
            stats.update(
                all_force_channels=[
                    "front_cd",
                    "front_cl",
                    "rear_cd",
                    "rear_cl",
                ],
                all_force_mean=[1.0, 0.2, 2.0, 0.4],
                all_force_std=[1.0, 1.0, 1.0, 1.0],
            )
        (root / "normalization.json").write_text(json.dumps(stats), encoding="utf-8")

    def test_four_force_stage_c_ledger_uses_complete_causal_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_data(root, all_forces=True)
            raw = FlowEnv(
                {
                    "flow": TandemSurrogateFlow,
                    "flow_config": {
                        "data_root": root,
                        "split": "validation",
                        "case": "phase",
                        "frame": 1,
                        "network": ConstantFNO(7),
                        "device": "cpu",
                        "reward_mode": "stage_c_total_drag",
                        "phase_baseline": {
                            "total_drag": 3.0,
                            "front_lift_rms": 0.1,
                            "rear_lift_rms": 0.2,
                            "source": "zero_action_same_phase_test_fixture",
                        },
                        "shedding_period": 0.2,
                    },
                    "solver": TandemFNOStepper,
                    "solver_config": {"dt": 0.1},
                    "max_steps": 3,
                }
            )
            env = TandemRewardAudit(raw)
            observation, _ = env.reset()
            self.assertEqual(observation.shape, (69,))

            _, first_reward, _, _, first_info = env.step([0.0])
            self.assertEqual(first_reward, 0.0)
            self.assertFalse(first_info["stage_c_ledger"]["window_ready"])

            _, second_reward, _, _, second_info = env.step([0.5])
            ledger = second_info["stage_c_ledger"]
            self.assertLess(second_reward, 0.0)
            self.assertTrue(ledger["window_ready"])
            self.assertEqual(ledger["window_samples"], 3)
            self.assertAlmostEqual(ledger["total_drag"], 3.0)
            self.assertAlmostEqual(ledger["front_lift_ratio"], 2.0, places=6)
            self.assertAlmostEqual(ledger["rear_lift_ratio"], 2.0, places=6)
            self.assertEqual(
                ledger["baseline_source"], "zero_action_same_phase_test_fixture"
            )
            components = [
                value for key, value in second_info.items() if key.startswith("reward_")
            ]
            self.assertAlmostEqual(sum(components), second_reward)
            self.assertLess(second_info["reward_front_lift_excess"], 0.0)
            self.assertLess(second_info["reward_rear_lift_excess"], 0.0)
            self.assertLess(second_info["reward_actuation"], 0.0)
            self.assertLess(second_info["reward_rate"], 0.0)
            self.assertEqual(second_info["predicted_front_cd"], 1.0)
            self.assertEqual(second_info["predicted_rear_cd"], 2.0)

    def test_rear_only_checkpoint_remains_legacy_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_data(root, all_forces=False)
            flow = TandemSurrogateFlow(
                data_root=root,
                split="validation",
                case="phase",
                frame=1,
                network=ConstantFNO(5),
                device="cpu",
            )
            self.assertEqual(flow.num_outputs, 67)
            self.assertEqual(flow.force_channels, ("rear_cd", "rear_cl"))
            self.assertEqual(flow.get_observations().shape, (67,))

    def test_stage_c_refuses_missing_baseline_and_wrong_checkpoint_width(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_data(root, all_forces=True)
            with self.assertRaisesRegex(ValueError, "phase_baseline"):
                TandemSurrogateFlow(
                    data_root=root,
                    split="validation",
                    case="phase",
                    frame=1,
                    network=ConstantFNO(7),
                    device="cpu",
                    reward_mode="stage_c_total_drag",
                )
            env = FlowEnv(
                {
                    "flow": TandemSurrogateFlow,
                    "flow_config": {
                        "data_root": root,
                        "split": "validation",
                        "case": "phase",
                        "frame": 1,
                        "network": ConstantFNO(5),
                        "device": "cpu",
                    },
                    "solver": TandemFNOStepper,
                    "solver_config": {"dt": 0.1},
                }
            )
            env.reset()
            with self.assertRaisesRegex(ValueError, "force schema requires 7"):
                env.step([0.0])


if __name__ == "__main__":
    unittest.main()
