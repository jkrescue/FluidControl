from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch

from fluid_control.canonical_joint_v1 import (
    canonical_joint_cost_components,
    causal_window_ledger,
)
from fluid_control.full40_canonical_hydrogym import make_full40_canonical_env
from fluid_control.openfoam_force_history import actual_causal_prehistory


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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_force(path: Path, cd: float, cl: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Time Cd x x Cl\n"
        + "".join(
            f"{141.9 + 0.1 * index:.8f} {cd} 0 0 {cl}\n"
            for index in range(62)
        ),
        encoding="utf-8",
    )


def make_fixture(root: Path) -> tuple[Path, Path]:
    data = root / "data"
    cases = root / "cases"
    case = "matched_start_acquisition_train_b00_zero"
    source = cases / "tandem_backward_dt005"
    front = source / "postProcessing/forceFront/0/coefficient.dat"
    rear = source / "postProcessing/forceRear/0/coefficient.dat"
    write_force(front, 1.0, 0.2)
    write_force(rear, 2.0, 0.4)
    config = {
        "case": case,
        "split": "train",
        "source_restart_case": source.name,
        "source_restart_time": 148.0,
        "source_force_sha256": {
            "forceFront": sha256(front),
            "forceRear": sha256(rear),
        },
        "action_target": 0.0,
    }
    case_root = cases / case
    case_root.mkdir(parents=True)
    (case_root / "case_config.json").write_text(json.dumps(config), encoding="utf-8")
    (data / "train").mkdir(parents=True)
    with h5py.File(data / f"train/{case}.h5", "w") as handle:
        handle.attrs["case"] = case
        handle.attrs["split"] = "train"
        handle.attrs["config_json"] = json.dumps(config)
        handle.create_dataset("state", data=np.zeros((1, 3, 2, 2), np.float32))
        handle.create_dataset("mask", data=np.ones((1, 1, 2, 2), np.float32))
        handle.create_dataset("omega", data=np.zeros((1, 1), np.float32))
        handle.create_dataset(
            "force", data=np.asarray([[1.0, 0.2, 2.0, 0.4]], np.float32)
        )
        handle.create_dataset("time", data=np.asarray([[148.0]]))
        handle.create_dataset("x", data=np.asarray([16.0, 18.0]))
        handle.create_dataset("y", data=np.asarray([5.0, 10.0]))
    (data / "splits").mkdir()
    split_manifest = data / "splits/train.json"
    split_manifest.write_text(
        json.dumps(
            {
                "split": "train",
                "cases": [case],
                "hdf5_sha256": {case: sha256(data / f"train/{case}.h5")},
            }
        )
    )
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "profile": "matched_start_full40_v1",
                "max_abs_omega": 0.75,
                "split_manifests": {
                    "train": {
                        "path": "splits/train.json",
                        "sha256": sha256(split_manifest),
                    }
                },
            }
        )
    )
    (data / "normalization.json").write_text(
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
    return data, cases


class Full40CanonicalHydroGymTests(unittest.TestCase):
    def make_env(self, root: Path):
        data, cases = make_fixture(root)
        initial_observation = np.zeros(69, dtype=np.float32)
        initial_observation[64:68] = [1.0, 0.2, 2.0, 0.4]
        patcher = patch(
            "fluid_control.tandem_hydrogym.total_drag_observation_at",
            return_value=(initial_observation, {}),
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        return make_full40_canonical_env(
            data=data,
            split="train",
            case="matched_start_acquisition_train_b00_zero",
            frame=0,
            network=ConstantFourForceFNO(),
            checkpoint_epoch=10,
            baseline={
                "total_drag": 3.0,
                "rear_cl_fluctuation_rms": 0.5,
                "source": "software_contract_fixture_not_cfd_benefit",
            },
            episode_steps=100,
            device="cpu",
            cases_root=cases,
        )

    def test_reset_is_absolute_window_ready_and_enforces_slew(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env = self.make_env(Path(directory))
            self.assertEqual(len(env.unwrapped.flow._reward_history), 62)
            self.assertTrue(env.unwrapped.flow.canonical_ledger()["window_ready"])
            observation, _ = env.reset()
            self.assertEqual(observation.shape, (69,))
            self.assertAlmostEqual(env.unwrapped.flow.t, 148.0)
            before = list(env.unwrapped.flow._reward_history)
            self.assertEqual(len(before), 62)
            self.assertEqual(len({row[0] for row in before}), 62)
            _, direct_forces, _ = actual_causal_prehistory(
                Path(directory) / "cases/tandem_backward_dt005",
                148.0,
                provenance_root=Path(directory),
            )
            np.testing.assert_array_equal(
                np.stack([row[1] for row in before]),
                np.asarray(direct_forces, dtype=np.float64),
            )
            self.assertNotEqual(
                float(before[0][1][1]),
                float(np.float32(before[0][1][1])),
            )
            self.assertTrue(env.unwrapped.flow.canonical_ledger()["window_ready"])
            _, reward, _, _, info = env.step([0.75])
            self.assertAlmostEqual(info["applied_omega"], 0.1)
            self.assertAlmostEqual(env.unwrapped.flow.t, 148.1)
            self.assertEqual(env.unwrapped.iter, 1)
            self.assertAlmostEqual(info["applied_abs_rate"], 1.0)
            self.assertTrue(info["rate_limited"])
            self.assertEqual(info["observation_dimension"], 69)
            self.assertTrue(info["canonical_joint_ledger"]["window_ready"])
            components = sum(
                value for key, value in info.items() if key.startswith("reward_")
            )
            self.assertAlmostEqual(components, reward)
            # The same explicit force/action sequence must give the exact
            # canonical components used by the direct-CFD environment.
            history = list(env.unwrapped.flow._reward_history)
            ledger = causal_window_ledger(
                np.asarray([row[0] for row in history]),
                np.stack([row[1] for row in history]),
                env.unwrapped.flow.canonical_baseline,
            )
            expected = canonical_joint_cost_components(
                ledger,
                omega=0.1,
                delta_omega=0.1,
                action_scale=0.75,
                max_delta_omega=0.1,
            )
            for key, value in expected.items():
                self.assertAlmostEqual(info[f"reward_{key}"], -0.1 * value)
            env.reset()
            after = list(env.unwrapped.flow._reward_history)
            self.assertEqual([row[0] for row in before], [row[0] for row in after])
            np.testing.assert_array_equal(
                np.stack([row[1] for row in before]),
                np.stack([row[1] for row in after]),
            )
            self.assertEqual(len(after), 62)

    def test_reset_rejects_different_time_or_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env = self.make_env(Path(directory))
            flow = env.unwrapped.flow
            with self.assertRaisesRegex(ValueError, "time differs"):
                flow.reset(t=147.9)
            changed = flow.copy_state()
            changed["force"][3] += 1.0
            with self.assertRaisesRegex(ValueError, "only for initial q0"):
                flow.reset(q0=changed)

    def test_canonical_manifest_binding_rejects_hdf_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, cases = make_fixture(root)
            split_path = data / "splits/train.json"
            split = json.loads(split_path.read_text(encoding="utf-8"))
            case = "matched_start_acquisition_train_b00_zero"
            split["hdf5_sha256"][case] = "0" * 64
            split_path.write_text(json.dumps(split), encoding="utf-8")
            manifest_path = data / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["split_manifests"]["train"]["sha256"] = sha256(split_path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            initial = np.zeros(69, dtype=np.float32)
            initial[64:68] = [1.0, 0.2, 2.0, 0.4]
            with patch(
                "fluid_control.tandem_hydrogym.total_drag_observation_at",
                return_value=(initial, {}),
            ), self.assertRaisesRegex(ValueError, "split/HDF5 manifest differs"):
                make_full40_canonical_env(
                    data=data,
                    split="train",
                    case=case,
                    frame=0,
                    network=ConstantFourForceFNO(),
                    checkpoint_epoch=10,
                    baseline={
                        "total_drag": 3.0,
                        "rear_cl_fluctuation_rms": 0.5,
                        "source": "fixture",
                    },
                    episode_steps=100,
                    device="cpu",
                    cases_root=cases,
                )


if __name__ == "__main__":
    unittest.main()
