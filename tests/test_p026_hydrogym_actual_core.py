"""Actual HydroGym-core lifecycle regression with synthetic data and a mock net.

This is not a model, CFD, policy, or scientific test.  It imports the pinned
HydroGym ``PDEBase``/``FlowEnv`` implementation and exercises only project
history/reset plumbing on a tiny temporary HDF fixture.
"""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import hydrogym
import hydrogym.core

import fluid_control.full40_canonical_hydrogym as canonical_module
import fluid_control.tandem_hydrogym as tandem_module
from fluid_control.full40_canonical_hydrogym import make_full40_canonical_env
from test_full40_canonical_hydrogym import make_fixture


EXPECTED_HYDROGYM_CORE_SHA256 = "153e8c4cebccefd30b123346c3457b0da5ebda1cb4f2d8163d448965327a9d15"
EXPECTED_HYDROGYM_INIT_SHA256 = "9a5f61695578b539b7af459d9aceb83be493b98e52230fff1a3f3a3db2c8aedb"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MockP026K4Network(torch.nn.Module):
    """Tiny deterministic dependency fixture, not an official-model claim."""

    history_length = 4

    def __init__(self):
        super().__init__()
        self.inputs: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, flow_inputs, aerodynamic_inputs):
        self.inputs.append((flow_inputs.clone(), aerodynamic_inputs.clone()))
        return torch.zeros(
            len(flow_inputs),
            7,
            *flow_inputs.shape[-2:],
            dtype=flow_inputs.dtype,
            device=flow_inputs.device,
        )


def runtime_identity():
    keys = (
        "dual_manifest_sha256",
        "flow_model_sha256",
        "flow_state_sha256",
        "aerodynamic_model_sha256",
        "aerodynamic_state_sha256",
        "history_state_module_sha256",
        "history_inference_module_sha256",
        "training_protocol_sha256",
        "config_sha256",
        "normalization_sha256",
    )
    return {
        "schema_version": 1,
        "profile": "p026_k4",
        "history_length": 4,
        "manifest_kind": "FC_P026_K4_HISTORY_FORCE_FNO",
        "flow_input_channels": 6,
        "aerodynamic_input_channels": 18,
        "left_padding": "trajectory_frame0",
        "autoregressive_state_source": "frozen_flow_prediction",
        "future_state_inputs": False,
        "future_force_inputs": False,
        **{key: format(index + 1, "x") * 64 for index, key in enumerate(keys)},
    }


class P026ActualHydroGymCoreTests(unittest.TestCase):
    def test_flowenv_constructor_step_and_reset_keep_explicit_history(self):
        repo_root = Path(__file__).resolve().parents[1]
        self.assertTrue(Path(tandem_module.__file__).resolve().is_relative_to(repo_root))
        self.assertTrue(Path(canonical_module.__file__).resolve().is_relative_to(repo_root))
        self.assertEqual(sha256(Path(hydrogym.core.__file__)), EXPECTED_HYDROGYM_CORE_SHA256)
        self.assertEqual(sha256(Path(hydrogym.__file__)), EXPECTED_HYDROGYM_INIT_SHA256)
        print(
            "P026_ACTUAL_CORE_IMPORTS",
            hydrogym.core.__file__,
            tandem_module.__file__,
            canonical_module.__file__,
            flush=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, cases = make_fixture(root)
            initial = np.zeros(69, dtype=np.float32)
            initial[64:68] = [1.0, 0.2, 2.0, 0.4]
            network = MockP026K4Network()
            with patch(
                "fluid_control.tandem_hydrogym.total_drag_observation_at",
                return_value=(initial, {}),
            ):
                env = make_full40_canonical_env(
                    data=data,
                    split="train",
                    case="matched_start_acquisition_train_b00_zero",
                    frame=0,
                    network=network,
                    checkpoint_epoch=1,
                    baseline={
                        "total_drag": 3.0,
                        "rear_cl_fluctuation_rms": 0.5,
                        "source": "synthetic_core_fixture_not_scientific_evidence",
                    },
                    episode_steps=100,
                    device="cpu",
                    cases_root=cases,
                    fno_history_runtime=runtime_identity(),
                )
                flow = env.unwrapped.flow
                self.assertEqual(len(flow._reward_history), 62)
                self.assertEqual(flow.fno_history.source, "generated_frame0_reset_padding")
                self.assertEqual(flow.fno_history.padding_mask.tolist(), [[True, True, True, False]])
                initial_history = flow.copy_state()["fno_history"]
                observation, _ = env.reset()
                self.assertEqual(observation.shape, (69,))
                self.assertEqual(len(flow._reward_history), 62)
                torch.testing.assert_close(flow.fno_history.states, initial_history.states)

                _, _, _, _, info = env.step([0.75])
                self.assertAlmostEqual(info["applied_omega"], 0.1)
                self.assertEqual(flow.fno_history.source, "autoregressive_prediction")
                self.assertEqual(flow.fno_history.padding_mask.tolist(), [[True, True, False, False]])
                self.assertAlmostEqual(float(flow.fno_history.actions[0, -1]), 0.1 / 0.75)
                flow_inputs, aerodynamic_inputs = network.inputs[-1]
                self.assertEqual(tuple(flow_inputs.shape), (1, 6, 2, 2))
                self.assertEqual(tuple(aerodynamic_inputs.shape), (1, 18, 2, 2))
                torch.testing.assert_close(flow.fno_history.states[0, -1], flow.q)

                env.reset()
                self.assertEqual(len(flow._reward_history), 62)
                self.assertEqual(flow.fno_history.source, "generated_frame0_reset_padding")
                torch.testing.assert_close(flow.fno_history.states, initial_history.states)
                self.assertEqual(flow.fno_history.padding_mask.tolist(), [[True, True, True, False]])
                env.close()


if __name__ == "__main__":
    unittest.main()
