from types import SimpleNamespace
import unittest
import os
import sys
import types
from pathlib import Path

import fluid_control

REPO = Path(os.environ.get("P064_REPO", Path(__file__).resolve().parents[1])).resolve()
fluid_control.__path__.append(str(REPO / "src/fluid_control"))
gym = types.ModuleType("gymnasium")
gym.Env = object
gym.Wrapper = object
sys.modules.setdefault("gymnasium", gym)
hydrogym = types.ModuleType("hydrogym")
hydrogym.PDEBase = object
hydrogym.TransientSolver = object
sys.modules.setdefault("hydrogym", hydrogym)

from fluid_control import dual_control_contract as contract
from fluid_control import tandem_hydrogym as tandem


def identity(arm: str):
    payload = {
        "kind": contract.P064_SYSTEM_KIND[arm],
        "history_input": {
            "schema_version": 1,
            "profile": "p026_k1",
            "history_length": 1,
            "flow_input_channels": 6,
            "aerodynamic_input_channels": 6,
            "left_padding": "trajectory_frame0",
            "autoregressive_state_source": "frozen_flow_prediction",
            "future_state_inputs": False,
            "future_force_inputs": False,
        },
        "history_state_module_sha256": contract.P026_HISTORY_STATE_SHA256,
        "history_inference_module_sha256": contract.P026_HISTORY_INFERENCE_SHA256,
        "training_protocol_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "normalization_sha256": "3" * 64,
    }
    return SimpleNamespace(
        payload=payload,
        manifest_sha256="4" * 64,
        flow=SimpleNamespace(model_sha256="5" * 64, state_sha256="6" * 64),
        aerodynamic=SimpleNamespace(model_sha256="7" * 64, state_sha256="8" * 64),
    )


class RuntimeCompatibilityTest(unittest.TestCase):
    def test_p064_identity_reuses_exact_k1_history_runtime(self):
        for arm in ("A", "B"):
            binding = contract.p026_runtime_binding(identity(arm))
            self.assertEqual(binding["profile"], "p026_k1")
            self.assertEqual(binding["history_length"], 1)
            self.assertEqual(binding["manifest_kind"], contract.P064_SYSTEM_KIND[arm])
            network = SimpleNamespace(history_length=1)
            self.assertEqual(
                tandem.TandemSurrogateFlow._validate_history_runtime(binding, network),
                binding,
            )

    def test_p064_kind_cannot_be_routed_as_k4(self):
        binding = contract.p026_runtime_binding(identity("B"))
        binding["profile"] = "p026_k4"
        binding["history_length"] = 4
        binding["aerodynamic_input_channels"] = 18
        with self.assertRaisesRegex(ValueError, "manifest kind"):
            tandem.TandemSurrogateFlow._validate_history_runtime(
                binding, SimpleNamespace(history_length=4)
            )

    def test_unrelated_kind_still_has_no_p026_runtime(self):
        value = identity("A")
        value.payload["kind"] = "UNRELATED"
        self.assertIsNone(contract.p026_runtime_binding(value))


if __name__ == "__main__":
    unittest.main()
