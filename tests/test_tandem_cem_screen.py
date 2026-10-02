import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from screen_tandem_cem_mpc import require_gate_b_pass, rollout_population


class ConstantSevenOutput(torch.nn.Module):
    def forward(self, inputs):
        output = torch.zeros(
            (inputs.shape[0], 7, inputs.shape[2], inputs.shape[3]),
            dtype=inputs.dtype,
            device=inputs.device,
        )
        output[:, 3:] = 1.0
        return output


class TandemCEMScreenTests(unittest.TestCase):
    def test_rollout_population_preserves_schema(self):
        state = torch.zeros((1, 3, 2, 3))
        mask = torch.ones((1, 1, 2, 3))
        actions = np.asarray([[0.1, 0.2], [-0.1, -0.2]])
        forces, bounds = rollout_population(
            ConstantSevenOutput(),
            state,
            mask,
            actions,
            current_omega=0.0,
            action_scale=5.0,
            force_mean=torch.zeros(4),
            force_std=torch.ones(4),
            batch_size=1,
        )
        self.assertEqual(forces.shape, (2, 2, 4))
        self.assertTrue(np.allclose(forces, 1.0))
        self.assertTrue(np.allclose(bounds, 0.0))

    def test_gate_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.json"
            path.write_text('{"status":"GATE_B_NEEDS_MULTISTEP_RETRAINING"}')
            with self.assertRaisesRegex(RuntimeError, "requires GATE_B_PASS"):
                require_gate_b_pass(path)


if __name__ == "__main__":
    unittest.main()
