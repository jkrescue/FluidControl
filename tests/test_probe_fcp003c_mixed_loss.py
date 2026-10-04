import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import torch


SCRIPT = Path(__file__).parents[1] / "scripts" / "probe_fcp003c_mixed_loss.py"
SPEC = importlib.util.spec_from_file_location("probe_fcp003c_mixed_loss", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TestFcp003cMixedLossProbe(unittest.TestCase):
    def test_regular_identity_requires_exact_seeded_batch_and_shapes(self):
        batch = {
            key: torch.zeros(shape) for key, shape in MODULE.EXPECTED_REGULAR_SHAPES.items()
        }
        MODULE.validate_regular_identity([dict(MODULE.EXPECTED_REGULAR_METADATA)], batch)
        changed = dict(MODULE.EXPECTED_REGULAR_METADATA)
        changed["step"] += 1
        with self.assertRaises(ValueError):
            MODULE.validate_regular_identity([changed], batch)
        batch["omega"] = torch.zeros(1, 100, 1)
        with self.assertRaises(ValueError):
            MODULE.validate_regular_identity([dict(MODULE.EXPECTED_REGULAR_METADATA)], batch)

    def test_pair_identity_requires_fixed_real_pair_and_h100_shapes(self):
        shapes = {
            "action_state": [1, 101, 3, 128, 256],
            "zero_state": [1, 101, 3, 128, 256],
            "action_omega": [1, 101, 1],
            "zero_omega": [1, 101, 1],
            "action_force": [1, 101, 4],
            "zero_force": [1, 101, 4],
            "mask": [1, 1, 128, 256],
        }
        pair = {key: torch.zeros(shape) for key, shape in shapes.items()}
        MODULE.validate_pair_identity([{"pair_id": "b00:multisine"}], pair)
        with self.assertRaises(ValueError):
            MODULE.validate_pair_identity([{"pair_id": "b00:prbs"}], pair)

    def test_counting_optimizer_refuses_second_step(self):
        model = torch.nn.Linear(2, 1)
        optimizer = MODULE.CountingAdamW(model.parameters(), lr=1e-3)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        self.assertEqual(optimizer.step_count, 1)
        with self.assertRaises(RuntimeError):
            optimizer.step()

    def test_model_state_hash_changes_only_when_tensor_changes(self):
        model = torch.nn.Linear(2, 1)
        first = MODULE.model_state_sha256(model)
        self.assertEqual(first, MODULE.model_state_sha256(model))
        with torch.no_grad():
            model.weight.add_(1)
        self.assertNotEqual(first, MODULE.model_state_sha256(model))

    def test_sha256_reads_exact_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value"
            path.write_bytes(b"abc")
            self.assertEqual(
                MODULE.sha256(path),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )


if __name__ == "__main__":
    unittest.main()
