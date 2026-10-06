import ast
import importlib.util
import os
from pathlib import Path
import unittest

import numpy as np


REPO = Path(os.environ.get("P064_REPO", Path(__file__).resolve().parents[1])).resolve()
OLD = REPO / "artifacts/exploratory_projected_32768_ppo_long_cfd_source_20261006_immutable/run_exploratory_projected_32768_ppo_long_cfd.py"
NEW = REPO / "scripts/run_p064_candidate_projected_32768_ppo_long_cfd.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def function_ast(path, name):
    tree = ast.parse(path.read_text())
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name)
    return ast.dump(node, include_attributes=False)


old = load("old_projected", OLD)
new = load("new_p064_projected", NEW)


class CandidateCFDTemplateTest(unittest.TestCase):
    def test_controller_numerical_helpers_are_byte_semantics_identical(self):
        for name in (
            "observation", "validate_vec", "predict", "reflect_physical69",
            "projected_request", "summarize", "values_summary",
        ):
            self.assertEqual(function_ast(NEW, name), function_ast(OLD, name), name)

    def test_reflection_projection_matches_retained_driver(self):
        obs = np.linspace(-0.7, 0.7, 69, dtype=np.float32)
        obs[-1] = np.float32(0.2)
        np.testing.assert_array_equal(
            new.reflect_physical69(obs), old.reflect_physical69(obs)
        )
        policy = lambda value: float(np.asarray(value, dtype=np.float32).sum() / 100)
        self.assertEqual(new.projected_request(policy, obs), old.projected_request(policy, obs))

    def test_training_receipt_requires_matching_p064_candidate(self):
        result = {
            "status": new.TRAINING_STATUS,
            "candidate_arm": "B",
            "candidate_manifest_sha256": "1" * 64,
            "timesteps": 32768,
            "ppo_n_updates": 256,
            "optimizer_steps": [{"optimizer_step": i} for i in range(1, 513)],
            "fno_tensors_unchanged": True,
            "artifacts": {"ppo_final.zip": "2" * 64, "vecnormalize.pkl": "3" * 64},
            "protocol": {
                "observation_dimension": 69, "action_limit": .75,
                "action_delta_limit": .1, "control_dt": .1,
                "norm_obs": False, "norm_reward": False, "reset_count": 24,
                "episode_steps": 5,
                "reset_selection": "deterministic_phase_cycle_no_reward_selection",
            },
            "scientific_admission": False,
            "reset_counts_by_phase": {p: [274,273,273,273,273,273]
                                      for p in ("00", "02", "04", "06")},
        }
        new.validate_training(result, "2" * 64, "3" * 64, "B", "1" * 64)
        with self.assertRaisesRegex(ValueError, "candidate identity"):
            new.validate_training(result, "2" * 64, "3" * 64, "A", "1" * 64)


if __name__ == "__main__":
    unittest.main()
