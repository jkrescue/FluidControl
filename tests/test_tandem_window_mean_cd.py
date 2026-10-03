from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_tandem_window_mean_cd.py"
WRAPPER = ROOT / "scripts" / "run_tandem_window_mean_cd_v4_validation_spark.sh"


def load_module():
    fake_torch = types.ModuleType("torch")
    fake_torch.Tensor = object
    fake_torch.cat = lambda values, dim: (tuple(values), dim)
    fake_torch.cuda = types.SimpleNamespace(set_per_process_memory_fraction=lambda *a, **k: None)

    fake_h5py = types.ModuleType("h5py")
    fake_physicsnemo = types.ModuleType("physicsnemo")
    fake_distributed = types.ModuleType("physicsnemo.distributed")
    fake_distributed.DistributedManager = object
    fake_utils = types.ModuleType("physicsnemo.utils")
    fake_utils.load_checkpoint = lambda *a, **k: 1
    fake_evaluate = types.ModuleType("evaluate_tandem_fno")
    fake_evaluate.load_composed_config = lambda path: None
    fake_train = types.ModuleType("train_tandem_fno")
    fake_train.build_model = lambda cfg: None

    replacements = {
        "torch": fake_torch,
        "h5py": fake_h5py,
        "physicsnemo": fake_physicsnemo,
        "physicsnemo.distributed": fake_distributed,
        "physicsnemo.utils": fake_utils,
        "evaluate_tandem_fno": fake_evaluate,
        "train_tandem_fno": fake_train,
    }
    saved = {name: sys.modules.get(name) for name in replacements}
    sys.modules.update(replacements)
    try:
        spec = importlib.util.spec_from_file_location("window_mean_cd", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


MODULE = load_module()


class WindowMeanCdTest(unittest.TestCase):
    def test_float32_cfd_time_rounding_is_not_irregular_sampling(self):
        time = (80.0 + np.arange(801, dtype=np.float64) * 0.1).astype(
            np.float32
        )[:, None]
        MODULE.validate_time_spacing(time)
        irregular = time.copy()
        irregular[100] += np.float32(0.01)
        with self.assertRaisesRegex(ValueError, "uniform"):
            MODULE.validate_time_spacing(irregular)

    def test_pooled_nrmse_uses_segment_weighted_sums(self):
        cases = [
            {"segments": 1, "window_mean_cd_rmse": 1.0, "window_mean_cd_target_rms": 2.0},
            {"segments": 3, "window_mean_cd_rmse": 2.0, "window_mean_cd_target_rms": 4.0},
        ]
        self.assertAlmostEqual(MODULE.pooled_nrmse(cases), (13.0 / 52.0) ** 0.5)

    def test_rollout_input_has_exact_training_channel_order(self):
        values, dim = MODULE.build_rollout_inputs("state", "mask", "now", "next")
        self.assertEqual(values, ("state", "mask", "now", "next"))
        self.assertEqual(dim, 1)

    def test_model_config_modes_are_compared_as_sequences(self):
        expected = {"in_channels": 6, "num_fno_modes": [32, 32]}
        MODULE.validate_model_config(
            {"in_channels": 6, "num_fno_modes": (32, 32)}, expected
        )
        with self.assertRaisesRegex(ValueError, "num_fno_modes"):
            MODULE.validate_model_config(
                {"in_channels": 6, "num_fno_modes": (16, 32)}, expected
            )

    def test_normalization_rejects_non_train_or_nonpositive_scale(self):
        valid = {
            "computed_from": "train split only",
            "all_force_channels": list(MODULE.FORCE_CHANNELS),
            "state_mean": [0.0, 0.0, 0.0],
            "state_std": [1.0, 1.0, 1.0],
            "all_force_mean": [0.0, 0.0, 0.0, 0.0],
            "all_force_std": [1.0, 1.0, 1.0, 1.0],
        }
        MODULE.validate_normalization(valid)
        invalid = dict(valid, all_force_std=[1.0, 1.0, 0.0, 1.0])
        with self.assertRaisesRegex(ValueError, "non-positive"):
            MODULE.validate_normalization(invalid)

    def test_exclusive_output_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            output.write_text('{"old": true}\n', encoding="utf-8")
            with self.assertRaises(FileExistsError):
                MODULE.write_json_exclusive(output, {"new": True})
            self.assertEqual(json.loads(output.read_text()), {"old": True})

    def test_wrapper_refuses_training_and_preserves_frozen_test(self):
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("pgrep -f '[t]rain_tandem_fno(_rollout)?", text)
        self.assertIn("WINDOW_MEAN_HOST_TRAINING_GUARD=passed", text)
        self.assertIn("tandem_cylinders_control_gap_v4", text)
        self.assertNotIn("--split test", text)
        self.assertNotIn("/test", text)

    def test_wrapper_keeps_memory_floor_and_fixed_allocator_cap(self):
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("20 * 1024 * 1024", text)
        self.assertIn("--min-free-gib 20", text)
        self.assertIn("--allocator-fraction 0.20", text)


if __name__ == "__main__":
    unittest.main()
