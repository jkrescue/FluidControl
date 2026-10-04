import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_fcp003c_mixed_probe_output.py"
SPEC = importlib.util.spec_from_file_location("validate_fcp003c_mixed", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class MixedProbeOutputValidatorTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "container_output/result_bundle").mkdir(parents=True)
        for name in (
            "resolved_config.yaml",
            "source_snapshot.sha256",
            "launch_receipt.json",
            "cpu_equivalence_receipt.json",
            "immutable_launcher.sh",
            "immutable_guard.py",
        ):
            (self.root / name).write_text(name)
        model = "a" * 64
        state = "b" * 64
        self.model, self.state, self.image = model, state, "sha256:image"
        self.result = {
            "status": "FC_P003C_MIXED_LOSS_TECHNICAL_PROBE_PASS",
            "scientific_result": False,
            "candidate_saved": False,
            "validation_or_frozen_accessed": False,
            "optimizer_steps": 1,
            "regular_metadata": MODULE.REGULAR_METADATA,
            "pair_metadata": MODULE.PAIR_METADATA,
            "regular_hdf_sha256": MODULE.REGULAR_HDF_SHA,
            "force_channels": MODULE.CHANNELS,
            "force_channel_weights_normalized": MODULE.WEIGHTS,
            "scratch_model_state_sha256_before": "c" * 64,
            "scratch_model_state_sha256_after": "d" * 64,
            "parent_file_sha256_before": {"model": model, "state": state},
            "parent_file_sha256_after": {"model": model, "state": state},
            "config_sha256": MODULE.sha256(self.root / "resolved_config.yaml"),
            "source_manifest_sha256": MODULE.sha256(self.root / "source_snapshot.sha256"),
            "launch_receipt_sha256": MODULE.sha256(self.root / "launch_receipt.json"),
            "cpu_equivalence_receipt_sha256": MODULE.sha256(self.root / "cpu_equivalence_receipt.json"),
            "image_id": self.image,
            "physicsnemo_version": "2.2.2",
            "regular_field_loss": 1.0,
            "regular_force_loss": 2.0,
            "cuda_peak_allocated_gib": 3.0,
            "cuda_peak_reserved_gib": 4.0,
            "minimum_mem_available_gib": 30.0,
            "metrics": {
                "loss": 1.0,
                "base_loss": 1.0,
                "paired_step_force_loss": 0.1,
                "paired_step_force_weighted_loss": 1.0,
                "preclip_gradient_norm": 2.0,
                "paired_step_force_per_channel_mse": [1.0] * 4,
                "paired_step_force_per_channel_weighted_contribution": [0.1] * 4,
                "optimizer_steps": 1,
                "chunk_size": 10,
                "chunk_count": 10,
            },
        }
        self.write()

    def tearDown(self):
        self.temp.cleanup()

    def write(self):
        path = self.root / "container_output/result_bundle/result.json"
        path.write_text(json.dumps(self.result))

    def test_accepts_exact_bound_result(self):
        value = MODULE.validate(self.root, self.model, self.state, self.image)
        self.assertEqual(value["status"], "FC_P003C_MIXED_LOSS_TECHNICAL_PROBE_COMPLETE")

    def test_rejects_each_tampered_binding_and_scope(self):
        changes = {
            "config_sha256": "e" * 64,
            "source_manifest_sha256": "e" * 64,
            "launch_receipt_sha256": "e" * 64,
            "cpu_equivalence_receipt_sha256": "e" * 64,
            "image_id": "wrong",
            "physicsnemo_version": "wrong",
            "regular_metadata": {},
            "pair_metadata": {},
            "force_channels": [],
            "force_channel_weights_normalized": [0.25] * 4,
            "scratch_model_state_sha256_after": "c" * 64,
            "candidate_saved": True,
        }
        for key, value in changes.items():
            with self.subTest(key=key):
                original = self.result[key]
                self.result[key] = value
                self.write()
                with self.assertRaises(ValueError):
                    MODULE.validate(self.root, self.model, self.state, self.image)
                self.result[key] = original

    def test_rejects_weighted_channel_or_step_tamper(self):
        for key, value in (
            ("paired_step_force_per_channel_weighted_contribution", [1.0] * 3),
            ("optimizer_steps", 2),
            ("chunk_count", 9),
        ):
            with self.subTest(key=key):
                original = self.result["metrics"][key]
                self.result["metrics"][key] = value
                self.write()
                with self.assertRaises(ValueError):
                    MODULE.validate(self.root, self.model, self.state, self.image)
                self.result["metrics"][key] = original


if __name__ == "__main__":
    unittest.main()
