"""Tests for the fail-closed v4 validation-only comparison audit."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_control_gap_v4_validation.py"
SPEC = importlib.util.spec_from_file_location("v4_validation_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ValidationOnlyAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.v4 = self.root / "tandem_cylinders_control_gap_v4"
        self.v3 = self.root / "tandem_cylinders_gate_b_aug_v3"
        self.candidate = self.root / "candidate"
        self.parent = self.root / "parent"
        for data in (self.v4, self.v3):
            (data / "validation").mkdir(parents=True)
            for index in range(4):
                (data / "validation" / f"validation_{index:02d}.h5").write_bytes(
                    f"real-cfd-{index}".encode()
                )
        (self.v4 / "manifest.json").write_text(
            json.dumps(
                {
                    "profile": "control_gap_v4",
                    "trajectory_counts": {"train": 28, "validation": 4, "test": 5},
                    "max_abs_omega": 5.0,
                }
            )
        )
        (self.v3 / "manifest.json").write_text(
            json.dumps(
                {
                    "profile": "gate_b_aug_v3",
                    "trajectory_counts": {"train": 26, "validation": 4, "test": 5},
                    "max_abs_omega": 5.0,
                }
            )
        )
        normalization = {
            "computed_from": "train split only",
            "state_mean": [0.0, 0.0, 0.0],
            "state_std": [1.0, 1.0, 1.0],
            "all_force_channels": [
                "front_cd",
                "front_cl",
                "rear_cd",
                "rear_cl",
            ],
            "all_force_mean": [0.0, 0.0, 0.0, 0.0],
            "all_force_std": [1.0, 1.0, 1.0, 1.0],
        }
        (self.v4 / "normalization.json").write_text(json.dumps(normalization))
        parent_normalization = normalization | {"state_mean": [0.1, 0.0, 0.0]}
        (self.v3 / "normalization.json").write_text(json.dumps(parent_normalization))
        architecture = {
            "in_channels": 6,
            "out_channels": 7,
            "latent_channels": 48,
            "num_fno_layers": 5,
            "num_fno_modes": [32, 32],
            "decoder_layers": 2,
            "decoder_layer_size": 128,
            "padding": 8,
            "coord_features": True,
        }
        for directory, data_root, epoch in (
            (self.candidate, self.v4, 5),
            (self.parent, self.v3, 9),
        ):
            (directory / "best").mkdir(parents=True)
            (directory / "resolved_config.yaml").write_text(
                json.dumps(
                    {
                        "data": {
                            "root": str(data_root),
                            "force_indices": [0, 1, 2, 3],
                        },
                        "model": architecture,
                    }
                )
            )
            (directory / "best" / f"FNO.0.{epoch}.mdlus").write_bytes(b"model")
            (directory / "best" / f"checkpoint.0.{epoch}.pt").write_bytes(b"state")
        (self.candidate / "training_history.json").write_text(
            json.dumps([{"epoch": epoch} for epoch in range(1, 6)])
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_preflight_pins_validation_and_checkpoint_provenance(self) -> None:
        report = MODULE.preflight(self.v4, self.v3, self.candidate, self.parent)
        self.assertEqual(report["status"], "VALIDATION_ONLY_PREFLIGHT_PASS")
        self.assertEqual(report["candidate_checkpoint"]["epoch"], 5)
        self.assertEqual(len(report["validation_trajectory_sha256"]), 4)
        self.assertIn(
            "its own train-only normalization", report["normalization_policy"]
        )
        self.assertNotEqual(
            report["v4_normalization_sha256"],
            report["v3_parent_normalization_sha256"],
        )

    def test_preflight_rejects_incomplete_training(self) -> None:
        (self.candidate / "training_history.json").write_text(
            json.dumps([{"epoch": epoch} for epoch in range(1, 5)])
        )
        with self.assertRaisesRegex(ValueError, "incomplete"):
            MODULE.preflight(self.v4, self.v3, self.candidate, self.parent)

    def test_preflight_rejects_changed_validation_data(self) -> None:
        (self.v4 / "validation" / "validation_03.h5").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "not byte-identical"):
            MODULE.preflight(self.v4, self.v3, self.candidate, self.parent)

    def evaluation(self, values: list[float], normalization: Path) -> dict:
        return {
            "split": "validation",
            "evaluation_data": str(self.v4),
            "normalization_data": str(normalization),
            "segment_stride": 25,
            "action_mode": "observed",
            "force_indices": [0, 1, 2, 3],
            "checkpoint_epoch": 5,
            "summary": {
                str(horizon): {
                    "total_drag_nrmse": value,
                    "stable": True,
                    "failed_segments": 0,
                }
                for horizon, value in zip(MODULE.HORIZONS, values, strict=True)
            },
        }

    def test_compare_uses_declared_horizons_and_checkpoint_normalizations(self) -> None:
        candidate_path = self.root / "candidate.json"
        parent_path = self.root / "parent.json"
        preflight_path = self.root / "preflight.json"
        candidate_path.write_text(
            json.dumps(self.evaluation([0.01, 0.02, 0.08, 0.09], self.v4))
        )
        parent_path.write_text(
            json.dumps(self.evaluation([0.02, 0.03, 0.09, 0.11], self.v3))
        )
        preflight_path.write_text(
            json.dumps(
                {
                    "candidate_checkpoint": {"epoch": 5},
                    "parent_checkpoint": {"epoch": 9},
                }
            )
        )
        report = MODULE.compare(
            candidate_path,
            parent_path,
            preflight_path,
            str(self.v4),
            str(self.v4),
            str(self.v3),
        )
        self.assertEqual(report["status"], "V4_VALIDATION_100STEP_IMPROVED")
        self.assertAlmostEqual(report["candidate_minus_parent_nrmse"]["100"], -0.02)
        self.assertIn("own training", report["normalization_policy"])
        self.assertIn("same physical-unit", report["common_denominator_basis"])

    def test_compare_rejects_non_validation_report(self) -> None:
        report = self.evaluation([0.01, 0.02, 0.08, 0.09], self.v4)
        report["split"] = "test"
        with self.assertRaisesRegex(ValueError, "split"):
            MODULE.validate_evaluation(report, str(self.v4), str(self.v4))


if __name__ == "__main__":
    unittest.main()
