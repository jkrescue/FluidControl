"""Tests for strict pooled Gate-B terminal-drag metric auditing."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py

SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_gate_b_metric_integrity.py"
SPEC = importlib.util.spec_from_file_location("gate_b_metric_integrity", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class GateBMetricIntegrityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.normalization = self.root / "normalization"
        (self.data / "test").mkdir(parents=True)
        self.normalization.mkdir()
        for name in ("case_a", "case_b"):
            with h5py.File(self.data / "test" / f"{name}.h5", "w") as handle:
                handle.create_dataset("force", data=[[0.0, 0.0, 0.0, 0.0]])
                handle.attrs["force_channels"] = json.dumps(MODULE.FORCE_CHANNELS)
                handle.attrs["split"] = "test"
        (self.data / "manifest.json").write_text(
            json.dumps(
                {
                    "profile": "gate_b_aug_v3",
                    "trajectory_counts": {"test": 2},
                    "max_abs_omega": 5.0,
                }
            )
        )
        (self.normalization / "manifest.json").write_text(
            json.dumps({"profile": "gate_b_aug_v3", "max_abs_omega": 5.0})
        )
        stats = {
            "computed_from": "train split only",
            "state_mean": [0.0, 0.0, 0.0],
            "state_std": [1.0, 1.0, 1.0],
            "all_force_channels": list(MODULE.FORCE_CHANNELS),
            "all_force_mean": [0.0, 0.0, 0.0, 0.0],
            "all_force_std": [1.0, 1.0, 1.0, 1.0],
        }
        self.normalization_path = self.normalization / "normalization.json"
        self.normalization_path.write_text(json.dumps(stats))
        self.checkpoint = self.root / "model" / "best" / "FNO.0.9.mdlus"
        self.checkpoint.parent.mkdir(parents=True)
        self.checkpoint.write_bytes(b"official-physicsnemo-checkpoint")
        self.report_path = self.root / "evaluation.json"
        self.report_path.write_text(json.dumps(self.report()))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def row(segments: int, rmse: float, target_rms: float = 1.0) -> dict:
        return {
            "segments": segments,
            "total_drag_rmse": rmse,
            "total_drag_target_rms": target_rms,
            "total_drag_nrmse": rmse / target_rms,
            "stable": True,
            "failed_segments": 0,
        }

    def report(self) -> dict:
        return {
            "split": "test",
            "evaluation_data": str(self.data),
            "normalization_data": str(self.normalization),
            "checkpoint_dir": str(self.checkpoint.parent),
            "checkpoint_epoch": 9,
            "checkpoint_metadata": {
                "action_scale": 5.0,
                "force_channels": list(MODULE.FORCE_CHANNELS),
                "force_indices": list(MODULE.FORCE_INDICES),
                "model_config": {"in_channels": 6, "out_channels": 7},
            },
            "action_scale": 5.0,
            "force_channels": list(MODULE.FORCE_CHANNELS),
            "force_indices": list(MODULE.FORCE_INDICES),
            "cases": [
                {"case": "case_a", "horizons": {"100": self.row(1, 0.05)}},
                {"case": "case_b", "horizons": {"100": self.row(9, 0.11)}},
            ],
            "summary": {
                "100": {
                    "total_drag_nrmse": 0.08,
                    "segments": 10,
                    "stable": True,
                    "failed_segments": 0,
                }
            },
        }

    def audit(self) -> dict:
        return MODULE.audit_report(
            self.report_path,
            self.data,
            self.normalization,
            self.checkpoint,
            expected_split="test",
            expected_profile="gate_b_aug_v3",
            expected_normalization_profile="gate_b_aug_v3",
            expected_case_count=2,
            expected_checkpoint_sha256=MODULE.sha256(self.checkpoint),
            expected_normalization_sha256=MODULE.sha256(self.normalization_path),
        )

    def test_pooled_metric_prevents_macro_false_pass(self) -> None:
        result = self.audit()
        self.assertAlmostEqual(result["macro_total_drag_nrmse"], 0.08)
        self.assertGreater(result["pooled_total_drag_nrmse"], 0.10)
        self.assertEqual(result["status"], "TERMINAL_FORCE_GATE_B_ACCURACY_FAIL")
        self.assertEqual(result["worst_case"]["case"], "case_b")

    def test_rejects_split_or_case_contract_mismatch(self) -> None:
        report = self.report()
        report["split"] = "validation"
        self.report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "split mismatch"):
            self.audit()

    def test_rejects_profile_mismatch(self) -> None:
        manifest = MODULE.load_json(self.data / "manifest.json")
        manifest["profile"] = "wrong"
        (self.data / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "profile mismatch"):
            self.audit()

    def test_rejects_checkpoint_digest_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "checkpoint SHA-256"):
            MODULE.audit_report(
                self.report_path,
                self.data,
                self.normalization,
                self.checkpoint,
                expected_split="test",
                expected_profile="gate_b_aug_v3",
                expected_normalization_profile="gate_b_aug_v3",
                expected_case_count=2,
                expected_checkpoint_sha256="0" * 64,
                expected_normalization_sha256=MODULE.sha256(self.normalization_path),
            )

    def test_rejects_normalization_contract_mismatch(self) -> None:
        stats = MODULE.load_json(self.normalization_path)
        stats["all_force_channels"] = ["rear_cd", "rear_cl"]
        self.normalization_path.write_text(json.dumps(stats))
        with self.assertRaisesRegex(ValueError, "force channel order"):
            self.audit()

    def test_rejects_hdf5_force_channel_order_mismatch(self) -> None:
        path = self.data / "test" / "case_b.h5"
        with h5py.File(path, "r+") as handle:
            handle.attrs["force_channels"] = json.dumps(
                ["rear_cd", "rear_cl", "front_cd", "front_cl"]
            )
        with self.assertRaisesRegex(ValueError, "HDF5 force_channels order"):
            self.audit()

    def test_main_refuses_to_overwrite_output(self) -> None:
        output = self.root / "existing.json"
        output.write_text("do-not-overwrite")
        with (
            patch.object(
                MODULE,
                "parse_args",
                return_value=SimpleNamespace(output=output),
            ),
            self.assertRaisesRegex(FileExistsError, "refusing to overwrite"),
        ):
            MODULE.main()
        self.assertEqual(output.read_text(), "do-not-overwrite")


if __name__ == "__main__":
    unittest.main()
