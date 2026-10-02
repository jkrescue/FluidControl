from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


def load_module():
    path = Path(__file__).parents[1] / "scripts" / "audit_tandem_control_objective.py"
    spec = importlib.util.spec_from_file_location("control_objective_audit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_case(path: Path, split: str, start: float, omega: list[float]) -> None:
    force = np.asarray(
        [
            [1.0, 0.1, 2.0, -1.0],
            [2.0, 0.2, 3.0, 0.0],
            [3.0, 0.3, 4.0, 1.0],
        ],
        dtype=np.float32,
    )
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "time", data=np.asarray([[start], [start + 0.5], [start + 1.0]])
        )
        handle.create_dataset("omega", data=np.asarray(omega)[:, None])
        handle.create_dataset("force", data=force)
        handle.attrs["config_json"] = json.dumps(
            {
                "source_restart_time": start,
                "schedule_kind": "unit_test",
                "split": split,
            }
        )


def write_baseline(case: Path) -> None:
    rows = [
        "# Time Cd Cd(f) Cd(r) Cl",
        "80 1 0 0 -0.5",
        "80.5 2 0 0 0.0",
        "81 3 0 0 0.5",
        "81.5 2 0 0 0.0",
        "82 1 0 0 -0.5",
        "82.5 2 0 0 0.0",
        "83 3 0 0 0.5",
    ]
    for object_name in ("forceFront", "forceRear"):
        path = case / "postProcessing" / object_name / "0" / "coefficient.dat"
        path.parent.mkdir(parents=True)
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")


class TandemControlObjectiveAuditTests(unittest.TestCase):
    def test_case_metrics_use_all_four_force_channels(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.h5"
            write_case(path, "train", 80.0, [0.0, 1.0, 2.0])
            baseline_case = Path(directory) / "baseline"
            write_baseline(baseline_case)
            report, series = module.case_report(
                path,
                "sample",
                "train",
                module.load_zero_baseline(baseline_case),
                80.0,
                4.0,
            )
        self.assertAlmostEqual(report["force_mean"]["front_cd"], 2.0)
        self.assertAlmostEqual(report["force_mean"]["rear_cd"], 3.0)
        self.assertAlmostEqual(report["total_cd"]["mean"], 5.0)
        self.assertAlmostEqual(report["rear_cl_mean"], 0.0)
        self.assertAlmostEqual(report["front_cl_mean"], 0.2)
        self.assertAlmostEqual(report["rear_cl_rms_fluctuation"], np.sqrt(2.0 / 3.0))
        self.assertAlmostEqual(report["mean_omega_sq"], 5.0 / 3.0)
        self.assertAlmostEqual(report["mean_delta_omega_sq"], 1.0)
        np.testing.assert_allclose(series.action_rate, [2.0, 2.0])
        self.assertAlmostEqual(
            report["relative_to_phase_matched_zero_action"]["total_cd_ratio"],
            1.25,
        )
        self.assertAlmostEqual(
            report["relative_to_phase_matched_zero_action"]["front_cl_rms_ratio"],
            0.2,
        )

    def test_full_audit_reports_phase_coverage_and_weight_scan(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            counts = {"train": 1, "validation": 1, "test": 1}
            for split in counts:
                (root / split).mkdir(parents=True)
            (root / "manifest.json").write_text(
                json.dumps({"profile": "synthetic", "trajectory_counts": counts}),
                encoding="utf-8",
            )
            write_case(root / "train" / "a.h5", "train", 80.0, [0, 1, 2])
            write_case(root / "validation" / "b.h5", "validation", 81.0, [0, -1, -2])
            write_case(root / "test" / "c.h5", "test", 82.0, [0, 0, 0])
            baseline_case = Path(directory) / "baseline"
            write_baseline(baseline_case)
            report = module.audit(
                [root],
                baseline_case=baseline_case,
                phase_reference_time=80.0,
                shedding_period=4.0,
            )
        self.assertEqual(report["status"], "REAL_CFD_CONTROL_OBJECTIVE_AUDITED")
        self.assertEqual(len(report["cases"]), 3)
        self.assertEqual(len(report["groups"]), 3)
        self.assertEqual(
            report["phase_definition"]["unique_initial_phase_cycles"],
            [0.0, 0.25, 0.5],
        )
        self.assertEqual(len(report["pareto"]["weight_scan"]), 80)
        self.assertFalse(report["provenance"]["flow_fields_loaded"])

    def test_histogram_accounts_for_out_of_range_values(self) -> None:
        module = load_module()
        result = module.histogram(np.asarray([-2.0, 0.5, 3.0]), np.asarray([-1, 0, 1]))
        self.assertEqual(result["counts"], [0, 1])
        self.assertEqual(result["underflow"], 1)
        self.assertEqual(result["overflow"], 1)


if __name__ == "__main__":
    unittest.main()
