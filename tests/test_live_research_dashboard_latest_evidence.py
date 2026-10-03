from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "serve_live_research_dashboard.py"
if not SCRIPT.exists():
    SCRIPT = ROOT / "serve_live_research_dashboard.py"


def load_module():
    spec = importlib.util.spec_from_file_location("live_research_dashboard", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = load_module()


class LatestEvidenceDashboardTests(unittest.TestCase):
    def test_latest_evidence_uses_only_declared_final_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixtures = {
                MODULE.V4_VALIDATION_DECISION: {"split": "validation"},
                MODULE.V4_WINDOW_MEAN_CD: {"metric": "window"},
                MODULE.V3_PARENT_WINDOW_MEAN_CD: {"metric": "parent-window"},
                MODULE.TWO_PHASE_ALTERNATING: {"status": "real-cfd"},
                MODULE.LONG_DWELL075: {"status": "long-dwell-real-cfd"},
                MODULE.LOW_ACTION_PHASE94_CANONICAL: {"status": "canonical-v3"},
            }
            for relative, value in fixtures.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value), encoding="utf-8")
            self.assertEqual(
                MODULE._latest_evidence(root),
                {
                    "v4_validation_decision": {"split": "validation"},
                    "v4_window_mean_cd": {"metric": "window"},
                    "v3_parent_window_mean_cd": {"metric": "parent-window"},
                    "two_phase_alternating": {"status": "real-cfd"},
                    "long_dwell075": {"status": "long-dwell-real-cfd"},
                    "low_action_phase94_canonical": {"status": "canonical-v3"},
                },
            )

    def test_missing_latest_artifacts_are_explicit_nulls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = MODULE._latest_evidence(Path(directory))
        self.assertEqual(set(result), {
            "v4_validation_decision",
            "v4_window_mean_cd",
            "v3_parent_window_mean_cd",
            "two_phase_alternating",
            "long_dwell075",
            "low_action_phase94_canonical",
        })
        self.assertTrue(all(value is None for value in result.values()))

    def test_page_distinguishes_evidence_scopes(self) -> None:
        page = MODULE.PAGE
        for label in (
            "v4 验证集 · 第 100 步终点总阻力",
            "v4 验证集 · 100 步窗口平均总阻力",
            "两相位交替旋转 · 真实 OpenFOAM CFD",
            "v4冻结测试未访问",
            "不是代理预测或闭环结果",
            "三项联合状态",
        ):
            self.assertIn(label, page)
        self.assertIn("candidate_terminal_100step", page)
        self.assertIn("pooled_window_mean_cd_nrmse", page)
        self.assertIn("v3_parent_window_mean_cd", page)
        self.assertIn("NO GAIN", page)
        self.assertIn("canonical_joint_pass_both_phases", page)
        self.assertIn("LOW_ACTION_PHASE94_CANONICAL_PHYSICAL_AUDIT_V3_COMPLETE", page)
        self.assertIn("真实OpenFOAM，仅t94单相位开环", page)
        self.assertIn("TWO_PHASE_LONG_DWELL075_OPENFOAM_AUDIT_COMPLETED", page)
        self.assertIn("低幅长驻留±0.75（T=40）真实OpenFOAM", page)
        self.assertIn("仅预声明开环物理筛查，不是代理或闭环达标", page)
        self.assertIn("仅验证集诊断，不是零控制收益或CFD闭环成功", page)
        self.assertIn("唯一严格同初态start=0", page)
        self.assertIn("Matched-start 九案 commissioning", page)
        self.assertIn("Worker 整机 CPU", page)
        self.assertIn("RAW_TRANSFER_VERIFIED", page)
        self.assertIn("不是训练结果或控制收益", page)
        self.assertIn("RAW 已回传验收", page)
        self.assertIn("RAW 待回传验收", page)
        self.assertIn("VTK_READY 801帧", page)
        self.assertIn("HDF staging 完成", page)
        self.assertIn("Curator 实际任务", page)
        self.assertIn("HDF 完成不等于模型已训练", page)

    def test_matched_start_progress_uses_solver_time_and_end_marker(self) -> None:
        lines = [
            "ignored",
            "__MATCHED_START__",
            "matched_start_acquisition_train_b00_zero|150|1|0",
            "matched_start_acquisition_train_b02_p075|186|0|1",
            "matched_start_acquisition_train_b04_m075|120.005|0|0",
        ]
        rows = {
            row["case"]: row for row in MODULE._matched_start_progress(lines)
        }
        self.assertEqual(len(rows), 9)
        self.assertEqual(
            rows["matched_start_acquisition_train_b00_zero"]["iteration"], 400
        )
        self.assertEqual(
            rows["matched_start_acquisition_train_b00_zero"]["status"], "running"
        )
        completed = rows["matched_start_acquisition_train_b02_p075"]
        self.assertEqual(completed["iteration"], 16_000)
        self.assertEqual(completed["status"], "complete")
        self.assertNotIn("raw_transfer_verified", completed)
        self.assertEqual(
            rows["matched_start_acquisition_train_b04_m075"]["status"],
            "stopped_incomplete",
        )
        self.assertEqual(
            rows["matched_start_acquisition_train_b04_zero"]["status"], "pending"
        )

    def test_transfer_receipts_require_status_case_and_phase_sha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt_dir = root / MODULE.MATCHED_START_RECEIPTS
            receipt_dir.mkdir(parents=True)
            valid_case = "matched_start_acquisition_train_b00_zero"
            wrong_sha_case = "matched_start_acquisition_train_b00_p075"
            wrong_case_case = "matched_start_acquisition_train_b00_m075"
            common = {
                "status": "RAW_TRANSFER_VERIFIED",
                "phase_manifest_sha256": MODULE.MATCHED_START_PHASE_MANIFEST_SHA256,
            }
            (receipt_dir / f"{valid_case}.json").write_text(
                json.dumps({**common, "case": valid_case}), encoding="utf-8"
            )
            (receipt_dir / f"{wrong_sha_case}.json").write_text(
                json.dumps(
                    {
                        **common,
                        "case": wrong_sha_case,
                        "phase_manifest_sha256": "wrong",
                    }
                ),
                encoding="utf-8",
            )
            (receipt_dir / f"{wrong_case_case}.json").write_text(
                json.dumps({**common, "case": "different-case"}), encoding="utf-8"
            )
            rows = {
                row["case"]: row
                for row in MODULE._matched_start_transfer_receipts(root)
            }
        self.assertTrue(rows[valid_case]["raw_transfer_verified"])
        self.assertFalse(rows[wrong_sha_case]["raw_transfer_verified"])
        self.assertFalse(rows[wrong_sha_case]["checks"]["phase_manifest_sha256"])
        self.assertFalse(rows[wrong_case_case]["raw_transfer_verified"])
        self.assertFalse(rows[wrong_case_case]["checks"]["case"])
        self.assertFalse(
            rows["matched_start_acquisition_train_b02_zero"][
                "raw_transfer_verified"
            ]
        )

    def test_pipeline_status_reads_markers_and_path_metadata_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ready_case = "matched_start_acquisition_train_b00_zero"
            writing_case = "matched_start_acquisition_train_b00_p075"
            started_case = "matched_start_acquisition_train_b00_m075"
            conflict_case = "matched_start_acquisition_train_b02_zero"
            marker_dir = root / MODULE.MATCHED_START_VTK_READY
            marker_dir.mkdir(parents=True)
            (marker_dir / f"{ready_case}.json").write_text(
                json.dumps(
                    {"status": "VTK_READY", "case": ready_case, "frames": 801}
                ),
                encoding="utf-8",
            )
            (marker_dir / f"{writing_case}.json").write_text(
                json.dumps(
                    {"status": "VTK_READY", "case": "wrong-case", "frames": 801}
                ),
                encoding="utf-8",
            )

            def staging(case: str) -> Path:
                path = root / MODULE.MATCHED_START_STAGING / case / "train"
                path.mkdir(parents=True)
                return path

            (staging(ready_case) / f"{ready_case}.h5").touch()
            (staging(writing_case) / f"{writing_case}.h5.tmp").touch()
            conflict = staging(conflict_case)
            (conflict / f"{conflict_case}.h5").touch()
            (conflict / f"{conflict_case}.h5.tmp").touch()
            logs = root / MODULE.MATCHED_START_CURATOR_LOGS
            logs.mkdir(parents=True)
            (logs / f"{started_case}.log").write_text("started\n", encoding="utf-8")
            rows = {
                row["case"]: row for row in MODULE._matched_start_pipeline_status(root)
            }
        self.assertTrue(rows[ready_case]["vtk_ready"])
        self.assertEqual(rows[ready_case]["hdf_staging_status"], "complete")
        self.assertFalse(rows[writing_case]["vtk_ready"])
        self.assertFalse(rows[writing_case]["vtk_checks"]["case"])
        self.assertEqual(rows[writing_case]["hdf_staging_status"], "writing")
        self.assertEqual(rows[started_case]["hdf_staging_status"], "started")
        self.assertEqual(rows[conflict_case]["hdf_staging_status"], "conflict")
        self.assertEqual(
            rows["matched_start_acquisition_train_b02_p075"][
                "hdf_staging_status"
            ],
            "pending",
        )

    def test_only_canonical_low_action_audit_is_loaded(self) -> None:
        self.assertEqual(
            MODULE.LOW_ACTION_PHASE94_CANONICAL.name,
            "low_action_phase94_physical_audit_v3_canonical.json",
        )
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn('"low_action_phase94_physical_audit.json"', source)
        self.assertNotIn('"low_action_phase94_physical_audit_v2_canonical.json"', source)

    def test_negative_drag_reduction_is_presented_as_increase(self) -> None:
        self.assertIn("value>=0?`降阻 ${pct(value)}`:`增阻 ${pct(-value)}`", MODULE.PAGE)

    def test_worker_single_step_reference_uses_pooled_nrmse(self) -> None:
        rows = [
            {"segments": 1, "total_drag_rmse": 1.0, "total_drag_target_rms": 2.0},
            {"segments": 3, "total_drag_rmse": 2.0, "total_drag_target_rms": 4.0},
        ]
        self.assertAlmostEqual(MODULE._pooled_terminal_nrmse(rows), 0.5)
        self.assertIsNone(MODULE._pooled_terminal_nrmse([]))
        self.assertIn("不是H20", MODULE.PAGE)

    def test_low_action_fno_summary_joins_audited_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pairwise = {
                "status": "LOW_ACTION_PAIRWISE_FNO_H100_AUDIT_COMPLETE",
                "models": {
                    name: {
                        "strict_common_initial_pairwise_absolute_error": error,
                        "strict_common_initial_ranking_correct": True,
                    }
                    for name, error in (("v3_parent", 0.09), ("v4_candidate", 0.11))
                },
            }
            pair_path = root / MODULE.LOW_ACTION_FNO_PAIRWISE
            pair_path.parent.mkdir(parents=True)
            pair_path.write_text(json.dumps(pairwise), encoding="utf-8")
            for directory_name, pooled_nrmse, mae in (
                ("v3_h20_parent", 0.16, 0.35),
                ("v4_h20_candidate", 0.18, 0.37),
            ):
                result_dir = root / MODULE.LOW_ACTION_FNO_ROOT / directory_name
                result_dir.mkdir(parents=True)
                evaluation = {
                    "split": "validation",
                    "action_mode": "observed",
                    "summary": {
                        "100": {
                            "total_drag_mae": mae,
                            "persistence_total_drag_mae": 0.22,
                        }
                    },
                }
                pooled = {
                    "split": "validation",
                    "horizons": {
                        "100": {"pooled": {"total_drag_nrmse_pooled": pooled_nrmse}}
                    },
                }
                (result_dir / "evaluation.json").write_text(
                    json.dumps(evaluation), encoding="utf-8"
                )
                (result_dir / "pooled_audit.json").write_text(
                    json.dumps(pooled), encoding="utf-8"
                )
            result = MODULE._low_action_fno_summary(root)
            self.assertEqual(result["split"], "validation")
            self.assertAlmostEqual(
                result["models"]["v3_parent"]["h100_pooled_total_drag_nrmse"],
                0.16,
            )
            self.assertAlmostEqual(
                result["models"]["v4_candidate"]["h100_mae_minus_persistence"],
                0.15,
            )
            self.assertAlmostEqual(
                result["models"]["v4_candidate"][
                    "strict_start0_pairwise_delta_absolute_error"
                ],
                0.11,
            )


if __name__ == "__main__":
    unittest.main()
