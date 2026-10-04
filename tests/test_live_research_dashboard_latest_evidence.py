from __future__ import annotations

import importlib.util
import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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
    def test_dynamic_candidate_requires_bound_gate_and_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "artifacts/tandem_fno_dynamic_paired_interleaved_lambda10_20261005/posteval_fc_p003b"
            (base / "validation10").mkdir(parents=True)
            self.assertIsNone(MODULE._completed_interleaved_candidate(root, dynamic=True))
            checkpoint = "b" * 64
            hashes = {}
            for relative, payload in (
                ("validation10/endpoint_gate.json", {"checkpoint_sha256": checkpoint}),
                ("development_gate.json", {"checkpoint_sha256": checkpoint,
                 "status": "DYNAMIC_FNO_DEVELOPMENT_ADMISSION_FAIL"}),
            ):
                raw = json.dumps(payload).encode()
                (base / relative).write_bytes(raw)
                hashes[relative] = hashlib.sha256(raw).hexdigest()
            receipt = {"status": "FC_P003B_POSTEVAL_COMPLETE", "checkpoint_sha256": checkpoint,
                       "candidate_kind": "dynamic_paired_interleaved_lambda10",
                       "frozen_test_accessed": False, "sha256": hashes}
            (base / "receipt.json").write_text(json.dumps(receipt))
            self.assertTrue(MODULE._completed_interleaved_candidate(root, dynamic=True)["receipt_bound"])
            receipt["frozen_test_accessed"] = True
            (base / "receipt.json").write_text(json.dumps(receipt))
            self.assertIsNone(MODULE._completed_interleaved_candidate(root, dynamic=True))
            receipt["frozen_test_accessed"] = False
            (base / "receipt.json").write_text(json.dumps(receipt))
            (base / "development_gate.json").write_text("{}")
            self.assertIsNone(MODULE._completed_interleaved_candidate(root, dynamic=True))

    def test_current_candidate_requires_complete_matching_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertIsNone(MODULE._completed_interleaved_candidate(root))
            base = root / "artifacts/tandem_fno_paired_stats_interleaved_lambda10_20261005/posteval_fc_p003"
            (base / "validation10").mkdir(parents=True)
            checkpoint = "a" * 64
            status = "DYNAMIC_FNO_DEVELOPMENT_ADMISSION_FAIL"
            hashes = {}
            for relative, payload in (
                ("validation10/endpoint_gate.json", {"checkpoint_sha256": checkpoint}),
                ("development_gate.json", {"checkpoint_sha256": checkpoint, "status": status}),
            ):
                raw = json.dumps(payload).encode()
                (base / relative).write_bytes(raw)
                hashes[relative] = hashlib.sha256(raw).hexdigest()
            receipt = {"status": "FC_P003_POSTEVAL_COMPLETE", "checkpoint_sha256": checkpoint,
                       "development_gate_status": status, "sha256": hashes}
            (base / "receipt.json").write_text(json.dumps(receipt))
            result = MODULE._completed_interleaved_candidate(root)
            self.assertTrue(result["receipt_bound"])
            self.assertEqual(result["development"]["status"], status)
            self.assertNotIn("ppo_authorized", result)
            (base / "development_gate.json").write_text("{}")
            self.assertIsNone(MODULE._completed_interleaved_candidate(root))

    def test_resource_sampler_detects_paired_trainer_not_guard(self) -> None:
        lines = ["cpu 1 2 3 4 5", "MemTotal: 128000000 kB", "MemAvailable: 64000000 kB",
                 "96, 64, 42", "__TASKS__",
                 "python python -u scripts/spark_gpu_guard.py -- python scripts/train_tandem_fno_paired_stats.py",
                 "python python -u scripts/train_tandem_fno_paired_stats.py --config-name tandem_fno_dynamic_paired_interleaved_h100"]
        for marker in ("__EPOCH__", "__V3_WORKER_EPOCH__", "__V3_ROLLOUT_EPOCH__",
                       "__V3_H20_EPOCH__", "__V3_PRIMARY_SEED_H20_EPOCH__",
                       "__V3_H20_REAR_DRAG_EPOCH__", "__V4_EPOCH__", "__V4_SINGLE_VALIDATION__"):
            lines.extend([marker, "0"])
        result, _ = MODULE._parse_host("\n".join(lines), None)
        self.assertEqual(result["tasks"], ["PhysicsNeMo FNO 动态配对训练"])
        self.assertEqual(result["task_count"], 1)

    def test_current_training_log_is_bounded_evidence_not_live_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertIsNone(MODULE._current_training_log(root)["batch_percent"])
            run = root / "artifacts/tandem_fno_paired_stats_interleaved_lambda10_20261005"
            run.mkdir(parents=True)
            (run / "training_history.json").write_text('[{"epoch": 1}]')
            (run / "train.log").write_text(
                '[2026-10-04 16:35:52,915][train][INFO] - [43.86%] Mini-Batch Losses: loss = 1.471e-02\n'
            )
            result = MODULE._current_training_log(root)
            self.assertEqual(result["completed_epochs"], 1)
            self.assertEqual(result["batch_percent"], 43.86)
            self.assertEqual(result["logged_at_utc"], "2026-10-04T16:35:52.915000+00:00")
            self.assertNotIn("running", result)
            (run / "train.log").write_text(
                '[2026-10-04 16:35:52,915][train][INFO] - [143.86%] Mini-Batch Losses: loss = 1\n'
            )
            self.assertIsNone(MODULE._current_training_log(root)["batch_percent"])

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
            "旧 v4 历史严格证据（非 full40 新链）",
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
        self.assertIn("Matched-start 新增 31 案采集", page)
        self.assertIn("求解完成不等于 RAW QC 通过", page)
        self.assertIn("冻结测试仅显示采集与验收状态", page)
        self.assertIn("31 案 CFD/RAW 也不代表 Curator、模型训练或精度提升", page)
        self.assertIn("PPO 禁止宣称收益", page)
        self.assertIn("Full40 / dev30 · 当前模型修复实验", page)
        self.assertIn("当前主线：官方 PhysicsNeMo FNO", page)
        self.assertIn("未训练即明确BLOCKED", page)
        self.assertIn("val rollout field/force", page)
        self.assertIn("Spark本机已验收HDF", page)
        self.assertIn("不是上方 v4 validation 评估", SCRIPT.read_text(encoding="utf-8"))
        self.assertIn("当前模型修复实验", page)
        self.assertIn("A · Spark · 训练 H20 / 验证 H100", page)
        self.assertIn("B · Worker · 训练 H50 / 验证 H100", page)
        self.assertIn("HydroGym + 真实 OpenFOAM PPO", page)
        self.assertIn("实现与测试中", page)
        self.assertIn("renderFreeAR(d)", page)
        self.assertIn("反归一化流场综合MAE / 四个力系数平均MAE", page)
        self.assertIn("不是减阻率或正式Gate", page)
        self.assertIn("首次后评估仅因可视化写入只读路径失败", page)
        self.assertIn("训练内stride100终点pooled总Cd NRMSE", page)
        self.assertIn("不等于stride25完整评估", page)
        self.assertIn("FNO未用于奖励", page)
        self.assertIn("原始未平滑时序", page)
        self.assertIn("canonical_physical_joint_check", page)
        self.assertIn("计划8、实际5", page)
        self.assertIn("动态H100", page)

    def test_dual_node_watchdog_is_exposed_without_replacing_science_metrics(self) -> None:
        page = MODULE.PAGE
        for label in (
            "双节点运行守护",
            "连续 300 秒无有效项目计算时告警",
            "Spark 保留 20 GiB、Worker 保留 40 GiB",
            "只告警，不执行",
            "正式闭环研究",
            "Train20 固定动作物理对照 · TRAIN ONLY",
            "不替代九案 commissioning",
        ):
            self.assertIn(label, page)
        for existing in (
            "旧 v4 历史严格证据",
            "真实 CFD 收益与代理决策是否一致",
            "FNO 训练和推理结果",
            "HydroGym 闭环控制",
        ):
            self.assertIn(existing, page)
        self.assertIn("renderDualWatchdog(d)", page)
        self.assertIn('data["dual_node_watchdog"]', SCRIPT.read_text(encoding="utf-8"))
        self.assertIn('data["full40_train20_physics"]', SCRIPT.read_text(encoding="utf-8"))
        self.assertIn("FULL40_TRAIN20_OPEN_LOOP_PHYSICS_SUMMARY", page)
        self.assertIn("validation/frozen结果读取", page)

    def test_full40_development_chain_reads_only_fixed_development_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nine = root / MODULE.MATCHED9_FINAL
            (nine / "train").mkdir(parents=True)
            (nine / "commissioning_qc.json").write_text(
                json.dumps(
                    {"status": "MATCHED_START_COMMISSIONING_NINE_CASE_QC_OK"}
                ),
                encoding="utf-8",
            )
            for case in MODULE.MATCHED_START_CASES:
                (nine / "train" / f"{case}.h5").touch()
            for case, (split, _, _) in MODULE.FULL40_CASES.items():
                if split not in {"train", "validation"}:
                    continue
                path = root / MODULE.FULL40_STAGING / case / split
                path.mkdir(parents=True)
                (path / f"{case}.h5").touch()

            one = (
                root
                / "artifacts"
                / f"{MODULE.DEV30_QUICKSCREEN_PREFIX}onestep_qs1"
            )
            h20 = (
                root / "artifacts" / f"{MODULE.DEV30_QUICKSCREEN_PREFIX}h20_qs1"
            )
            one.mkdir(parents=True)
            h20.mkdir(parents=True)
            (one / "training_history.json").write_text(
                json.dumps(
                    [
                        {"epoch": index, "force_mae_normalized": 0.1 / index}
                        for index in range(1, 11)
                    ]
                ),
                encoding="utf-8",
            )
            (h20 / "training_history.json").write_text(
                json.dumps(
                    [
                        {"epoch": index, "selection_score": 0.05 / index}
                        for index in range(1, 4)
                    ]
                ),
                encoding="utf-8",
            )
            full40 = {
                "cases": [
                    {"raw_qc_verified": index < 29} for index in range(31)
                ]
            }
            result = MODULE._full40_development_chain(root, full40)
        self.assertEqual(result["raw_qc"]["full40_verified"], 29)
        self.assertTrue(result["raw_qc"]["commissioning_qc_pass"])
        self.assertEqual(result["development_hdf"]["train_ready"], 20)
        self.assertEqual(result["development_hdf"]["validation_ready"], 10)
        self.assertFalse(result["dev30_release"]["published"])
        self.assertEqual(result["quickscreen"]["onestep"]["epoch"], 10)
        self.assertEqual(result["quickscreen"]["h20"]["epoch"], 3)
        self.assertEqual(result["canonical_ppo"]["status"], "BLOCKED_NOT_STARTED")
        self.assertFalse(result["frozen_hdf_enumerated_or_opened"])

    def test_dev30_release_and_diagnostic_require_strict_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release = root / MODULE.DEV30_RELEASE
            release.mkdir(parents=True)
            (release / "manifest.json").write_text(
                json.dumps(
                    {
                        "profile": "matched_start_full40_v1",
                        "release_kind": "immutable_development_train20_validation10",
                        "materialized_trajectory_counts": {
                            "train": 20,
                            "validation": 10,
                        },
                        "frozen_test_materialized": False,
                    }
                ),
                encoding="utf-8",
            )
            diagnostic = (
                root
                / "artifacts/tandem_cylinders"
                / f"{MODULE.DEV30_DIAGNOSTIC_PREFIX}qs1"
            )
            diagnostic.mkdir(parents=True)
            (diagnostic / "diagnostic.json").write_text(
                json.dumps(
                    {
                        "status": "DEV30_VALIDATION_DIAGNOSTIC_COMPLETE",
                        "formal_gate": False,
                        "horizons": {"100": {"pooled_total_cd_nrmse": 0.09}},
                    }
                ),
                encoding="utf-8",
            )
            result = MODULE._full40_development_chain(root, {"cases": []})
        self.assertTrue(result["dev30_release"]["published"])
        self.assertEqual(
            result["validation_diagnostic"]["horizons"]["100"][
                "pooled_total_cd_nrmse"
            ],
            0.09,
        )

    def test_free_ar_ablation_reads_only_fixed_runs_and_sync_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            h20 = (
                root
                / "artifacts/tandem_fno_full40_free_ar_h20_ar20_20261003"
            )
            h50 = (
                root
                / "artifacts/tandem_fno_full40_free_ar_h50_ar50_20261003"
            )
            h20.mkdir(parents=True)
            h50.mkdir(parents=True)
            formal_train16 = (
                root / "artifacts/tandem_fno_control_train16_h100_20261004"
            )
            formal_train16.mkdir(parents=True)
            (formal_train16 / "training_history.json").write_text(
                json.dumps([{"epoch": 1, "selection_score": 0.125}]),
                encoding="utf-8",
            )
            probe = root / "artifacts/tandem_fno_control_train16_h100_probe_20261004"
            probe.mkdir(parents=True)
            (probe / "training_history.json").write_text(
                json.dumps([{"epoch": 99, "selection_score": 0.0}]),
                encoding="utf-8",
            )
            (h20 / "training_history.json").write_text(
                json.dumps(
                    [
                        {
                            "epoch": 2,
                            "selection_score": 1.25,
                            "terminal_state_mae": 0.5,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (h50 / "training_history.json").write_text(
                json.dumps(
                    [
                        {
                            "epoch": 1,
                            "selection_score": 2.5,
                            "terminal_force_mae": 0.75,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (h50 / "worker_epoch_01_sync.json").write_text(
                "{}", encoding="utf-8"
            )
            dynamic = (
                root
                / "artifacts/tandem_cylinders/full40_dynamic6_fno_e5_20261003"
            )
            dynamic.mkdir(parents=True)
            (dynamic / "diagnostic.json").write_text(
                json.dumps(
                    {
                        "status": "DYNAMIC6_FNO_DIAGNOSTIC_FAIL",
                        "pooled_h100_total_cd_nrmse": 45660.0,
                        "ppo_authorized": False,
                        "frozen_test_accessed": False,
                    }
                ),
                encoding="utf-8",
            )
            h50_validation = (
                root
                / "artifacts/tandem_fno_full40_free_ar_h50_epoch1_eval_20261003/validation10"
            )
            h50_validation.mkdir(parents=True)
            (h50_validation / "diagnostic.json").write_text(
                json.dumps(
                    {
                        "status": "DEV30_VALIDATION_DIAGNOSTIC_COMPLETE",
                        "horizons": {
                            "100": {
                                "pooled_total_cd_nrmse": 0.115,
                                "macro_total_cd_nrmse": 0.071,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(MODULE, "_service_state", return_value="active"):
                result = MODULE._free_ar_ablation(root)
        self.assertEqual(result["h20"]["epoch"], 2)
        self.assertEqual(result["h20"]["last_metrics"]["selection_score"], 1.25)
        self.assertEqual(result["h20"]["previous_failed_service_state"], "active")
        self.assertEqual(result["h50"]["epoch"], 1)
        self.assertEqual(result["h50"]["synced_epochs"], 1)
        self.assertFalse(result["h50"]["final_sync_complete"])
        self.assertEqual(result["dynamic_h100"]["expected_epochs"], 2)
        self.assertEqual(result["control_train16_h100"]["epoch"], 1)
        self.assertEqual(
            result["control_train16_h100"]["last_metrics"]["selection_score"],
            0.125,
        )
        self.assertFalse(
            result["control_train16_h100"]["technical_probe_formal_candidate"]
        )
        self.assertEqual(
            result["h50"]["validation10"]["horizons"]["100"][
                "pooled_total_cd_nrmse"
            ],
            0.115,
        )
        self.assertEqual(
            result["dynamic6_fno"]["status"],
            "DYNAMIC6_FNO_DIAGNOSTIC_FAIL",
        )
        self.assertFalse(result["direct_cfd_ppo"]["physical_result_available"])
        self.assertFalse(result["frozen_hdf_opened_or_enumerated"])

    def test_free_ar_marks_worker_epoch5_as_intentional_reallocation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = root / "artifacts/worker_audit/B_E5_REALLOCATION_RECEIPT.json"
            receipt.parent.mkdir(parents=True)
            receipt.write_text(
                json.dumps(
                    {"status": "B_E5_FROZEN_FOR_DYNAMIC_H100_REALLOCATION"}
                ),
                encoding="utf-8",
            )
            h50 = root / "artifacts/tandem_fno_full40_free_ar_h50_ar50_20261003"
            h50.mkdir(parents=True)
            (h50 / "training_history.json").write_text(
                json.dumps([{"epoch": 5, "selection_score": 0.041}]),
                encoding="utf-8",
            )
            with patch.object(MODULE, "_service_state", return_value="inactive"):
                result = MODULE._free_ar_ablation(root)
        self.assertEqual(
            result["h50"]["status"], "INTENTIONALLY_REALLOCATED_AT_EPOCH5"
        )
        self.assertEqual(result["h50"]["epoch"], 5)

    def test_direct_cfd_ppo_reads_tail_and_after_update_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "artifacts/direct_cfd/directppo2048_v1"
            training = run / "training"
            training.mkdir(parents=True)
            for index, step in ((0, 41), (1, 40)):
                (run / f"worker_env{index}.jsonl").write_text(
                    "not-json\n"
                    + json.dumps({"event": "reset", "episode": 2})
                    + "\n"
                    + json.dumps(
                        {
                            "event": "step",
                            "env_index": index,
                            "episode": 2,
                            "step": step,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
            (training / "progress.json").write_text(
                json.dumps(
                    {
                        "status": "DIRECT_REAL_CFD_PPO_RUNNING",
                        "completed_transitions": 256,
                        "checkpoints": [
                            {
                                "timesteps": 256,
                                "ppo_update_count": 1,
                                "raw_physical_reward": {"mean": -0.25},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            result = MODULE._direct_cfd_ppo_status(root)
        self.assertTrue(result["running"])
        self.assertEqual(result["completed_transitions"], 256)
        self.assertEqual(result["live_collection_steps"], 81)
        self.assertEqual(result["ppo_update_count"], 1)
        self.assertEqual(
            result["last_checkpoint"]["raw_physical_reward"]["mean"], -0.25
        )
        self.assertFalse(result["fno_used_for_reward"])
        self.assertFalse(result["frozen_test_accessed"])

    def test_h20_posteval_distinguishes_retry_from_training_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            retrying = MODULE._h20_posteval_stage(run, active=True)
            self.assertEqual(retrying["status"], "RETRY_RUNNING")
            self.assertTrue(retrying["model_training_complete"])
            validation = run / "validation10"
            validation.mkdir()
            (validation / "evaluation.json").write_text(
                json.dumps(
                    {"summary": {str(value): {} for value in (1, 10, 50, 100)}}
                ),
                encoding="utf-8",
            )
            complete = MODULE._h20_posteval_stage(run, active=False)
        self.assertEqual(complete["status"], "COMPLETE")
        self.assertTrue(complete["complete"])

    def test_direct_cfd_ppo_exposes_completed_training_and_pair_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "artifacts/direct_cfd/directppo2048_v1"
            training = run / "training"
            training.mkdir(parents=True)
            (run / "worker_env0.jsonl").write_text(
                json.dumps({"event": "step", "step": 128}) + "\n",
                encoding="utf-8",
            )
            (training / "progress.json").write_text(
                json.dumps(
                    {
                        "completed_transitions": 2048,
                        "checkpoints": [{"timesteps": 2048}],
                    }
                ),
                encoding="utf-8",
            )
            (training / "result.json").write_text(
                json.dumps({"status": "DIRECT_REAL_CFD_PPO_TRAINING_COMPLETE"}),
                encoding="utf-8",
            )
            pair = root / "artifacts/direct_cfd/directppo2048_b00_eval80_v1"
            (pair / "rollout").mkdir(parents=True)
            (pair / "rollout/progress.json").write_text(
                json.dumps({"completed_steps_per_branch": 300}), encoding="utf-8"
            )
            (pair / "physical_result.json").write_text(
                json.dumps(
                    {
                        "status": "DIRECT_CFD_B00_FROZEN_PPO_PAIR_EVALUATED",
                        "comparison": {"canonical_physical_joint_check": False},
                    }
                ),
                encoding="utf-8",
            )
            (pair / "raw_pair_timeseries.png").write_bytes(b"png")
            independent = (
                root / "artifacts/direct_cfd/directppo2048_b01_eval80_v1"
            )
            independent.mkdir(parents=True)
            (independent / "physical_result.json").write_text(
                json.dumps(
                    {
                        "status": "DIRECT_CFD_B01_FROZEN_PPO_PAIR_EVALUATED",
                        "comparison": {
                            "canonical_physical_joint_check": True,
                            "total_drag_reduction": 0.0425,
                        },
                    }
                ),
                encoding="utf-8",
            )
            replay = root / "artifacts/direct_cfd/b00seq_b01_openloop_v1"
            replay.mkdir(parents=True)
            (replay / "audit_receipt.json").write_text(
                json.dumps(
                    {
                        "status": "B00_ACTIONS_B01_OPENLOOP_INDEPENDENT_AUDIT_PASS",
                        "physical_summary": {
                            "openloop_drag_reduction_vs_zero": -0.00744,
                            "feedback_drag_reduction_vs_zero": 0.0425,
                        },
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(MODULE, "_service_state", return_value="active"):
                result = MODULE._direct_cfd_ppo_status(root)
        self.assertTrue(result["training_complete"])
        self.assertFalse(result["running"])
        self.assertEqual(
            result["paired_80d_evaluation"]["completed_steps_per_branch"], 300
        )
        self.assertEqual(
            result["paired_80d_evaluation"]["physical_result"]["status"],
            "DIRECT_CFD_B00_FROZEN_PPO_PAIR_EVALUATED",
        )
        self.assertTrue(result["paired_80d_evaluation"]["figure"]["available"])
        self.assertEqual(
            result["paired_80d_evaluation"]["figure"]["path"],
            "/direct-cfd-pair.png",
        )
        self.assertEqual(
            result["independent_b01_evaluation"]["physical_result"]["status"],
            "DIRECT_CFD_B01_FROZEN_PPO_PAIR_EVALUATED",
        )
        self.assertIn(
            "statistical independence is unproven",
            result["independent_b01_evaluation"]["scope"],
        )
        self.assertEqual(
            result["b00_sequence_b01_replay"]["status"],
            "B00_ACTIONS_B01_OPENLOOP_INDEPENDENT_AUDIT_PASS",
        )

    def test_current_cards_label_b5_screen_and_executed_action_ramp(self) -> None:
        page = MODULE.PAGE
        self.assertIn("B5有限筛查 FAIL", page)
        self.assertIn("动作效应符号", page)
        self.assertIn("b01未用于训练的另一启动时刻", page)
        self.assertIn("统计独立性尚未证明", page)
        self.assertIn("不是阶梯保持", page)
        self.assertIn("这是epoch内validation10，不是dynamic6或最终门槛", page)
        self.assertIn("四力${", page)
        self.assertIn("此起点上反馈对减阻有附加价值", page)
        self.assertIn("固定序列更抑制升力波动", page)
        self.assertNotIn("feedback_vs_openloop_reference", page)
        self.assertIn("串联双圆柱流动控制 · 三层目标现状", page)
        self.assertIn('id="goal-real-cfd"', page)
        self.assertIn('id="goal-fno"', page)
        self.assertIn('id="goal-surrogate-control"', page)
        self.assertIn("评估配置修复后的完整推理", page)
        self.assertIn("Dynamic6 正式验收 FAIL", page)
        self.assertIn("16条真实PPO交互轨迹已通过官方DataPipe读取", page)
        self.assertIn('id="train16-formal-progress"', page)
        self.assertIn("one-batch技术probe明确排除", page)
        self.assertIn("tandem_fno_control_train16_h100_20261004", page)
        self.assertIn("红色阻塞", page)
        self.assertIn("后评估运行中", page)
        self.assertIn("历史superseded失败", page)
        self.assertIn("未知故障需要agent分析", page)
        self.assertIn("模型仍需改进", page)
        self.assertIn("auto_recovery_enabled", page)
        self.assertIn("paired统计DataPipe", page)
        self.assertIn("不虚称GPU在训", page)
        self.assertIn("paired后评估待批准", page)
        self.assertIn("FC-P001 paired posteval尚未执行", page)
        self.assertIn("FC-P001 Lead已批准 · 实现/血缘预检中", page)
        self.assertIn("不等待用户确认", page)
        self.assertIn("不预填结果", page)
        self.assertIn("training_evaluation_watchdog", page)

    def test_training_evaluation_watchdog_uses_fixed_latest_path(self) -> None:
        self.assertEqual(
            MODULE.TRAINING_EVALUATION_WATCHDOG,
            Path("artifacts/monitor/training_evaluation_watchdog/latest.json"),
        )
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('data["training_evaluation_watchdog"]', source)

    def test_dual_node_watchdog_reads_only_valid_latest_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / MODULE.DUAL_NODE_WATCHDOG
            path.parent.mkdir(parents=True)
            payload = {
                "status": "MONITORING",
                "alerts": [],
                "nodes": {
                    "spark": {"node_cpu_utilization_pct": 55.0},
                    "worker78": {"node_cpu_utilization_pct": 21.0},
                },
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(MODULE._dual_node_watchdog(root), payload)
            path.write_text(json.dumps({"status": "UNKNOWN"}), encoding="utf-8")
            self.assertIsNone(MODULE._dual_node_watchdog(root))

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

    def test_full40_status_separates_solver_completion_from_strict_raw_qc(self) -> None:
        self.assertEqual(len(MODULE.FULL40_CASES), 31)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            completed = "matched_start_acquisition_train_b00_m0375"
            verified = "matched_start_acquisition_validation_b01_zero"
            scheduler = root / MODULE.FULL40_SCHEDULER_STATE
            scheduler.parent.mkdir(parents=True)
            scheduler.write_text(
                json.dumps(
                    {
                        "status": "FULL40_EXTENSION_WATCH_ACTIVE",
                        "receipt_count": 1,
                        "snapshot": {
                            "cases": {
                                completed: {"status": "COMPLETED", "alive": False},
                                verified: {"status": "COMPLETED", "alive": False},
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            receipt_dir = root / MODULE.FULL40_RECEIPTS
            receipt_dir.mkdir(parents=True)
            receipt_dir.joinpath(f"{verified}.json").write_text(
                json.dumps(
                    {
                        "status": "FULL40_RAW_TRANSFER_VERIFIED",
                        "case": verified,
                        "split": "validation",
                        "phase_bin": 1,
                        "action_target": 0.0,
                        "full40_predeclaration_sha256": (
                            MODULE.FULL40_PREDECLARATION_SHA256
                        ),
                        "full40_extension_authorization_sha256": (
                            MODULE.FULL40_AUTHORIZATION_SHA256
                        ),
                        "worker_raw_manifest_sha256": "a" * 64,
                        "raw_file_count": 4,
                    }
                ),
                encoding="utf-8",
            )
            rows = {
                row["case"]: row
                for row in MODULE._full40_extension_status(root)["cases"]
            }
        self.assertTrue(rows[completed]["solver_completed"])
        self.assertFalse(rows[completed]["raw_qc_verified"])
        self.assertTrue(rows[verified]["solver_completed"])
        self.assertTrue(rows[verified]["raw_qc_verified"])

    def test_full40_receipt_fails_closed_on_wrong_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = "matched_start_acquisition_frozen_test_b03_p075"
            receipt_dir = root / MODULE.FULL40_RECEIPTS
            receipt_dir.mkdir(parents=True)
            receipt_dir.joinpath(f"{case}.json").write_text(
                json.dumps(
                    {
                        "status": "FULL40_RAW_TRANSFER_VERIFIED",
                        "case": case,
                        "split": "validation",
                        "phase_bin": 3,
                        "action_target": 0.75,
                        "full40_predeclaration_sha256": (
                            MODULE.FULL40_PREDECLARATION_SHA256
                        ),
                        "full40_extension_authorization_sha256": (
                            MODULE.FULL40_AUTHORIZATION_SHA256
                        ),
                        "worker_raw_manifest_sha256": "b" * 64,
                        "raw_file_count": 3,
                    }
                ),
                encoding="utf-8",
            )
            row = next(
                row
                for row in MODULE._full40_extension_status(root)["cases"]
                if row["case"] == case
            )
        self.assertFalse(row["raw_qc_verified"])
        self.assertFalse(row["receipt_checks"]["split"])

    def test_nine_case_physics_summary_is_train_only_and_reduced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / MODULE.MATCHED_START_PHYSICS_SUMMARY
            path.parent.mkdir(parents=True)
            comparison = {
                "m075": {
                    "total_drag_reduction_fraction_positive_is_better": 0.02,
                    "abs_mean_rear_cl_over_zero_fluctuation_rms": 0.6,
                    "canonical_joint_diagnostic_pass": False,
                },
                "p075": {
                    "total_drag_reduction_fraction_positive_is_better": 0.03,
                    "abs_mean_rear_cl_over_zero_fluctuation_rms": 0.7,
                    "canonical_joint_diagnostic_pass": False,
                },
            }
            path.write_text(
                json.dumps(
                    {
                        "status": (
                            "MATCHED_START_9_CASE_TRAIN_COMMISSIONING_PHYSICS_SUMMARY"
                        ),
                        "phases": {
                            "b00": {
                                "split": "train",
                                "same_phase_zero_comparisons": comparison,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            result = MODULE._matched_start_physics_summary(root)
        self.assertEqual(result["scope"], "train_phase_open_loop_commissioning_only")
        self.assertEqual(result["comparison_count"], 2)
        self.assertEqual(result["joint_pass_count"], 0)

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


    def test_dynamic6_runtime_reads_bounded_log_and_exact_service(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = root / "cfd/tandem_cylinders/cases"
            for name, phase, profile, start in (
                ("full40_dynamic_validation_b01_zero", 1, "zero", 130.0),
                ("full40_dynamic_validation_b01_minus", 1, "minus", 130.0),
                ("full40_dynamic_validation_b01_plus", 1, "plus", 130.0),
                ("full40_dynamic_validation_b05_zero", 5, "zero", 102.0),
                ("full40_dynamic_validation_b05_minus", 5, "minus", 102.0),
                ("full40_dynamic_validation_b05_plus", 5, "plus", 102.0),
            ):
                case = cases / name
                case.mkdir(parents=True)
                (case / "case_config.json").write_text(
                    json.dumps(
                        {
                            "phase_bin": phase,
                            "profile": profile,
                            "start_time": start,
                        }
                    ),
                    encoding="utf-8",
                )
            current = cases / "full40_dynamic_validation_b01_zero"
            (current / "log.pimpleFoam.full40_dynamic_validation").write_text(
                "Time = 130.005\nTime = 135\n", encoding="utf-8"
            )
            with patch.object(
                MODULE.subprocess,
                "run",
                return_value=SimpleNamespace(stdout="active\n", returncode=0),
            ):
                result = MODULE._dynamic6_runtime(root)
            self.assertEqual(result["status"], "RUNNING")
            self.assertEqual(result["current_case"]["steps"], 1000)
            self.assertEqual(result["current_case"]["phase_bin"], 1)
            self.assertEqual(result["ai_training_processes"], 0)
            self.assertEqual(result["cfd_solver_processes"], 0)
            self.assertFalse(result["frozen_hdf_opened_or_enumerated"])

    def test_dashboard_exposes_dynamic6_and_interrupt_record(self) -> None:
        page = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('id="chain-dynamic"', page)
        self.assertIn('data["dynamic6_runtime"]', page)
        self.assertIn("AI训练叶进程", page)
        self.assertIn("post-hoc per-phase oracle", page)
        self.assertIn("展开 0–10 / 10–20D/U 分段明细（8行）", page)
        self.assertIn("comparisons.elapsed_0_20", page)
        self.assertIn('data["dynamic6_physical_qc"]', page)
        self.assertIn("full40_dev30_quickscreen_qs1_interrupt_recovery.json", page)

    def test_current_chain_is_promoted_and_history_is_collapsed(self) -> None:
        page = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('id="current-focus"', page)
        self.assertNotIn("$('page-guide').before($('current-focus'))", page)
        self.assertIn('id="lead-overview"', page)
        self.assertIn('id="legacy-details"', page)
        self.assertLess(page.index('id="lead-overview"'), page.index('id="legacy-details"'))
        self.assertIn("历史 v4（非当前 full40/dev30）", page)
        self.assertEqual(page.count('<details class="archive">'), 3)
        self.assertNotIn('<details class="archive" open>', page)
        self.assertIn("历史 v3/v4 与旧开环动作证据（非当前", page)
        self.assertIn("历史数据生产与 commissioning 明细", page)
        identifiers = re.findall(r'id="([^"]+)"', page)
        self.assertEqual(len(identifiers), len(set(identifiers)))

    def test_current_candidate_missing_evidence_is_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = MODULE._research_overview(Path(directory))
            for candidate in ("lambda0", "lambda10"):
                self.assertIsNone(result[candidate]["endpoint"])
                self.assertIsNone(result[candidate]["updated_at"])
                self.assertIn("posteval_fc_p001", result[candidate]["path"])

    def test_current_candidate_endpoint_does_not_imply_closed_loop_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relative = MODULE._research_overview(root)["lambda0"]["path"]
            path = root / relative
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"status": "FULL40_VALIDATION_SURROGATE_READINESS_PASS"}))
            result = MODULE._research_overview(root)
            self.assertIsNotNone(result["lambda0"]["updated_at"])
            self.assertNotIn("project_goal_complete", result["lambda0"])
            self.assertIsNone(result["lambda10"]["endpoint"])
        self.assertIn("不代表动态／窗口检验通过", MODULE.PAGE)
        self.assertIn("尚未启动本轮 MPC 实验", MODULE.PAGE)
        self.assertIn("不是智能体实时心跳", MODULE.PAGE)

    def test_compact_chinese_reading_guide_defines_scientific_layers(self) -> None:
        page = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('id="page-guide"', page)
        self.assertIn("怎么看这页 / 术语说明", page)
        for term in (
            "三层结论",
            "epoch",
            "H20 / H100",
            "matched-start",
            "full40 / dev30",
            "VTK / HDF",
            "QC",
            "joint gate",
            "macro",
            "D/U",
            "历史 v4",
            "RAW",
            "不是 rollout",
            "低幅 H100",
            "不是低频",
            "t=120–160",
            "时间平均窗口",
        ):
            self.assertIn(term, page)


if __name__ == "__main__":
    unittest.main()
