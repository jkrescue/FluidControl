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


if __name__ == "__main__":
    unittest.main()
