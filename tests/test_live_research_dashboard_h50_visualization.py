from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "serve_live_research_dashboard.py"
if not SCRIPT.exists():
    SCRIPT = Path("/tmp/serve_live_research_dashboard.py")


def load_module():
    spec = importlib.util.spec_from_file_location("dashboard_h50", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = load_module()


class H50VisualizationTests(unittest.TestCase):
    def test_page_defaults_to_fixed_h100_three_action_comparison(self) -> None:
        page = MODULE.PAGE
        self.assertIn('id="h50-horizon"', page)
        self.assertIn('<option value="100" selected>', page)
        for role in ("zero", "minus", "plus"):
            self.assertIn(f'id="h50-{role}"', page)
        for label in (
            "局部 ROI",
            "不是完整 CFD 计算域",
            "u/U∞",
            "v/U∞",
            "p/(ρU∞²)",
            "1–99 百分位显示范围",
            "Dynamic6 正式动作差值门槛未通过",
            "16 条真实 PPO 交互轨迹已通过官方 DataPipe 读取",
        ):
            self.assertIn(label, page)

    def test_h50_figure_route_is_strictly_whitelisted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = (
                root
                / MODULE.H50_FIXED_FIGURES
                / "full40_dynamic_validation_b01_zero"
                / "horizon_100_start_0000.png"
            )
            image.parent.mkdir(parents=True)
            image.write_bytes(b"valid-png-fixture")

            handler = MODULE.Handler.__new__(MODULE.Handler)
            handler.root = root
            replies = []
            handler._send = lambda body, mime, status=200: replies.append(
                (body, mime, status)
            )
            handler.path = "/figure/h50-dynamic/zero/100.png"
            handler.do_GET()
            self.assertEqual(replies[-1], (b"valid-png-fixture", "image/png", 200))

            for path in (
                "/figure/h50-dynamic/zero/999.png",
                "/figure/h50-dynamic/../../etc/passwd",
                "/figure/h50-dynamic/probe/100.png",
                "/figure/h50-dynamic/zero/100.png/extra",
            ):
                handler.path = path
                handler.do_GET()
                self.assertEqual(replies[-1][2], 404)

    def test_api_contract_distinguishes_probe_from_formal_candidate(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('"technical_probe_is_formal_candidate": False', source)
        self.assertIn("DIRECTPPO_TRAIN16_MANIFEST", source)
        self.assertIn("validation_only_roi_not_full_cfd_domain_", source)
        self.assertNotIn("tandem_fno_control_train16_h100_probe_20261004", source)


if __name__ == "__main__":
    unittest.main()
