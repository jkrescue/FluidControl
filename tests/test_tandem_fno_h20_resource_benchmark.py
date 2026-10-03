from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


def load_module():
    path = Path(__file__).parents[1] / "scripts" / "benchmark_tandem_fno_h20_resource.py"
    spec = importlib.util.spec_from_file_location("resource_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ResourceBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_plan_keeps_required_order_and_h20_guard(self) -> None:
        plan = self.module.benchmark_plan()
        self.assertEqual(
            [(row.latent_channels, row.batch_size) for row in plan],
            [(48, 4), (48, 8), (64, 4)],
        )
        self.assertTrue(self.module.reserve_ok(20.0))
        self.assertFalse(self.module.reserve_ok(19.999))


if __name__ == "__main__":
    unittest.main()
