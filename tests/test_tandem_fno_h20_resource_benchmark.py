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
            [
                (row.name, row.latent_channels, row.batch_size, row.num_fno_modes)
                for row in plan
            ],
            [
                ("width48_batch4", 48, 4, (32, 32)),
                ("width48_batch8", 48, 8, (32, 32)),
                ("width64_batch4", 64, 4, (32, 32)),
            ],
        )
        self.assertEqual(
            [(row.name, row.num_fno_modes) for row in plan[:3]],
            [
                ("width48_batch4", (32, 32)),
                ("width48_batch8", (32, 32)),
                ("width64_batch4", (32, 32)),
            ],
        )
        self.assertTrue(self.module.reserve_ok(20.0))
        self.assertFalse(self.module.reserve_ok(19.999))

    def test_modes48_is_explicit_opt_in_after_original_plan(self) -> None:
        default = self.module.benchmark_plan()
        expanded = self.module.benchmark_plan(include_modes48=True)
        self.assertEqual(expanded[:3], default)
        self.assertEqual(len(default), 3)
        self.assertEqual(len(expanded), 4)
        self.assertEqual(
            (
                expanded[-1].name,
                expanded[-1].latent_channels,
                expanded[-1].batch_size,
                expanded[-1].num_fno_modes,
            ),
            ("modes48_width48_batch4", 48, 4, (48, 48)),
        )


if __name__ == "__main__":
    unittest.main()
