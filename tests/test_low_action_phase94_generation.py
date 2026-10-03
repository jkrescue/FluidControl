from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


def load_module():
    path = Path(__file__).parents[1] / "cfd" / "tandem_cylinders" / "generate_low_action_phase94_validation.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("low_action_generation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LowActionGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_pair_is_mirrored_and_within_frozen_support(self) -> None:
        positive = self.module.action_points(1.0)
        negative = self.module.action_points(-1.0)
        self.assertEqual(len(positive), 801)
        self.assertEqual((positive[0][0], positive[-1][0]), (94.0, 174.0))
        for left, right in zip(positive, negative, strict=True):
            self.assertEqual(left[0], right[0])
            self.assertAlmostEqual(left[1], -right[1], places=12)
        metrics = self.module.action_metrics(positive)
        self.assertEqual(metrics["max_abs_omega"], 0.75)
        self.assertLessEqual(metrics["max_abs_domega_dt"], 0.3750001)


if __name__ == "__main__":
    unittest.main()
