from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


def load_module():
    path = Path(__file__).parents[1] / "scripts" / "finalize_low_action_phase94_curated.py"
    spec = importlib.util.spec_from_file_location("finalize_low_action", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FinalizeLowActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_float32_time_grid_is_accepted_but_corruption_is_not(self) -> None:
        times = np.linspace(94.0, 174.0, 801).astype(np.float32).astype(np.float64)
        self.module.validate_time_grid(times)
        times[400] += 5.0e-4
        with self.assertRaisesRegex(ValueError, "time mismatch|nonuniform"):
            self.module.validate_time_grid(times)


if __name__ == "__main__":
    unittest.main()
