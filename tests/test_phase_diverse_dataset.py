from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


def load_module():
    root = Path(__file__).parents[1]
    script_dir = root / "cfd" / "tandem_cylinders"
    sys.path.insert(0, str(script_dir))
    path = script_dir / "make_phase_diverse_control_dataset.py"
    spec = importlib.util.spec_from_file_location("phase_diverse", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PhaseDiverseDatasetTests(unittest.TestCase):
    def test_phase_panel_has_independent_restarts_and_splits(self) -> None:
        module = load_module()
        self.assertEqual(
            [case.source_time for case in module.PHASE_CASES], [82.0, 84.0, 86.0, 88.0]
        )
        self.assertEqual(
            [case.split for case in module.PHASE_CASES],
            ["train", "train", "validation", "test"],
        )

    def test_every_schedule_is_bounded_and_starts_from_zero(self) -> None:
        module = load_module()
        audit = module.validate()
        self.assertEqual(len(audit["cases"]), 4)
        for case in module.PHASE_CASES:
            points = module.schedule_points(case)
            metrics = module.schedule_metrics(points)
            self.assertAlmostEqual(points[0][1], 0.0)
            self.assertAlmostEqual(points[-1][0] - points[0][0], module.DURATION)
            self.assertLessEqual(metrics["max_abs_domega_dt"], module.RATE_LIMIT)
            self.assertLessEqual(
                max(abs(metrics["omega_min"]), abs(metrics["omega_max"])),
                module.OMEGA_LIMIT,
            )


if __name__ == "__main__":
    unittest.main()
