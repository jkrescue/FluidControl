"""RMS arithmetic and fail-closed sample checks."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cfd" / "tandem_cylinders"))
from audit_lift_rms_definitions import force_metrics  # noqa: E402


class LiftRmsTests(unittest.TestCase):
    def test_total_rms_includes_mean(self) -> None:
        rows = [(index * 0.005, 0.0, 2.0 + (1 if index % 2 else -1)) for index in range(2000)]
        metrics = force_metrics(rows, 0.0, 10.0)
        self.assertAlmostEqual(metrics["cl_mean"], 2.0)
        self.assertAlmostEqual(metrics["cl_fluctuation_rms"], 1.0)
        self.assertAlmostEqual(metrics["cl_total_rms"], math.sqrt(5))

    def test_refuses_too_few_samples(self) -> None:
        with self.assertRaisesRegex(ValueError, "insufficient"):
            force_metrics([(0.0, 0.0, 1.0)], 0.0, 1.0)


if __name__ == "__main__":
    unittest.main()
