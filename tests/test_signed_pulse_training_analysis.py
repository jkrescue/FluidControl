"""Train-only paired CFD audit must fail on leakage/mismatched restart."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cfd" / "tandem_cylinders"))
from validate_signed_pulse_training_pair import compare  # noqa: E402


class SignedPulseAnalysisTests(unittest.TestCase):
    def test_rejects_nonpaired_source(self) -> None:
        rows = [
            {"sign": 1.0, "source_restart_u_sha256": "a", "source_restart_p_sha256": "a", "rear_force": {"cl_rms": 1.0}},
            {"sign": -1.0, "source_restart_u_sha256": "b", "source_restart_p_sha256": "a", "rear_force": {"cl_rms": 1.0}},
        ]
        with self.assertRaisesRegex(ValueError, "share source"):
            compare(rows)


if __name__ == "__main__":
    unittest.main()
