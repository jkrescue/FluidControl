"""Checks for paired onset-comparison bookkeeping."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cfd" / "tandem_cylinders"))
from analyze_control_onset_replication import compare  # noqa: E402


class OnsetAnalysisTests(unittest.TestCase):
    def test_rejects_unmatched_restart(self) -> None:
        rows = []
        for onset, omega in (("ramp", 0.0), ("ramp", 1.0), ("ramp", -1.0), ("instant", 1.0), ("instant", -1.0)):
            rows.append({
                "onset": onset, "omega_final": omega,
                "source_restart_u_sha256": "wrong" if onset == "instant" and omega == -1 else "same",
                "source_restart_p_sha256": "same",
                "rear": {"cl_rms": 1.0, "cl_mean": 0.0},
                "total_cd_mean": 2.0,
                "total_cd_block_means": [2.0] * 6,
            })
        with self.assertRaisesRegex(ValueError, "same restart"):
            compare(rows)


if __name__ == "__main__":
    unittest.main()
