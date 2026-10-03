"""Versioned Curator profile includes only the two new train-only cases."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from curate_tandem_cfd import PROFILE_ACTION_LIMITS, PROFILE_COUNTS, case_records  # noqa: E402


class ControlGapProfileTests(unittest.TestCase):
    def test_v4_extends_v3_without_changing_heldout_splits(self) -> None:
        root = Path(__file__).resolve().parents[1] / "cfd/tandem_cylinders/cases"
        v3 = case_records(root, "gate_b_aug_v3")
        v4 = case_records(root, "control_gap_v4")
        by_split = lambda rows, split: {row["name"] for row in rows if row["split"] == split}
        self.assertEqual(by_split(v3, "validation"), by_split(v4, "validation"))
        self.assertEqual(by_split(v3, "test"), by_split(v4, "test"))
        self.assertEqual(len(by_split(v4, "train") - by_split(v3, "train")), 2)
        self.assertEqual(PROFILE_COUNTS["control_gap_v4"], {"train": 28, "validation": 4, "test": 5})
        self.assertEqual(PROFILE_ACTION_LIMITS["control_gap_v4"], 5.0)


if __name__ == "__main__":
    unittest.main()
