"""Exact fixed three-criterion objective must reject partial improvements."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cfd" / "tandem_cylinders"))
from audit_existing_open_loop_acceptance import accept  # noqa: E402


class ExistingOpenLoopAcceptanceTests(unittest.TestCase):
    def test_joint_rule(self) -> None:
        self.assertTrue(accept(0.03, 1.04, 0.09)["joint_pass"])
        self.assertFalse(accept(0.03, 1.06, 0.09)["joint_pass"])
        self.assertFalse(accept(0.03, 1.04, 0.11)["joint_pass"])
        self.assertFalse(accept(0.01, 1.04, 0.09)["joint_pass"])


if __name__ == "__main__":
    unittest.main()
