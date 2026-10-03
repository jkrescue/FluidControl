"""Cache-hint target validation does not permit traversal or active jobs."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from reclaim_completed_cfd_cache import case_path  # noqa: E402


class CacheTargetTests(unittest.TestCase):
    def test_rejects_path_traversal(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid case name"):
            case_path("../../unrelated")

    def test_rejects_noncase_name(self) -> None:
        with self.assertRaises(ValueError):
            case_path("definitely_not_a_project_case_20261003")


if __name__ == "__main__":
    unittest.main()
