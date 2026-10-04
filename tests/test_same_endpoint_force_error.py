import importlib.util
import unittest
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "scripts/audit_same_endpoint_force_error.py"
SPEC = importlib.util.spec_from_file_location("same_endpoint", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture():
    rows = []
    for phase in ("b01", "b05"):
        for action in ("minus", "zero", "plus"):
            for start in range(101):
                for horizon, offset, error in ((100, 0, .2), (1, 99, .1)):
                    rows.append(dict(case=f"full40_dynamic_validation_{phase}_{action}",
                                     horizon=horizon, start=start + offset,
                                     target_total_drag=float(start), rear_cl_mae=error))
    return dict(split="validation", action_mode="observed", segments=rows)


class SameEndpointTests(unittest.TestCase):
    def test_aligned_panel(self):
        result = MODULE.compare(fixture())
        self.assertEqual(result["matched_endpoints"], 606)
        self.assertAlmostEqual(next(iter(result["cases"].values()))["true_state_h1_mae"], .1)

    def test_reject_missing_duplicate_mismatched_nonfinite_and_negative(self):
        for fault in ("missing", "duplicate", "target", "nan", "negative", "split"):
            with self.subTest(fault=fault):
                payload = fixture()
                if fault == "missing":
                    payload["segments"].pop()
                elif fault == "duplicate":
                    payload["segments"].append(payload["segments"][0])
                elif fault == "target":
                    payload["segments"][0]["target_total_drag"] = 999
                elif fault == "split":
                    payload["split"] = "test"
                else:
                    payload["segments"][0]["rear_cl_mae"] = float("nan") if fault == "nan" else -1
                with self.assertRaises(ValueError):
                    MODULE.compare(payload)


if __name__ == "__main__":
    unittest.main()
