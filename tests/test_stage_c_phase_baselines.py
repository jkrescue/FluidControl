import unittest

from build_stage_c_phase_baselines import build_phase_baselines


class StageCPhaseBaselineTests(unittest.TestCase):
    def fixture(self):
        return {
            "cases": [
                {
                    "dataset": "real_cfd",
                    "split": "test",
                    "case": "phase_00",
                    "phase_matched_zero_action_baseline": {
                        "window": [80.0, 160.0],
                        "samples": 801,
                        "total_cd_mean": 2.3,
                        "front_cl": {"root_mean_square": 0.31},
                        "rear_cl": {"root_mean_square": 1.18},
                    },
                }
            ]
        }

    def test_builds_runtime_baseline_from_raw_rms(self):
        result = build_phase_baselines(self.fixture(), "audit.json")
        self.assertEqual(result["case_count"], 1)
        row = result["cases"]["real_cfd/test/phase_00"]
        self.assertEqual(row["total_drag"], 2.3)
        self.assertEqual(row["front_lift_rms"], 0.31)
        self.assertEqual(row["rear_lift_rms"], 1.18)
        self.assertIn("phase_matched_zero_action", row["source"])

    def test_rejects_duplicate_case_identity(self):
        audit = self.fixture()
        audit["cases"].append(audit["cases"][0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_phase_baselines(audit, "audit.json")


if __name__ == "__main__":
    unittest.main()
