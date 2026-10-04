import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import torch

P = (
    Path(__file__).resolve().parents[1]
    / "scripts/diagnose_fcp003c_train_vs_validation_true_state_h1.py"
)
S = importlib.util.spec_from_file_location("d015", P)
M = importlib.util.module_from_spec(S)
S.loader.exec_module(M)


class Tests(unittest.TestCase):
    def test_windows(self):
        self.assertEqual(M.endpoint_indices("train_paired_window"), list(range(1, 101)))
        self.assertEqual(M.endpoint_indices("train_late_window"), list(range(100, 201)))
        with self.assertRaises(ValueError):
            M.endpoint_indices("bad")

    def test_summary(self):
        x = M.summarize([-2.0, 1.0])
        self.assertEqual(x["mae"], 1.5)
        self.assertEqual(x["bias"], -0.5)
        self.assertAlmostEqual(x["rmse"], math.sqrt(2.5))
        for bad in ([], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                M.summarize(bad)

    def test_rows(self):
        c = {
            x: {"action_error": 1.0, "zero_error": -0.5, "delta_error": 1.5}
            for x in M.CHANNELS
        }
        r = M.summarize_rows(
            [
                {
                    "panel": "train_paired_window",
                    "phase": "b00",
                    "profile": "prbs",
                    "channels": c,
                }
            ],
            [{"panel": "train_paired_window", "phase": "b00", "channels": c}],
        )
        self.assertEqual(
            r["action_profiles"]["train_paired_window/b00/prbs/rear_cl"]["delta_error"][
                "mae"
            ],
            1.5,
        )
        self.assertEqual(
            r["deduplicated_zero_phases"]["train_paired_window/b00/rear_cl"]["count"], 1
        )

    def test_micro_counts_exclude_validation_zero_delta_and_dedup_zero_absolute(self):
        def row(panel, profile=None):
            value = {
                c: {"action_error": 1.0, "zero_error": 2.0, "delta_error": 3.0}
                for c in M.CHANNELS
            }
            result = {"panel": panel, "phase": "b00", "channels": value}
            if profile is not None:
                result["profile"] = profile
            return result

        action = [
            row("train_paired_window", "prbs"),
            row("train_late_window", "prbs"),
            row("validation_late_window", "minus"),
            row("validation_late_window", "zero"),
        ]
        zero = [
            row("train_paired_window"),
            row("train_late_window"),
            row("validation_late_window"),
        ]
        result = M.summarize_rows(action, zero)["panel_micro"]
        self.assertEqual(
            result["train_paired_window"]["absolute_micro"]["endpoint_count"], 2
        )
        self.assertEqual(
            result["validation_late_window"]["absolute_micro"]["endpoint_count"], 2
        )
        self.assertEqual(
            result["validation_late_window"]["nonzero_action_delta_micro"][
                "endpoint_count"
            ],
            1,
        )

    def test_normalization_and_predict_chunk_cover_endpoint_200(self):
        frames = 201
        h = w = 2
        action = {
            "state": np.zeros((frames, 3, h, w), dtype=np.float32),
            "mask": np.ones((frames, 1, h, w), dtype=np.float32),
            "omega": np.arange(frames, dtype=np.float32)[:, None] / 400,
            "force": np.zeros((frames, 4), dtype=np.float32),
            "time": np.arange(frames, dtype=np.float32)[:, None] / 10,
        }
        zero = {k: v.copy() for k, v in action.items()}
        zero["omega"][:] = 0
        action["state"][:, 0] = np.arange(frames, dtype=np.float32)[:, None, None]
        pair = M._normalized_pair(
            action,
            zero,
            list(range(100, 201)),
            {"state_mean": [0, 0, 0], "state_std": [1, 1, 1]},
            "cpu",
        )
        self.assertEqual(tuple(pair["action_state"].shape), (1, 102, 3, 2, 2))
        self.assertEqual(float(pair["action_state"][0, -1, 0, 0, 0]), 200.0)
        fake = types.ModuleType("train_tandem_fno")

        def predict(_model, x, _mask):
            force = x[:, :1].mean((-2, -1)).repeat(1, 4)
            return x[:, :3], force

        fake.predict = predict
        old = sys.modules.get("train_tandem_fno")
        sys.modules["train_tandem_fno"] = fake
        try:
            pa, _pz = M._predict_pair(torch.nn.Identity(), pair, 10)
        finally:
            if old is None:
                sys.modules.pop("train_tandem_fno", None)
            else:
                sys.modules["train_tandem_fno"] = old
        self.assertEqual(tuple(pa.shape), (101, 4))
        self.assertEqual(float(pa[-1, 0]), 199.0)

    def test_segments(self):
        rows = []
        seg = []
        for phase in ("b01", "b05"):
            for profile in ("minus", "zero", "plus"):
                case = f"full40_dynamic_validation_{phase}_{profile}"
                for target in range(100, 201):
                    rows.append(
                        {
                            "phase": phase,
                            "profile": profile,
                            "target_index": target,
                            "channels": {
                                c: {"action_error": -0.25} for c in M.CHANNELS
                            },
                        }
                    )
                    seg.append(
                        {
                            "case": case,
                            "horizon": 1,
                            "start": target - 1,
                            "force_channel_absolute_error": {
                                c: 0.25 for c in M.CHANNELS
                            },
                        }
                    )
        payload = {"split": "validation", "action_mode": "observed", "segments": seg}
        self.assertEqual(
            M.validate_existing_segments(payload, rows)[
                "max_absolute_error_difference"
            ],
            0.0,
        )
        for fault in ("missing", "duplicate", "split", "channels"):
            b = {
                "split": "validation",
                "action_mode": "observed",
                "segments": [dict(x) for x in seg],
            }
            if fault == "missing":
                b["segments"].pop()
            elif fault == "duplicate":
                b["segments"].append(b["segments"][0])
            elif fault == "split":
                b["split"] = "train"
            else:
                b["segments"][0] = dict(
                    b["segments"][0], force_channel_absolute_error={"rear_cl": 0.25}
                )
            with self.assertRaises(ValueError):
                M.validate_existing_segments(b, rows)


if __name__ == "__main__":
    unittest.main()
