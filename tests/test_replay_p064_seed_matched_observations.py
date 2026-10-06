import importlib.util
import json
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "scripts/replay_p064_seed_matched_observations.py"
spec = importlib.util.spec_from_file_location("replay", SOURCE)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def test_fixed_windows_cover_expected_rows():
    rows = [{"end_time": round(148 + 0.1 * step, 10)} for step in range(1, 801)]
    assert {name: len(replay.select_window(rows, window)) for name, window in replay.WINDOWS.items()} == {
        "early_12p4": 124,
        "early_first_6p2": 62,
        "early_trailing_6p2": 62,
        "primary_final_60": 600,
        "historical_inclusive_final_60": 601,
        "full_80": 800,
    }


def test_odd_even_decomposition_is_exact():
    raw, reflected = 0.6, -0.2
    odd = 0.5 * (raw - reflected)
    even = 0.5 * (raw + reflected)
    assert odd == 0.4 and even == 0.19999999999999998
    assert np.isclose(odd + even, raw) and np.isclose(even - odd, reflected)


def test_summary_keeps_policies_separate():
    rows = [
        {"seed20261006": {"raw": 1.0, "reflected": -1.0, "odd": 1.0, "even": 0.0}, "seed20261007": {"raw": 0.2, "reflected": 0.2, "odd": 0.0, "even": 0.2}},
        {"seed20261006": {"raw": -1.0, "reflected": 1.0, "odd": -1.0, "even": 0.0}, "seed20261007": {"raw": -0.2, "reflected": -0.2, "odd": 0.0, "even": -0.2}},
    ]
    result = replay.summarize(rows)
    assert result["seed20261006"]["odd"]["rms"] == 1.0
    assert result["seed20261007"]["odd"]["rms"] == 0.0
    assert result["paired"]["odd_mean_abs_difference"] == 1.0


def test_worker_has_no_filter_fno_or_cfd_execution():
    text = SOURCE.read_text()
    assert "apply_action_rate_limit" not in text
    assert "PairSolvers" not in text
    assert '"fno_loaded": False' in text
    assert '"filter_state_propagated": False' in text
