import json
from pathlib import Path

import numpy as np
import pytest

from fluid_control import p064_b_continuation_contract as c


def terminal_result():
    obs0 = np.zeros(69, np.float32)
    obs0[-1] = np.float32(.29905773401260382)
    obs1 = np.zeros(69, np.float32)
    obs1[-1] = np.float32(c.SAVED_APPLIED_OMEGA)
    return {"cycles": 800, "rows": [None] * 799 + [{
        "step": 800, "start_time": 327.9, "end_time": 328.0,
        "input_observation": obs0.tolist(), "output_observation": obs1.tolist(),
        "applied_omega": c.SAVED_APPLIED_OMEGA,
        "observation_sources": {"time": 328.0,
          "probe_sources": ["x/327.9/U"],
          "force_sources": ["x/327.9/coefficient.dat"],
          "front_force_sources": ["y/327.9/coefficient.dat"]},
    }]}


def test_seed_uses_output_and_double_limiter_state():
    obs, previous = c.seed_from_e109(terminal_result())
    assert obs.dtype == np.float32 and obs.shape == (69,)
    assert previous == c.SAVED_APPLIED_OMEGA
    assert float(obs[-1]) != previous


def test_input_observation_cannot_replace_output():
    value = terminal_result()
    value["rows"][-1]["output_observation"] = value["rows"][-1]["input_observation"]
    with pytest.raises(ValueError, match="representation"):
        c.seed_from_e109(value)


def test_restart_contract_and_missing_old_time(tmp_path: Path):
    for name in c.RESTART_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
    assert set(c.require_restart_tree(tmp_path, "case_mpc")) == set(c.RESTART_FILES)
    (tmp_path / "U_0").unlink()
    with pytest.raises(ValueError, match="missing"):
        c.require_restart_tree(tmp_path, "case_mpc")


def test_windows_are_fixed_and_cover_tail_without_warmup_drop():
    rows = c.prospective_windows()
    assert rows[:4] == [("tail_block_1", 328.0, 348.0),
                       ("tail_block_2", 348.0, 368.0),
                       ("tail_block_3", 368.0, 388.0),
                       ("tail_block_4", 388.0, 408.0)]
    assert ("tail_full_80", 328.0, 408.0) in rows
    assert ("joined_full_160", 248.0, 408.0) in rows
    assert ("joined_post_transition_140", 268.0, 408.0) in rows
