from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load_module():
    path = Path(__file__).parents[1] / "scripts/build_fc_p003_dynamic8_pair_candidate.py"
    spec = importlib.util.spec_from_file_location("fc_p003_dynamic8", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sample(offset: float = 0.0):
    state = np.zeros((101, 3, 2, 2), dtype=np.float32)
    state[0, 0, 0, 0] = offset
    return {"state": state, "mask": np.ones((101, 1, 2, 2), dtype=np.float32),
            "omega": np.zeros((101, 1), dtype=np.float32),
            "force": np.zeros((101, 4), dtype=np.float32),
            "time": np.arange(101, dtype=np.float32)[:, None] * 0.1}


def test_accepts_declared_float32_curator_tolerance():
    result = load_module().audit_pair(sample(2.3841858e-7), sample())
    assert result["state0_max_abs_difference"] <= 3e-7


def test_rejects_larger_initial_state_difference():
    with pytest.raises(ValueError, match="state0 exceeds atol"):
        load_module().audit_pair(sample(3.1e-7), sample())


def test_rejects_nonzero_reference_action():
    zero = sample(); zero["omega"][5] = 0.01
    with pytest.raises(ValueError, match="zero reference"):
        load_module().audit_pair(sample(), zero)
