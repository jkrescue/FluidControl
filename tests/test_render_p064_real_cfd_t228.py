"""CPU-only rendering arithmetic; no source CFD reads or exports."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/render_p064_real_cfd_t228.py'
spec = importlib.util.spec_from_file_location('physical_display', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_masked_cells_excluded():
    state = np.zeros((3, 128, 256))
    state[0] = 3
    state[1] = 4
    mask = np.ones((1, 128, 256))
    mask[0, 0, 0] = 0
    state[:, 0, 0] = 999
    field = module.masked_quantity({'state': state, 'mask': mask}, 'speed')
    assert np.isnan(field[0, 0])
    assert np.all(field[1:] == 5)


def test_shared_range_uses_all_panels():
    assert module.common_range([np.array([1., 2., np.nan]), np.array([-3., 8.])]) == (-3., 8.)


def test_degenerate_range_rejected():
    with pytest.raises(AssertionError):
        module.common_range([np.ones((2, 2))])
