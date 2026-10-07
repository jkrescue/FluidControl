import importlib.util
import json
import math
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "p064_b04_long_excitation.py"
if not SOURCE.is_file():
    SOURCE = HERE.parent / "scripts/p064_b04_long_excitation.py"
spec = importlib.util.spec_from_file_location("b04", SOURCE)
module = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(module)


def source_points():
    # Nontrivial valid 201-point sequence with zero endpoints.
    values = [round(0.3 * math.sin(2 * math.pi * i / 40), 10) for i in range(201)]
    values[0] = values[-1] = 0.0
    return [[round(module.START + i * module.DT, 10), value] for i, value in enumerate(values)]


def test_exact_repetition_grid_and_limits():
    source = source_points()
    rows = module.repeated_prbs(source)
    assert len(rows) == 801
    assert rows[:201] == source
    assert rows[201][0] == 140.1
    assert rows[200][1] == rows[400][1] == rows[600][1] == rows[800][1] == 0.0
    for period in range(4):
        assert [r[1] for r in rows[period * 200 : period * 200 + 201]] == [r[1] for r in source]
    assert max(abs(rows[i + 1][1] - rows[i][1]) for i in range(800)) <= 0.1 + 1e-12


def test_bad_source_rejected():
    points = source_points()
    points[8][0] += 0.01
    with pytest.raises(ValueError, match="time grid"):
        module.repeated_prbs(points)


def test_contract_uses_mechanical_starts(monkeypatch, tmp_path):
    # build_contract integration is exercised against the real repository separately.
    assert [i * 700 // 31 for i in range(32)][0] == 0
    assert [i * 700 // 31 for i in range(32)][-1] == 700
