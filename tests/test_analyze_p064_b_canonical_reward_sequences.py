from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).with_name("analyze_b_canonical_reward_sequences.py")
if not SOURCE.is_file():
    SOURCE = Path(__file__).resolve().parents[1] / "scripts/analyze_p064_b_canonical_reward_sequences.py"
SPEC = importlib.util.spec_from_file_location("reward_replay", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_total_cost_requires_exact_frozen_component_schema():
    values = {
        "drag_screen": -0.1,
        "drag_gate_violation": 0.2,
        "rear_cl_fluctuation_gate_violation": 0.3,
        "rear_cl_mean_bias_gate_violation": 0.4,
        "actuation": 0.5,
        "rate": 0.6,
    }
    assert MODULE.total_cost(values) == 1.9
    broken = dict(values)
    broken["invented"] = 1.0
    try:
        MODULE.total_cost(broken)
    except ValueError as error:
        assert "schema" in str(error)
    else:
        raise AssertionError("extra reward component was accepted")


def test_validation_baseline_uses_fixed_final60_and_both_force_components(tmp_path):
    case = tmp_path / "case"
    for name, cd, cl in (("forceFront", 1.25, 4.0), ("forceRear", 0.75, -2.0)):
        path = case / "postProcessing" / name / "0" / "coefficient.dat"
        path.parent.mkdir(parents=True)
        times = 20.0 + 0.005 * np.arange(12001)
        lines = [f"{time:.8f} {cd:.8f} 0 0 {cl + 0.1*np.sin(time):.12f}\n" for time in times]
        path.write_text("".join(lines), encoding="utf-8")
    baseline, sources = MODULE.validation_baseline(case, 80.0)
    assert baseline["total_drag"] == 2.0
    assert baseline["rear_cl_fluctuation_rms"] > 0.0
    assert set(sources) == {"forceFront", "forceRear"}


def test_exact_ties_are_not_broken_by_role_order():
    costs = {"minus": 1.0, "zero": 1.0, "plus": 2.0}
    minimum = min(costs.values())
    assert [role for role in MODULE.ROLES if costs[role] == minimum] == ["minus", "zero"]
