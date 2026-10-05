"""Diagnostic aggregate compatibility only; no model/HDF loads or GPU work."""
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("p026_aggregate_audit", HERE.parent / "scripts/audit_fcp026_candidate.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
REPO = HERE.parent


def advance(value, count):
    for _ in range(count):
        value = math.nextafter(value, math.inf)
    return value


@pytest.mark.parametrize("steps", [0, 1, 2, 3, 4])
def test_four_ulp_inclusive_boundary(steps):
    audit.equal_aggregate({"values": [1.0, None], "count": 3},
                          {"values": [advance(1.0, steps), None], "count": 3}, "aggregate")


@pytest.mark.parametrize("steps", [5, 6, 100])
def test_more_than_four_ulp_rejected(steps):
    with pytest.raises(ValueError):
        audit.equal_aggregate(1.0, advance(1.0, steps), "aggregate")


@pytest.mark.parametrize("left,right", [
    (True, True), (False, 0), (1, 1.0), (None, 0), (3, 4),
    (float("nan"), float("nan")), (float("inf"), float("inf")),
    (-float("inf"), -float("inf")), ({"a": 1.0}, {"b": 1.0}),
    ([1.0], [1.0, 2.0]), ([1.0], (1.0,)), ("x", "x"),
])
def test_structure_types_and_nonfinite_rejected(left, right):
    with pytest.raises(ValueError):
        audit.equal_aggregate(left, right, "aggregate")


def test_original_exact_equality_unchanged():
    with pytest.raises(ValueError):
        audit.equal({"x": 1.0}, {"x": advance(1.0, 1)}, "strict")


def test_actual_four_panels_and_subgroups():
    result_path = REPO / "artifacts/fcp026_history_training_k1_20261005/candidate/result.json"
    if not result_path.is_file():
        pytest.skip("actual preserved K1 diagnostic JSON unavailable")
    namespace = {}
    for relative, name, expected_sha in [
        ("scripts/probe_fcp020_symmetric_statistics.py", "aggregate",
         "139dee2c3dcd4ee97de78a7a0e343ca9ab9d9708a1cafab447f0d9e4988b5d8c"),
        ("scripts/train_fcp026_history.py", "grouped_panel",
         "562d268545ba5cd2559374f4bd8bd34bf59a2e49f2e886e4e286bb12aac5d49e"),
    ]:
        source = REPO / relative
        raw = source.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected_sha
        function = next(n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name == name)
        # Compile only the reviewed pure aggregation function, not trainer imports.
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    result_raw = result_path.read_bytes()
    assert hashlib.sha256(result_raw).hexdigest() == "ec5476c389112ea4f27a2adf2f82ac3fad93ba6dcf9bb5625b60cfd80fdf5071"
    result = json.loads(result_raw)
    panels = result["fixed_train_panels"]
    assert [p["consumed_windows"] for p in panels] == [0, 456, 912, 1368]
    def leaves(value):
        if isinstance(value, dict):
            return sum(leaves(v) for v in value.values())
        return 1
    assert sum(leaves(p["aggregate"]) + leaves(p["history_subgroups"]) for p in panels) == 136
    for panel in panels:
        for key, function in [("aggregate", "aggregate"), ("history_subgroups", "grouped_panel")]:
            audit.equal_aggregate(panel[key], namespace[function](panel["rows"]), key)
    # Explanatory precise-sum cross-check: every saved leaf matches exactly.
    namespace["sum"] = math.fsum
    for panel in panels:
        assert panel["aggregate"] == namespace["aggregate"](panel["rows"])
        assert panel["history_subgroups"] == namespace["grouped_panel"](panel["rows"])
