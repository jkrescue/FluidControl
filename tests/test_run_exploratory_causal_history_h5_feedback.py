import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/run_exploratory_causal_history_h5_feedback.py"
SPEC = importlib.util.spec_from_file_location("run_causal_h5", SCRIPT)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_explicit_h5_status_and_mode():
    assert M.STATUS == "EXPLORATORY_PAIRED_CANONICAL_HISTORY_H5_REAL_CFD_EXECUTION_APPROVED"
    assert "canonical_causal_history_h5_v1" in SCRIPT.read_text()


def test_driver_runs_one_shared_rollout_then_new_selector_and_actual_only_append():
    tree = ast.parse(SCRIPT.read_text())
    execute = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == "execute")
    names = [getattr(node.func, "id", getattr(node.func, "attr", ""))
             for node in ast.walk(execute) if isinstance(node, ast.Call)]
    assert names.count("rollout_five_held_horizon") == 1
    assert names.count("select_canonical_history_horizon") == 1
    assert names.count("append_actual_endpoint") == 1
    source = SCRIPT.read_text()
    assert 'if role == "mpc"' in source
    assert "actual_force=actual" in source
    assert "persistent_history_actual_endpoint_only" in source


def test_spec_keeps_old_protocol_fixed_but_requires_new_approval(tmp_path):
    payload = {
        "status": M.STATUS, "execution_authorized": True, "steps": 10,
        "planning_horizon": 5,
        "start_time": 148.0, "inference_device": "cpu",
        "state_abs_limit": M.STATE_ABS_LIMIT, "deadline_seconds": 900,
        "original_long_ar_gate_passed": False, "scientific_admission": False,
        "source_files": {"x": "a" * 64},
        "inputs": {"x": {"path": "x", "sha256": "b" * 64}},
        "causal_history_sources": {
            "forceFront": [{"path": "front", "sha256": "c" * 64}],
            "forceRear": [{"path": "rear", "sha256": "d" * 64}],
        },
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(payload))
    M.validate_spec(payload, path, hashlib.sha256(path.read_bytes()).hexdigest())
    payload["status"] = "EXPLORATORY_PAIRED_SHORT_H2_REAL_CFD_EXECUTION_APPROVED"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="approval"):
        M.validate_spec(payload, path, hashlib.sha256(path.read_bytes()).hexdigest())
