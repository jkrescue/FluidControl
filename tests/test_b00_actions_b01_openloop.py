import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


PATH = Path(__file__).parents[1] / "scripts" / "run_b00_actions_b01_openloop.py"
SPEC = importlib.util.spec_from_file_location("openloop", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_rollout(path: Path, actions: np.ndarray) -> None:
    rows = []
    for step, action in enumerate(actions, 1):
        rows.append(
            {
                "step": step,
                "role": "ppo",
                "cfd_time": 148.0 + 0.1 * step,
                "applied_omega": float(action),
            }
        )
    path.write_text(json.dumps({"rows": rows}))


def test_action_sequence_contract_and_sha(tmp_path, monkeypatch):
    actions = np.linspace(0.0, 0.4, MODULE.STEPS)
    path = tmp_path / "rollout.json"
    write_rollout(path, actions)
    monkeypatch.setattr(MODULE, "SOURCE_ROLLOUT_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(MODULE, "ACTION_SEQUENCE_SHA256", MODULE.action_sequence_sha256(actions))
    np.testing.assert_array_equal(MODULE.load_action_sequence(path), actions)


def test_action_sequence_rejects_slew_violation(tmp_path, monkeypatch):
    actions = np.zeros(MODULE.STEPS)
    actions[10] = 0.2
    path = tmp_path / "rollout.json"
    write_rollout(path, actions)
    monkeypatch.setattr(MODULE, "SOURCE_ROLLOUT_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(MODULE, "ACTION_SEQUENCE_SHA256", MODULE.action_sequence_sha256(actions))
    with pytest.raises(ValueError, match="action endpoint contract"):
        MODULE.load_action_sequence(path)


def test_predeclaration_binds_protocol(tmp_path, monkeypatch):
    actions = np.zeros(MODULE.STEPS)
    predecl = {
        "status": "B00_ACTIONS_B01_OPENLOOP_PREDECLARED_NOT_EXECUTED",
        "run_id": "b00seq_b01_openloop_v1",
        "source_rollout_sha256": MODULE.SOURCE_ROLLOUT_SHA256,
        "action_sequence_sha256_float64_le": MODULE.action_sequence_sha256(actions),
        "action_endpoints": 800,
        "source_phase": "b00_train",
        "replay_phase": "b01_validation",
        "source_restart_time": 130.0,
        "end_time": 210.0,
        "analysis_window": [150.0, 210.0],
        "control_dt": 0.1,
        "solver_dt": 0.005,
        "frozen_test_access": False,
    }
    path = tmp_path / "predecl.json"
    path.write_text(json.dumps(predecl))
    monkeypatch.setattr(MODULE, "PREDECLARATION", path)
    assert MODULE.validate_predeclaration("b00seq_b01_openloop_v1", actions) == predecl
    predecl["analysis_window"] = [149.0, 210.0]
    path.write_text(json.dumps(predecl))
    with pytest.raises(ValueError, match="predeclaration differs"):
        MODULE.validate_predeclaration("b00seq_b01_openloop_v1", actions)
