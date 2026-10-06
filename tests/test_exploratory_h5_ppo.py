"""Pure CPU contracts; not real CFD/model/training evidence."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
from train_exploratory_h5_ppo import (
    BASELINE_SHA,
    CONFIG_SHA,
    K1_SHA,
    NORM_SHA,
    PROTOCOL,
    REQUIRED_IMPORTS,
    validate_spec,
)

from exploratory_h5_hydrogym import TRAIN_CASES, audit_transition


def spec():
    return {
        "status": "EXPLORATORY_H5_PPO_EXECUTION_APPROVED",
        "execution_authorized": True,
        "protocol": json.loads(json.dumps(PROTOCOL)),
        "data_root": "/data",
        "train_cases": list(TRAIN_CASES),
        "train_hdf": [{"path": f"/data/train/{c}.h5"} for c in TRAIN_CASES],
        "runtime_packages": {"stable-baselines3": "2.7.1", "gymnasium": "1.2.3"},
        "import_bindings": dict.fromkeys(REQUIRED_IMPORTS, "/source.py"),
        "inputs": {
            k: {"sha256": v, "path": f"/data/{k}.json"}
            for k, v in {
                "manifest": K1_SHA,
                "config": CONFIG_SHA,
                "normalization": NORM_SHA,
                "baseline": BASELINE_SHA,
            }.items()
        },
    }


def test_fixed_spec():
    assert validate_spec(spec())["protocol"]["timesteps"] == 4096


@pytest.mark.parametrize(
    "key,value",
    [
        ("timesteps", 4097),
        ("episode_steps", 100),
        ("norm_obs", True),
        ("seed", 1),
        ("environments", True),
        ("device", "cpu"),
    ],
)
def test_protocol_changes_rejected(key, value):
    s = spec()
    s["protocol"][key] = value
    with pytest.raises(ValueError):
        validate_spec(s)


def test_non_train_path_rejected():
    s = spec()
    s["train_hdf"][0]["path"] = "/data/validation/a.h5"
    with pytest.raises(ValueError):
        validate_spec(s)


@pytest.mark.parametrize("step", [1, 2, 3, 4, 5])
def test_exact_five_truncation(step):
    assert audit_transition(
        step=step,
        reward=-0.1,
        components={"drag": -0.1},
        state_bound=1.0,
        state_limit=2.0,
    ) == (False, step == 5)


def test_divergence_is_not_bootstrappable_timeout():
    assert audit_transition(
        step=5, reward=-0.1, components={"drag": -0.1}, state_bound=3.0, state_limit=2.0
    ) == (True, False)


def test_unrelated_termination_preserved():
    assert audit_transition(
        step=5,
        reward=-0.1,
        components={"drag": -0.1},
        state_bound=1.0,
        state_limit=2.0,
        terminated=True,
    ) == (True, False)


def test_nonfinite_reward_rejected():
    with pytest.raises(ValueError):
        audit_transition(
            step=1,
            reward=float("nan"),
            components={"drag": -0.1},
            state_bound=1.0,
            state_limit=2.0,
        )


def test_no_formal_wrapper_gate_edit_or_inheritance():
    source = (SCRIPTS / "exploratory_h5_hydrogym.py").read_text()
    assert "class ExploratoryH5Audit(gym.Wrapper)" in source
    assert "class ExploratoryH5Audit(Full40CanonicalRewardAudit)" not in source
