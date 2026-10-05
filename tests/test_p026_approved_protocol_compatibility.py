"""Tiny protocol-only tests; no model/archive/HDF imports or candidate execution."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "p026_protocol_audit", Path(__file__).resolve().parents[1] / "scripts/audit_fcp026_candidate.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
REPO = Path(__file__).resolve().parents[1]
APPROVAL = REPO / "artifacts/fcp026_history_training_k1_20261005/execution_approval.json"
APPROVAL_SHA = "1088285e4e13c7e3553009dd511436e9a5da976e93eed44d6bb0c28ab9515aa3"
PROTOCOL_SHA = "daf22b2464744509260f1eb9e0b20d3b80da484c8985f3a22887293bbb40cb30"


@pytest.fixture
def actual_protocol():
    # Portable synthetic protocol: ordinary compatibility tests never need artifacts.
    return {
        "training_experiment": "FC-P026",
        "gradient_clip_norm": 1,
        "optimizer_steps": 171,
        "training_windows": 1368,
        "learning_rate": 1.5625e-7,
        "validation_accessed": False,
        "history_input": {"profile": "p026_k1", "history_length": 1,
                          "aerodynamic_input_channels": 6},
    }


@pytest.mark.parametrize("k", [1, 4])
@pytest.mark.parametrize("clip", [1, 1.0])
def test_only_clip_spelling_compatible_without_mutation(actual_protocol, k, clip):
    # Both arms here are synthetic; actual preserved evidence is tested separately.
    actual = copy.deepcopy(actual_protocol)
    actual["history_input"].update(profile=f"p026_k{k}", history_length=k,
                                    aerodynamic_input_channels=6 if k == 1 else 18)
    actual["gradient_clip_norm"] = clip
    expected = copy.deepcopy(actual)
    expected["gradient_clip_norm"] = 1.0
    before = json.dumps(actual, sort_keys=True)
    audit.validate_approved_protocol(actual, expected)
    assert json.dumps(actual, sort_keys=True) == before


def test_actual_candidate_protocol_and_approval_bytes():
    path = APPROVAL.parent / "candidate/training_protocol.json"
    if not APPROVAL.is_file() or not path.is_file():
        pytest.skip("actual preserved Main approval/candidate JSON unavailable")
    content = APPROVAL.read_bytes()
    assert hashlib.sha256(content).hexdigest() == APPROVAL_SHA
    approval = json.loads(content)
    assert approval["effective_protocol_sha256"] == PROTOCOL_SHA
    actual_protocol = approval["effective_protocol"]
    assert type(actual_protocol["gradient_clip_norm"]) is int
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PROTOCOL_SHA
    expected = json.loads(raw)
    with pytest.raises(ValueError, match="approved protocol"):
        audit.equal(actual_protocol, expected, "approved protocol")
    audit.validate_approved_protocol(actual_protocol, expected)


@pytest.mark.parametrize("value", [True, False, float("nan"), float("inf"),
                                   -float("inf"), 0, 2, 1.0000000000000002,
                                   "1", None])
def test_reject_invalid_clip(actual_protocol, value):
    expected = {**actual_protocol, "gradient_clip_norm": 1.0}
    with pytest.raises(ValueError):
        audit.validate_approved_protocol({**actual_protocol, "gradient_clip_norm": value}, expected)


def test_reject_missing_clip(actual_protocol):
    actual = copy.deepcopy(actual_protocol)
    del actual["gradient_clip_norm"]
    with pytest.raises(ValueError):
        audit.validate_approved_protocol(actual, {**actual_protocol, "gradient_clip_norm": 1.0})


@pytest.mark.parametrize("field,value", [("optimizer_steps", 171.0),
                                         ("training_windows", True),
                                         ("learning_rate", 1e-5),
                                         ("validation_accessed", 0)])
def test_other_fields_remain_strict(actual_protocol, field, value):
    expected = {**actual_protocol, "gradient_clip_norm": 1.0}
    actual = {**actual_protocol, field: value}
    with pytest.raises(ValueError):
        audit.validate_approved_protocol(actual, expected)


def test_extra_key_remains_rejected(actual_protocol):
    with pytest.raises(ValueError):
        audit.validate_approved_protocol({**actual_protocol, "extra": 1},
                                         {**actual_protocol, "gradient_clip_norm": 1.0})
