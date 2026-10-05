import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from fluid_control.calibrated_checkpoint import validate_calibrated_epoch_zero


ROOT = Path(__file__).parents[1]


def load_script(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint(tmp_path, **metadata_updates):
    model = tmp_path / "FNO.0.0.mdlus"
    state = tmp_path / "checkpoint.0.0.pt"
    model.write_bytes(b"official-model-placeholder")
    metadata = {
        "status": "FC_P008_TRAIN_ONLY_FORCE_ROW_CANDIDATE",
        "candidate_checkpoint_epoch": 0,
        "parent_checkpoint_epoch": 2,
        "calibration_generation": 1,
        "validation_accessed": False,
        "frozen_test_accessed": False,
        "ppo_executed": False,
    }
    metadata.update(metadata_updates)
    torch.save({"metadata": metadata}, state)
    return model, state


def kwargs(model, state):
    return {
        "allow": True,
        "expected_model_sha256": sha(model),
        "expected_state_sha256": sha(state),
    }


def test_epoch_zero_requires_explicit_exact_pair_and_metadata(tmp_path):
    model, state = checkpoint(tmp_path)
    with pytest.raises(ValueError, match="not explicitly allowed"):
        validate_calibrated_epoch_zero(tmp_path, 0)
    result = validate_calibrated_epoch_zero(tmp_path, 0, **kwargs(model, state))
    assert result["calibrated_epoch_zero"] is True
    assert result["checkpoint_epoch"] == 0
    assert result["checkpoint_sha256"] == sha(model)
    assert result["checkpoint_state_sha256"] == sha(state)


def test_positive_epoch_preserves_historical_path_without_calibration_files(tmp_path):
    assert validate_calibrated_epoch_zero(tmp_path, 2) == {
        "checkpoint_epoch": 2,
        "calibrated_epoch_zero": False,
    }


@pytest.mark.parametrize(
    "update",
    [
        {"status": "wrong"},
        {"candidate_checkpoint_epoch": 2},
        {"parent_checkpoint_epoch": 1},
        {"calibration_generation": 2},
        {"validation_accessed": True},
    ],
)
def test_epoch_zero_rejects_wrong_metadata(tmp_path, update):
    model, state = checkpoint(tmp_path, **update)
    with pytest.raises(ValueError, match="metadata differs"):
        validate_calibrated_epoch_zero(tmp_path, 0, **kwargs(model, state))


def test_epoch_zero_rejects_missing_or_wrong_sha(tmp_path):
    model, state = checkpoint(tmp_path)
    state.unlink()
    with pytest.raises(ValueError, match="pair differs"):
        validate_calibrated_epoch_zero(
            tmp_path,
            0,
            allow=True,
            expected_model_sha256=sha(model),
            expected_state_sha256="0" * 64,
        )


def test_existing_auditors_directly_validate_epoch_zero_identity(tmp_path):
    model, state = checkpoint(tmp_path)
    common = {
        "allow_calibrated_epoch_zero": True,
        "expected_calibrated_model_sha256": sha(model),
        "expected_calibrated_state_sha256": sha(state),
    }
    dev = load_script("audit_dev30_validation_diagnostic")
    dev_result = dev.validate_checkpoint({"checkpoint_epoch": 0}, tmp_path, **common)
    assert dev_result["calibrated_epoch_zero"] is True
    full = load_script("audit_full40_validation_gate")
    report = {"checkpoint_epoch": 0, "checkpoint_dir": str(tmp_path.resolve())}
    full_result = full.validate_checkpoint(report, tmp_path, **common)
    assert full_result["calibrated_epoch_zero"] is True
    with pytest.raises(ValueError, match="not explicitly allowed"):
        dev.validate_checkpoint({"checkpoint_epoch": 0}, tmp_path)
