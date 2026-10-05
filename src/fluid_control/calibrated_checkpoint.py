"""Strict identity check for the approved FC-P008 epoch-zero calibration."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


FC_P008_STATUS = "FC_P008_TRAIN_ONLY_FORCE_ROW_CANDIDATE"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_calibrated_epoch_zero(
    checkpoint_dir: Path,
    epoch: int,
    *,
    allow: bool = False,
    expected_model_sha256: str | None = None,
    expected_state_sha256: str | None = None,
) -> dict:
    """Accept epoch zero only for the exact official FC-P008 checkpoint pair.

    PhysicsNeMo returns ``0`` both for a valid checkpoint saved with epoch zero
    and for a missing training state. This function removes that ambiguity.
    Positive epochs preserve the historical behavior and need no exception.
    """
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        raise ValueError("checkpoint epoch is invalid")
    if epoch > 0:
        return {"checkpoint_epoch": epoch, "calibrated_epoch_zero": False}
    if not allow:
        raise ValueError("epoch-zero calibration was not explicitly allowed")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_model_sha256 or ""):
        raise ValueError("expected calibrated model SHA is required")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_state_sha256 or ""):
        raise ValueError("expected calibrated state SHA is required")
    models = sorted(checkpoint_dir.glob("FNO.0.*.mdlus"))
    states = sorted(checkpoint_dir.glob("checkpoint.0.*.pt"))
    if [path.name for path in models] != ["FNO.0.0.mdlus"] or [path.name for path in states] != ["checkpoint.0.0.pt"]:
        raise ValueError("calibrated epoch-zero checkpoint pair differs")
    model_sha, state_sha = _sha256(models[0]), _sha256(states[0])
    if model_sha != expected_model_sha256 or state_sha != expected_state_sha256:
        raise ValueError("calibrated epoch-zero checkpoint SHA differs")
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("torch is required for calibrated checkpoint identity") from error
    payload = torch.load(states[0], map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or set(payload) != {"metadata"}:
        raise ValueError("calibrated training-state payload differs")
    metadata = payload["metadata"]
    required = {
        "status": FC_P008_STATUS,
        "candidate_checkpoint_epoch": 0,
        "parent_checkpoint_epoch": 2,
        "calibration_generation": 1,
        "validation_accessed": False,
        "frozen_test_accessed": False,
        "ppo_executed": False,
    }
    if not isinstance(metadata, dict) or any(metadata.get(key) != value for key, value in required.items()):
        raise ValueError("calibrated training-state metadata differs")
    return {
        "checkpoint_epoch": 0,
        "calibrated_epoch_zero": True,
        "checkpoint_model_file": models[0].name,
        "checkpoint_state_file": states[0].name,
        "checkpoint_sha256": model_sha,
        "checkpoint_state_sha256": state_sha,
        "calibration_generation": 1,
        "parent_checkpoint_epoch": 2,
    }
