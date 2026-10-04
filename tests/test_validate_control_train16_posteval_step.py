"""Focused corruption tests for post-evaluation artifact reuse."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validate_control_train16_posteval_step.py"
spec = importlib.util.spec_from_file_location("posteval_validator", SCRIPT)
assert spec is not None and spec.loader is not None
MODULE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MODULE)


def test_truncated_json_is_rejected(tmp_path) -> None:
    path = tmp_path / "truncated.json"
    path.write_text('{"status":', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        MODULE.load_object(path)


def test_wrong_checkpoint_is_rejected() -> None:
    with pytest.raises(ValueError, match="checkpoint SHA differs"):
        MODULE.require_checkpoint({"checkpoint_sha256": "a" * 64}, "b" * 64)


def test_complete_receipt_rejects_tampered_file(tmp_path) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"finite": true}\n', encoding="utf-8")
    checkpoint = "c" * 64
    receipt = {
        "status": "CONTROL_TRAIN16_POSTEVAL_COMPLETE",
        "checkpoint_sha256": checkpoint,
        "sha256": {"evidence.json": MODULE.sha256(evidence)},
    }
    (tmp_path / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    MODULE.validate_complete(tmp_path, checkpoint)
    evidence.write_text('{"finite": false}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="hash table differs"):
        MODULE.validate_complete(tmp_path, checkpoint)
