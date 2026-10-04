from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_fcp003b_candidate_lineage.py"
SPEC = importlib.util.spec_from_file_location("fcp003b_lineage", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(tmp_path: Path, monkeypatch) -> argparse.Namespace:
    candidate = tmp_path / "candidate"
    files = {
        "approval": tmp_path / "approval",
        "dynamic_manifest": tmp_path / "manifest",
        "parent_model": tmp_path / "parent/FNO.0.2.mdlus",
        "parent_state": tmp_path / "parent/checkpoint.0.2.pt",
    }
    for path in files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name)
    completion = {
        "checkpoint_epoch": 2,
        "checkpoint_model_sha256": "model",
        "checkpoint_state_sha256": "state",
        "checkpoint_generation_payload_sha256": {"model.pt": "payload"},
    }
    dump(candidate / "completion_receipt.json", completion)
    dump(candidate / "launch_receipt.json", {})
    dump(candidate / "resolved_config.yaml", {})
    dump(candidate / "training_history.json", [
        {"epoch": 1, "paired_identities": list(range(16)), "paired_identity_passes": [list(range(8)), list(range(8))]},
        {"epoch": 2, "paired_identities": list(range(16)), "paired_identity_passes": [list(range(8)), list(range(8))]},
    ])
    source_receipt = tmp_path / "source_receipt.json"; source_receipt.write_text("{}")
    source_required = tmp_path / "required.json"; source_required.write_text("{}")
    real = tmp_path / "real.json"; real.write_text("{}")
    baseline = tmp_path / "baseline.json"; baseline.write_text("{}")
    validator = SimpleNamespace(
        SOURCE_COMMIT="commit",
        APPROVAL_SHA=MODULE.sha256(files["approval"]),
        DYNAMIC_MANIFEST_SHA=MODULE.sha256(files["dynamic_manifest"]),
        PARENT_MODEL_SHA=MODULE.sha256(files["parent_model"]),
        PARENT_STATE_SHA=MODULE.sha256(files["parent_state"]),
        validate_output=lambda *_: completion,
        validate_source=lambda *_: {"file_count": 284},
        validate_sampling=lambda *_: {"regular_sequence_sha256": "regular"},
    )
    monkeypatch.setattr(MODULE, "load_validator", lambda _: validator)
    return argparse.Namespace(
        candidate=candidate, validator=tmp_path / "validator.py", source=tmp_path,
        source_receipt=source_receipt, source_required=source_required,
        approval=files["approval"], dynamic_manifest=files["dynamic_manifest"],
        real_sampling=real, baseline_order=baseline, parent=tmp_path / "parent",
    )


def test_lineage_binds_dynamic_candidate_and_two_passes(tmp_path: Path, monkeypatch) -> None:
    args = fixture(tmp_path, monkeypatch)
    result = MODULE.build(args)
    assert result["status"] == "FC_P003B_DYNAMIC_PAIR_CANDIDATE_LINEAGE_PASS"
    assert result["candidate_kind"] == "dynamic_paired_interleaved_lambda10"
    assert result["paired_updates_per_epoch"] == [16, 16]
    assert len(result["paired_identity_passes_per_epoch"]) == 2


def test_lineage_rejects_tampered_completion(tmp_path: Path, monkeypatch) -> None:
    args = fixture(tmp_path, monkeypatch)
    dump(args.candidate / "completion_receipt.json", {"checkpoint_epoch": 1})
    with pytest.raises(ValueError, match="completion receipt"):
        MODULE.build(args)
