import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "dashboard_fcp008", Path(__file__).resolve().parents[1] / "scripts/serve_live_research_dashboard.py"
)
dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard)


@pytest.fixture
def calibrated(tmp_path, monkeypatch):
    base = tmp_path / dashboard.FCP008
    path = base / "candidate_build/result.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "candidate_model_sha256": "a" * 64,
        "full_train_parent_native_metrics_physical": {
            "rear_cd": {"mae": .025}, "rear_cl": {"mae": .064}},
        "full_train_native_metrics_physical": {
            "rear_cd": {"mae": .010}, "rear_cl": {"mae": .017}},
    }))
    monkeypatch.setattr(dashboard, "FCP008_RESULT_SHA", hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(dashboard, "_service_state", lambda unit: "active")
    return tmp_path, base, path


def test_active_stage_is_not_admission(calibrated):
    root, _, _ = calibrated
    result = dashboard._full_train_calibration(root)
    assert result["ready"] and result["service_state"] == "active"
    assert result["stage"] == "validation10"
    assert result["admission"] is False and result["formal_complete"] is False


def test_missing_or_tampered_result_is_hidden(calibrated):
    root, _, path = calibrated
    path.write_text("{}")
    assert dashboard._full_train_calibration(root) == {"ready": False}
    path.unlink()
    assert dashboard._full_train_calibration(root) == {"ready": False}


def test_completed_step_does_not_imply_live_job(calibrated, monkeypatch):
    root, base, _ = calibrated
    steps = base / "posteval_fc_p008/step_receipts"
    steps.mkdir(parents=True)
    (steps / "validation10.json").write_text("{}")
    monkeypatch.setattr(dashboard, "_service_state", lambda unit: "inactive")
    result = dashboard._full_train_calibration(root)
    assert result["stage"] == "dynamic6"
    assert result["service_state"] == "inactive"
    assert result["admission"] is False


def test_completion_record_is_model_bound_and_not_scientific_pass(calibrated):
    root, base, _ = calibrated
    receipt = base / "posteval_fc_p008/receipt.json"
    receipt.parent.mkdir()
    receipt.write_text(json.dumps({"status": "FC_P008_POSTEVAL_COMPLETE", "checkpoint_sha256": "b" * 64}))
    assert not dashboard._full_train_calibration(root)["formal_complete"]
    receipt.write_text(json.dumps({"status": "FC_P008_POSTEVAL_COMPLETE", "checkpoint_sha256": "a" * 64}))
    result = dashboard._full_train_calibration(root)
    assert result["formal_complete"] and result["admission"] is False
