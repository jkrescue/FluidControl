import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "dashboard_fcp009_formal", Path(__file__).resolve().parents[1] / "scripts/serve_live_research_dashboard.py"
)
dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard)


def test_actual_service_not_historical_files(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "_service_state", lambda unit: "inactive")
    base = tmp_path / "artifacts/fcp009_joint_force_row_candidate_20261005/posteval_fc_p009"
    (base / "step_receipts").mkdir(parents=True)
    receipt = {"status": "FC_P009_POSTEVAL_STEP_COMPLETE", "step": "validation10",
               "checkpoint_sha256": "dc41fc91d42476e052970b39fc66aed22fa72aa8b6f218a341a3abb095f42e31"}
    (base / "step_receipts/validation10.json").write_text(json.dumps(receipt))
    result = dashboard._fcp009_formal_status(tmp_path)
    assert result["stage"] == "dynamic6" and result["service_state"] == "inactive"
    assert result["complete_recorded"] is False and result["admission"] is False
    assert result["verified_fail"] is False
    receipt["checkpoint_sha256"] = "wrong"
    (base / "step_receipts/validation10.json").write_text(json.dumps(receipt))
    assert dashboard._fcp009_formal_status(tmp_path)["stage"] == "validation10"
