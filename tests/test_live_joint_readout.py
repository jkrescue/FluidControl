import hashlib
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "dashboard_joint", Path(__file__).resolve().parents[1] / "scripts/serve_live_research_dashboard.py"
)
dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard)


def test_bound_joint_metrics_not_admission(tmp_path, monkeypatch):
    path = tmp_path / "artifacts/fcp009_free_ar_force_readout_cache_20261005/cpu_joint_50_50_analysis.json"
    path.parent.mkdir(parents=True)
    reports = {}
    for name, value in (("parent_predict_free_ar", .07), ("joint_fit_predict_free_ar", .04)):
        reports[name] = {"pooled": {"relative_horizons": {
            h: {"physical": {k: {"mae": value} for k in ("total_cd", "rear_cl")}}
            for h in ("H1", "H100")}}}
    path.write_text(json.dumps({
        "status": "FC_P009_FIXED_50_50_JOINT_READOUT_DIAGNOSTIC_COMPLETE",
        "source_result_sha256": "1321c30a1e12172b85b269405c208981390f952bdf6f03a7fe2dd21f3bb91daf",
        "reports": reports,
    }))
    monkeypatch.setattr(dashboard, "FCP009_JOINT_SHA", hashlib.sha256(path.read_bytes()).hexdigest())
    result = dashboard._joint_readout_diagnostic(tmp_path)
    assert result["ready"] and len(result["rows"]) == 4
    assert result["admission"] is False and result["candidate_verified"] is False
    assert result["rows"][0]["before"] == .07
    path.write_text("{}")
    assert dashboard._joint_readout_diagnostic(tmp_path) == {"ready": False}


def test_missing_joint_not_displayed(tmp_path):
    assert dashboard._joint_readout_diagnostic(tmp_path) == {"ready": False}
