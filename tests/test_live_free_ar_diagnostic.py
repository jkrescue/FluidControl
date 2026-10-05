import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "dashboard_free_ar", Path(__file__).resolve().parents[1] / "scripts/serve_live_research_dashboard.py"
)
dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard)


def test_progress_requires_actual_service_observation(tmp_path, monkeypatch):
    path = tmp_path / "artifacts/fcp009_free_ar_force_readout_cache_20261005/run.log"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"event": "fcp009_extract_progress", "windows": 12, "rows": 1200, "finite": True}) + "\npartial")
    monkeypatch.setattr(dashboard, "_service_state", lambda unit: "inactive")
    result = dashboard._free_ar_diagnostic(tmp_path)
    assert result["ready"] and result["windows"] == 12
    assert result["service_state"] == "inactive" and result["admission"] is False


def test_malformed_progress_not_accepted(tmp_path, monkeypatch):
    path = tmp_path / "artifacts/fcp009_free_ar_force_readout_cache_20261005/run.log"
    path.parent.mkdir(parents=True)
    monkeypatch.setattr(dashboard, "_service_state", lambda unit: "active")
    for windows, rows in [(True, 100), (1369, 136900), (4, 499), (-1, -100)]:
        path.write_text(json.dumps({"event": "fcp009_extract_progress", "windows": windows, "rows": rows}))
        assert dashboard._free_ar_diagnostic(tmp_path) == {"ready": False}


def test_missing_progress_not_running(tmp_path):
    assert dashboard._free_ar_diagnostic(tmp_path) == {"ready": False}
