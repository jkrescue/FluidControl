"""Dashboard software fixtures, not scientific evaluation results."""
import hashlib
import json
from types import SimpleNamespace

from test_live_fcp013_training import dashboard


def live():
    return ("ActiveState=active\nSubState=running\nMainPID=123\n"
            "InvocationID=7235b2f06282435a89b84964e384c60f\n"
            "ExecStart=/usr/bin/env FCP_POSTEVAL_PROFILE=p013 "
            "/repo/artifacts/p013_posteval_chain_f95048c3a786_immutable/scripts/run_fcp008_posteval_spark.sh --execute\n")


def test_formal_state_needs_exact_running_invocation():
    assert dashboard._parse_fcp013_posteval_live(live())["running"]
    for old, new in (("MainPID=123", "MainPID=0"), ("SubState=running", "SubState=exited"),
                     ("7235b2f06282435a89b84964e384c60f", "another"),
                     ("FCP_POSTEVAL_PROFILE=p013", "FCP_POSTEVAL_PROFILE=p009")):
        assert not dashboard._parse_fcp013_posteval_live(live().replace(old, new))["running"]
    assert not dashboard._parse_fcp013_posteval_live("")["admission"]


def fixture(tmp_path, monkeypatch):
    base = tmp_path / "artifacts/fcp013_independent_force_fno_training_r2_20261005"
    hashes = {}
    def write(name, value):
        path = base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(value).encode()
        path.write_bytes(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    write("candidate_audit.json", {"dual_manifest_sha256": "fixture-manifest"})
    write("completion_receipt.json", {"status": "FC_P013_TRAINING_COMPLETE_NOT_ADMISSION",
          "candidate_audit_sha256": hashes["candidate_audit.json"], "dual_manifest_sha256": "fixture-manifest"})
    rows = [{"identity": {"window": i}, "true_state_h1_force_mae": [0, 0, 0, 0.1],
             "free_ar_force_mae": [0, 0, 0, 0.1], "true_state_h1_field_relative_l2_uvp": [0.1]*3,
             "free_ar_field_relative_l2_uvp": [0.2]*3} for i in range(6)]
    altered = json.loads(json.dumps(rows))
    for row in altered:
        row["true_state_h1_force_mae"][3] = 0.2
    write("fixed_six_diagnostics/result.json", {
        "status": "FC_P013_FIXED_TRAIN_DIAGNOSTICS_COMPLETE_NOT_ADMISSION",
        "dual_manifest_sha256": "fixture-manifest", "tensor_sha256_before": {"x": "same"},
        "tensor_sha256_after": {"x": "same"},
        "panels": {"p009_parent": {"windows": rows}, "p013_terminal": {"windows": altered}}})
    monkeypatch.setattr(dashboard, "FCP013_TERMINAL_FILES", hashes)
    monkeypatch.setattr(dashboard.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=live()))
    return base


def test_terminal_diagnostic_counts_are_not_admission(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    value = dashboard._fcp013_terminal_progress(tmp_path)
    assert value["verified"] and value["formal"]["running"]
    assert value["h1_regressions"] == 6 and value["ar_regressions"] == 0
    assert value["fields_unchanged"] and not value["admission"]


def test_changed_artifact_is_not_displayed_as_verified(tmp_path, monkeypatch):
    base = fixture(tmp_path, monkeypatch)
    (base / "fixed_six_diagnostics/result.json").write_text("{}")
    assert dashboard._fcp013_terminal_progress(tmp_path) == {"verified": False}


def test_missing_artifacts_are_not_completion(tmp_path):
    assert dashboard._fcp013_terminal_progress(tmp_path) == {"verified": False}
