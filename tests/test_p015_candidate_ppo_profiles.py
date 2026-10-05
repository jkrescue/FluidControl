"""Software fixtures only: no P015 scientific admission or policy execution."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("p015_fixture_helpers", Path(__file__).with_name("test_p013_candidate_ppo_plumbing.py"))
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)
KIND = "fcp015_window_accumulation_dual_fno"
EXPERIMENT = dict(training_experiment="FC-P015", optimizer_steps=171,
                  training_windows=1368, accumulation_windows=8)


def fixture(tmp_path, monkeypatch):
    args = P.provenance_fixture(tmp_path, monkeypatch)
    args["profile"] = "P015"
    root = args["candidate_root"]
    lineage = args["lineage"]
    lineage.update(EXPERIMENT, candidate_kind=KIND,
                   status="FC_P015_DUAL_CANDIDATE_LINEAGE_PASS_NOT_ADMISSION")
    for filename, key in (("execution_approval.json", "training_approval_sha256"),
                          ("running_execution_evidence.json", "execution_observation_sha256")):
        P.READY.write_json(root / filename, {"fixture": filename})
        lineage[key] = P.READY.digest(root / filename)
    P.READY.write_json(root / "candidate_audit.json", lineage)
    path = root / "completion_receipt.json"
    completion = P.READY.MODULE.load(path)
    completion.update(EXPERIMENT, status="FC_P015_TRAINING_COMPLETE_NOT_ADMISSION",
                      candidate_audit_sha256=P.READY.digest(root / "candidate_audit.json"))
    for key in ("checkpoint_sha256", "checkpoint_state_sha256", "training_approval_sha256", "execution_observation_sha256"):
        completion[key] = lineage[key]
    del completion["aerodynamic_model_sha256"], completion["aerodynamic_state_sha256"]
    P.READY.write_json(path, completion)
    P.READY.write_json(root / "dual_reload_receipt.json", {"software_fixture": True})
    approval_path = args["receipt_path"].parent / "evidence/formal_evaluation_approval.json"
    approval = P.READY.MODULE.load(approval_path)
    approval.update(candidate_kind=KIND, candidate_completion_receipt_sha256=P.READY.digest(path),
                    dual_reload_receipt_sha256=P.READY.digest(root / "dual_reload_receipt.json"))
    P.READY.write_json(approval_path, approval)
    precision_path = args["receipt_path"].parent / "precision.json"
    precision = P.READY.MODULE.load(precision_path)
    precision["status"] = "FC_P015_FORMAL_EVALUATION_DEFAULT_TF32_HIGH"
    P.READY.write_json(precision_path, precision)
    return args


def test_p015_actual_completion_schema(tmp_path, monkeypatch):
    result = P.READY.MODULE.p013_candidate_identity(**fixture(tmp_path, monkeypatch))
    assert result["epoch"] == 1
    assert result["checkpoint_relative_directory"] == Path("candidate/aerodynamic")


@pytest.mark.parametrize("key,value", [("optimizer_steps", 1368), ("training_windows", 171),
    ("accumulation_windows", 1), ("candidate_kind", P.KIND), ("training_experiment", "FC-P013")])
def test_rejects_mixed_profile(tmp_path, monkeypatch, key, value):
    args = fixture(tmp_path, monkeypatch)
    args["lineage"][key] = value
    with pytest.raises(ValueError):
        P.READY.MODULE.p013_candidate_identity(**args)


@pytest.mark.parametrize("filename", ["execution_approval.json", "running_execution_evidence.json",
                                      "completion_receipt.json", "dual_reload_receipt.json"])
def test_rejects_changed_terminal_evidence(tmp_path, monkeypatch, filename):
    args = fixture(tmp_path, monkeypatch)
    P.READY.write_json(args["candidate_root"] / filename, {"changed": True})
    with pytest.raises(ValueError):
        P.READY.MODULE.p013_candidate_identity(**args)


@pytest.mark.parametrize("mode", ["dry-run", "execute"])
def test_p015_dual_command(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(P, "KIND", KIND)
    P.test_dual_command_has_all_identities_and_vecnormalize(tmp_path, mode)


@pytest.mark.parametrize("key", ["endpoint_gate_path", "window_gate_path", "dynamic_gate_path", "development_gate_path"])
def test_p015_keeps_all_scientific_failures(tmp_path, monkeypatch, key):
    monkeypatch.setattr(P, "KIND", KIND)
    P.test_p013_does_not_bypass_any_legacy_numeric_gate(tmp_path, monkeypatch, key)


def test_p015_host_export(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "KIND", KIND)
    P.test_export_preserves_binding_and_translates_only_policy(tmp_path)
