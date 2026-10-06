"""Small-receipt generator fixtures only; no model, HDF, or GPU access."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


STAGE = Path(__file__).resolve().parents[1]
SCRIPT = STAGE / "scripts/prepare_p028_h10_comparison_spec.py"
if not SCRIPT.is_file():
    SCRIPT = STAGE / "spec/prepare_p028_h10_comparison_spec.py"


def load_module():
    definition = importlib.util.spec_from_file_location("p029_spec_generator_test", SCRIPT)
    module = importlib.util.module_from_spec(definition)
    sys.modules[definition.name] = module
    definition.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inputs(tmp_path, profile):
    base = tmp_path / "base.json"
    base.write_text(json.dumps({
        "status": "P027_OFFLINE_DIAGNOSTIC_EXECUTION_APPROVED",
        "candidates": {"1": {}, "4": {}},
        "data": {"base": {}, "train8": {}, "train16": {}},
    }))
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    manifest = candidate / "dual_model_manifest.json"
    manifest.write_text(json.dumps({"kind": f"fixture-{profile}"}))
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({
        "status": f"FC_{profile.upper()}_CANDIDATE_INTEGRITY_VERIFIED_NOT_ADMISSION",
        "scientific_admission": False,
        "ppo_authorized": False,
        "candidate_sha256": {"candidate/dual_model_manifest.json": digest(manifest)},
    }))
    closure = tmp_path / "closure"
    closure.mkdir()
    (closure / "diagnostic.py").write_text("pass\n")
    source = closure / "source_manifest.json"
    source.write_text(json.dumps({
        "status": f"{profile.upper()}_H10_SOURCE_CLOSURE_FROZEN",
        "files_sha256": {"diagnostic.py": digest(closure / "diagnostic.py")},
    }))
    return base, source, audit, candidate


def run_generator(monkeypatch, tmp_path, profile, legacy_alias=False):
    module = load_module()
    base, source, audit, candidate = inputs(tmp_path, profile)
    output = tmp_path / "pending.json"
    audit_flag = "--p028-candidate-audit" if legacy_alias else "--candidate-audit"
    audit_sha_flag = ("--p028-candidate-audit-sha256" if legacy_alias
                      else "--candidate-audit-sha256")
    monkeypatch.setattr(sys, "argv", [
        "generator", "--profile", profile,
        "--base-p027-spec", str(base), "--base-p027-spec-sha256", digest(base),
        "--source-manifest", str(source), "--source-manifest-sha256", digest(source),
        audit_flag, str(audit), audit_sha_flag, digest(audit),
        "--candidate", str(candidate), "--output", str(output),
    ])
    module.main()
    return json.loads(output.read_text()), audit, candidate


def test_p029_pending_spec_binds_actual_small_receipts_without_authorizing(monkeypatch, tmp_path):
    value, audit, candidate = run_generator(monkeypatch, tmp_path, "p029")
    assert value["status"] == "P029_MATCHED_H10_COMPARISON_PENDING_LEAD_APPROVAL"
    assert value["intended_authorized_status"] == "P029_MATCHED_H10_COMPARISON_EXECUTION_APPROVED"
    assert value["execution_authorized"] is False
    assert value["comparison_profile"] == "p029"
    assert "p028_candidate" not in value and "p028_terminal_proof" not in value
    assert value["p029_candidate"]["manifest"] == str(
        (candidate / "dual_model_manifest.json").resolve())
    assert value["p029_terminal_proof"]["path"] == str(audit.resolve())
    assert value["p029_terminal_proof"]["reviewed_by_lead"] is False
    assert value["comparison_contract"]["original_p027_k1_k4_arms_preserved"] is True
    assert value["comparison_contract"]["formal_or_scientific_admission"] is False


def test_p028_legacy_alias_and_output_schema_are_preserved(monkeypatch, tmp_path):
    value, _, _ = run_generator(monkeypatch, tmp_path, "p028", legacy_alias=True)
    assert value["status"] == "P028_MATCHED_H10_COMPARISON_PENDING_LEAD_APPROVAL"
    assert "comparison_profile" not in value
    assert "p028_candidate" in value and "p028_terminal_proof" in value
    assert "p029_candidate" not in value


def test_p029_rejects_p028_audit_identity(monkeypatch, tmp_path):
    module = load_module()
    base, source, audit, candidate = inputs(tmp_path, "p029")
    payload = json.loads(audit.read_text())
    payload["status"] = "FC_P028_CANDIDATE_INTEGRITY_VERIFIED_NOT_ADMISSION"
    audit.write_text(json.dumps(payload))
    monkeypatch.setattr(sys, "argv", [
        "generator", "--profile", "p029",
        "--base-p027-spec", str(base), "--base-p027-spec-sha256", digest(base),
        "--source-manifest", str(source), "--source-manifest-sha256", digest(source),
        "--candidate-audit", str(audit), "--candidate-audit-sha256", digest(audit),
        "--candidate", str(candidate), "--output", str(tmp_path / "pending.json"),
    ])
    with pytest.raises(ValueError, match="P029 audit status"):
        module.main()
