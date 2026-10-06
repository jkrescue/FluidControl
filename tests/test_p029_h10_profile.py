"""Synthetic profile/identity fixtures only; no model, HDF, or container access."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


STAGE = Path(__file__).resolve().parents[1]


def load(name, path):
    definition = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(definition)
    sys.modules[name] = module
    definition.loader.exec_module(module)
    return module


probe = load("p029_profile_probe", STAGE / "scripts/diagnose_p028_short_horizon_comparison.py")
launcher = load("p029_profile_launcher", STAGE / "scripts/run_p028_h10_comparison.py")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidate(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")
    return {"manifest": str(path), "manifest_sha256": digest(path)}


def approval(tmp_path, profile):
    source = tmp_path / "source"
    source.mkdir()
    entry = source / launcher.ENTRY_BASENAME
    entry.write_text("pass\n")
    config = tmp_path / "config.yaml"
    baseline = tmp_path / "baseline.json"
    audit = tmp_path / "audit.json"
    phase = tmp_path / "phase.json"
    predecl = tmp_path / "predecl.json"
    guard = tmp_path / launcher.GUARD_BASENAME
    for path in (config, baseline, audit, phase, predecl, guard):
        path.write_text("{}")
    data = {}
    for family in ("base", "train8", "train16"):
        root = tmp_path / family
        (root / "train").mkdir(parents=True)
        (root / "manifest.json").write_text("{}")
        (root / "normalization.json").write_text("{}")
        data[family] = {
            "root": str(root), "manifest_sha256": "a" * 64,
            "normalization_sha256": "b" * 64, "train_files": {},
        }
    spec = {
        "status": launcher.PROFILES[profile]["approval_status"],
        "comparison_profile": profile,
        "execution_authorized": True,
        "source_files": [{"path": str(entry), "sha256": digest(entry)},
                         {"path": str(guard), "sha256": digest(guard)}],
        "config": {"path": str(config), "sha256": digest(config)},
        "baselines_file": {"path": str(baseline), "sha256": digest(baseline)},
        "train_audit": {"path": str(audit)},
        "train16_predeclaration": {"path": str(predecl), "sha256": digest(predecl)},
        "source_phase_mapping": {"path": str(phase), "sha256": digest(phase)},
        "candidates": {str(k): candidate(tmp_path / f"k{k}/manifest.json") for k in (1, 4)},
        launcher.PROFILES[profile]["candidate_key"]: candidate(
            tmp_path / f"{profile}/manifest.json"),
        "data": data,
    }
    path = tmp_path / "approval.json"
    path.write_text(json.dumps(spec))
    return spec, path


def test_historical_p028_default_is_unchanged():
    spec = {"status": "P028_MATCHED_H10_COMPARISON_EXECUTION_APPROVED"}
    profile = probe.comparison_profile(spec)
    assert profile["label"] == "p028"
    assert profile["candidate_kind"] == "FC_P028_FLOW_ROLLOUT_REPAIR"


def test_p029_requires_explicit_matching_status_and_kind():
    spec = {
        "status": "P029_MATCHED_H10_COMPARISON_EXECUTION_APPROVED",
        "comparison_profile": "p029",
        "config": {"sha256": "c" * 64},
    }
    profile = probe.comparison_profile(spec, "p029")
    identity = SimpleNamespace(payload={
        "kind": "FC_P029_CONTROL_AWARE_FLOW_REPAIR",
        "config_sha256": "c" * 64,
        "normalization_sha256": "f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1",
    })
    probe.validate_repair_identity(identity, profile, spec)
    identity.payload["kind"] = "FC_P028_FLOW_ROLLOUT_REPAIR"
    with pytest.raises(ValueError, match="P029 candidate identity"):
        probe.validate_repair_identity(identity, profile, spec)
    spec["status"] = "P028_MATCHED_H10_COMPARISON_EXECUTION_APPROVED"
    with pytest.raises(ValueError, match="approval"):
        probe.comparison_profile(spec, "p029")


def test_profile_rejected_before_runtime_access(monkeypatch, tmp_path):
    class RuntimeReached(RuntimeError):
        pass

    monkeypatch.setattr(
        probe, "runtime_dependencies",
        lambda: (_ for _ in ()).throw(RuntimeReached("runtime reached")),
    )
    approved = {"status": "P029_MATCHED_H10_COMPARISON_EXECUTION_APPROVED",
                "comparison_profile": "p029"}
    with pytest.raises(RuntimeReached):
        probe.execute(approved, tmp_path / "result.json", "p029")
    with pytest.raises(ValueError, match="profiles differ"):
        probe.execute(approved, tmp_path / "result.json", "p028")


def test_launcher_p029_mount_command_and_status_are_distinct(tmp_path, monkeypatch):
    spec, path = approval(tmp_path, "p029")
    assert launcher.load_spec(path, digest(path)) == spec
    monkeypatch.setattr(launcher, "GUARD_PATH", Path(spec["source_files"][1]["path"]))
    command = launcher.create_command(spec, path, digest(path), tmp_path / "output")
    joined = " ".join(command)
    assert launcher.CONTAINER == "fcp029-matched-h10-comparison-20261006"
    assert "--profile p029" in joined
    assert str(Path(spec["p029_candidate"]["manifest"]).parent) in joined
    assert "p028_candidate" not in spec
    assert "timeout -k 20 900" in joined


@pytest.mark.parametrize("profile", ["p028", "p029"])
def test_launcher_result_cannot_masquerade_as_other_profile(tmp_path, profile):
    spec, _ = approval(tmp_path, profile)
    result = {
        "status": launcher.PROFILES[profile]["result_status"],
        "comparison_profile": profile,
        "source_spec": spec,
        "scientific_admission": False,
        "optimizer_created": False,
        "model_saved": False,
        "validation_accessed": False,
        "frozen_test_accessed": False,
    }
    launcher.validate_result(result, spec)
    other = "p029" if profile == "p028" else "p028"
    result["status"] = launcher.PROFILES[other]["result_status"]
    with pytest.raises(RuntimeError, match="profile/status"):
        launcher.validate_result(result, spec)
