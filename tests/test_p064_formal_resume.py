import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).parents[1]
RUNNER = ROOT / "scripts/run_fcp064_formal_resume.py"
GENERATOR = ROOT / "scripts/prepare_fcp064_formal_resume_approval.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


resume = load("test_p064_resume", RUNNER)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reused_fixture(tmp_path):
    expected = {}
    for relative in resume.REUSED:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative.endswith("_terminal.json"):
            path.write_text(json.dumps({"State": {"ExitCode": 0, "OOMKilled": False, "Running": False}}))
        else:
            path.write_text(relative)
        expected[relative] = digest(path)
    return expected


def test_reused_proof_and_copy_are_exact(tmp_path):
    prior = tmp_path / "prior"
    output = tmp_path / "output"
    expected = reused_fixture(prior)
    resume.validate_reused(prior, expected)
    resume.copy_reused(prior, output, expected)
    assert {name: digest(output / name) for name in resume.REUSED} == expected


def test_reused_proof_rejects_hash_and_terminal_mismatch(tmp_path):
    prior = tmp_path / "prior"
    expected = reused_fixture(prior)
    bad = dict(expected)
    bad["precision.log"] = "0" * 64
    with pytest.raises(RuntimeError, match="reused source differs"):
        resume.validate_reused(prior, bad)
    terminal = prior / "evidence/validation10_container_terminal.json"
    terminal.write_text(json.dumps({"State": {"ExitCode": 1, "OOMKilled": False, "Running": False}}))
    expected["evidence/validation10_container_terminal.json"] = digest(terminal)
    with pytest.raises(RuntimeError, match="reused terminal container differs"):
        resume.validate_reused(prior, expected)


def test_execution_identity_binds_unit_invocation_and_pid(monkeypatch):
    monkeypatch.setenv("INVOCATION_ID", "a" * 32)
    monkeypatch.setattr(resume.os, "getpid", lambda: 123)
    monkeypatch.setattr(resume.subprocess, "check_output", lambda *a, **k:
        "MainPID=123\nActiveState=active\nSubState=running\nInvocationID=" + "a" * 32 + "\n")
    assert resume.execution_unit("unit.service") == {
        "unit": "unit.service", "invocation": "a" * 32, "main_pid": 123,
    }


def test_execution_identity_rejects_wrong_main_pid(monkeypatch):
    monkeypatch.setenv("INVOCATION_ID", "b" * 32)
    monkeypatch.setattr(resume.os, "getpid", lambda: 123)
    monkeypatch.setattr(resume.subprocess, "check_output", lambda *a, **k:
        "MainPID=124\nActiveState=active\nSubState=running\nInvocationID=" + "b" * 32 + "\n")
    with pytest.raises(RuntimeError, match="actual R3 unit identity differs"):
        resume.execution_unit("unit.service")


def test_production_plan_skips_only_validation10_and_records_provenance():
    tree = ast.parse(RUNNER.read_text())
    source = ast.unparse(tree)
    assert "remaining = plan[1:]" in source
    assert "['precision', 'validation10']" in source
    assert "startup_memory_samples" in source
    assert "planned_execution_unit" in source
    assert "resource.RESOURCE_CONTRACT" in source
