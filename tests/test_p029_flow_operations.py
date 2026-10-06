"""Isolated source/profile tests; no Docker, model, HDF, or real freeze."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def load(name):
    path = Path(__file__).with_name(name + ".py")
    if not path.exists():
        path = Path(__file__).resolve().parents[1] / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cpu = load("run_p028_official_cpu_reload")
freeze = load("prepare_fcp028_formal_source_freeze")


def test_exact_closure_without_numeric_additions():
    old = freeze.cpu_closure("FC-P028")
    new = freeze.cpu_closure("FC-P029")
    assert len(old) == 10 and len(new) == 12
    assert "scripts/flow_repair_profiles.py" in old
    assert set(new) - set(old) == {"scripts/audit_fcp029_candidate.py",
                                  "scripts/verify_fcp029_dual_reload.py"}
    assert len(freeze.OVERLAYS) == 7
    args = SimpleNamespace(experiment="FC-P029", repo=Path("/repo"),
                           numerical_runner=Path("/numerical.py"), formal_runner=Path("/p029.py"))
    sources = freeze.orchestration_sources(args)
    assert set(sources) == {"scripts/run_fcp026_posteval.py", "scripts/run_fcp028_posteval.py",
                            "scripts/run_fcp029_posteval.py", "scripts/flow_repair_profiles.py"}


def test_explicit_future_actual_pins_and_no_gpu(tmp_path):
    profile = cpu.p029_profile("a" * 64, "b" * 32)
    assert profile.source_count == 12
    assert profile.config == profile.source.parent / "numerical_source/training_config.yaml"
    assert str(profile.candidate).endswith("fcp029_control_aware_flow_training_20261006/payload")
    spec = {"audit_path": "/audit.json", "audit_sha": "c" * 64}
    command = cpu.create_command(spec, tmp_path, profile)
    assert "--gpus" not in command and "NVIDIA_VISIBLE_DEVICES=void" in command
    assert command[command.index("--runtime") + 1] == "runc"
    assert command[command.index("--memory") + 1] == "8g"
    assert command[command.index("timeout") + 3] == "300"
    assert str(profile.source / "scripts/verify_fcp029_dual_reload.py") in command
    assert len(cpu.readonly_mounts(spec, profile=profile)) == 4
    assert all(row[2] is True for row in cpu.readonly_mounts(spec, profile=profile))


@pytest.mark.parametrize("sha,inv", [(None, "a"*32), ("pending", "a"*32), ("a"*64, None), ("a"*64, "pending")])
def test_unobserved_placeholder_pins_rejected(sha, inv):
    with pytest.raises(RuntimeError):
        cpu.p029_profile(sha, inv)


def test_p029_terminal_does_not_accept_p028(monkeypatch):
    profile = cpu.p029_profile("a"*64, "b"*32)
    fields = dict(LoadState="loaded", ActiveState="active", SubState="exited", Result="success",
                  ExecMainCode="1", ExecMainStatus="0", MainPID="0", InvocationID=cpu.INVOCATION)
    monkeypatch.setattr(cpu.subprocess, "check_output", lambda *a, **k: "\n".join(f"{k}={v}" for k,v in fields.items()))
    with pytest.raises(RuntimeError):
        cpu.require_training_terminal(profile)


def test_unknown_freeze_profile_rejected():
    with pytest.raises(ValueError):
        freeze.cpu_closure("FC-P030")
