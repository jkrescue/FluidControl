"""CPU mock launcher tests; no Docker, model, HDF, or GPU access."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


STAGE = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get(
    "P031_CANONICAL_REPO", "/workspace/fluid_control"))
SCRIPT = STAGE / "scripts/run_p031_control_aware_flow.py"
definition = importlib.util.spec_from_file_location("p031_launcher_tested", SCRIPT)
launcher = importlib.util.module_from_spec(definition)
sys.modules[definition.name] = launcher
definition.loader.exec_module(launcher)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path, text="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return {"path": str(path), "sha256": digest(path)}


def fixture_spec(tmp_path, mode):
    source = tmp_path / "source"
    entry = source / "scripts/train_p031_control_aware_flow_h25.py"
    entry.parent.mkdir(parents=True)
    entry.write_text("pass\n")
    data = {}
    for family in ("base", "train8", "train16"):
        root = tmp_path / family
        (root / "train").mkdir(parents=True)
        (root / "manifest.json").write_text("{}")
        (root / "normalization.json").write_text("{}")
        data[family] = {"root": str(root),
                        "manifest_sha256": digest(root / "manifest.json"),
                        "normalization_sha256": digest(root / "normalization.json"),
                        "train_files": {"a.h5": "a" * 64}}
    parent = tmp_path / "parent/dual_model_manifest.json"
    parent.parent.mkdir()
    parent.write_text("{}")
    value = {
        "status": "FC_P031_EXECUTION_APPROVED", "mode": mode,
        "protocol": launcher.PROTOCOL, "resources": {"allocator_fraction": 0.06},
        "source_root": str(source),
        "source_sha256": {"scripts/" + entry.name: digest(entry)},
        "config": record(tmp_path / "config.yaml"),
        "parent_manifest": {"path": str(parent), "sha256": digest(parent)},
        "train_audit": record(tmp_path / "audit.json", "{}"), "data": data,
        "fixed_scales": {"field": 0.001456146538716282,
                         "force": 0.003364271827125755},
        "scales_receipt": record(tmp_path / "scales.json", "{}"),
    }
    if mode == "train":
        value["resource_probe_receipt"] = record(tmp_path / "probe.json", "{}")
    path = tmp_path / f"{mode}.json"
    path.write_text(json.dumps(value))
    return value, path, digest(path)


@pytest.mark.parametrize("mode,deadline,proofs", [
    ("resource-probe", "900", 1), ("train", "14400", 2),
])
def test_exact_mode_command_and_readonly_mounts(tmp_path, mode, deadline, proofs):
    base = launcher.load_base(REPO / "scripts/run_p028_flow_training.py")
    value, path, sha = fixture_spec(tmp_path, mode)
    assert launcher.load_spec(base, path, sha, mode) == value
    base.CONTAINER = f"fcp031-{mode}-20261006"
    command = launcher.create_command(base, value, path, sha, tmp_path / "output")
    assert command[:4] == ["docker", "create", "--name", base.CONTAINER]
    assert command[command.index("--gpus") + 1] == "device=0"
    assert command[command.index("--memory") + 1] == "12g"
    assert command[command.index("--memory-swap") + 1] == "12g"
    assert command[command.index("timeout") + 3] == deadline
    assert command[command.index("--mode") + 1] == mode
    mounts = launcher.readonly_mounts(base, value, path)
    assert all(read_only for _, _, read_only in mounts)
    mounted = {str(source) for source, _, _ in mounts}
    assert sum(str(tmp_path / name) in mounted for name in ("scales.json", "probe.json")) == proofs
    assert all("validation" not in item and "frozen" not in item for item in mounted)


@pytest.mark.parametrize("mode", launcher.PROFILES)
def test_result_contract(tmp_path, mode):
    value, _, _ = fixture_spec(tmp_path, mode)
    profile = launcher.PROFILES[mode]
    result = {"status": profile["status"], "mode": mode,
              "optimizer_steps": profile["optimizer_steps"],
              "training_windows": profile["training_windows"],
              "scientific_admission": False, "validation_accessed": False,
              "frozen_test_accessed": False, "source_spec": value,
              "optimizer_created": mode == "train", "model_saved": mode == "train",
              "sampler_order_sha256": (launcher.RESOURCE_PROBE_ORDER_SHA256
                                        if mode == "resource-probe" else launcher.FULL_ORDER_SHA256),
              "fixed_scales": value["fixed_scales"],
              "records": [{}] * (171 if mode == "train" else 1)}
    if mode == "train":
        result["official_fresh_reload_verified"] = True
    launcher.validate_result(result, value)
    result["validation_accessed"] = True
    with pytest.raises(RuntimeError, match="isolation"):
        launcher.validate_result(result, value)


def test_wrong_status_horizon_scale_and_missing_proofs_rejected(tmp_path):
    base = launcher.load_base(REPO / "scripts/run_p028_flow_training.py")
    value, path, _ = fixture_spec(tmp_path, "resource-probe")
    for mutate in ("status", "horizon", "scale", "receipt"):
        bad = json.loads(json.dumps(value))
        if mutate == "status":
            bad["status"] = "FC_P029_EXECUTION_APPROVED"
        elif mutate == "horizon":
            bad["protocol"]["horizon"] = 10
        elif mutate == "scale":
            bad["fixed_scales"]["field"] *= 2
        else:
            del bad["scales_receipt"]
        path.write_text(json.dumps(bad))
        with pytest.raises(RuntimeError):
            launcher.load_spec(base, path, digest(path), "resource-probe")


def test_dry_run_never_calls_subprocess(tmp_path, monkeypatch, capsys):
    _, path, sha = fixture_spec(tmp_path, "resource-probe")
    def forbidden(*_args, **_kwargs):
        raise AssertionError("dry-run invoked process/container")
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "check_output", forbidden)
    monkeypatch.setattr(sys, "argv", ["launcher", "--base-launcher",
        str(REPO / "scripts/run_p028_flow_training.py"), "--spec", str(path),
        "--spec-sha256", sha, "--mode", "resource-probe",
        "--output", str(tmp_path / "output")])
    launcher.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "FC_P031_RESOURCE_PROBE_LAUNCH_PREPARED_NO_DOCKER_NO_GPU"
    assert payload["deadline_seconds"] == 900
    assert not (tmp_path / "output").exists()
