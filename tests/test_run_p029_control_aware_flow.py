import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


STAGE = Path(__file__).resolve().parents[1]
REPO = (STAGE if (STAGE / "scripts/run_p028_flow_training.py").is_file()
        else Path("/workspace/fluid_control"))


def load_launcher():
    path = STAGE / "scripts/run_p029_control_aware_flow.py"
    definition = importlib.util.spec_from_file_location("p029_launcher_tested", path)
    module = importlib.util.module_from_spec(definition)
    sys.modules[definition.name] = module
    definition.loader.exec_module(module)
    return module


@pytest.fixture
def launcher():
    return load_launcher()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_file(path, text="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return {"path": str(path), "sha256": digest(path)}


def fixture_spec(tmp_path, launcher, mode):
    source = tmp_path / "source"
    entry = source / "scripts/train_p029_control_aware_flow.py"
    entry.parent.mkdir(parents=True)
    entry.write_text("pass\n")
    records = {"scripts/train_p029_control_aware_flow.py": digest(entry)}
    data = {}
    for family in ("base", "train8", "train16"):
        root = tmp_path / family
        (root / "train").mkdir(parents=True)
        (root / "manifest.json").write_text("{}")
        (root / "normalization.json").write_text("{}")
        (root / "train/a.h5").write_text("not opened")
        data[family] = {
            "root": str(root),
            "manifest_sha256": digest(root / "manifest.json"),
            "normalization_sha256": digest(root / "normalization.json"),
            "train_files": {"a.h5": digest(root / "train/a.h5")},
        }
    parent = tmp_path / "parent/dual_model_manifest.json"
    parent.parent.mkdir()
    parent.write_text("{}")
    spec = {
        "status": "FC_P029_EXECUTION_APPROVED",
        "mode": mode,
        "protocol": launcher.PROTOCOL,
        "resources": {"allocator_fraction": 0.06},
        "source_root": str(source),
        "source_sha256": records,
        "config": make_file(tmp_path / "config.yaml"),
        "parent_manifest": {"path": str(parent), "sha256": digest(parent)},
        "train_audit": make_file(tmp_path / "train_audit.json", "{}"),
        "data": data,
    }
    if mode != "scales":
        spec["fixed_scales"] = {"field": 1.25, "force": 2.5}
        spec["scales_receipt"] = make_file(tmp_path / "scales.json", "{}")
    if mode == "train":
        spec["resource_probe_receipt"] = make_file(tmp_path / "probe.json", "{}")
    path = tmp_path / f"{mode}-approval.json"
    path.write_text(json.dumps(spec))
    return spec, path, digest(path)


@pytest.mark.parametrize("mode,deadline,proof_count", [
    ("scales", "900", 0),
    ("resource-probe", "900", 1),
    ("train", "14400", 2),
])
def test_mode_contract_and_command(tmp_path, launcher, mode, deadline, proof_count):
    base = launcher.load_base(REPO / "scripts/run_p028_flow_training.py")
    spec, path, sha = fixture_spec(tmp_path, launcher, mode)
    assert launcher.load_spec(base, path, sha, mode) == spec
    base.CONTAINER = f"fcp029-{mode}-20261006"
    command = launcher.create_command(base, spec, path, sha, tmp_path / "output")
    assert command[:4] == ["docker", "create", "--name", base.CONTAINER]
    assert launcher.IMAGE in command
    assert command[command.index("--memory") + 1] == "12g"
    assert command[command.index("--memory-swap") + 1] == "12g"
    assert command[command.index("--gpus") + 1] == "device=0"
    assert command[command.index("timeout") + 3] == deadline
    assert command[command.index("--mode") + 1] == mode
    mounts = launcher.readonly_mounts(base, spec, path)
    mounted_sources = {str(row[0]) for row in mounts}
    assert sum(name in mounted_sources for name in (
        str(tmp_path / "scales.json"), str(tmp_path / "probe.json"))) == proof_count
    assert all("validation" not in str(source) and "frozen" not in str(source)
               for source, _, _ in mounts)
    assert all(read_only for _, _, read_only in mounts)


@pytest.mark.parametrize("mode", ["scales", "resource-probe", "train"])
def test_result_contract(tmp_path, launcher, mode):
    spec, _, _ = fixture_spec(tmp_path, launcher, mode)
    profile = launcher.PROFILES[mode]
    result = {
        "status": profile["status"],
        "mode": mode,
        "optimizer_steps": profile["optimizer_steps"],
        "training_windows": profile["training_windows"],
        "scientific_admission": False,
        "validation_accessed": False,
        "frozen_test_accessed": False,
        "source_spec": spec,
        "optimizer_created": mode == "train",
        "model_saved": mode == "train",
        "sampler_order_sha256": (launcher.RESOURCE_PROBE_ORDER_SHA256
                                  if mode == "resource-probe"
                                  else launcher.FULL_ORDER_SHA256),
        "fixed_scales": {"field": 1.25, "force": 2.5},
        "records": [{}] * profile["training_windows"] if mode == "scales" else (
            [{}] * 171 if mode == "train" else [{}]),
    }
    if mode == "train":
        result["official_fresh_reload_verified"] = True
    launcher.validate_result(result, spec)
    result["validation_accessed"] = True
    with pytest.raises(RuntimeError, match="isolation"):
        launcher.validate_result(result, spec)


def test_resource_probe_rejects_full_sampler_order(tmp_path, launcher):
    spec, _, _ = fixture_spec(tmp_path, launcher, "resource-probe")
    profile = launcher.PROFILES["resource-probe"]
    result = {
        "status": profile["status"],
        "mode": "resource-probe",
        "optimizer_steps": 0,
        "training_windows": 1,
        "scientific_admission": False,
        "validation_accessed": False,
        "frozen_test_accessed": False,
        "source_spec": spec,
        "optimizer_created": False,
        "model_saved": False,
        "sampler_order_sha256": launcher.RESOURCE_PROBE_ORDER_SHA256,
        "fixed_scales": spec["fixed_scales"],
        "records": [{}],
    }
    launcher.validate_result(result, spec)
    result["sampler_order_sha256"] = launcher.FULL_ORDER_SHA256
    with pytest.raises(RuntimeError, match="observed sampler"):
        launcher.validate_result(result, spec)


def test_each_mode_requires_separate_approval_and_prior_proofs(tmp_path, launcher):
    base = launcher.load_base(REPO / "scripts/run_p028_flow_training.py")
    spec, path, sha = fixture_spec(tmp_path, launcher, "scales")
    with pytest.raises(RuntimeError, match="mode approval"):
        launcher.load_spec(base, path, sha, "train")
    spec, path, _ = fixture_spec(tmp_path / "resource", launcher, "resource-probe")
    spec.pop("scales_receipt")
    path.write_text(json.dumps(spec))
    with pytest.raises(RuntimeError, match="scales_receipt"):
        launcher.load_spec(base, path, digest(path), "resource-probe")
    spec, path, _ = fixture_spec(tmp_path / "train", launcher, "train")
    spec.pop("resource_probe_receipt")
    path.write_text(json.dumps(spec))
    with pytest.raises(RuntimeError, match="resource_probe_receipt"):
        launcher.load_spec(base, path, digest(path), "train")


def test_protocol_and_lifecycle_sha_fail_closed(tmp_path, launcher):
    base = launcher.load_base(REPO / "scripts/run_p028_flow_training.py")
    spec, path, _ = fixture_spec(tmp_path, launcher, "scales")
    spec["protocol"] = {**spec["protocol"], "field_weight": 0.6}
    path.write_text(json.dumps(spec))
    with pytest.raises(RuntimeError, match="protocol"):
        launcher.load_spec(base, path, digest(path), "scales")
    wrong = tmp_path / "base.py"
    wrong.write_text("pass\n")
    with pytest.raises(RuntimeError, match="lifecycle SHA"):
        launcher.load_base(wrong)


def test_dry_run_does_not_call_subprocess(tmp_path, launcher, monkeypatch, capsys):
    spec, path, sha = fixture_spec(tmp_path, launcher, "scales")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dry-run attempted a process/container operation")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "check_output", forbidden)
    monkeypatch.setattr(sys, "argv", [
        "launcher", "--base-launcher", str(REPO / "scripts/run_p028_flow_training.py"),
        "--spec", str(path), "--spec-sha256", sha, "--mode", "scales",
        "--output", str(tmp_path / "output"),
    ])
    launcher.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"].endswith("NO_DOCKER_NO_GPU")
    assert payload["deadline_seconds"] == 900
    assert payload["command"][0:2] == ["docker", "create"]
    assert not (tmp_path / "output").exists()


def test_outer_deadline_is_mode_specific_cleanup_escape(launcher, monkeypatch, tmp_path):
    class Base:
        def execute(self, *_args):
            handler = signal.getsignal(signal.SIGALRM)
            handler(signal.SIGALRM, None)

    import signal
    spec = {"mode": "resource-probe"}
    with pytest.raises(launcher.OuterDeadline):
        launcher.execute(Base(), spec, tmp_path / "spec", "0" * 64, tmp_path / "out")
    assert signal.alarm(0) == 0
