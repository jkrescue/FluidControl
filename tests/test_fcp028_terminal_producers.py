import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from test_run_fcp028_posteval import fixture


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name + "_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = load("audit_fcp028_candidate")
RELOAD = load("verify_fcp028_dual_reload")


def test_terminal_audit_produces_exact_formal_schema(tmp_path, monkeypatch):
    candidate, _, _ = fixture(tmp_path)
    loader = tmp_path / "loader.py"
    loader.write_text(
        "import json\nfrom pathlib import Path\nfrom types import SimpleNamespace\n"
        "def validate_dual_fno_manifest(path, expected_sha256=None):\n"
        " return SimpleNamespace(payload=json.loads(Path(path).read_text()), manifest_sha256=expected_sha256)\n"
    )
    invocation = "8" * 32
    terminal = {
        "LoadState": "loaded",
        "ActiveState": "active",
        "SubState": "exited",
        "Result": "success",
        "ExecMainCode": "1",
        "ExecMainStatus": "0",
        "MainPID": "0",
        "InvocationID": invocation,
    }
    monkeypatch.setattr(
        AUDIT.subprocess,
        "check_output",
        lambda *args, **kwargs: "\n".join(f"{key}={value}" for key, value in terminal.items()),
    )
    result = AUDIT.validate(
        candidate,
        loader,
        AUDIT.sha(loader),
        "fluid-control-fcp028-training-20261006.service",
        invocation,
    )
    assert result["status"] == AUDIT.STATUS
    assert result["flow_trained"] is True
    assert result["aerodynamic_frozen"] is True
    assert result["candidate_sha256"] == AUDIT.candidate_files(candidate)
    assert result["terminal_evidence"] == terminal


def test_cpu_reload_binding_crosslinks_terminal_audit_and_source(tmp_path):
    source = tmp_path / "source"
    required = {
        "src/fluid_control/dual_fno.py",
        "src/fluid_control/calibrated_checkpoint.py",
        "scripts/train_tandem_fno.py",
        "scripts/p026_history_inference.py",
        "scripts/p026_state_history.py",
        "scripts/audit_fcp028_candidate.py",
        "scripts/verify_fcp028_dual_reload.py",
    }
    mapping = {}
    for name in required:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name == "scripts/verify_fcp028_dual_reload.py":
            path.write_bytes((ROOT / name).read_bytes())
        else:
            path.write_text(name)
        mapping[name] = RELOAD.sha(path)
    manifest = tmp_path / "source.json"
    manifest.write_text(json.dumps(mapping))
    config = tmp_path / "config.yaml"
    config.write_text("model: {}\n")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    candidate_result = candidate / "result.json"
    candidate_result.write_text("{}")
    protocol = candidate / "training_protocol.json"
    protocol.write_text("{}")
    (candidate / "dual_model_manifest.json").write_text(
        json.dumps({"config_sha256": RELOAD.sha(config)})
    )
    audit = {
        "status": "FC_P028_CANDIDATE_INTEGRITY_VERIFIED_NOT_ADMISSION",
        "training_protocol_sha256": RELOAD.sha(protocol),
        "dual_manifest_sha256": RELOAD.sha(candidate / "dual_model_manifest.json"),
        "candidate_result_sha256": RELOAD.sha(candidate_result),
        "candidate_sha256": {
            "candidate/result.json": RELOAD.sha(candidate_result),
            "candidate/dual_model_manifest.json": RELOAD.sha(candidate / "dual_model_manifest.json"),
            "candidate/training_protocol.json": RELOAD.sha(protocol),
        },
        "tensor_sha256": {"flow": "4" * 64, "aerodynamic": "5" * 64},
        "role_loader_sha256": mapping["src/fluid_control/dual_fno.py"],
        "scientific_admission": False,
        "ppo_authorized": False,
    }
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit))
    args = type(
        "Args",
        (),
        {
            "source_root": source,
            "runtime_source_manifest": manifest,
            "runtime_source_manifest_sha256": RELOAD.sha(manifest),
            "candidate_audit": audit_path,
            "candidate_audit_sha256": RELOAD.sha(audit_path),
            "config": config,
            "candidate": candidate,
        },
    )()
    result = RELOAD.bindings(args)
    assert result["status"] == RELOAD.STATUS
    assert result["candidate_audit_sha256"] == RELOAD.sha(audit_path)
    assert result["runtime_source_sha256"] == mapping
    assert result["forward_performed"] is False


def test_cpu_reload_explicitly_establishes_parent_precision():
    fake = SimpleNamespace(
        value="highest",
        set_float32_matmul_precision=lambda value: setattr(fake, "value", value),
        get_float32_matmul_precision=lambda: fake.value,
        backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
            cudnn=SimpleNamespace(allow_tf32=True),
        ),
    )
    assert RELOAD.configure_precision(fake)["float32_matmul_precision"] == "high"


def test_terminal_auditor_imports_actual_dataclass_loader_before_contract_rejection(
    tmp_path, monkeypatch
):
    candidate, _, _ = fixture(tmp_path)
    loader = Path(
        "/workspace/fluid_control/src/fluid_control/dual_fno.py"
    )
    invocation = "8" * 32
    terminal = {
        "LoadState": "loaded",
        "ActiveState": "active",
        "SubState": "exited",
        "Result": "success",
        "ExecMainCode": "1",
        "ExecMainStatus": "0",
        "MainPID": "0",
        "InvocationID": invocation,
    }
    monkeypatch.setattr(
        AUDIT.subprocess,
        "check_output",
        lambda *args, **kwargs: "\n".join(f"{key}={value}" for key, value in terminal.items()),
    )
    try:
        AUDIT.validate(candidate, loader, AUDIT.sha(loader), "unit.service", invocation)
    except ValueError as error:
        assert "manifest" in str(error) or "identity" in str(error)
    else:
        raise AssertionError("incomplete synthetic manifest unexpectedly passed actual loader")
