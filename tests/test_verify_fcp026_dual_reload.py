"""Small CPU software fixtures only; no official full-size model load."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import verify_fcp026_dual_reload as verifier


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)


def test_receipt_persist_exclusive(tmp_path):
    path = tmp_path/"result.json"
    verifier.persist(path, {"scientific_admission": False})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        verifier.persist(path, {"scientific_admission": True})
    assert path.read_bytes() == before


def fixture_source(tmp_path):
    mapping = {}
    for name in verifier.REQUIRED:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(Path(verifier.__file__).read_bytes() if name.endswith("verify_fcp026_dual_reload.py") else b"fixture")
        mapping[name] = verifier.sha(path)
    manifest = tmp_path/"source_manifest.json"
    manifest.write_text(json.dumps(mapping))
    return SimpleNamespace(source_root=tmp_path, runtime_source_manifest=manifest,
                           runtime_source_manifest_sha256=verifier.sha(manifest))


def test_external_source_manifest_and_actual_verifier(tmp_path):
    args = fixture_source(tmp_path)
    assert verifier.REQUIRED == set(verifier.verify_sources(args))
    (tmp_path/"scripts/p026_state_history.py").write_text("changed")
    with pytest.raises(ValueError):
        verifier.verify_sources(args)


def test_source_closure_missing_history_rejected(tmp_path):
    args = fixture_source(tmp_path)
    mapping = verifier.read(args.runtime_source_manifest)
    del mapping["scripts/p026_history_inference.py"]
    args.runtime_source_manifest.write_text(json.dumps(mapping))
    args.runtime_source_manifest_sha256 = verifier.sha(args.runtime_source_manifest)
    with pytest.raises(ValueError):
        verifier.verify_sources(args)


def test_explicit_cpu_visibility_required_before_model_loading(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        verifier.execute_cpu(SimpleNamespace(), {})


@pytest.mark.parametrize("mutation", ["none", "cross_arm", "file_changed", "protocol", "steps", "admission"])
def test_audit_candidate_bindings_fail_closed(monkeypatch, tmp_path, mutation):
    import audit_fcp026_candidate as audit
    import fluid_control.dual_fno as dual
    for name in audit.FILES:
        p = tmp_path/"candidate"/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("fixture")
    result = dict(flow_tensor_sha256="a"*64, aerodynamic_terminal_tensor_sha256="b"*64)
    (tmp_path/"candidate/result.json").write_text(json.dumps(result))
    config = tmp_path/"config.yaml"
    config.write_text("model: fixture")
    files = audit.candidate_files(tmp_path)
    checked = dict(status="FC_P026_K1_CANDIDATE_INTEGRITY_VERIFIED_NOT_ADMISSION", history_k=1,
        actual_optimizer_steps=171, actual_training_windows=1368, accumulation_windows=8,
        scientific_admission=False, ppo_authorized=False, candidate_sha256=files,
        training_protocol_sha256=files["candidate/training_protocol.json"],
        dual_manifest_sha256=files["candidate/dual_model_manifest.json"],
        candidate_result_sha256=files["candidate/result.json"],
        tensor_sha256=dict(flow="a"*64, aerodynamic="b"*64))
    if mutation == "cross_arm":
        checked["history_k"] = 4
    elif mutation == "file_changed":
        (tmp_path/"candidate/flow/FNO.0.0.mdlus").write_text("changed")
    elif mutation == "protocol":
        checked["training_protocol_sha256"] = "c"*64
    elif mutation == "steps":
        checked["actual_optimizer_steps"] = 1368
    elif mutation == "admission":
        checked["scientific_admission"] = True
    receipt = tmp_path/"audit.json"
    receipt.write_text(json.dumps(checked))
    source = {"scripts/audit_fcp026_candidate.py": verifier.sha(audit.__file__),
              "src/fluid_control/dual_fno.py": verifier.sha(dual.__file__)}
    monkeypatch.setattr(verifier, "verify_sources", lambda args: source)
    monkeypatch.setattr(dual, "validate_dual_fno_manifest", lambda path: SimpleNamespace(
        manifest_sha256=files["candidate/dual_model_manifest.json"],
        payload=dict(kind="FC_P026_K1_HISTORY_FORCE_FNO", config_sha256=verifier.sha(config))))
    args = SimpleNamespace(root=tmp_path, candidate_audit=receipt,
        candidate_audit_sha256=verifier.sha(receipt), config=config,
        history_k=1, runtime_source_manifest_sha256="d"*64)
    if mutation == "none":
        payload = verifier.bindings(args)
        assert payload["official_dual_reload_verified"] is True  # Expected contract only; not persisted without execute_cpu.
        assert payload["scientific_admission"] is False
    else:
        with pytest.raises(ValueError):
            verifier.bindings(args)
