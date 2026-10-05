"""CPU software contracts for the P026 readiness identity branch."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


STAGE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "p026_readiness_under_test", STAGE / "scripts/audit_candidate_ppo_readiness.py"
)
readiness = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(readiness)
WRAPPER_SPEC = importlib.util.spec_from_file_location(
    "p026_wrapper_under_test", STAGE / "scripts/run_candidate_full40_canonical_ppo.py"
)
wrapper = importlib.util.module_from_spec(WRAPPER_SPEC)
WRAPPER_SPEC.loader.exec_module(wrapper)


def write(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def fixture(tmp_path, monkeypatch, *, k=1, stale=False):
    repo = tmp_path
    training_root = repo / "artifacts/run"
    candidate = training_root / "candidate"
    posteval = training_root / "formal"
    manifest = write(candidate / "dual_model_manifest.json", {"fixture": True})
    model = write(candidate / "aerodynamic/FNO.0.1.mdlus", {"model": True})
    state = write(candidate / "aerodynamic/checkpoint.0.1.pt", {"state": True})
    flow_model = write(candidate / "flow/FNO.0.0.mdlus", {"flow": True})
    flow_state = write(candidate / "flow/checkpoint.0.0.pt", {"flow_state": True})
    write(repo / readiness.FC_P013_CONFIG, {"config": True})
    normalization = write(repo / "normalization.json", {"normalization": True})
    data = write(repo / "train.json", {"train": True})
    lineage = {
        "status": f"FC_P026_K{k}_CANDIDATE_INTEGRITY_VERIFIED_NOT_ADMISSION",
        "history_k": k,
    }
    write(candidate / "result.json", {"input_sha256": {"train": readiness.sha256(data)}})
    approval = {
        "history_k": k,
        "candidate_sha256": {"dual_model_manifest.json": readiness.sha256(manifest)},
        "audit": ({**lineage, "history_k": 4 if k == 1 else 1} if stale else lineage),
    }
    approval_path = write(posteval / "evidence/formal_approval.json", approval)
    receipt_path = write(
        posteval / "receipt.json",
        {
            "history_k": k,
            "formal_approval_sha256": readiness.sha256(approval_path),
            "sha256": {"evidence/formal_approval.json": readiness.sha256(approval_path)},
        },
    )
    runner = repo / "scripts/run_fcp026_posteval.py"
    runner.parent.mkdir(parents=True, exist_ok=True)
    runner.write_text(
        "def validate_terminal_proofs(repo,candidate,approval):\n"
        " return {'independent_terminal_audit': approval['audit']}\n",
        encoding="utf-8",
    )
    kind = f"FC_P026_K{k}_HISTORY_FORCE_FNO"
    identity = SimpleNamespace(
        payload={"kind": kind},
        manifest_sha256=readiness.sha256(manifest),
        flow=SimpleNamespace(
            model=flow_model,
            state=flow_state,
            model_sha256=readiness.sha256(flow_model),
            state_sha256=readiness.sha256(flow_state),
        ),
        aerodynamic=SimpleNamespace(
            directory=model.parent,
            model=model,
            state=state,
            model_sha256=readiness.sha256(model),
            state_sha256=readiness.sha256(state),
        ),
    )
    import fluid_control.dual_fno as dual
    import fluid_control.dual_control_contract as control

    monkeypatch.setattr(dual, "validate_dual_fno_manifest", lambda *a, **kw: identity)
    monkeypatch.setattr(dual, "validate_dual_runtime_files", lambda *a, **kw: None)
    runtime = {"profile": f"p026_k{k}", "history_length": k}
    monkeypatch.setattr(
        control,
        "verify_dual_control_binding",
        lambda **kwargs: {"fno_history_runtime": runtime},
    )
    monkeypatch.setattr(
        control,
        "verify_p026_terminal_chain",
        lambda root, receipt, identity: {
            "independent_terminal_audit": approval["audit"]
        },
    )
    return dict(
        repo_root=repo,
        candidate_root=training_root,
        lineage=lineage,
        receipt_path=receipt_path,
        normalization_path=normalization,
        data_artifacts={"train": data},
    )


@pytest.mark.parametrize("k", [1, 4])
def test_exact_p026_arm_terminal_and_runtime_binding(monkeypatch, tmp_path, k):
    result = readiness.p026_candidate_identity(**fixture(tmp_path, monkeypatch, k=k))
    assert result["kind"] == f"FC_P026_K{k}_HISTORY_FORCE_FNO"
    assert result["fno_history_runtime"] == {"profile": f"p026_k{k}", "history_length": k}
    assert result["checkpoint_relative_directory"] == Path("candidate/aerodynamic")


def test_cross_arm_or_stale_reviewed_terminal_proof_rejected(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="reviewed terminal audit"):
        readiness.p026_candidate_identity(
            **fixture(tmp_path, monkeypatch, k=1, stale=True)
        )


def test_formal_approval_copy_tamper_rejected(monkeypatch, tmp_path):
    values = fixture(tmp_path, monkeypatch, k=1)
    approval = values["receipt_path"].parent / "evidence/formal_approval.json"
    approval.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="approval/arm"):
        readiness.p026_candidate_identity(**values)


def test_launcher_rejects_missing_history_runtime_even_on_both_sides(monkeypatch):
    monkeypatch.setattr(wrapper, "sha256", lambda path: "a" * 64)
    identity = {
        "candidate_kind": "FC_P026_K1_HISTORY_FORCE_FNO",
        "checkpoint_epoch": 1,
        "checkpoint_sha256": "b" * 64,
        "checkpoint_state_sha256": "c" * 64,
        "resolved_config_sha256": "d" * 64,
        "precision_protocol": {},
        "endpoint_evaluation_config_sha256": "a" * 64,
        "dual_control_binding": {"fno_history_runtime": None},
        "dual_manifest_path": "candidate/dual_model_manifest.json",
        "dual_posteval_receipt_path": "formal/receipt.json",
        "fno_history_runtime": None,
    }
    args = SimpleNamespace(
        repo=Path("/repo"),
        data=Path("/data"),
        endpoint_evaluation_config=Path("/config"),
        baselines=Path("/baselines"),
        promotion_receipt=Path("/promotion"),
        validation_report=Path("/evaluation"),
        validation_segments=Path("/segments"),
        predeclaration=Path("/predecl"),
        episode_steps=100,
        timesteps=8192,
        checkpoint_interval=2048,
        seed=20261003,
        gpu_memory_fraction=0.2,
    )
    with pytest.raises(ValueError, match="history runtime"):
        wrapper.command_contract(args, {"candidate_identity": identity})


def test_low_level_dual_binding_source_pins_terminal_proof_runner(monkeypatch, tmp_path):
    import fluid_control.dual_control_contract as control

    candidate = tmp_path / "artifacts/run/candidate"
    manifest = write(candidate / "dual_model_manifest.json", {"manifest": True})
    root = tmp_path / "artifacts/run/formal"
    approval = write(
        root / "evidence/formal_approval.json",
        {"candidate_sha256": {"dual_model_manifest.json": readiness.sha256(manifest)}},
    )
    receipt = {
        "formal_approval_sha256": readiness.sha256(approval),
        "sha256": {"evidence/formal_approval.json": readiness.sha256(approval)},
    }
    runner = tmp_path / "scripts/run_fcp026_posteval.py"
    runner.parent.mkdir(parents=True)
    runner.write_text(
        "def validate_terminal_proofs(repo,candidate,approval):\n"
        " return {'independent_terminal_audit': {'status':'fixture'}}\n",
        encoding="utf-8",
    )
    identity = SimpleNamespace(
        manifest_path=manifest, manifest_sha256=readiness.sha256(manifest)
    )
    monkeypatch.setattr(control, "P026_FORMAL_RUNNER_SHA256", readiness.sha256(runner))
    result = control.verify_p026_terminal_chain(root, receipt, identity)
    assert result["independent_terminal_audit"]["status"] == "fixture"
    runner.write_text(runner.read_text() + "# tamper\n", encoding="utf-8")
    with pytest.raises(ValueError, match="formal runner differs"):
        control.verify_p026_terminal_chain(root, receipt, identity)
