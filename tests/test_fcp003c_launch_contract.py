from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_module(relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolved() -> dict:
    return {
        "data": {
            "root": "/workspace/base",
            "additional_train_roots": ["/workspace/train8", "/workspace/train16"],
            "paired_dataset_kind": "dynamic8",
            "paired_action_root": "/workspace/train8",
            "paired_zero_root": "/workspace/base",
            "paired_manifest": "/workspace/dynamic_pair_manifest.json",
        },
        "model": {"in_channels": 6, "out_channels": 7, "latent_channels": 48, "num_fno_layers": 5, "num_fno_modes": [32, 32], "padding": 8, "coord_features": True},
        "training": {
            "epochs": 2, "batch_size": 1, "rollout_steps": 100, "validation_rollout_steps": 100,
            "seed": 20261003, "train_stride": 20, "additional_train_stride": 2,
            "initial_checkpoint": "/workspace/parent", "paired_batch_size": 1,
            "paired_batches_per_epoch": 16, "max_paired_eval_batches": 8,
            "paired_batch_schedule": "interleaved", "expected_regular_batches": 1368,
            "paired_dataset_repetitions": 2, "paired_objective_kind": "true_state_step_force",
            "paired_step_chunk_size": 10, "force_channel_weights": [1.0, 1.0, 4.0, 1.0],
            "paired_stat_loss_weight": 10.0, "force_loss_weight": 0.2,
            "teacher_forcing_start": 0.0, "teacher_forcing_end": 0.0,
            "learning_rate": 1e-5, "gradient_clip_norm": 1.0,
        },
    }


def test_resolved_contract_accepts_only_true_state_single_factor(tmp_path: Path):
    module = load_module("scripts/audit_fcp003c_candidate.py", "candidate_audit")
    path = tmp_path / "resolved.yaml"
    path.write_text(yaml.safe_dump(resolved(), sort_keys=False))
    result = module.validate_resolved(path)
    assert result["status"] == "FC_P003C_RESOLVED_CONFIG_PASS"
    value = resolved()
    value["training"]["paired_objective_kind"] = "paired_statistics"
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="training contract"):
        module.validate_resolved(path)


def test_resolved_contract_rejects_schedule_or_architecture_change(tmp_path: Path):
    module = load_module("scripts/audit_fcp003c_candidate.py", "candidate_audit_negative")
    for key, value in (("paired_dataset_repetitions", 1), ("seed", 9)):
        config = resolved()
        config["training"][key] = value
        path = tmp_path / f"{key}.yaml"
        path.write_text(yaml.safe_dump(config))
        with pytest.raises(ValueError):
            module.validate_resolved(path)
    config = resolved()
    config["model"]["latent_channels"] = 64
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="architecture"):
        module.validate_resolved(path)


def test_execution_evidence_rejects_tampered_probe_or_approval(tmp_path: Path, monkeypatch):
    module = load_module("scripts/audit_fcp003c_candidate.py", "candidate_execution_evidence")
    evidence = tmp_path / "launch_evidence"
    evidence.mkdir()
    result = evidence / "mixed_probe_result.json"
    result.write_text(json.dumps({"status": "FC_P003C_MIXED_LOSS_TECHNICAL_PROBE_PASS", "optimizer_steps": 1, "candidate_saved": False, "validation_or_frozen_accessed": False}))
    result_sha = sha(result)
    completion = evidence / "mixed_probe_completion_receipt.json"
    completion.write_text(json.dumps({"status": "FC_P003C_MIXED_LOSS_TECHNICAL_PROBE_COMPLETE", "result_sha256": result_sha}))
    approval = evidence / "full_training_approval.json"
    approval.write_text(json.dumps({"status": "FC_P003C_FULL_TRAINING_APPROVED", "gpu_execution_authorized": True}))
    monkeypatch.setattr(module, "MIXED_RESULT", result_sha)
    monkeypatch.setattr(module, "MIXED_COMPLETION", sha(completion))
    launch = {"full_execution_approval_sha256": sha(approval)}
    module.validate_execution_evidence(tmp_path, launch)
    result.write_text("{}")
    with pytest.raises(ValueError, match="mixed-probe evidence"):
        module.validate_execution_evidence(tmp_path, launch)
    result.write_text(json.dumps({"status": "FC_P003C_MIXED_LOSS_TECHNICAL_PROBE_PASS", "optimizer_steps": 1, "candidate_saved": False, "validation_or_frozen_accessed": False}))
    approval.write_text(json.dumps({"status": "FC_P003C_FULL_TRAINING_APPROVED", "gpu_execution_authorized": False}))
    launch["full_execution_approval_sha256"] = sha(approval)
    with pytest.raises(ValueError, match="full approval"):
        module.validate_execution_evidence(tmp_path, launch)


def test_immutable_chain_receipt_rejects_tampered_helper(tmp_path: Path):
    module = load_module("scripts/audit_fcp003c_candidate.py", "candidate_chain_evidence")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    helper = scripts / "audit_fcp003c_candidate.py"
    helper.write_text("reviewed\n")
    commit = "1" * 40
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"status": "FC_P003C_IMMUTABLE_POSTEVAL_CHAIN_STAGED", "git_commit": commit, "git_tree": "2" * 40, "sha256": {"scripts/audit_fcp003c_candidate.py": sha(helper)}}))
    module.validate_immutable_chain(receipt, scripts, commit)
    helper.write_text("tampered\n")
    with pytest.raises(ValueError, match="immutable posteval chain"):
        module.validate_immutable_chain(receipt, scripts, commit)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_complete(tmp_path: Path) -> tuple[Path, Path, str]:
    candidate, out = tmp_path / "candidate", tmp_path / "candidate/posteval_fc_p003c"
    out.mkdir(parents=True)
    checkpoint = "a" * 64
    paths = {
        "validation10": ("validation10/evaluation.json", "validation10/segments.json", "validation10/diagnostic.json", "validation10/endpoint_gate.json"),
        "dynamic6": ("dynamic6/evaluation.json", "dynamic6/segments.json", "dynamic6/diagnostic.json"),
        "force_window": ("force_window/result.json", "development_gate.json"),
    }
    for names in paths.values():
        for name in names:
            path = out / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n")
    (out / "lineage.json").write_text(json.dumps({"status": "FC_P003C_CANDIDATE_LINEAGE_PASS", "checkpoint_sha256": checkpoint}))
    development = {"status": "DYNAMIC_FNO_DEVELOPMENT_ADMISSION_FAIL", "ppo_authorized": False, "frozen_test_accessed": False}
    (out / "development_gate.json").write_text(json.dumps(development))
    receipts = out / "step_receipts"
    receipts.mkdir()
    for step, names in paths.items():
        actual = {str(out / name): sha(out / name) for name in names}
        (receipts / f"{step}.json").write_text(json.dumps({"status": "FC_P003C_POSTEVAL_STEP_COMPLETE", "step": step, "checkpoint_sha256": checkpoint, "sha256": actual}))
    files = {str(path.relative_to(out)): sha(path) for path in sorted(out.rglob("*")) if path.is_file() and path.name not in {"receipt.json", "outer.log"}}
    receipt = {"status": "FC_P003C_POSTEVAL_COMPLETE", "checkpoint_sha256": checkpoint, "ppo_auto_launched": False, "frozen_test_accessed": False, "sha256": files}
    (out / "receipt.json").write_text(json.dumps(receipt))
    return candidate, out, checkpoint


def test_complete_receipt_accepts_scientific_fail_but_not_missing_file(tmp_path: Path):
    module = load_module("scripts/validate_fcp003c_posteval.py", "posteval_validator")
    candidate, out, checkpoint = make_complete(tmp_path)
    module.validate_complete(candidate, out, checkpoint)
    (out / "dynamic6/segments.json").unlink()
    with pytest.raises(ValueError):
        module.validate_complete(candidate, out, checkpoint)


def test_step_receipt_rejects_outside_or_wrong_file(tmp_path: Path):
    module = load_module("scripts/validate_fcp003c_posteval.py", "posteval_validator_path")
    _, out, checkpoint = make_complete(tmp_path)
    receipt = out / "step_receipts/dynamic6.json"
    value = json.loads(receipt.read_text())
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    value["sha256"] = {str(outside): sha(outside)}
    receipt.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="hashes"):
        module.validate_step_receipt(receipt, out, "dynamic6", checkpoint)


def test_launchers_default_to_dry_run_and_keep_protocol_paths_distinct():
    training = (ROOT / "scripts/run_fcp003c_training_spark.sh").read_text()
    posteval = (ROOT / "scripts/run_fcp003c_posteval_spark.sh").read_text()
    assert 'mode="${1:---dry-run}"' in training
    assert 'mode="${1:---dry-run}"' in posteval
    assert "FCP003C_MIXED_PROBE_RECEIPT" in training
    assert "FCP003C_MIXED_PROBE_RESULT" in training
    assert "FC_P003C_MIXED_LOSS_TECHNICAL_PROBE_COMPLETE" in training
    assert "FC_P003C_MIXED_LOSS_TECHNICAL_PROBE_PASS" in training
    assert "FCP003C_FULL_APPROVAL" in training
    assert '-v "$dev30:/workspace/devdata:ro"' in posteval
    assert '-v "$full40/validation:/workspace/devdata/validation:ro"' in posteval
    assert "--data /workspace/devdata" in posteval
    assert "/workspace/full40" not in posteval
    assert "--segment-stride 25 --evaluation-batch-size 4" in posteval
    assert "--segment-stride 1 --evaluation-batch-size 8" in posteval
