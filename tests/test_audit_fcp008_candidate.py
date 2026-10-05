from __future__ import annotations

import importlib.util
import io
import json
import zipfile
from collections import OrderedDict
from pathlib import Path

import pytest
import torch

MODULE_PATH = Path(__file__).parents[1] / "scripts/audit_fcp008_candidate.py"
SPEC = importlib.util.spec_from_file_location("audit_fcp008_candidate", MODULE_PATH)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def model_archive(path: Path, state: OrderedDict, metadata: dict) -> None:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("model.pt", buffer.getvalue())
        archive.writestr("args.json", "{}\n")
        archive.writestr("metadata.json", json.dumps(metadata))


def state_dict(force_delta: float = 0.0, other_delta: float = 0.0) -> OrderedDict:
    return OrderedDict(
        [
            ("decoder_net.layers.0.linear.weight", torch.ones(2, 2) + other_delta),
            ("decoder_net.final_layer.linear.weight", torch.arange(7 * 128).reshape(7, 128).float()),
            ("decoder_net.final_layer.linear.bias", torch.arange(7).float()),
        ]
    ) | OrderedDict()


def changed_state(other_delta: float = 0.0) -> OrderedDict:
    value = state_dict(other_delta=other_delta)
    value["decoder_net.final_layer.linear.weight"] = value[
        "decoder_net.final_layer.linear.weight"
    ].clone()
    value["decoder_net.final_layer.linear.bias"] = value[
        "decoder_net.final_layer.linear.bias"
    ].clone()
    value["decoder_net.final_layer.linear.weight"][3:] += 0.25
    value["decoder_net.final_layer.linear.bias"][3:] += 0.5
    return value


def build_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, str]:
    repo = tmp_path / "repo"
    candidate = repo / "artifacts" / audit.CANDIDATE_NAME
    static_files = {
        "APPROVAL_CONTRACT_SHA256": repo / "docs/FC_P008_APPROVAL_20261005.md",
        "BASE_MANIFEST": repo / "data/curated/tandem_cylinders_matched_start_full40_dev30_v1/manifest.json",
        "BASE_SPLIT": repo / "data/curated/tandem_cylinders_matched_start_full40_dev30_v1/splits/train.json",
        "TRAIN8": repo / "data/curated/tandem_cylinders_dynamic_train8_v1/manifest.json",
        "TRAIN16": repo / "data/curated/tandem_cylinders_directppo_train16_v1/manifest.json",
        "NORM": repo / "data/curated/tandem_cylinders_matched_start_full40_dev30_v1/normalization.json",
        "CONFIG": repo / "artifacts/tandem_fno_true_state_paired_step_lambda10_20261005/resolved_config.yaml",
        "MAPPING": repo / "artifacts/fcp003c_full_train_source_phase_mapping_20261005/mapping.json",
        "PARENT_STATE": repo / "artifacts/tandem_fno_true_state_paired_step_lambda10_20261005/best/checkpoint.0.2.pt",
    }
    for index, (constant, path) in enumerate(static_files.items()):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"static-{index}".encode())
        monkeypatch.setattr(audit, constant, audit.sha256(path))
    parent_model = (
        repo
        / "artifacts/tandem_fno_true_state_paired_step_lambda10_20261005/best/FNO.0.2.mdlus"
    )
    model_archive(parent_model, state_dict(), {"epoch": 2})
    monkeypatch.setattr(audit, "PARENT_MODEL", audit.sha256(parent_model))

    source = candidate / "source_snapshot"
    for name in audit.REQUIRED_SOURCE_FILES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"source:{name}\n", encoding="utf-8")
    implementation = source / "scripts/build_fcp008_force_readout_candidate.py"
    implementation_sha = audit.sha256(implementation)
    manifest = candidate / "source_snapshot.sha256"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "".join(
            f"{audit.sha256(path)}  {path.relative_to(source).as_posix()}\n"
            for path in sorted(source.rglob("*"))
            if path.is_file()
        ),
        encoding="utf-8",
    )

    changed = [
        "decoder_net.final_layer.linear.weight",
        "decoder_net.final_layer.linear.bias",
    ]
    approval = {
        "status": audit.APPROVAL_STATUS,
        "implementation_sha256": implementation_sha,
        "formal_evaluation_authorized": False,
        "validation_accessed": False,
        "frozen_test_accessed": False,
        "ppo_executed": False,
    }
    approval_path = candidate / "launch_evidence/fc_p008_execution_approval.json"
    write_json(approval_path, approval)
    approval_sha = audit.sha256(approval_path)
    launch = {
        "status": audit.LAUNCH_STATUS,
        "execution_approval_sha256": approval_sha,
        "implementation_sha256": implementation_sha,
        "runtime_image_id": audit.IMAGE_ID,
        "resolved_config_sha256": audit.CONFIG,
        "normalization_sha256": audit.NORM,
        "parent_model_sha256": audit.PARENT_MODEL,
        "parent_training_state_sha256": audit.PARENT_STATE,
        "source_phase_mapping_sha256": audit.MAPPING,
        "base_manifest_sha256": audit.BASE_MANIFEST,
        "base_train_split_sha256": audit.BASE_SPLIT,
        "train8_manifest_sha256": audit.TRAIN8,
        "train16_manifest_sha256": audit.TRAIN16,
        "source_snapshot_manifest_sha256": audit.sha256(manifest),
        "source_commit": "1" * 40,
        "validation_accessed": False,
        "frozen_test_accessed": False,
        "ppo_executed": False,
    }
    write_json(candidate / "launch_receipt.json", launch)
    build = candidate / "candidate_build"
    model = build / "candidate/FNO.0.0.mdlus"
    metadata = {
        "status": "FC_P008_TRAIN_ONLY_FORCE_ROW_CANDIDATE",
        "parent_model_sha256": audit.PARENT_MODEL,
        "selected_alpha": 0.01,
        "changed_parameters": changed,
        "validation_accessed": False,
        "frozen_test_accessed": False,
        "ppo_executed": False,
    }
    model_archive(model, changed_state(), metadata)
    state = build / "candidate/checkpoint.0.0.pt"
    torch.save({"metadata": {"status": "candidate-epoch0"}}, state)
    inventory = {
        "status": "FC_P008_TRAIN_ONLY_SOURCE_INVENTORY_PASS",
        "trajectory_count": 44,
        "endpoint_count": 19648,
        "family_endpoint_counts": audit.FAMILY_ENDPOINTS,
        "validation_accessed": False,
        "frozen_test_accessed": False,
    }
    write_json(build / "source_inventory.json", inventory)
    (build / "train_features.npz").write_bytes(b"feature-cache")
    parent_tensor_sha = audit.model_tensor_sha256(audit.model_state(parent_model))
    candidate_tensor_sha = audit.model_tensor_sha256(audit.model_state(model))
    result = {
        "status": audit.RESULT_STATUS,
        "source_inventory": inventory,
        "source_phase_mapping_sha256": audit.MAPPING,
        "approval_contract_commit": audit.APPROVAL_CONTRACT_COMMIT,
        "approval_contract_sha256": audit.APPROVAL_CONTRACT_SHA256,
        "execution_approval_sha256": approval_sha,
        "family_exposures": audit.FAMILY_EXPOSURES,
        "family_endpoint_counts": audit.FAMILY_ENDPOINTS,
        "precision_protocol": audit.PRECISION,
        "parent_model_sha256": audit.PARENT_MODEL,
        "parent_training_state_sha256": audit.PARENT_STATE,
        "parent_checkpoint_epoch": 2,
        "calibration_generation": 1,
        "candidate_checkpoint_epoch": 0,
        "official_training_state_epoch_field_present": False,
        "fresh_official_load_checkpoint_return_epoch": 0,
        "fresh_official_reload_tensor_sha256": candidate_tensor_sha,
        "calibration_fit_performed": True,
        "no_optimizer_training": True,
        "parent_tensor_sha256": parent_tensor_sha,
        "candidate_tensor_sha256": candidate_tensor_sha,
        "implementation_sha256": implementation_sha,
        "optimizer_steps": 0,
        "architecture_changed": False,
        "validation_accessed": False,
        "frozen_test_accessed": False,
        "ppo_executed": False,
        "formal_evaluation_authorized": False,
        "state_rows_0_3_byte_identical": True,
        "all_other_tensors_byte_identical": True,
        "selected_alpha": 0.01,
        "changed_parameter_names": changed,
        "parent_native_pointwise_wiring_max_abs": 1e-6,
        "candidate_native_pointwise_wiring_max_abs": 1e-6,
        "state_output_sha256_by_trajectory": {str(index): "a" * 64 for index in range(44)},
        "candidate_model_sha256": audit.sha256(model),
        "candidate_state_sha256": audit.sha256(state),
        "candidate_model_archive_member_sha256": audit.archive_payload(model),
        "candidate_training_state_metadata": {"status": "candidate-epoch0"},
        "input_sha256": {
            "base_manifest": audit.BASE_MANIFEST,
            "base_train_split": audit.BASE_SPLIT,
            "train8_manifest": audit.TRAIN8,
            "train16_manifest": audit.TRAIN16,
            "normalization": audit.NORM,
            "resolved_config": audit.CONFIG,
            "source_phase_mapping": audit.MAPPING,
            "parent_model": audit.PARENT_MODEL,
            "parent_training_state": audit.PARENT_STATE,
            "execution_approval": approval_sha,
            "implementation": implementation_sha,
        },
        "full_train_native_metrics_normalized": {"mae": [0.1] * 4},
    }
    write_json(build / "result.json", result)
    required = [
        "candidate_build/result.json",
        "candidate_build/source_inventory.json",
        "candidate_build/train_features.npz",
        "candidate_build/candidate/FNO.0.0.mdlus",
        "candidate_build/candidate/checkpoint.0.0.pt",
        "launch_receipt.json",
        "source_snapshot.sha256",
        "run.log",
    ]
    (candidate / "run.log").write_text("complete\n", encoding="utf-8")
    write_json(
        candidate / "completion_receipt.json",
        {
            "status": audit.COMPLETION_STATUS,
            "sha256": {name: audit.sha256(candidate / name) for name in required},
        },
    )
    return repo, candidate, approval_sha


def test_real_lineage_is_calibration_not_fake_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, candidate, approval_sha = build_candidate(tmp_path, monkeypatch)
    result = audit.validate_candidate(repo, candidate, approval_sha)
    assert result["checkpoint_epoch"] == 0
    assert result["parent_checkpoint_epoch"] == 2
    assert result["calibration_generation"] == 1
    assert result["calibration_fit_performed"] is True
    assert result["optimizer_training_performed"] is False
    assert result["training_performed"] is False


def test_lineage_rejects_nonforce_tensor_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, candidate, approval_sha = build_candidate(tmp_path, monkeypatch)
    model = candidate / "candidate_build/candidate/FNO.0.0.mdlus"
    with zipfile.ZipFile(model) as archive:
        metadata = json.loads(archive.read("metadata.json"))
    changed = changed_state(other_delta=1.0)
    model_archive(model, changed, metadata)
    result_path = candidate / "candidate_build/result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["candidate_model_sha256"] = audit.sha256(model)
    result["candidate_model_archive_member_sha256"] = audit.archive_payload(model)
    result["candidate_tensor_sha256"] = audit.model_tensor_sha256(audit.model_state(model))
    result["fresh_official_reload_tensor_sha256"] = result["candidate_tensor_sha256"]
    write_json(result_path, result)
    completion = json.loads((candidate / "completion_receipt.json").read_text())
    completion["sha256"]["candidate_build/candidate/FNO.0.0.mdlus"] = audit.sha256(model)
    completion["sha256"]["candidate_build/result.json"] = audit.sha256(result_path)
    write_json(candidate / "completion_receipt.json", completion)
    with pytest.raises(ValueError, match="non-force tensor changed"):
        audit.validate_candidate(repo, candidate, approval_sha)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("precision_protocol", {"float32_matmul_precision": "highest"}),
        ("parent_checkpoint_epoch", 0),
        ("formal_evaluation_authorized", True),
    ],
)
def test_lineage_rejects_result_scope_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    repo, candidate, approval_sha = build_candidate(tmp_path, monkeypatch)
    result_path = candidate / "candidate_build/result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result[field] = value
    write_json(result_path, result)
    completion = json.loads((candidate / "completion_receipt.json").read_text())
    completion["sha256"]["candidate_build/result.json"] = audit.sha256(result_path)
    write_json(candidate / "completion_receipt.json", completion)
    with pytest.raises(ValueError):
        audit.validate_candidate(repo, candidate, approval_sha)
