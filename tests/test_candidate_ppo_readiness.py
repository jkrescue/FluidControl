from __future__ import annotations

import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_candidate_ppo_readiness.py"
SPEC = importlib.util.spec_from_file_location("candidate_ppo_readiness", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path: Path) -> dict:
    candidate = tmp_path / "artifacts/candidate"
    posteval = candidate / "posteval"
    model = candidate / "best/FNO.0.2.mdlus"
    state = candidate / "best/checkpoint.0.2.pt"
    model_config = {
        "in_channels": 6, "out_channels": 7, "latent_channels": 48,
        "num_fno_layers": 5, "num_fno_modes": [32, 32],
        "decoder_layers": 2, "decoder_layer_size": 128,
        "padding": 8, "coord_features": True,
    }
    model.parent.mkdir(parents=True)
    with zipfile.ZipFile(model, "w") as archive:
        for name, content in (
            ("model.pt", b"model"),
            ("args.json", json.dumps({"__args__": {**model_config, "dimension": 2}})),
            ("metadata.json", b"{}"),
        ):
            archive.writestr(name, content)
    state.write_bytes(b"state")
    config = candidate / "resolved_config.yaml"
    config.write_text(yaml.safe_dump({
        "model": model_config,
        "training": {"rollout_steps": 100, "validation_rollout_steps": 100},
    }))
    launch, completion = candidate / "launch_receipt.json", candidate / "completion_receipt.json"
    write_json(launch, {"status": "fixture"})
    write_json(completion, {"status": "fixture"})
    normalization = tmp_path / "normalization.json"
    manifest = tmp_path / "validation_manifest.json"
    dev30, train8 = tmp_path / "dev30.json", tmp_path / "train8.json"
    write_json(manifest, {
        "profile": MODULE.PROFILE, "max_abs_omega": 0.75,
        "max_delta_omega": 0.1,
        "trajectory_counts": {"train": 20, "validation": 10, "frozen_test": 10},
    })
    for path, content in (
        (normalization, b"norm"), (dev30, b"dev30"), (train8, b"train8"),
    ):
        path.write_bytes(content)
    lineage = posteval / "lineage.json"
    write_json(lineage, {
        "status": "FIXTURE_CANDIDATE_LINEAGE_PASS",
        "candidate_kind": "fixture_h100",
        "candidate_root": "artifacts/candidate",
        "checkpoint_epoch": 2,
        "checkpoint_sha256": digest(model),
        "checkpoint_state_sha256": digest(state),
        "checkpoint_generation_payload_sha256": MODULE.archive_payload(model),
        "resolved_config_sha256": digest(config),
        "launch_receipt_sha256": digest(launch),
        "completion_receipt_sha256": digest(completion),
        "data_lineage": {
            "dev30": digest(dev30), "train8": digest(train8),
            "normalization": digest(normalization),
        },
        "frozen_test_opened_or_enumerated": False,
        "ppo_auto_launch": False,
        "training_performed": False,
    })
    producer, evidence = tmp_path / "producer.py", tmp_path / "evidence.json"
    producer.write_text("# fixture\n")
    evidence.write_text("{}\n")
    common = {
        "profile": MODULE.PROFILE,
        "checkpoint_sha256": digest(model),
        "normalization_sha256": digest(normalization),
        "data_manifest_sha256": digest(manifest),
        "frozen_test_accessed": False,
    }
    endpoint = posteval / "validation10/endpoint_gate.json"
    write_json(endpoint, {
        **common, "status": MODULE.ENDPOINT_STATUS, "split": "validation",
        "joint_terminal_readiness": True,
        "h100_force_gate": {"beats_persistence": True},
        "physicsnemo_image_id": MODULE.IMAGE_ID,
        "model_config_sha256": digest(config),
    })
    evidence_common = {
        **common, "validation_phases": ["b01", "b05"],
        "producer_script": str(producer),
        "producer_script_sha256": digest(producer),
        "evidence_path": str(evidence),
        "evidence_sha256": digest(evidence),
    }
    window = tmp_path / "window_gate.json"
    write_json(window, {
        **evidence_common, "status": MODULE.WINDOW_STATUS,
        "causal_window_seconds": 6.15,
        "total_drag_window_fidelity_pass": True,
        "rear_cl_fluctuation_window_fidelity_pass": True,
        "rear_cl_mean_bias_window_fidelity_pass": True,
    })
    dynamic = tmp_path / "dynamic_gate.json"
    write_json(dynamic, {
        **evidence_common, "status": MODULE.DYNAMIC_STATUS,
        "max_abs_omega": 0.75, "max_delta_omega": 0.1,
        "minimum_horizon_steps": 100, "dynamic_action_validation_pass": True,
    })
    force = posteval / "force_window/result.json"
    force.parent.mkdir(parents=True)
    force.write_text("{}\n")
    development = posteval / "development_gate.json"
    write_json(development, {
        "status": MODULE.DEVELOPMENT_STATUS,
        "checkpoint_sha256": digest(model),
        "frozen_test_accessed": False,
        "ppo_authorized": False,
        "window_gate": {"status": "PASS"},
        "dynamic_action_gate": {"status": "PASS"},
        "source_force_window_sha256": digest(force),
    })
    receipt = posteval / "receipt.json"
    write_json(receipt, {
        "status": "FIXTURE_POSTEVAL_COMPLETE",
        "checkpoint_sha256": digest(model),
        "frozen_test_accessed": False,
        "ppo_auto_launched": False,
        "sha256": {
            "lineage.json": digest(lineage),
            "validation10/endpoint_gate.json": digest(endpoint),
            "development_gate.json": digest(development),
            "force_window/result.json": digest(force),
        },
    })
    return {
        "repo_root": tmp_path, "candidate_root": candidate, "lineage_path": lineage,
        "posteval_receipt_path": receipt, "endpoint_gate_path": endpoint,
        "window_gate_path": window, "dynamic_gate_path": dynamic,
        "development_gate_path": development,
        "validation_manifest_path": manifest, "normalization_path": normalization,
        "data_artifacts": {"dev30": dev30, "train8": train8},
    }


def test_complete_candidate_is_ready_but_does_not_authorize_or_run_ppo(tmp_path: Path) -> None:
    result = MODULE.audit(**fixture(tmp_path))
    assert result["status"] == "CANDIDATE_PPO_CPU_DRY_RUN_READY"
    assert result["blockers"] == []
    assert result["training_executed"] is False
    assert result["policy_created"] is False
    assert result["ppo_execution_authorized"] is False
    assert result["frozen_test_directory_enumerated_or_opened"] is False


@pytest.mark.parametrize("key", ["window_gate_path", "dynamic_gate_path", "development_gate_path"])
def test_each_gate_is_independently_required(tmp_path: Path, key: str) -> None:
    values = fixture(tmp_path)
    values[key] = tmp_path / "missing.json"
    result = MODULE.audit(**values)
    assert result["status"].endswith("BLOCKED")
    assert any(row["kind"] == "MISSING" for row in result["blockers"])


def test_development_gate_cannot_substitute_for_canonical_window(tmp_path: Path) -> None:
    values = fixture(tmp_path)
    values["window_gate_path"] = values["development_gate_path"]
    result = MODULE.audit(**values)
    assert result["status"].endswith("BLOCKED")
    assert result["blockers"][0]["kind"] == "SCIENTIFIC_FAIL"


def test_failed_development_gate_is_scientific_failure(tmp_path: Path) -> None:
    values = fixture(tmp_path)
    path = values["development_gate_path"]
    gate = json.loads(path.read_text())
    gate["status"] = "DYNAMIC_FNO_DEVELOPMENT_ADMISSION_FAIL"
    write_json(path, gate)
    receipt = json.loads(values["posteval_receipt_path"].read_text())
    receipt["sha256"]["development_gate.json"] = digest(path)
    write_json(values["posteval_receipt_path"], receipt)
    result = MODULE.audit(**values)
    assert result["status"].endswith("BLOCKED")
    assert result["blockers"][0]["kind"] == "SCIENTIFIC_FAIL"


def test_relative_gate_evidence_is_resolved_from_repo(tmp_path: Path) -> None:
    values = fixture(tmp_path)
    for key in ("window_gate_path", "dynamic_gate_path"):
        path = values[key]
        gate = json.loads(path.read_text())
        gate["producer_script"] = "producer.py"
        gate["evidence_path"] = "evidence.json"
        write_json(path, gate)
    assert MODULE.audit(**values)["status"].endswith("READY")


@pytest.mark.parametrize("fault", ["missing", "tampered", "outside"])
def test_complete_receipt_validates_every_confined_hash(tmp_path: Path, fault: str) -> None:
    values = fixture(tmp_path)
    extra = values["posteval_receipt_path"].parent / "extra.json"
    extra.write_text("{}\n")
    receipt = json.loads(values["posteval_receipt_path"].read_text())
    if fault == "outside":
        receipt["sha256"]["../outside.json"] = "0" * 64
    else:
        receipt["sha256"]["extra.json"] = digest(extra)
        if fault == "missing":
            extra.unlink()
        else:
            extra.write_text('{"tampered":true}\n')
    write_json(values["posteval_receipt_path"], receipt)
    result = MODULE.audit(**values)
    assert result["status"].endswith("BLOCKED")
    assert result["blockers"][0]["kind"] == "SCHEMA_ERROR"


def test_candidate_root_requires_exact_repo_relative_identity(tmp_path: Path) -> None:
    values = fixture(tmp_path)
    lineage = json.loads(values["lineage_path"].read_text())
    lineage["candidate_root"] = "other/location/candidate"
    write_json(values["lineage_path"], lineage)
    receipt = json.loads(values["posteval_receipt_path"].read_text())
    receipt["sha256"]["lineage.json"] = digest(values["lineage_path"])
    write_json(values["posteval_receipt_path"], receipt)
    result = MODULE.audit(**values)
    assert result["blockers"][0]["kind"] == "SCHEMA_ERROR"


@pytest.mark.parametrize("fault", ["model", "state", "config", "data", "normalization"])
def test_candidate_identity_tampering_is_blocked(tmp_path: Path, fault: str) -> None:
    values = fixture(tmp_path)
    targets = {
        "model": values["candidate_root"] / "best/FNO.0.2.mdlus",
        "state": values["candidate_root"] / "best/checkpoint.0.2.pt",
        "config": values["candidate_root"] / "resolved_config.yaml",
        "data": values["data_artifacts"]["dev30"],
        "normalization": values["normalization_path"],
    }
    with targets[fault].open("ab") as stream:
        stream.write(b"tamper")
    result = MODULE.audit(**values)
    assert result["status"].endswith("BLOCKED")
    assert result["blockers"][0]["kind"] == "SCHEMA_ERROR"


@pytest.mark.parametrize(
    "gate_key", ["endpoint_gate_path", "window_gate_path", "dynamic_gate_path"]
)
def test_scientific_gate_failure_is_not_schema_pass(tmp_path: Path, gate_key: str) -> None:
    values = fixture(tmp_path)
    path = values[gate_key]
    gate = json.loads(path.read_text())
    gate["status"] = "FAIL"
    write_json(path, gate)
    if gate_key == "endpoint_gate_path":
        receipt = json.loads(values["posteval_receipt_path"].read_text())
        receipt["sha256"]["validation10/endpoint_gate.json"] = digest(path)
        write_json(values["posteval_receipt_path"], receipt)
    result = MODULE.audit(**values)
    assert result["status"].endswith("BLOCKED")
    assert result["blockers"][0]["kind"] == "SCIENTIFIC_FAIL"


def test_wrong_image_and_frozen_scope_are_blocked(tmp_path: Path) -> None:
    values = fixture(tmp_path)
    result = MODULE.audit(**values, official_image_id="sha256:wrong")
    assert result["blockers"][0]["kind"] == "SCHEMA_ERROR"
    values = fixture(tmp_path / "second")
    path = values["dynamic_gate_path"]
    gate = json.loads(path.read_text())
    gate["frozen_test_accessed"] = True
    write_json(path, gate)
    result = MODULE.audit(**values)
    assert result["blockers"][0]["kind"] == "SCHEMA_ERROR"


def test_parse_data_artifacts_rejects_duplicates_and_malformed() -> None:
    with pytest.raises(ValueError):
        MODULE.parse_data_artifacts(["dev30=a", "dev30=b"])
    with pytest.raises(ValueError):
        MODULE.parse_data_artifacts(["missing-separator"])
