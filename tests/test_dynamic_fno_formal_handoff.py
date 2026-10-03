"""Focused tests for dynamic-candidate lineage and development evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    lineage = load_module("audit_dynamic_fno_candidate_lineage")
    repo = tmp_path / "repo"
    candidate = repo / "artifacts/tandem_fno_dynamic_train8_h50_fixture"
    for relative, content in (
        ("data/curated/tandem_cylinders_matched_start_full40_dev30_v1/manifest.json", "dev"),
        ("data/curated/tandem_cylinders_dynamic_train8_v1/manifest.json", "train8"),
        ("data/curated/tandem_cylinders_matched_start_full40_v1/manifest.json", "formal"),
    ):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    for relative in (
        "data/curated/tandem_cylinders_matched_start_full40_dev30_v1/normalization.json",
        "data/curated/tandem_cylinders_dynamic_train8_v1/normalization.json",
        "data/curated/tandem_cylinders_matched_start_full40_v1/normalization.json",
    ):
        path = repo / relative
        path.write_text("normalization", encoding="utf-8")
    monkeypatch.setattr(lineage, "DEV30_MANIFEST_SHA", digest(repo / "data/curated/tandem_cylinders_matched_start_full40_dev30_v1/manifest.json"))
    monkeypatch.setattr(lineage, "TRAIN8_MANIFEST_SHA", digest(repo / "data/curated/tandem_cylinders_dynamic_train8_v1/manifest.json"))
    monkeypatch.setattr(lineage, "FULL40_MANIFEST_SHA", digest(repo / "data/curated/tandem_cylinders_matched_start_full40_v1/manifest.json"))
    monkeypatch.setattr(lineage, "NORMALIZATION_SHA", digest(repo / "data/curated/tandem_cylinders_matched_start_full40_dev30_v1/normalization.json"))

    config = {
        "training": {
            "rollout_steps": 50,
            "validation_rollout_steps": 100,
            "epochs": 4,
            "teacher_forcing_start": 0.0,
            "teacher_forcing_end": 0.0,
            "seed": 20261003,
        },
        "data": {
            "root": "/workspace/base",
            "additional_train_roots": ["/workspace/train8"],
            "force_indices": [0, 1, 2, 3],
        },
    }
    candidate.mkdir(parents=True)
    (candidate / "resolved_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    history = [
        {
            "epoch": epoch,
            "teacher_forcing_ratio": 0.0,
            "train_rollout_steps": 50,
            "validation_rollout_steps": 100,
            "selection_score": score,
        }
        for epoch, score in enumerate((4.0, 3.0, 2.0, 1.0), 1)
    ]
    write_json(candidate / "training_history.json", history)
    write_json(
        candidate / "training_data_sources.json",
        {
            "base_root": "/workspace/base",
            "base_windows": 760,
            "total_windows": 1368,
            "additional_sources": [{
                "root": "/workspace/train8",
                "windows": 608,
                "stride": 2,
                "manifest_sha256": lineage.TRAIN8_MANIFEST_SHA,
                "normalization_sha256": lineage.NORMALIZATION_SHA,
            }],
        },
    )
    for subdirectory in ("best", "checkpoints"):
        directory = candidate / subdirectory
        directory.mkdir()
        (directory / "FNO.0.4.mdlus").write_bytes(b"model")
        (directory / "checkpoint.0.4.pt").write_bytes(b"state")
    parent = candidate / "immutable_parent"
    parent.mkdir()
    (parent / "parent.mdlus").write_bytes(b"parent-model")
    (parent / "parent.pt").write_bytes(b"parent-state")
    write_json(candidate / "parent_receipt.json", {
        "status": "IMMUTABLE_PARENT_COPIED",
        "sha256": {name: digest(parent / name) for name in ("parent.mdlus", "parent.pt")},
    })
    for relative in (
        "scripts/train_tandem_fno.py",
        "scripts/train_tandem_fno_rollout.py",
        "scripts/evaluate_tandem_fno.py",
        "src/fluid_control/augmented_datapipe.py",
        "conf/tandem_fno_dynamic_train8_h50.yaml",
    ):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    return lineage, repo, candidate


def force_document(module, checkpoint: str, *, corrupt_case: str | None = None):
    cases = []
    times = [0.1 * index for index in range(101)]
    for case_index, name in enumerate(module.CASES):
        profile = name.rsplit("_", 1)[-1]
        shift = {"minus": -0.05, "zero": 0.0, "plus": 0.05}[profile]
        truth = []
        prediction = []
        for step in range(101):
            rear_cl = 0.2 * __import__("math").sin(step / 10)
            row = [1.0 + shift, 0.0, 0.5 + shift, rear_cl]
            truth.append(row)
            predicted_row = list(row)
            if name == corrupt_case and step >= 39:
                predicted_row[3] += 0.5
            prediction.append(predicted_row)
        omega = [max(-0.75, min(0.75, shift * 10))] * 101
        cases.append({
            "case": name,
            "horizon_steps": 100,
            "times": times,
            "omega_endpoints": omega,
            "true_forces": truth,
            "predicted_forces": prediction,
        })
    return {
        "status": "SAMPLED_FORCE_WINDOW_DIAGNOSTIC_COMPLETE",
        "model_sha256": checkpoint,
        "manifest_sha256": module.MANIFEST_SHA,
        "normalization_sha256": module.NORMALIZATION_SHA,
        "force_channels": module.FORCE_CHANNELS,
        "cases": cases,
        "frozen_test_accessed": False,
        "ppo_authorized": False,
    }


def test_h50_lineage_binds_unique_best_and_detects_mismatch(tmp_path, monkeypatch):
    module, repo, candidate = make_candidate(tmp_path, monkeypatch)
    result = module.build(repo, candidate)
    assert result["status"] == "DYNAMIC_FNO_CANDIDATE_LINEAGE_PASS"
    assert result["selected_epoch"] == 4
    assert result["source_snapshot_recorded_at_launch"] is False
    assert result["training_implementation_sha256_at_launch"] is None
    assert "src/fluid_control/augmented_datapipe.py" in (
        result["current_recovery_and_validation_implementation_sha256"]
    )
    (candidate / "best/FNO.0.4.mdlus").write_bytes(b"different")
    with pytest.raises(ValueError, match="best files differ"):
        module.build(repo, candidate)


def test_development_gate_passes_exact_stepwise_evidence(tmp_path):
    module = load_module("audit_dynamic_fno_development_gates")
    checkpoint = "a" * 64
    source = tmp_path / "forces.json"
    write_json(source, force_document(module, checkpoint))
    result = module.audit(source, checkpoint)
    assert result["status"] == "DYNAMIC_FNO_DEVELOPMENT_ADMISSION_PASS"
    assert result["window_gate"]["status"] == "PASS"
    assert all(row["truth"]["sample_count"] == 62 for row in result["window_gate"]["branches"])
    assert all(row["truth"]["sample_span_D_over_U"] == pytest.approx(6.1) for row in result["window_gate"]["branches"])
    assert result["dynamic_action_gate"]["metrics"]["non_tie_sign_pairs"] == 4
    assert result["ppo_authorized"] is False


def test_development_gate_keeps_failing_branch_quantitative_gap(tmp_path):
    module = load_module("audit_dynamic_fno_development_gates")
    checkpoint = "b" * 64
    source = tmp_path / "forces.json"
    failed = module.CASES[0]
    write_json(source, force_document(module, checkpoint, corrupt_case=failed))
    result = module.audit(source, checkpoint)
    assert result["status"] == "DYNAMIC_FNO_DEVELOPMENT_ADMISSION_FAIL"
    row = next(item for item in result["window_gate"]["branches"] if item["case"] == failed)
    assert row["metric_pass"]["rear_cl_mean"] is False
    assert row["absolute_errors"]["rear_cl_mean"] > row["maximum_errors"]["rear_cl_mean"]
    assert row["margin_to_maximum"]["rear_cl_mean"] < 0.0


def test_development_gate_rejects_wrong_data_identity(tmp_path):
    module = load_module("audit_dynamic_fno_development_gates")
    checkpoint = "c" * 64
    source = tmp_path / "forces.json"
    document = force_document(module, checkpoint)
    document["manifest_sha256"] = "0" * 64
    write_json(source, document)
    with pytest.raises(ValueError, match="identity differs"):
        module.audit(source, checkpoint)


def audit_modified_document(tmp_path, module, mutator):
    checkpoint = "9" * 64
    document = force_document(module, checkpoint)
    mutator(document)
    source = tmp_path / "modified.json"
    write_json(source, document)
    return module.audit(source, checkpoint)


@pytest.mark.parametrize(
    ("channel", "modifier", "expected_metric"),
    (
        (0, lambda step: 0.1, "total_cd"),
        (3, lambda step: 0.5 if step % 2 else -0.5, "rear_cl_fluctuation_rms"),
        (3, lambda step: 0.5, "rear_cl_mean"),
    ),
)
def test_each_window_metric_can_fail_independently(
    tmp_path, channel, modifier, expected_metric
):
    module = load_module("audit_dynamic_fno_development_gates")
    failed_case = module.CASES[0]

    def mutate(document):
        row = next(item for item in document["cases"] if item["case"] == failed_case)
        for step in range(39, 101):
            row["predicted_forces"][step][channel] += modifier(step)

    result = audit_modified_document(tmp_path, module, mutate)
    row = next(item for item in result["window_gate"]["branches"] if item["case"] == failed_case)
    assert row["metric_pass"][expected_metric] is False
    assert row["margin_to_maximum"][expected_metric] < 0.0


def test_endpoint_delta_sign_and_order_failures_are_quantified(tmp_path):
    module = load_module("audit_dynamic_fno_development_gates")

    def mutate(document):
        row = next(item for item in document["cases"] if item["case"].endswith("b01_minus"))
        row["predicted_forces"][-1][0] += 0.3

    result = audit_modified_document(tmp_path, module, mutate)
    gate = result["dynamic_action_gate"]
    assert gate["metric_pass"]["zero_relative_delta_cd_mae"] is False
    assert gate["metric_pass"]["non_tie_sign_accuracy"] is False
    assert gate["metric_pass"]["non_tie_cross_action_ordering_accuracy"] is False
    assert gate["quantitative_margins"]["zero_relative_endpoint_delta_cd_mae_to_maximum"] < 0.0


@pytest.mark.parametrize(
    ("mutator", "message"),
    (
        (
            lambda document: document["cases"][0]["omega_endpoints"].__setitem__(0, 0.76),
            "action magnitude differs",
        ),
        (
            lambda document: document["cases"][0]["omega_endpoints"].__setitem__(1, 0.75),
            "action slew differs",
        ),
        (
            lambda document: document["cases"][0]["times"].__setitem__(2, 0.21),
            "time grid differs",
        ),
        (
            lambda document: document["cases"][0]["predicted_forces"][50].__setitem__(0, math.nan),
            "non-finite force",
        ),
    ),
)
def test_development_gate_rejects_invalid_stepwise_contract(tmp_path, mutator, message):
    module = load_module("audit_dynamic_fno_development_gates")
    checkpoint = "8" * 64
    document = force_document(module, checkpoint)
    mutator(document)
    source = tmp_path / "invalid.json"
    write_json(source, document)
    with pytest.raises(ValueError, match=message):
        module.audit(source, checkpoint)


def test_development_gate_rejects_duplicate_case(tmp_path):
    module = load_module("audit_dynamic_fno_development_gates")
    checkpoint = "7" * 64
    document = force_document(module, checkpoint)
    document["cases"].append(dict(document["cases"][0]))
    source = tmp_path / "duplicate.json"
    write_json(source, document)
    with pytest.raises(ValueError, match="exactly dynamic6"):
        module.audit(source, checkpoint)


def test_handoff_requires_endpoint_and_development_evidence(tmp_path, monkeypatch):
    module = load_module("audit_dynamic_fno_formal_handoff")
    checkpoint = "d" * 64
    recomputed = {
        "status": "DYNAMIC_FNO_CANDIDATE_LINEAGE_PASS",
        "candidate_kind": "dynamic_train8_h100",
        "checkpoint_sha256": checkpoint,
        "data_lineage": {
            "formal_full40_manifest_sha256": "e" * 64,
            "normalization_sha256": "f" * 64,
        },
    }

    class Lineage:
        @staticmethod
        def build(repo, candidate):
            return recomputed

    monkeypatch.setattr(module, "module", lambda path, name: Lineage)
    lineage = tmp_path / "lineage.json"
    endpoint = tmp_path / "endpoint.json"
    development = tmp_path / "development.json"
    write_json(lineage, recomputed)
    write_json(endpoint, {
        "status": "FULL40_VALIDATION_SURROGATE_READINESS_PASS",
        "joint_terminal_readiness": True,
        "checkpoint_sha256": checkpoint,
        "data_manifest_sha256": "e" * 64,
        "normalization_sha256": "f" * 64,
        "frozen_test_accessed": False,
    })
    blocked = module.audit(tmp_path, tmp_path / "candidate", lineage, endpoint)
    assert blocked["candidate_specific_ppo_preflight_status"] == "BLOCKED"
    assert blocked["ppo_authorized"] is False
    write_json(development, {
        "status": "DYNAMIC_FNO_DEVELOPMENT_ADMISSION_PASS",
        "checkpoint_sha256": checkpoint,
        "window_gate": {"status": "PASS"},
        "dynamic_action_gate": {"status": "PASS"},
        "frozen_test_accessed": False,
        "ppo_authorized": False,
    })
    ready = module.audit(
        tmp_path, tmp_path / "candidate", lineage, endpoint, development
    )
    assert ready["candidate_specific_ppo_preflight_status"] == (
        "READY_FOR_SEPARATE_REVIEWED_PPO_LAUNCHER_IMPLEMENTATION"
    )
    assert ready["surrogate_ppo_episode_steps_required"] == 100
    assert ready["ppo_authorized"] is False
