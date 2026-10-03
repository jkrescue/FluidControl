"""Focused tests for dynamic-candidate lineage and development evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
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


def make_train16_candidate(tmp_path, monkeypatch):
    module, repo, old = make_candidate(tmp_path, monkeypatch)
    candidate = repo / "artifacts/tandem_fno_control_train16_h100_20261004"
    old.rename(candidate)
    train16 = repo / "data/curated/tandem_cylinders_directppo_train16_v1"
    train16.mkdir(parents=True)
    (train16 / "normalization.json").write_text("normalization", encoding="utf-8")
    manifest = {
        "status": "DIRECTPPO_TRAIN16_TRAIN_ONLY_CURATED",
        "profile": "directppo_train16_v1",
        "trajectory_counts": {"train": 16, "validation": 0, "frozen_test": 0},
        "frames_per_trajectory": 129,
        "normalization_sha256": module.NORMALIZATION_SHA,
        "validation_or_frozen_accessed": False,
    }
    write_json(train16 / "manifest.json", manifest)
    monkeypatch.setattr(module, "TRAIN16_MANIFEST_SHA", digest(train16 / "manifest.json"))
    config = {
        "training": {
            "rollout_steps": 100,
            "validation_rollout_steps": 100,
            "epochs": 2,
            "batch_size": 2,
            "learning_rate": 1.0e-5,
            "train_stride": 20,
            "additional_train_stride": 2,
            "initial_checkpoint": "/workspace/parent",
            "teacher_forcing_start": 0.0,
            "teacher_forcing_end": 0.0,
            "seed": 20261003,
        },
        "data": {
            "root": "/workspace/base",
            "additional_train_roots": ["/workspace/train8", "/workspace/train16"],
            "force_indices": [0, 1, 2, 3],
        },
    }
    (candidate / "resolved_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    write_json(candidate / "training_history.json", [
        {
            "epoch": epoch,
            "teacher_forcing_ratio": 0.0,
            "train_rollout_steps": 100,
            "validation_rollout_steps": 100,
            "selection_score": score,
        }
        for epoch, score in ((1, 2.0), (2, 1.0))
    ])
    shutil.rmtree(candidate / "best")
    shutil.rmtree(candidate / "checkpoints")
    for directory_name in ("best", "checkpoints"):
        directory = candidate / directory_name
        directory.mkdir()
        (directory / "FNO.0.2.mdlus").write_bytes(b"trained-model")
        (directory / "checkpoint.0.2.pt").write_bytes(b"trained-state")
    shutil.rmtree(candidate / "immutable_parent")
    parent = candidate / "immutable_parent"
    parent.mkdir()
    (parent / "FNO.0.2.mdlus").write_bytes(b"parent-model")
    (parent / "checkpoint.0.2.pt").write_bytes(b"parent-state")
    parent_hashes = {
        name: digest(parent / name)
        for name in ("FNO.0.2.mdlus", "checkpoint.0.2.pt")
    }
    write_json(candidate / "parent_receipt.json", {
        "status": "IMMUTABLE_PARENT_COPIED",
        "sha256": parent_hashes,
    })
    write_json(candidate / "training_data_sources.json", {
        "base_root": "/workspace/base",
        "base_windows": 720,
        "total_windows": 1368,
        "additional_sources": [
            {
                "root": "/workspace/train8",
                "windows": 408,
                "stride": 2,
                "manifest_sha256": module.TRAIN8_MANIFEST_SHA,
                "normalization_sha256": module.NORMALIZATION_SHA,
                "release_status": "DYNAMIC_TRAIN8_TRAIN_ONLY_CURATED",
                "trajectory_count": 8,
            },
            {
                "root": "/workspace/train16",
                "windows": 240,
                "stride": 2,
                "manifest_sha256": module.TRAIN16_MANIFEST_SHA,
                "normalization_sha256": module.NORMALIZATION_SHA,
                "release_status": "DIRECTPPO_TRAIN16_TRAIN_ONLY_CURATED",
                "trajectory_count": 16,
            },
        ],
    })
    required = (
        "scripts/train_tandem_fno.py",
        "scripts/train_tandem_fno_rollout.py",
        "scripts/probe_directppo_train16_datapipe.py",
        "scripts/run_fno_control_train16_spark.sh",
        "scripts/evaluate_tandem_fno.py",
        "src/fluid_control/augmented_datapipe.py",
        "src/fluid_control/tandem_datapipe.py",
        "conf/tandem_fno_control_train16_h100.yaml",
    )
    for relative in required:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "scripts", "src", "conf"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"],
        cwd=repo,
        check=True,
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=repo, text=True).strip()
    snapshot = candidate / "source_snapshot"
    snapshot.mkdir()
    snapshot_hashes = {}
    for relative in required:
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / relative, target)
        snapshot_hashes[relative] = digest(target)
    probe = {
        "status": "DIRECTPPO_TRAIN16_OFFICIAL_DATAPIPE_PROBE_PASS",
        "training_executed": False,
        "validation_or_frozen_hdf_opened": False,
        "rollout_steps": 100,
        "base_windows": 720,
        "total_windows": 1368,
        "additional_sources": [
            {
                "root": "/workspace/train8", "windows": 408, "stride": 2,
                "manifest_sha256": module.TRAIN8_MANIFEST_SHA,
                "normalization_sha256": module.NORMALIZATION_SHA,
                "trajectory_count": 8,
                "release_status": "DYNAMIC_TRAIN8_TRAIN_ONLY_CURATED",
            },
            {
                "root": "/workspace/train16", "windows": 240, "stride": 2,
                "manifest_sha256": module.TRAIN16_MANIFEST_SHA,
                "normalization_sha256": module.NORMALIZATION_SHA,
                "trajectory_count": 16,
                "release_status": "DIRECTPPO_TRAIN16_TRAIN_ONLY_CURATED",
            },
        ],
        "samples": [
            {
                "metadata": {
                    "dataset_index": index, "split": "train", "rollout_steps": 100
                },
                "shapes": {
                    "state": [1, 3, 128, 256],
                    "target_state": [1, 100, 3, 128, 256],
                    "target_force": [1, 100, 4],
                    "omega": [1, 101, 1],
                    "mask": [1, 1, 128, 256],
                },
                "max_abs_omega": 0.5,
                "max_delta_omega": 0.1,
                "action_slew_representation_tolerance": (
                    2.0e-6 if index == 2 else 2.0e-5
                ),
            }
            for index in (0, 1, 2, 2)
        ],
        "script_sha256": snapshot_hashes["scripts/probe_directppo_train16_datapipe.py"],
        "implementation_sha256": {
            relative: snapshot_hashes[relative]
            for relative in (
                "src/fluid_control/augmented_datapipe.py",
                "src/fluid_control/tandem_datapipe.py",
            )
        },
    }
    probe_path = repo / module.TRAIN16_PROBE
    write_json(probe_path, probe)
    write_json(candidate / "source_receipt.json", {
        "status": "CONTROL_TRAIN16_H100_STAGED_NOT_EXECUTED",
        "candidate_kind": "control_train16_h100",
        "source_snapshot_commit": commit,
        "source_snapshot_tree": tree,
        "source_snapshot_sha256": snapshot_hashes,
        "physicsnemo_image_id": module.PHYSICSNEMO_IMAGE_ID,
        "datapipe_probe_sha256": digest(probe_path),
        "data": {
            "dev30": {
                "manifest_sha256": module.DEV30_MANIFEST_SHA,
                "normalization_sha256": module.NORMALIZATION_SHA,
            },
            "train8": {
                "manifest_sha256": module.TRAIN8_MANIFEST_SHA,
                "normalization_sha256": module.NORMALIZATION_SHA,
                "file_count": 8,
            },
            "train16": {
                "manifest_sha256": module.TRAIN16_MANIFEST_SHA,
                "normalization_sha256": module.NORMALIZATION_SHA,
                "file_count": 16,
            },
            "frozen_test_transferred_or_opened": False,
        },
        "parent": {
            "candidate_kind": "dynamic_train8_h100",
            "checkpoint_epoch": 2,
            "sha256": parent_hashes,
            "optimizer_loaded": False,
            "new_optimizer": "AdamW",
        },
    })
    return module, repo, candidate


def test_train16_lineage_binds_three_sources_parent_snapshot_and_probe(
    tmp_path, monkeypatch
):
    module, repo, candidate = make_train16_candidate(tmp_path, monkeypatch)
    result = module.build(repo, candidate)
    assert result["candidate_kind"] == "control_train16_h100"
    assert result["data_lineage"]["train16_windows"] == 240
    assert result["data_lineage"]["train16_manifest_sha256"] == module.TRAIN16_MANIFEST_SHA
    assert result["training_implementation_sha256_at_launch"] is not None
    probe_path = repo / module.TRAIN16_PROBE
    probe = json.loads(probe_path.read_text())
    probe["samples"][0]["max_delta_omega"] = 0.2
    write_json(probe_path, probe)
    source = json.loads((candidate / "source_receipt.json").read_text())
    source["datapipe_probe_sha256"] = digest(probe_path)
    write_json(candidate / "source_receipt.json", source)
    with pytest.raises(ValueError, match="DataPipe sample contract"):
        module.build(repo, candidate)
    probe["samples"][0]["max_delta_omega"] = 0.1
    write_json(probe_path, probe)
    source = json.loads((candidate / "source_receipt.json").read_text())
    source["datapipe_probe_sha256"] = digest(probe_path)
    write_json(candidate / "source_receipt.json", source)
    assert module.build(repo, candidate)["candidate_kind"] == "control_train16_h100"
    source = json.loads((candidate / "source_receipt.json").read_text())
    source["parent"]["checkpoint_epoch"] = 1
    write_json(candidate / "source_receipt.json", source)
    with pytest.raises(ValueError, match="launch source/data/parent"):
        module.build(repo, candidate)


def test_control_train16_runner_is_fail_closed_and_training_only():
    source = (ROOT / "scripts/run_fno_control_train16_spark.sh").read_text()
    assert "--dry-run|--probe|--execute" in source
    assert "tandem_fno_control_train16_h100_20261004" in source
    assert "ef95ff96582983680800710679258a6705eb2294fab6a43dfa38163a606ed0c8" in source
    assert "63e88160faef1db50140c6aee859ef4de50b84d68e0f675af83f3bc5fb35c63a" in source
    assert "--allocator-fraction 0.45" in source
    assert "--min-free-gib 20" in source
    assert "DIRECTPPO_TRAIN16_OFFICIAL_DATAPIPE_PROBE_PASS" in source
    assert "DYNAMIC_FNO_DEVELOPMENT_ADMISSION_FAIL" in source
    assert "frozen_test" not in "\n".join(
        line for line in source.splitlines() if "--mount" in line
    )
    assert "train_full40_hydrogym_ppo" not in source


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
