from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest
import yaml


SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_fcp003b_worker_output.py"
SPEC = importlib.util.spec_from_file_location("validate_fcp003b", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def model(path: Path, marker: bytes = b"same") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("model.pt", marker)
        archive.writestr("args.json", b"{}")
        archive.writestr("metadata.json", b"{}")


def make_output(root: Path, *, nonfinite: bool = False, omit_state: bool = False) -> None:
    identities = sorted(MODULE.EXPECTED_IDENTITIES)
    dump(
        root / "launch_receipt.json",
        {
            "status": "FC_P003B_WORKER_LAUNCH_STAGED",
            "mode": "--probe",
            "git_commit": MODULE.SOURCE_COMMIT,
            "approval_sha256": MODULE.APPROVAL_SHA,
            "dynamic_pair_manifest_sha256": MODULE.DYNAMIC_MANIFEST_SHA,
            "cpu_probe_sha256": MODULE.CPU_PROBE_SHA,
            "fc_p003_order_receipt_sha256": MODULE.BASELINE_ORDER_SHA,
            "real_sampling_receipt_sha256": MODULE.REAL_SAMPLING_SHA,
            "source_receipt_sha256": MODULE.SOURCE_RECEIPT_SHA,
            "official_image_id": MODULE.IMAGE_ID,
            "parent_model_sha256": MODULE.PARENT_MODEL_SHA,
            "parent_state_sha256": MODULE.PARENT_STATE_SHA,
            "dev30_manifest_sha256": MODULE.DATA_MANIFESTS["dev30"],
            "train8_manifest_sha256": MODULE.DATA_MANIFESTS["train8"],
            "train16_manifest_sha256": MODULE.DATA_MANIFESTS["train16"],
            "single_factor": "paired_supervision_content_static16_vs_dynamic8_repeated_twice",
            "paired_dataset_kind": "dynamic8",
            "unique_pair_count": 8,
            "paired_updates_per_epoch": 8,
            "paired_dataset_repetitions": 1,
            "validation_or_frozen_training_accessed": False,
            "ppo_auto_launch": False,
        },
    )
    dump(
        root / "runtime_metadata.json",
        {
            "rollout_steps": 100,
            "initial_checkpoint": "/workspace/parent",
            "paired_stat_loss_weight": 10.0,
            "paired_stat_horizons": [20, 50, 100],
            "gpu_memory_fraction": 0.45,
            "paired_manifest": "/workspace/dynamic_pair_manifest.json",
        },
    )
    dump(
        root / "training_data_sources.json",
        {
            "base_windows": 720,
            "total_windows": 1368,
            "additional_sources": [
                {
                    "root": path,
                    "manifest_sha256": digest,
                    "normalization_sha256": MODULE.NORMALIZATION_SHA,
                }
                for path, digest in MODULE.EXPECTED_MANIFESTS.items()
            ],
        },
    )
    dump(
        root / "training_history.json",
        [
            {
                "epoch": 1,
                "selection_score": float("nan") if nonfinite else 0.5,
                "paired_dataset_kind": "dynamic8",
                "paired_batch_schedule": "interleaved",
                "paired_batch_indices": list(range(8)),
                "paired_identity_passes": [identities],
                "paired_identities": identities,
            }
        ],
    )
    config = {
        "data": {
            "root": "/workspace/base",
            "additional_train_roots": ["/workspace/train8", "/workspace/train16"],
            "paired_dataset_kind": "dynamic8",
            "paired_action_root": "/workspace/train8",
            "paired_zero_root": "/workspace/base",
            "paired_manifest": "/workspace/dynamic_pair_manifest.json",
        },
        "training": {
            "batch_size": 1,
            "rollout_steps": 100,
            "validation_rollout_steps": 100,
            "seed": 20261003,
            "train_stride": 20,
            "additional_train_stride": 2,
            "initial_checkpoint": "/workspace/parent",
            "paired_batch_size": 1,
            "paired_stat_horizons": [20, 50, 100],
            "paired_stat_loss_weight": 10.0,
            "paired_batch_schedule": "interleaved",
            "learning_rate": 1.0e-5,
            "force_channel_weights": [1.0, 1.0, 4.0, 1.0],
            "teacher_forcing_start": 0.0,
            "teacher_forcing_end": 0.0,
            "gpu_memory_fraction": 0.45,
            "validation_stride": 100,
            "epochs": 1,
            "max_train_batches": 8,
            "max_validation_batches": 1,
            "expected_regular_batches": 8,
            "paired_dataset_repetitions": 1,
            "paired_batches_per_epoch": 8,
            "max_paired_eval_batches": 1,
        },
    }
    config["model"] = {
        "in_channels": 6,
        "out_channels": 7,
        "latent_channels": 48,
        "num_fno_layers": 5,
        "num_fno_modes": [32, 32],
        "decoder_layers": 2,
        "decoder_layer_size": 128,
        "padding": 8,
        "coord_features": True,
    }
    (root / "resolved_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (root / "train.log").write_text("done\n", encoding="utf-8")
    model(root / "best/FNO.0.1.mdlus")
    model(root / "checkpoints/FNO.0.1.mdlus")
    (root / "best/checkpoint.0.1.pt").write_bytes(b"state")
    if not omit_state:
        (root / "checkpoints/checkpoint.0.1.pt").parent.mkdir(parents=True, exist_ok=True)
        (root / "checkpoints/checkpoint.0.1.pt").write_bytes(b"state")


def test_output_passes_and_records_exact_checkpoint_pair(tmp_path: Path) -> None:
    make_output(tmp_path)
    result = MODULE.validate_output(tmp_path, "--probe")
    assert result["paired_updates_per_epoch"] == 8
    assert result["checkpoint_epoch"] == 1
    assert "best/FNO.0.1.mdlus" in result["sha256"]
    assert "best/checkpoint.0.1.pt" in result["sha256"]


def test_output_rejects_tampered_launch_or_model_contract(tmp_path: Path) -> None:
    make_output(tmp_path)
    launch = json.loads((tmp_path / "launch_receipt.json").read_text())
    launch["approval_sha256"] = "tampered"
    dump(tmp_path / "launch_receipt.json", launch)
    with pytest.raises(ValueError, match="launch receipt"):
        MODULE.validate_output(tmp_path, "--probe")
    make_output(tmp_path)
    config = yaml.safe_load((tmp_path / "resolved_config.yaml").read_text())
    config["model"]["latent_channels"] = 64
    (tmp_path / "resolved_config.yaml").write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="model contract"):
        MODULE.validate_output(tmp_path, "--probe")


@pytest.mark.parametrize("failure", ["nan", "state", "payload"])
def test_output_fails_closed_for_invalid_scientific_artifacts(tmp_path: Path, failure: str) -> None:
    make_output(tmp_path, nonfinite=failure == "nan", omit_state=failure == "state")
    if failure == "payload":
        model(tmp_path / "checkpoints/FNO.0.1.mdlus", b"different")
    with pytest.raises(ValueError):
        MODULE.validate_output(tmp_path, "--probe")


def test_source_receipt_covers_every_snapshot_file(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "src").mkdir(parents=True)
    (source / "src/a.py").write_text("x=1\n")
    mapping = MODULE.file_map(source)
    dump(
        tmp_path / "receipt.json",
        {
            "status": "FC_P003B_IMMUTABLE_SOURCE_STAGED_V2",
            "git_commit": "abc",
            "source_files": mapping,
        },
    )
    dump(tmp_path / "required.json", mapping)
    MODULE.validate_source(source, tmp_path / "receipt.json", "abc", tmp_path / "required.json")
    (source / "src/a.py").write_text("x=2\n")
    with pytest.raises(ValueError):
        MODULE.validate_source(source, tmp_path / "receipt.json", "abc", tmp_path / "required.json")


def test_sampling_requires_bound_baseline(tmp_path: Path) -> None:
    identities = sorted(MODULE.EXPECTED_IDENTITIES)
    sequence = "sequence"
    real = {
        "status": "FC_P003B_REAL_OFFICIAL_DATALOADER_SAMPLING_PASS",
        "regular_sequence_sha256": sequence,
        "fc_p003_regular_sequence_sha256": sequence,
        "global_torch_rng_sha256": "rng",
        "pair_identity_passes": [identities, identities],
        "paired_batch_indices": MODULE.EXPECTED_POSITIONS,
        "regular_count": 1368,
        "validation_or_frozen_accessed": False,
    }
    baseline = {
        "status": "FC_P003_OFFICIAL_DATALOADER_ORDER_COUNTERFACTUAL_PASS",
        "regular_sequence_sha256": sequence,
        "global_torch_rng_sha256": "rng",
    }
    dump(tmp_path / "baseline.json", baseline)
    real["fc_p003_order_receipt_sha256"] = MODULE.sha256(tmp_path / "baseline.json")
    dump(tmp_path / "real.json", real)
    MODULE.validate_sampling(tmp_path / "real.json", tmp_path / "baseline.json")
    baseline["regular_sequence_sha256"] = "tampered"
    dump(tmp_path / "baseline.json", baseline)
    with pytest.raises(ValueError):
        MODULE.validate_sampling(tmp_path / "real.json", tmp_path / "baseline.json")
