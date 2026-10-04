from __future__ import annotations

import importlib.util
import json
import copy
from pathlib import Path

import pytest
import torch


SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts/diagnose_fcp003c_gradient_balance.py"
)
LAUNCHER = (
    Path(__file__).resolve().parents[1]
    / "scripts/run_fcp003c_gradient_diagnostic_spark.sh"
)
VALIDATOR = (
    Path(__file__).resolve().parents[1]
    / "scripts/validate_fcp003c_gradient_diagnostic.py"
)


def module():
    spec = importlib.util.spec_from_file_location("gradient_diagnostic", SCRIPT)
    assert spec and spec.loader
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def validator_module():
    spec = importlib.util.spec_from_file_location("gradient_validator", VALIDATOR)
    assert spec and spec.loader
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_positions_are_exact_full_epoch_interleaving():
    value = module()
    assert value.POSITIONS == (
        0,
        91,
        182,
        273,
        364,
        455,
        546,
        637,
        729,
        820,
        911,
        1002,
        1093,
        1184,
        1275,
        1367,
    )


def test_gradient_vector_math_and_direction_metrics():
    value = module()
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    field = (parameter.square()).sum()
    force = parameter.sum()
    g_field = value.gradients(field, [parameter], retain_graph=True)
    g_force = value.gradients(0.2 * force, [parameter])
    regular = value.add(g_field, g_force)
    assert torch.allclose(g_field[0], torch.tensor([2.0, -4.0]))
    assert torch.allclose(g_force[0], torch.tensor([0.2, 0.2]))
    assert torch.allclose(regular[0], torch.tensor([2.2, -3.8]))
    stats = value.vector_stats(regular)
    assert stats["element_count"] == 2
    assert stats["finite_count"] == 2
    assert stats["nonzero_count"] == 2
    comparison = value.compare(g_field, g_force)
    assert comparison["cosine_defined"] is True
    assert comparison["sign_conflict_fraction"] == pytest.approx(0.5)
    assert -1 <= comparison["cosine"] <= 1
    undefined = value.compare([torch.zeros(2)], [torch.zeros(2)])
    assert undefined["cosine_defined"] is False
    assert undefined["cosine"] is None
    assert undefined["sign_conflict_fraction"] is None
    assert undefined["left_over_right_l2"] is None


def test_gradient_decomposition_is_checked_and_emits_numeric_evidence(capsys):
    value = module()
    expected = [torch.tensor([1.0, 2.0])]
    residual = value.assert_gradient_close(
        [torch.tensor([1.0, 2.0])], expected, label="test"
    )
    assert residual["l2"] == 0
    with pytest.raises(RuntimeError, match="gradient decomposition differs"):
        value.assert_gradient_close([torch.tensor([2.0, 2.0])], expected, label="test")
    evidence = json.loads(capsys.readouterr().out)
    assert evidence["status"] == "GRADIENT_DECOMPOSITION_MISMATCH"
    assert evidence["label"] == "test"
    assert evidence["residual_l2"] > 0
    assert evidence["reference_l2"] > 0
    assert evidence["actual_l2"] > 0
    assert evidence["relative_residual_l2"] > evidence["relative_tolerance"]


def test_gradient_aggregation_rejects_shape_mismatch_and_summarizes_even_rows():
    value = module()
    with pytest.raises(ValueError, match="counts differ"):
        value.add([torch.zeros(1)], [torch.zeros(1), torch.zeros(1)])
    rows = [{"ratio": number} for number in (4.0, 1.0, 3.0, 2.0)]
    assert value.summarize_rows(rows, "ratio") == {
        "min": 1.0,
        "median": 2.5,
        "max": 4.0,
    }
    with pytest.raises(FloatingPointError, match="nonfinite"):
        value.vector_stats([torch.tensor([float("nan")])])


def test_model_state_hash_covers_parameters_and_buffers_without_mutation():
    value = module()
    model = torch.nn.BatchNorm1d(2)
    before = value.model_state_sha256(model)
    assert value.model_state_sha256(model) == before
    with torch.no_grad():
        model.running_mean[0] = 1
    assert value.model_state_sha256(model) != before


def test_script_has_no_optimizer_or_checkpoint_write_path():
    source = SCRIPT.read_text()
    assert "torch.optim" not in source
    assert ".step()" not in source
    assert "save_checkpoint" not in source
    assert '"validation"' not in source
    assert '"frozen_test"' not in source
    assert '"optimizer_created_or_stepped": False' in source
    assert '"model_state_sha256_before"' in source
    assert '"model_state_sha256_after"' in source
    assert "regular_force_decomposition_gradient_residual" in source
    assert 'sampling["pair_identity_passes"]' in source
    assert 'pair_identity.get("pair_id") != expected_pair_ids[position_index]' in source
    assert "choices=(1, 16)" in source
    assert "FC_P003C_TRAIN_ONLY_GRADIENT_DIAGNOSTIC_DEBUG_ONLY" in source


def test_launcher_is_bounded_train_only_and_execution_gated():
    source = LAUNCHER.read_text()
    assert "timeout --signal=TERM --kill-after=30s 20m" in source
    assert "--min-free-gib 20" in source
    assert "--allocator-fraction 0.35" in source
    assert "EXECUTE_REVIEWED_FCP003C_TRAIN_GRADIENT_DIAGNOSTIC" in source
    assert 'config="$source_root/config.yaml"' in source
    assert 'config_sha="07e55fd11' in source
    assert "/home/USER/.venvs/physicsnemo-94dbdf82-cu13/bin/python" in source
    assert "/home/USER" not in source
    assert '"$base/train:/workspace/base/train:ro"' in source
    assert '"$train8/train:/workspace/train8/train:ro"' in source
    assert '"$train16/train:/workspace/train16/train:ro"' in source
    assert ":/workspace/validation" not in source
    assert ":/workspace/frozen" not in source
    assert "torch.optim" not in source


def test_strict_validator_binds_real_files_and_rejects_tamper(tmp_path):
    value = validator_module()
    roots = {}
    for name in ("base", "train8", "train16"):
        root = tmp_path / name
        (root / "splits").mkdir(parents=True)
        (root / "manifest.json").write_text(f"{name}-manifest")
        (root / "normalization.json").write_text("normalization")
        (root / "splits/train.json").write_text("base-split")
        roots[name] = root
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "FNO.0.2.mdlus").write_text("model")
    (parent / "checkpoint.0.2.pt").write_text("state")
    config = tmp_path / "resolved.yaml"
    config.write_text("resolved")
    pair_manifest = tmp_path / "pairs.json"
    pair_manifest.write_text("pairs")
    source_manifest = tmp_path / "source.sha256"
    source_manifest.write_text("source")
    pair_ids = [f"pair-{index}" for index in range(16)]
    sampling = tmp_path / "sampling.json"
    pair_indices = list(range(16))
    sampling.write_text(
        json.dumps(
            {
                "pair_identity_passes": [pair_ids[:8], pair_ids[8:]],
                "pair_pass_indices": [pair_indices[:8], pair_indices[8:]],
            }
        )
    )
    inputs = {
        "base_manifest": value.sha256(roots["base"] / "manifest.json"),
        "base_train_split_manifest": value.sha256(roots["base"] / "splits/train.json"),
        "base_normalization": value.sha256(roots["base"] / "normalization.json"),
        "train8_manifest": value.sha256(roots["train8"] / "manifest.json"),
        "train8_normalization": value.sha256(roots["train8"] / "normalization.json"),
        "train16_manifest": value.sha256(roots["train16"] / "manifest.json"),
        "train16_normalization": value.sha256(roots["train16"] / "normalization.json"),
        "pair_manifest": value.sha256(pair_manifest),
        "source_manifest": value.sha256(source_manifest),
    }
    stats = {
        "l2": 1.0,
        "linf": 1.0,
        "finite_count": 2,
        "nonzero_count": 2,
        "element_count": 2,
    }
    comparison = {
        "cosine_defined": True,
        "cosine": 0.5,
        "sign_conflict_defined": True,
        "sign_conflict_fraction": 0.25,
        "ratio_defined": True,
        "left_over_right_l2": 1.0,
    }
    channels = {name: 1.0 for name in ("front_cd", "front_cl", "rear_cd", "rear_cl")}
    channel_gradients = {name: dict(stats) for name in channels}
    rows = []
    for index, position in enumerate(value.POSITIONS):
        row = {
            "position": position,
            "regular_dataset_index": 100 + index,
            "regular_metadata": {"split": "train", "rollout_steps": 100},
            "pair_dataset_index": pair_indices[index],
            "pair_metadata": {
                "pair_id": pair_ids[index],
                "split": "train",
                "start": 0,
                "horizon": 100,
            },
            "losses": {
                name: 1.0
                for name in (
                    "regular_total",
                    "regular_field",
                    "regular_force_raw",
                    "regular_force_0p2",
                    "old_pair_raw",
                    "old_pair_lambda10",
                    "true_state_pair_raw",
                )
            },
            "gradient": {
                name: dict(stats)
                for name in (
                    "field",
                    "regular_force_0p2",
                    "regular",
                    "old_pair_lambda10",
                    "true_state_pair_lambda10",
                )
            },
            "regular_total_gradient_residual": dict(stats),
            "regular_total_gradient_residual_relative": 1e-6,
            "regular_force_channel_loss_0p2": dict(channels),
            "regular_force_channel_gradient": dict(channel_gradients),
            "regular_force_decomposition_gradient_residual": dict(stats),
            "regular_force_decomposition_gradient_residual_relative": 1e-6,
            "old_statistic_gradient": {
                name: dict(stats)
                for name in ("mean_total_cd", "mean_rear_cl", "rear_cl_rms")
            },
            "old_statistic_gradient_residual": dict(stats),
            "old_statistic_gradient_residual_relative": 1e-6,
            "true_state_channel_loss": dict(channels),
            "true_state_channel_weighted_lambda10_loss": dict(channels),
            "true_state_channel_gradient": dict(channel_gradients),
            "true_state_chunk_gradient_residual_l2_max": 1e-6,
            "true_state_chunk_gradient_residual_relative_max": 1e-6,
            "old_vs_regular": dict(comparison),
            "new_vs_regular": dict(comparison),
            "new_vs_regular_force": dict(comparison),
            "old_vs_new": dict(comparison),
            "mixed_old": {**stats, "clip_scale_at_1": 0.5},
            "mixed_new": {**stats, "clip_scale_at_1": 0.5},
            "new_over_regular_l2": 1.0,
        }
        rows.append(row)
    result = {
        "status": "FC_P003C_TRAIN_ONLY_GRADIENT_DIAGNOSTIC_COMPLETE",
        "scientific_admission": False,
        "optimizer_created_or_stepped": False,
        "candidate_saved": False,
        "validation_or_frozen_accessed": False,
        "image_id": "image",
        "config_sha256": value.sha256(config),
        "sampling_receipt_sha256": value.sha256(sampling),
        "parent_model_sha256": value.sha256(parent / "FNO.0.2.mdlus"),
        "parent_state_sha256": value.sha256(parent / "checkpoint.0.2.pt"),
        "model_state_sha256_before": "same",
        "model_state_sha256_after": "same",
        "input_sha256": inputs,
        "positions": list(value.POSITIONS),
        "rows": rows,
        "mean_gradient": {
            name: dict(stats)
            for name in (
                "field",
                "regular_force_0p2",
                "regular",
                "old_pair_lambda10",
                "true_state_pair_lambda10",
            )
        },
        "mean_gradient_directions": {
            name: dict(comparison)
            for name in ("old_vs_regular", "new_vs_regular", "old_vs_new")
        },
        "summaries": {"new_over_regular_l2": {"min": 0.5, "median": 1.0, "max": 1.5}},
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result))
    progress_path = tmp_path / "progress.jsonl"
    progress_path.write_text(
        "".join(
            json.dumps(
                {
                    "status": "INCOMPLETE_PROGRESS",
                    "completed_positions": index + 1,
                    "position": position,
                    "finite": True,
                    "elapsed_seconds": float(index + 1),
                    "regular_gradient_l2": 1.0,
                    "old_pair_gradient_l2": 1.0,
                    "true_state_pair_gradient_l2": 1.0,
                }
            )
            + "\n"
            for index, position in enumerate(value.POSITIONS)
        )
    )
    arguments = {
        "result_path": result_path,
        "progress_path": progress_path,
        "config": config,
        "sampling_receipt": sampling,
        "parent": parent,
        "source_manifest": source_manifest,
        "base": roots["base"],
        "train8": roots["train8"],
        "train16": roots["train16"],
        "pair_manifest": pair_manifest,
        "image_id": "image",
    }
    bindings = value.validate(**arguments)
    assert bindings["result_sha256"] == value.sha256(result_path)
    validation_receipt = {
        "status": "FC_P003C_TRAIN_GRADIENT_DIAGNOSTIC_VALIDATED",
        "sha256": bindings,
    }
    value.validate_receipt_bindings(
        validation_receipt,
        result_path=result_path,
        progress_path=progress_path,
        config=config,
        sampling_receipt=sampling,
        source_manifest=source_manifest,
    )
    result_path.write_text("{}")
    with pytest.raises(ValueError, match="stale or incomplete"):
        value.validate_receipt_bindings(
            validation_receipt,
            result_path=result_path,
            progress_path=progress_path,
            config=config,
            sampling_receipt=sampling,
            source_manifest=source_manifest,
        )
    result_path.write_text(json.dumps(result))
    numeric_mutations = [
        (lambda payload: payload["rows"][0].pop("gradient"), "numeric field set"),
        (
            lambda payload: payload["rows"][0].pop("regular_total_gradient_residual"),
            "numeric field set",
        ),
        (
            lambda payload: payload["rows"][0].__setitem__("pair_dataset_index", 99),
            "identity contract",
        ),
        (
            lambda payload: payload["rows"][0]["gradient"]["regular"].__setitem__(
                "finite_count", 1
            ),
            "gradient-stat values",
        ),
        (
            lambda payload: payload["rows"][0]["old_vs_regular"].__setitem__(
                "cosine", None
            ),
            "defined metric",
        ),
    ]
    for mutation, message in numeric_mutations:
        invalid = copy.deepcopy(result)
        mutation(invalid)
        result_path.write_text(json.dumps(invalid))
        with pytest.raises(ValueError, match=message):
            value.validate(**arguments)
    result_path.write_text(json.dumps(result))
    result["config_sha256"] = "0" * 64
    result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="top-level"):
        value.validate(**arguments)
    result["config_sha256"] = value.sha256(config)
    result_path.write_text(json.dumps(result))
    pair_manifest.write_text("tampered-pairs")
    with pytest.raises(ValueError, match="input SHA"):
        value.validate(**arguments)
