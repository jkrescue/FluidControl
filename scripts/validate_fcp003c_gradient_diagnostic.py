#!/usr/bin/env python3
"""Strict CPU validator for the train-only gradient diagnostic result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

POSITIONS = tuple(index * 1367 // 15 for index in range(16))
CHANNELS = {"front_cd", "front_cl", "rear_cd", "rear_cl"}
GRADIENTS = {
    "field",
    "regular_force_0p2",
    "regular",
    "old_pair_lambda10",
    "true_state_pair_lambda10",
}
STATS = {"l2", "linf", "finite_count", "nonzero_count", "element_count"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _finite(item) for key, item in value.items()
        )
    return False


def _gradient_stats(value: Any, label: str) -> None:
    if not isinstance(value, dict) or set(value) != STATS:
        raise ValueError(f"{label} gradient-stat keys differ")
    counts = [
        value[name] for name in ("finite_count", "nonzero_count", "element_count")
    ]
    if (
        any(isinstance(item, bool) or not isinstance(item, int) for item in counts)
        or value["element_count"] < 1
        or value["finite_count"] != value["element_count"]
        or not 0 <= value["nonzero_count"] <= value["element_count"]
        or value["l2"] < 0
        or value["linf"] < 0
    ):
        raise ValueError(f"{label} gradient-stat values differ")


def _comparison(value: Any, label: str) -> None:
    expected = {
        "cosine_defined",
        "cosine",
        "sign_conflict_defined",
        "sign_conflict_fraction",
        "ratio_defined",
        "left_over_right_l2",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} comparison keys differ")
    contracts = (
        ("cosine_defined", "cosine", -1.000001, 1.000001),
        ("sign_conflict_defined", "sign_conflict_fraction", 0.0, 1.0),
        ("ratio_defined", "left_over_right_l2", 0.0, math.inf),
    )
    for flag, metric, lower, upper in contracts:
        if not isinstance(value[flag], bool):
            raise ValueError(f"{label} defined flag differs")
        if value[flag]:
            if (
                isinstance(value[metric], bool)
                or not isinstance(value[metric], (int, float))
                or not lower <= value[metric] <= upper
            ):
                raise ValueError(f"{label} defined metric differs")
        elif value[metric] is not None:
            raise ValueError(f"{label} undefined metric must be null")


def _scalar_map(value: Any, expected: set[str], label: str) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != expected
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(item)
            or item < 0
            for item in value.values()
        )
    ):
        raise ValueError(f"{label} scalar map differs")


def _validate_numeric_payload(result: dict[str, Any]) -> None:
    if result.get("positions") != list(POSITIONS):
        raise ValueError("top-level positions differ")
    rows = result["rows"]
    expected_row_keys = {
        "position",
        "regular_dataset_index",
        "regular_metadata",
        "pair_dataset_index",
        "pair_metadata",
        "losses",
        "gradient",
        "regular_total_gradient_residual",
        "regular_total_gradient_residual_relative",
        "regular_force_channel_loss_0p2",
        "regular_force_channel_gradient",
        "regular_force_decomposition_gradient_residual",
        "regular_force_decomposition_gradient_residual_relative",
        "old_statistic_gradient",
        "old_statistic_gradient_residual",
        "old_statistic_gradient_residual_relative",
        "true_state_channel_loss",
        "true_state_channel_weighted_lambda10_loss",
        "true_state_channel_gradient",
        "true_state_chunk_gradient_residual_l2_max",
        "true_state_chunk_gradient_residual_relative_max",
        "old_vs_regular",
        "new_vs_regular",
        "new_vs_regular_force",
        "old_vs_new",
        "mixed_old",
        "mixed_new",
        "new_over_regular_l2",
    }
    losses = {
        "regular_total",
        "regular_field",
        "regular_force_raw",
        "regular_force_0p2",
        "old_pair_raw",
        "old_pair_lambda10",
        "true_state_pair_raw",
    }
    old_groups = {"mean_total_cd", "mean_rear_cl", "rear_cl_rms"}
    residual_relative = {
        "regular_total_gradient_residual_relative",
        "regular_force_decomposition_gradient_residual_relative",
        "old_statistic_gradient_residual_relative",
        "true_state_chunk_gradient_residual_relative_max",
    }
    for row_index, row in enumerate(rows):
        if set(row) != expected_row_keys:
            raise ValueError("per-position numeric field set differs")
        _scalar_map(row["losses"], losses, "losses")
        if set(row["gradient"]) != GRADIENTS:
            raise ValueError("gradient component set differs")
        for name, stats in row["gradient"].items():
            _gradient_stats(stats, f"gradient.{name}")
        for key in (
            "regular_total_gradient_residual",
            "regular_force_decomposition_gradient_residual",
            "old_statistic_gradient_residual",
        ):
            _gradient_stats(row[key], key)
        for key in residual_relative:
            if not 0 <= row[key] <= 2e-5:
                raise ValueError(f"{key} exceeds implementation tolerance")
        if row["true_state_chunk_gradient_residual_l2_max"] < 0:
            raise ValueError("true-state absolute residual differs")
        _scalar_map(row["regular_force_channel_loss_0p2"], CHANNELS, "regular force")
        _scalar_map(row["true_state_channel_loss"], CHANNELS, "true-state force")
        _scalar_map(
            row["true_state_channel_weighted_lambda10_loss"],
            CHANNELS,
            "weighted true-state force",
        )
        for key, names in (
            ("regular_force_channel_gradient", CHANNELS),
            ("true_state_channel_gradient", CHANNELS),
            ("old_statistic_gradient", old_groups),
        ):
            if set(row[key]) != names:
                raise ValueError(f"{key} component set differs")
            for name, stats in row[key].items():
                _gradient_stats(stats, f"{key}.{name}")
        for key in (
            "old_vs_regular",
            "new_vs_regular",
            "new_vs_regular_force",
            "old_vs_new",
        ):
            _comparison(row[key], key)
        for key in ("mixed_old", "mixed_new"):
            mixed = row[key]
            if not isinstance(mixed, dict) or set(mixed) != STATS | {"clip_scale_at_1"}:
                raise ValueError(f"{key} keys differ")
            _gradient_stats({name: mixed[name] for name in STATS}, key)
            if not 0 < mixed["clip_scale_at_1"] <= 1:
                raise ValueError(f"{key} clip scale differs")
        if row["new_over_regular_l2"] < 0:
            raise ValueError("new/regular gradient ratio differs")
        if row_index >= 16:
            raise AssertionError("unreachable row count")
    if set(result.get("mean_gradient", {})) != GRADIENTS:
        raise ValueError("mean gradient component set differs")
    for name, stats in result["mean_gradient"].items():
        _gradient_stats(stats, f"mean_gradient.{name}")
    directions = result.get("mean_gradient_directions", {})
    if set(directions) != {"old_vs_regular", "new_vs_regular", "old_vs_new"}:
        raise ValueError("mean gradient direction set differs")
    for name, value in directions.items():
        _comparison(value, f"mean_gradient_directions.{name}")
    summary = result.get("summaries", {}).get("new_over_regular_l2")
    if (
        set(result.get("summaries", {})) != {"new_over_regular_l2"}
        or not isinstance(summary, dict)
        or set(summary) != {"min", "median", "max"}
        or not 0 <= summary["min"] <= summary["median"] <= summary["max"]
    ):
        raise ValueError("gradient-ratio summary differs")


def validate_receipt_bindings(
    receipt: dict[str, Any],
    *,
    result_path: Path,
    progress_path: Path,
    config: Path,
    sampling_receipt: Path,
    source_manifest: Path,
) -> None:
    """Reject a once-valid validation receipt after any bound file changes."""
    expected = {
        "result_sha256": sha256(result_path),
        "progress_sha256": sha256(progress_path),
        "config_sha256": sha256(config),
        "sampling_receipt_sha256": sha256(sampling_receipt),
        "source_manifest_sha256": sha256(source_manifest),
    }
    if (
        receipt.get("status") != "FC_P003C_TRAIN_GRADIENT_DIAGNOSTIC_VALIDATED"
        or receipt.get("sha256") != expected
    ):
        raise ValueError("validation receipt bindings are stale or incomplete")


def validate(
    *,
    result_path: Path,
    progress_path: Path,
    config: Path,
    sampling_receipt: Path,
    parent: Path,
    source_manifest: Path,
    base: Path,
    train8: Path,
    train16: Path,
    pair_manifest: Path,
    image_id: str,
) -> dict[str, str]:
    result = json.loads(result_path.read_text())
    progress = [json.loads(line) for line in progress_path.read_text().splitlines()]
    if not _finite(result) or not _finite(progress):
        raise ValueError("result contains nonfinite or unsupported JSON values")
    required = {
        "status": "FC_P003C_TRAIN_ONLY_GRADIENT_DIAGNOSTIC_COMPLETE",
        "scientific_admission": False,
        "optimizer_created_or_stepped": False,
        "candidate_saved": False,
        "validation_or_frozen_accessed": False,
        "image_id": image_id,
        "config_sha256": sha256(config),
        "sampling_receipt_sha256": sha256(sampling_receipt),
        "parent_model_sha256": sha256(parent / "FNO.0.2.mdlus"),
        "parent_state_sha256": sha256(parent / "checkpoint.0.2.pt"),
    }
    if any(result.get(key) != value for key, value in required.items()):
        raise ValueError("top-level result identity or safety contract differs")
    if result.get("model_state_sha256_before") != result.get(
        "model_state_sha256_after"
    ):
        raise ValueError("model state changed")
    expected_inputs = {
        "base_manifest": sha256(base / "manifest.json"),
        "base_train_split_manifest": sha256(base / "splits" / "train.json"),
        "base_normalization": sha256(base / "normalization.json"),
        "train8_manifest": sha256(train8 / "manifest.json"),
        "train8_normalization": sha256(train8 / "normalization.json"),
        "train16_manifest": sha256(train16 / "manifest.json"),
        "train16_normalization": sha256(train16 / "normalization.json"),
        "pair_manifest": sha256(pair_manifest),
        "source_manifest": sha256(source_manifest),
    }
    if result.get("input_sha256") != expected_inputs:
        raise ValueError("result input SHA bindings differ")
    rows = result.get("rows")
    if not isinstance(rows, list) or len(rows) != 16:
        raise ValueError("result must contain 16 position rows")
    _validate_numeric_payload(result)
    sampling = json.loads(sampling_receipt.read_text())
    expected_pair_ids = [
        pair_id
        for identity_pass in sampling["pair_identity_passes"]
        for pair_id in identity_pass
    ]
    expected_pair_indices = [
        pair_index
        for index_pass in sampling["pair_pass_indices"]
        for pair_index in index_pass
    ]
    regular_indices: list[int] = []
    for index, row in enumerate(rows):
        regular = row.get("regular_metadata", {})
        pair = row.get("pair_metadata", {})
        if (
            row.get("position") != POSITIONS[index]
            or row.get("pair_dataset_index") != expected_pair_indices[index]
            or pair.get("pair_id") != expected_pair_ids[index]
            or pair.get("split") != "train"
            or pair.get("start") != 0
            or pair.get("horizon") != 100
            or regular.get("split") != "train"
            or regular.get("rollout_steps") != 100
        ):
            raise ValueError("per-position identity contract differs")
        regular_index = row.get("regular_dataset_index")
        if (
            isinstance(regular_index, bool)
            or not isinstance(regular_index, int)
            or regular_index < 0
        ):
            raise ValueError("regular dataset index differs")
        regular_indices.append(regular_index)
    if len(set(regular_indices)) != 16:
        raise ValueError("regular dataset indices are not unique")
    if len(progress) != 16:
        raise ValueError("progress must contain exactly 16 flushed rows")
    for index, row in enumerate(progress):
        if (
            row.get("status") != "INCOMPLETE_PROGRESS"
            or row.get("completed_positions") != index + 1
            or row.get("position") != POSITIONS[index]
            or row.get("finite") is not True
            or not all(
                isinstance(row.get(key), (int, float))
                and not isinstance(row.get(key), bool)
                and math.isfinite(row[key])
                and row[key] >= 0
                for key in (
                    "elapsed_seconds",
                    "regular_gradient_l2",
                    "old_pair_gradient_l2",
                    "true_state_pair_gradient_l2",
                )
            )
        ):
            raise ValueError("progress sequence differs")
    return {
        "result_sha256": sha256(result_path),
        "progress_sha256": sha256(progress_path),
        "config_sha256": sha256(config),
        "sampling_receipt_sha256": sha256(sampling_receipt),
        "source_manifest_sha256": sha256(source_manifest),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sampling-receipt", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--train8", type=Path, required=True)
    parser.add_argument("--train16", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--image-id", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "status": "FC_P003C_TRAIN_GRADIENT_DIAGNOSTIC_VALIDATED",
                "sha256": validate(
                    result_path=args.result,
                    progress_path=args.progress,
                    config=args.config,
                    sampling_receipt=args.sampling_receipt,
                    parent=args.parent,
                    source_manifest=args.source_manifest,
                    base=args.base,
                    train8=args.train8,
                    train16=args.train16,
                    pair_manifest=args.pair_manifest,
                    image_id=args.image_id,
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
