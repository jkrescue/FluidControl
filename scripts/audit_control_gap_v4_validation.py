#!/usr/bin/env python3
"""Fail-closed preflight and comparison for v4 validation-only FNO evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import yaml
from audit_gate_b_metric_integrity import (
    MAX_TOTAL_DRAG_NRMSE,
    recompute_terminal_drag_metrics,
)

HORIZONS = (1, 10, 50, 100)
ARCHITECTURE_KEYS = (
    "in_channels",
    "out_channels",
    "latent_channels",
    "num_fno_layers",
    "num_fno_modes",
    "decoder_layers",
    "decoder_layer_size",
    "padding",
    "coord_features",
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a YAML mapping: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validation_hashes(root: Path) -> dict[str, str]:
    validation = root / "validation"
    if not validation.is_dir():
        raise FileNotFoundError(f"validation directory is missing: {validation}")
    paths = sorted(validation.glob("*.h5"))
    if len(paths) != 4:
        raise ValueError(f"expected exactly four validation trajectories: {validation}")
    return {path.name: sha256(path) for path in paths}


def checkpoint_pair(best: Path) -> dict[str, Any]:
    models = sorted(best.glob("FNO.*.mdlus"))
    states = sorted(best.glob("checkpoint.*.pt"))
    if len(models) != 1 or len(states) != 1:
        raise ValueError(f"expected one model/state checkpoint pair in {best}")
    model_match = re.fullmatch(r"FNO\.0\.(\d+)\.mdlus", models[0].name)
    state_match = re.fullmatch(r"checkpoint\.0\.(\d+)\.pt", states[0].name)
    if (
        not model_match
        or not state_match
        or model_match.group(1) != state_match.group(1)
    ):
        raise ValueError(f"checkpoint epochs do not match in {best}")
    return {
        "epoch": int(model_match.group(1)),
        "model": str(models[0]),
        "model_sha256": sha256(models[0]),
        "state": str(states[0]),
        "state_sha256": sha256(states[0]),
    }


def validate_normalization(normalization: dict[str, Any]) -> None:
    expected_channels = ["front_cd", "front_cl", "rear_cd", "rear_cl"]
    if normalization.get("computed_from") != "train split only":
        raise ValueError("normalization is not declared train-only")
    if normalization.get("all_force_channels") != expected_channels:
        raise ValueError("unexpected all-force channel order")
    for key, length in (
        ("state_mean", 3),
        ("state_std", 3),
        ("all_force_mean", 4),
        ("all_force_std", 4),
    ):
        values = normalization.get(key)
        if not isinstance(values, list) or len(values) != length:
            raise ValueError(f"invalid normalization field: {key}")
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError(f"non-finite normalization field: {key}")
        if key.endswith("_std") and not all(float(value) > 0 for value in values):
            raise ValueError(f"non-positive normalization standard deviation: {key}")


def preflight(
    v4_data: Path,
    v3_data: Path,
    candidate: Path,
    parent: Path,
    expected_epochs: int = 5,
) -> dict[str, Any]:
    if v4_data.name != "tandem_cylinders_control_gap_v4":
        raise ValueError("refusing a non-v4 evaluation data root")
    if v3_data.name != "tandem_cylinders_gate_b_aug_v3":
        raise ValueError("refusing an unexpected v3 comparison data root")

    manifest_path = v4_data / "manifest.json"
    normalization_path = v4_data / "normalization.json"
    parent_manifest_path = v3_data / "manifest.json"
    parent_normalization_path = v3_data / "normalization.json"
    manifest = load_json(manifest_path)
    normalization = load_json(normalization_path)
    parent_manifest = load_json(parent_manifest_path)
    parent_normalization = load_json(parent_normalization_path)
    if manifest.get("profile") != "control_gap_v4":
        raise ValueError("unexpected v4 manifest profile")
    counts = manifest.get("trajectory_counts", {})
    if counts.get("train") != 28 or counts.get("validation") != 4:
        raise ValueError("unexpected v4 train/validation trajectory counts")
    if float(manifest.get("max_abs_omega", -1)) != 5.0:
        raise ValueError("unexpected v4 action support")
    if parent_manifest.get("profile") != "gate_b_aug_v3":
        raise ValueError("unexpected v3 parent manifest profile")
    if float(parent_manifest.get("max_abs_omega", -1)) != 5.0:
        raise ValueError("unexpected v3 parent action support")
    validate_normalization(normalization)
    validate_normalization(parent_normalization)

    v4_validation = validation_hashes(v4_data)
    v3_validation = validation_hashes(v3_data)
    if v4_validation != v3_validation:
        raise ValueError("v4 and v3 validation trajectories are not byte-identical")

    history_path = candidate / "training_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(history, list) or len(history) != expected_epochs:
        raise ValueError("candidate training history is incomplete")
    epochs = [row.get("epoch") for row in history if isinstance(row, dict)]
    if epochs != list(range(1, expected_epochs + 1)):
        raise ValueError("candidate training history has unexpected epochs")

    candidate_config = load_yaml(candidate / "resolved_config.yaml")
    parent_config = load_yaml(parent / "resolved_config.yaml")
    if candidate_config.get("data", {}).get("root") != str(v4_data):
        raise ValueError("candidate resolved config does not reference v4 data")
    if parent_config.get("data", {}).get("root") != str(v3_data):
        raise ValueError("parent resolved config does not reference v3 data")
    expected_force_indices = [0, 1, 2, 3]
    for label, config in (("candidate", candidate_config), ("parent", parent_config)):
        if config.get("data", {}).get("force_indices") != expected_force_indices:
            raise ValueError(f"{label} force target order is incompatible")
    candidate_architecture = {
        key: candidate_config.get("model", {}).get(key) for key in ARCHITECTURE_KEYS
    }
    parent_architecture = {
        key: parent_config.get("model", {}).get(key) for key in ARCHITECTURE_KEYS
    }
    if candidate_architecture != parent_architecture:
        raise ValueError("candidate and parent FNO architectures differ")

    return {
        "status": "VALIDATION_ONLY_PREFLIGHT_PASS",
        "scientific_scope": "development comparison only; frozen test not accessed",
        "v4_manifest": str(manifest_path),
        "v4_manifest_sha256": sha256(manifest_path),
        "v4_normalization": str(normalization_path),
        "v4_normalization_sha256": sha256(normalization_path),
        "v3_parent_manifest": str(parent_manifest_path),
        "v3_parent_manifest_sha256": sha256(parent_manifest_path),
        "v3_parent_normalization": str(parent_normalization_path),
        "v3_parent_normalization_sha256": sha256(parent_normalization_path),
        "normalization_policy": (
            "each checkpoint is evaluated with its own train-only normalization; "
            "both use byte-identical raw validation trajectories"
        ),
        "validation_trajectory_sha256": v4_validation,
        "candidate_history_sha256": sha256(history_path),
        "candidate_checkpoint": checkpoint_pair(candidate / "best"),
        "parent_checkpoint": checkpoint_pair(parent / "best"),
        "architecture": candidate_architecture,
        "force_indices": expected_force_indices,
        "horizons": list(HORIZONS),
        "segment_stride": 25,
        "action_mode": "observed",
    }


def validate_evaluation(
    report: dict[str, Any], expected_data: str, expected_normalization: str
) -> None:
    expected = {
        "split": "validation",
        "evaluation_data": expected_data,
        "normalization_data": expected_normalization,
        "segment_stride": 25,
        "action_mode": "observed",
        "force_indices": [0, 1, 2, 3],
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"evaluation mismatch for {key}: {report.get(key)!r}")
    if set(report.get("summary", {})) != {str(value) for value in HORIZONS}:
        raise ValueError("evaluation does not contain exactly the declared horizons")
    if int(report.get("checkpoint_epoch", 0)) <= 0:
        raise ValueError("PhysicsNeMo checkpoint was not loaded")
    for horizon in HORIZONS:
        row = report["summary"][str(horizon)]
        value = float(row["total_drag_nrmse"])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid total-drag NRMSE at horizon {horizon}")
        if row.get("stable") is not True or int(row.get("failed_segments", -1)) != 0:
            raise ValueError(f"unstable rollout at horizon {horizon}")


def compare(
    candidate_path: Path,
    parent_path: Path,
    preflight_path: Path,
    expected_data: str,
    candidate_normalization: str,
    parent_normalization: str,
) -> dict[str, Any]:
    candidate = load_json(candidate_path)
    parent = load_json(parent_path)
    preflight_report = load_json(preflight_path)
    validate_evaluation(candidate, expected_data, candidate_normalization)
    validate_evaluation(parent, expected_data, parent_normalization)
    candidate_macro_nrmse = {
        str(horizon): float(candidate["summary"][str(horizon)]["total_drag_nrmse"])
        for horizon in HORIZONS
    }
    parent_macro_nrmse = {
        str(horizon): float(parent["summary"][str(horizon)]["total_drag_nrmse"])
        for horizon in HORIZONS
    }
    macro_delta = {
        str(horizon): (
            candidate_macro_nrmse[str(horizon)] - parent_macro_nrmse[str(horizon)]
        )
        for horizon in HORIZONS
    }
    candidate_terminal = recompute_terminal_drag_metrics(candidate)
    parent_terminal = recompute_terminal_drag_metrics(parent)
    candidate_pooled = candidate_terminal["pooled_total_drag_nrmse"]
    parent_pooled = parent_terminal["pooled_total_drag_nrmse"]
    pooled_delta = candidate_pooled - parent_pooled
    improved = pooled_delta < 0
    meets_threshold = candidate_pooled <= MAX_TOTAL_DRAG_NRMSE
    if improved and meets_threshold:
        status = "V4_VALIDATION_IMPROVED_AND_FIXED_THRESHOLD_MET"
    elif improved:
        status = "V4_VALIDATION_IMPROVED_BUT_ABOVE_FIXED_THRESHOLD"
    else:
        status = "V4_VALIDATION_100STEP_NO_GAIN"
    return {
        "status": status,
        "split": "validation",
        "selection_metric": "pooled 100-step terminal total-drag NRMSE",
        "selection_rule": "strictly lower is better; no threshold changed",
        "fixed_gate_b_threshold": MAX_TOTAL_DRAG_NRMSE,
        "candidate_meets_fixed_threshold": meets_threshold,
        "horizons": list(HORIZONS),
        "candidate_macro_total_drag_nrmse": candidate_macro_nrmse,
        "v3_parent_macro_total_drag_nrmse": parent_macro_nrmse,
        "candidate_minus_parent_macro_nrmse": macro_delta,
        "candidate_terminal_100step": candidate_terminal,
        "v3_parent_terminal_100step": parent_terminal,
        "candidate_minus_parent_pooled_100step_nrmse": pooled_delta,
        "evaluation_data": expected_data,
        "candidate_normalization_data": candidate_normalization,
        "parent_normalization_data": parent_normalization,
        "normalization_policy": (
            "each checkpoint uses the train-only normalization from its own training "
            "profile; NRMSE is compared after de-normalization in physical force units"
        ),
        "common_denominator_basis": (
            "byte-identical raw validation force targets provide the same physical-unit "
            "total-drag RMS denominator"
        ),
        "scientific_scope": (
            "validation-only terminal-force development comparison; not 100-step "
            "window-mean Cd accuracy, no frozen-test access, and no CFD control-benefit claim"
        ),
        "candidate_report": str(candidate_path),
        "candidate_report_sha256": sha256(candidate_path),
        "parent_report": str(parent_path),
        "parent_report_sha256": sha256(parent_path),
        "preflight_report": str(preflight_path),
        "preflight_report_sha256": sha256(preflight_path),
        "checkpoint_provenance": {
            "candidate": preflight_report["candidate_checkpoint"],
            "parent": preflight_report["parent_checkpoint"],
        },
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--v4-data", type=Path, required=True)
    preflight_parser.add_argument("--v3-data", type=Path, required=True)
    preflight_parser.add_argument("--candidate", type=Path, required=True)
    preflight_parser.add_argument("--parent", type=Path, required=True)
    preflight_parser.add_argument("--expected-epochs", type=int, default=5)
    preflight_parser.add_argument("--output", type=Path, required=True)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--candidate-report", type=Path, required=True)
    compare_parser.add_argument("--parent-report", type=Path, required=True)
    compare_parser.add_argument("--preflight-report", type=Path, required=True)
    compare_parser.add_argument("--expected-data", required=True)
    compare_parser.add_argument("--candidate-normalization", required=True)
    compare_parser.add_argument("--parent-normalization", required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "preflight":
        result = preflight(
            args.v4_data,
            args.v3_data,
            args.candidate,
            args.parent,
            args.expected_epochs,
        )
    else:
        result = compare(
            args.candidate_report,
            args.parent_report,
            args.preflight_report,
            args.expected_data,
            args.candidate_normalization,
            args.parent_normalization,
        )
    write_json(args.output, result)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
