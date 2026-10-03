#!/usr/bin/env python3
"""Strict provenance and pooled terminal-drag audit for a Gate-B JSON report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import h5py

HORIZON = "100"
MAX_TOTAL_DRAG_NRMSE = 0.10
FORCE_CHANNELS = ("front_cd", "front_cl", "rear_cd", "rear_cl")
FORCE_INDICES = (0, 1, 2, 3)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"missing numeric {label}")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"invalid {label}: {value}")
    return result


def project_path(value: str) -> str:
    prefix = "/workspace/"
    return value.removeprefix(prefix)


def recompute_terminal_drag_metrics(
    report: dict[str, Any], horizon: str = HORIZON
) -> dict[str, Any]:
    rows = []
    for case in report.get("cases", []):
        name = case.get("case")
        row = case.get("horizons", {}).get(horizon, {})
        segments = row.get("segments")
        if isinstance(segments, bool) or not isinstance(segments, int) or segments < 1:
            raise ValueError(f"{name}: invalid segment count at horizon {horizon}")
        rmse = finite_nonnegative(row.get("total_drag_rmse"), f"{name} RMSE")
        target_rms = finite_nonnegative(
            row.get("total_drag_target_rms"), f"{name} target RMS"
        )
        nrmse = finite_nonnegative(row.get("total_drag_nrmse"), f"{name} NRMSE")
        if target_rms <= 0:
            raise ValueError(f"{name}: total-drag target RMS must be positive")
        expected_nrmse = rmse / target_rms
        if not math.isclose(nrmse, expected_nrmse, rel_tol=1e-8, abs_tol=1e-10):
            raise ValueError(f"{name}: inconsistent total-drag NRMSE")
        if row.get("stable") is not True or row.get("failed_segments") != 0:
            raise ValueError(f"{name}: unstable terminal rollout")
        rows.append(
            {
                "case": name,
                "segments": segments,
                "total_drag_rmse": rmse,
                "total_drag_target_rms": target_rms,
                "total_drag_nrmse": nrmse,
            }
        )
    if not rows:
        raise ValueError("report has no cases")
    total_segments = sum(row["segments"] for row in rows)
    error_ss = sum(row["segments"] * row["total_drag_rmse"] ** 2 for row in rows)
    target_ss = sum(row["segments"] * row["total_drag_target_rms"] ** 2 for row in rows)
    if target_ss <= 0:
        raise ValueError("pooled total-drag target sum of squares must be positive")
    macro = sum(row["total_drag_nrmse"] for row in rows) / len(rows)
    pooled = math.sqrt(error_ss / target_ss)
    worst = max(rows, key=lambda row: row["total_drag_nrmse"])
    return {
        "horizon_steps": int(horizon),
        "case_count": len(rows),
        "segments": total_segments,
        "error_sum_squares": error_ss,
        "target_sum_squares": target_ss,
        "pooled_total_drag_nrmse": pooled,
        "macro_total_drag_nrmse": macro,
        "worst_case": worst,
        "cases": rows,
    }


def validate_normalization(stats: dict[str, Any]) -> None:
    if stats.get("computed_from") != "train split only":
        raise ValueError("normalization is not declared train-only")
    if tuple(stats.get("all_force_channels", ())) != FORCE_CHANNELS:
        raise ValueError("normalization force channel order mismatch")
    for key, count in (
        ("state_mean", 3),
        ("state_std", 3),
        ("all_force_mean", 4),
        ("all_force_std", 4),
    ):
        values = stats.get(key)
        if not isinstance(values, list) or len(values) != count:
            raise ValueError(f"invalid normalization field: {key}")
        numbers = [float(value) for value in values]
        if not all(math.isfinite(value) for value in numbers):
            raise ValueError(f"non-finite normalization field: {key}")
        if key.endswith("_std") and not all(value > 0 for value in numbers):
            raise ValueError(f"non-positive normalization field: {key}")


def decode_attribute(value: Any, label: str) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str) and label == "force_channels":
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("invalid HDF5 force_channels JSON") from error
    return value


def validate_hdf5_contract(path: Path, expected_split: str) -> None:
    with h5py.File(path, "r") as handle:
        if "force" not in handle:
            raise ValueError(f"{path}: force dataset is missing")
        force = handle["force"]
        if force.ndim != 2 or force.shape[1] != len(FORCE_CHANNELS):
            raise ValueError(f"{path}: force dataset must have shape (frames, 4)")
        channels = decode_attribute(
            handle.attrs.get("force_channels"), "force_channels"
        )
        if tuple(channels or ()) != FORCE_CHANNELS:
            raise ValueError(f"{path}: HDF5 force_channels order mismatch")
        split = decode_attribute(handle.attrs.get("split"), "split")
        if split != expected_split:
            raise ValueError(f"{path}: HDF5 split attribute mismatch")


def audit_report(
    report_path: Path,
    data: Path,
    normalization_data: Path,
    checkpoint_model: Path,
    *,
    expected_split: str,
    expected_profile: str,
    expected_normalization_profile: str,
    expected_case_count: int,
    expected_checkpoint_sha256: str,
    expected_normalization_sha256: str,
) -> dict[str, Any]:
    report = load_json(report_path)
    manifest_path = data / "manifest.json"
    normalization_manifest_path = normalization_data / "manifest.json"
    normalization_path = normalization_data / "normalization.json"
    manifest = load_json(manifest_path)
    normalization_manifest = load_json(normalization_manifest_path)
    stats = load_json(normalization_path)

    if expected_split not in {"validation", "test"}:
        raise ValueError("expected split must be validation or test")
    if report.get("split") != expected_split:
        raise ValueError("report split mismatch")
    if project_path(str(report.get("evaluation_data"))) != str(data):
        raise ValueError("report evaluation data mismatch")
    if project_path(str(report.get("normalization_data"))) != str(normalization_data):
        raise ValueError("report normalization data mismatch")
    if manifest.get("profile") != expected_profile:
        raise ValueError("evaluation manifest profile mismatch")
    if normalization_manifest.get("profile") != expected_normalization_profile:
        raise ValueError("normalization manifest profile mismatch")
    if manifest.get("trajectory_counts", {}).get(expected_split) != expected_case_count:
        raise ValueError("manifest split count mismatch")

    split_paths = sorted((data / expected_split).glob("*.h5"))
    for path in split_paths:
        validate_hdf5_contract(path, expected_split)
    expected_cases = [path.stem for path in split_paths]
    report_cases = [case.get("case") for case in report.get("cases", [])]
    if len(split_paths) != expected_case_count or report_cases != expected_cases:
        raise ValueError("report cases do not exactly match the declared split")
    if len(set(report_cases)) != len(report_cases):
        raise ValueError("duplicate cases in report")

    if tuple(report.get("force_channels", ())) != FORCE_CHANNELS:
        raise ValueError("report force channel order mismatch")
    if tuple(report.get("force_indices", ())) != FORCE_INDICES:
        raise ValueError("report force indices mismatch")
    validate_normalization(stats)
    action_scale = finite_nonnegative(report.get("action_scale"), "action scale")
    manifest_action_scale = finite_nonnegative(
        normalization_manifest.get("max_abs_omega"), "manifest action scale"
    )
    if not math.isclose(action_scale, manifest_action_scale):
        raise ValueError("report action scale mismatches normalization manifest")

    metadata = report.get("checkpoint_metadata")
    if not isinstance(metadata, dict):
        raise TypeError("checkpoint metadata is missing")
    if tuple(metadata.get("force_channels", ())) != FORCE_CHANNELS:
        raise ValueError("checkpoint force channel order mismatch")
    if tuple(metadata.get("force_indices", ())) != FORCE_INDICES:
        raise ValueError("checkpoint force indices mismatch")
    if not math.isclose(float(metadata.get("action_scale", math.nan)), action_scale):
        raise ValueError("checkpoint action scale mismatch")
    model_config = metadata.get("model_config", {})
    if model_config.get("in_channels") != 6 or model_config.get("out_channels") != 7:
        raise ValueError("checkpoint model channel contract mismatch")

    if not checkpoint_model.is_file() or checkpoint_model.suffix != ".mdlus":
        raise FileNotFoundError(f"checkpoint model is missing: {checkpoint_model}")
    checkpoint_sha = sha256(checkpoint_model)
    if checkpoint_sha != expected_checkpoint_sha256:
        raise ValueError("checkpoint SHA-256 mismatch")
    normalization_sha = sha256(normalization_path)
    if normalization_sha != expected_normalization_sha256:
        raise ValueError("normalization SHA-256 mismatch")
    report_checkpoint = project_path(str(report.get("checkpoint_dir")))
    if report_checkpoint != str(checkpoint_model.parent):
        raise ValueError("report checkpoint directory mismatch")
    match = re.fullmatch(r"FNO\.0\.(\d+)\.mdlus", checkpoint_model.name)
    if not match or int(match.group(1)) != int(report.get("checkpoint_epoch", -1)):
        raise ValueError("checkpoint filename epoch mismatches report")

    metrics = recompute_terminal_drag_metrics(report)
    summary = report.get("summary", {}).get(HORIZON, {})
    reported_macro = finite_nonnegative(
        summary.get("total_drag_nrmse"), "reported macro NRMSE"
    )
    if not math.isclose(
        reported_macro,
        metrics["macro_total_drag_nrmse"],
        rel_tol=1e-8,
        abs_tol=1e-10,
    ):
        raise ValueError("summary NRMSE is not the macro mean of case NRMSE values")
    if summary.get("segments") != metrics["segments"]:
        raise ValueError("summary segment count mismatch")

    pooled = metrics["pooled_total_drag_nrmse"]
    return {
        "schema_version": 2,
        "status": (
            "TERMINAL_FORCE_GATE_B_ACCURACY_PASS"
            if pooled <= MAX_TOTAL_DRAG_NRMSE
            else "TERMINAL_FORCE_GATE_B_ACCURACY_FAIL"
        ),
        "threshold": MAX_TOTAL_DRAG_NRMSE,
        "threshold_policy": "fixed at 10%; this audit cannot relax it",
        "metric_scope": (
            "100-step terminal instantaneous total-drag force only; not the mean "
            "Cd_total over the 100-step rollout window"
        ),
        "report": str(report_path),
        "report_sha256": sha256(report_path),
        "split": expected_split,
        "evaluation_profile": expected_profile,
        "normalization_profile": expected_normalization_profile,
        "manifest_sha256": sha256(manifest_path),
        "normalization_sha256": normalization_sha,
        "hdf5_contract": {
            "force_shape": "(frames, 4)",
            "force_channels": list(FORCE_CHANNELS),
            "split_attribute": expected_split,
        },
        "checkpoint_model": str(checkpoint_model),
        "checkpoint_sha256": checkpoint_sha,
        "reported_summary_macro_total_drag_nrmse": reported_macro,
        **metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--normalization-data", type=Path, required=True)
    parser.add_argument("--checkpoint-model", type=Path, required=True)
    parser.add_argument(
        "--expected-split", choices=("validation", "test"), required=True
    )
    parser.add_argument("--expected-profile", required=True)
    parser.add_argument("--expected-normalization-profile", required=True)
    parser.add_argument("--expected-case-count", type=int, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-normalization-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {args.output}")
    result = audit_report(
        args.report,
        args.data,
        args.normalization_data,
        args.checkpoint_model,
        expected_split=args.expected_split,
        expected_profile=args.expected_profile,
        expected_normalization_profile=args.expected_normalization_profile,
        expected_case_count=args.expected_case_count,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        expected_normalization_sha256=args.expected_normalization_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
