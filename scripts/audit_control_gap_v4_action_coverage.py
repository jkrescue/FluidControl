#!/usr/bin/env python3
"""Audit control_gap_v4 action coverage without reading flow-field arrays.

Only the curated profile manifest and each train/validation HDF5 ``omega``
dataset are read.  The frozen-test directory is deliberately not enumerated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np

PROFILE = "control_gap_v4"
TARGET_MAX_ABS_OMEGA = 0.75
THRESHOLDS = (0.375, 0.75, 1.0, 2.0)
QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)
AUDITED_SPLITS = ("train", "validation")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    counts = manifest.get("trajectory_counts", {})
    if (
        manifest.get("profile") != PROFILE
        or not isinstance(manifest.get("frames_per_trajectory"), int)
        or manifest["frames_per_trajectory"] <= 0
        or any(not isinstance(counts.get(split), int) for split in AUDITED_SPLITS)
    ):
        raise ValueError("manifest does not describe the expected control_gap_v4 profile")
    return manifest


def read_omega_only(path: Path, expected_frames: int) -> np.ndarray:
    """Read the sole audited HDF5 payload; never dereference field datasets."""
    with h5py.File(path, "r") as handle:
        if "omega" not in handle:
            raise ValueError(f"missing omega dataset: {path}")
        omega = np.asarray(handle["omega"][:], dtype=np.float64).reshape(-1)
    if omega.shape != (expected_frames,) or not np.isfinite(omega).all():
        raise ValueError(f"invalid omega shape or values: {path}")
    return omega


def trajectory_summary(path: Path, omega: np.ndarray) -> dict:
    absolute = np.abs(omega)
    target_mask = absolute <= TARGET_MAX_ABS_OMEGA
    return {
        "file": path.name,
        "frame_count": len(omega),
        "min_omega": float(np.min(omega)),
        "max_omega": float(np.max(omega)),
        "max_abs_omega": float(np.max(absolute)),
        "median_abs_omega": float(np.median(absolute)),
        "target_range_frame_count": int(np.count_nonzero(target_mask)),
        "target_range_frame_fraction": float(np.mean(target_mask)),
        "entire_trajectory_within_target_range": bool(np.all(target_mask)),
        "majority_frames_within_target_range": bool(np.mean(target_mask) >= 0.5),
    }


def split_summary(
    profile_dir: Path, split: str, expected_count: int, expected_frames: int
) -> dict:
    split_dir = profile_dir / split
    paths = sorted(split_dir.glob("*.h5"))
    if len(paths) != expected_count:
        raise ValueError(
            f"{split} manifest/file count mismatch: expected {expected_count}, got {len(paths)}"
        )
    arrays = []
    trajectories = []
    for path in paths:
        omega = read_omega_only(path, expected_frames)
        arrays.append(omega)
        trajectories.append(trajectory_summary(path, omega))
    if not arrays:
        raise ValueError(f"empty audited split: {split}")
    omega = np.concatenate(arrays)
    absolute = np.abs(omega)
    target_mask = absolute <= TARGET_MAX_ABS_OMEGA
    return {
        "trajectory_count": len(trajectories),
        "frame_count": len(omega),
        "target_range_abs_omega_le_0p75": {
            "frame_count": int(np.count_nonzero(target_mask)),
            "frame_fraction": float(np.mean(target_mask)),
            "entirely_target_range_trajectory_count": sum(
                row["entire_trajectory_within_target_range"] for row in trajectories
            ),
            "majority_target_range_trajectory_count": sum(
                row["majority_frames_within_target_range"] for row in trajectories
            ),
        },
        "frame_fraction_abs_omega_le_threshold": {
            str(threshold): float(np.mean(absolute <= threshold))
            for threshold in THRESHOLDS
        },
        "frame_fraction_abs_omega_gt_2": float(np.mean(absolute > 2.0)),
        "abs_omega_quantiles": {
            str(quantile): float(value)
            for quantile, value in zip(QUANTILES, np.quantile(absolute, QUANTILES))
        },
        "trajectories": trajectories,
    }


def build_report(profile_dir: Path) -> dict:
    manifest_path = profile_dir / "manifest.json"
    manifest = read_manifest(manifest_path)
    counts = manifest["trajectory_counts"]
    expected_frames = manifest["frames_per_trajectory"]
    splits = {
        split: split_summary(
            profile_dir, split, int(counts[split]), int(expected_frames)
        )
        for split in AUDITED_SPLITS
    }
    return {
        "status": "CONTROL_GAP_V4_ACTION_COVERAGE_AUDIT",
        "profile": PROFILE,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256(manifest_path),
        "read_contract": {
            "hdf5_dataset_read": "omega only",
            "audited_splits": list(AUDITED_SPLITS),
            "excluded_split": "test/frozen test was not enumerated or opened",
            "flow_fields_read": False,
        },
        "target_control_range": {
            "definition": "abs(omega) <= 0.75",
            "inclusive_boundary": True,
        },
        "splits": splits,
        "key_findings": {
            "train_target_range_frame_fraction": splits["train"][
                "target_range_abs_omega_le_0p75"
            ]["frame_fraction"],
            "train_entirely_target_range_trajectory_count": splits["train"][
                "target_range_abs_omega_le_0p75"
            ]["entirely_target_range_trajectory_count"],
            "validation_target_range_frame_fraction": splits["validation"][
                "target_range_abs_omega_le_0p75"
            ]["frame_fraction"],
            "validation_entirely_target_range_trajectory_count": splits["validation"][
                "target_range_abs_omega_le_0p75"
            ]["entirely_target_range_trajectory_count"],
        },
        "scientific_limitations": [
            (
                "Coverage mismatch is evidence of an action-distribution gap, but by "
                "itself does not prove that the gap caused any particular prediction error."
            ),
            (
                "Frames within a trajectory are temporally correlated; frame fractions "
                "are not counts of independent training examples."
            ),
            (
                "Low-action frames inside high-amplitude trajectories do not replace "
                "trajectories whose full action history remains in the target range."
            ),
            (
                "This audit measures omega coverage only; it does not measure state, wake "
                "phase, force-response, or action-history coverage."
            ),
            (
                "No frozen-test data, flow-field tensors, retraining, or model selection "
                "was performed."
            ),
        ],
        "interpretation": (
            "Use this as a dataset-design diagnostic. Targeted matched-start low-action "
            "train and independent validation trajectories are needed before making "
            "low-action surrogate or closed-loop accuracy claims."
        ),
    }


def write_json_exclusive_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(payload, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise ValueError(f"refusing to overwrite {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=repo / "data/curated/tandem_cylinders_control_gap_v4",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            repo
            / "artifacts/tandem_cylinders/control_gap_v4_action_coverage_20261003.json"
        ),
    )
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to((repo / "artifacts").resolve()):
        parser.error("output must be inside the repository artifacts directory")
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    report = build_report(args.profile_dir)
    write_json_exclusive_atomic(args.output, report)
    print(json.dumps({"status": report["status"], **report["key_findings"]}, indent=2))


if __name__ == "__main__":
    main()
