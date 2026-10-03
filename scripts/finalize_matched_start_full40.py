#!/usr/bin/env python3
"""Validate and atomically assemble the leakage-safe matched-start full40 HDF set."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
PROFILE = "matched_start_full40_v1"
PREDECLARATION = (
    REPO / "artifacts/tandem_cylinders/matched_start_full40_predeclared_20261003.json"
)
PREDECLARATION_SHA256 = "d7ff174ef10194a8739357376335ca13ff9b45c8079970846bb71f15a715d24b"
NINE_FINAL = (
    REPO / "data/curated/tandem_cylinders_matched_start_commissioning_train9_v1"
)
REMAINDER_STAGING = REPO / "data/curated/.staging/matched_start_full40_v1"
FINAL = REPO / "data/curated/tandem_cylinders_matched_start_full40_v1"
SPLITS = ("train", "validation", "frozen_test")
EXPECTED_COUNTS = {"train": 20, "validation": 10, "frozen_test": 10}
STATE_CHANNELS = ["u", "v", "gauge_pressure"]
FORCE_CHANNELS = ["front_cd", "front_cl", "rear_cd", "rear_cl"]
MAX_ABS_OMEGA = 0.75
EXECUTION_REVIEWED = False
EXECUTION_TOKEN = "FINALIZE_REVIEWED_MATCHED_START_FULL40"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_matrix(repo: Path = REPO) -> dict[str, dict]:
    path = repo / PREDECLARATION.relative_to(REPO)
    if sha256(path) != PREDECLARATION_SHA256:
        raise ValueError("full40 predeclaration SHA-256 differs")
    cases = json.loads(path.read_text(encoding="utf-8"))["cases"]
    counts = {
        split: sum(row["split"] == split for row in cases.values()) for split in SPLITS
    }
    if len(cases) != 40 or counts != EXPECTED_COUNTS:
        raise ValueError("full40 matrix/split contract differs")
    return cases


def split_manifests(matrix: dict[str, dict]) -> dict[str, list[str]]:
    manifests = {
        split: sorted(name for name, row in matrix.items() if row["split"] == split)
        for split in SPLITS
    }
    if {split: len(names) for split, names in manifests.items()} != EXPECTED_COUNTS:
        raise ValueError("explicit split manifest counts differ")
    phase_sets = {
        split: {matrix[name]["phase_bin"] for name in names}
        for split, names in manifests.items()
    }
    if phase_sets != {
        "train": {0, 2, 4, 6},
        "validation": {1, 5},
        "frozen_test": {3, 7},
    }:
        raise ValueError("phase bins leak across explicit split manifests")
    return manifests


def validate_nine_reuse(repo: Path, matrix: dict[str, dict]) -> dict[str, Path]:
    root = repo / NINE_FINAL.relative_to(REPO)
    manifest_path = root / "manifest.json"
    qc_path = root / "commissioning_qc.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    expected = {
        name
        for name, row in matrix.items()
        if row["disposition"] == "existing_nine_case_commissioning"
    }
    qc_rows = qc.get("cases", [])
    qc_by_case = {row.get("case"): row for row in qc_rows}
    if (
        len(expected) != 9
        or manifest.get("profile") != "matched_start_commissioning_train9_v1"
        or manifest.get("trajectory_counts") != {"train": 9, "validation": 0, "test": 0}
        or qc.get("status") != "MATCHED_START_COMMISSIONING_NINE_CASE_QC_OK"
        or len(qc_rows) != 9
        or set(qc_by_case) != expected
        or set(manifest.get("hdf5_sha256", {})) != expected
    ):
        raise ValueError("finalized nine-case reuse contract differs")
    result = {}
    for name in expected:
        path = root / "train" / f"{name}.h5"
        if not path.is_file() or sha256(path) != manifest["hdf5_sha256"][name]:
            raise ValueError(f"nine-case HDF5 hash differs: {name}")
        row = qc_by_case[name]
        expected_phase = f"b{matrix[name]['phase_bin']:02d}"
        if (
            row.get("split") != "train"
            or row.get("phase_bin") != expected_phase
            or row.get("source_state_sha256") != matrix[name]["source_state_sha256"]
            or row.get("hdf5_sha256") != manifest["hdf5_sha256"][name]
        ):
            raise ValueError(f"nine-case final QC provenance differs: {name}")
        result[name] = path
    return result


def validate_remainder_layout(repo: Path, matrix: dict[str, dict]) -> dict[str, Path]:
    root = repo / REMAINDER_STAGING.relative_to(REPO)
    expected = {
        name: root / name / row["split"] / f"{name}.h5"
        for name, row in matrix.items()
        if row["disposition"] == "planned_new_remainder_case"
    }
    if len(expected) != 31:
        raise ValueError("full40 remainder is not exactly 31 cases")
    actual = set(root.glob("*/*/*.h5"))
    temporary = list(root.glob("*/*/*.h5.tmp"))
    if actual != set(expected.values()) or temporary:
        raise ValueError("remainder staging is incomplete, extra, or non-atomic")
    return expected


def load_existing_finalizer(repo: Path):
    path = repo / "scripts/finalize_matched_start_commissioning.py"
    spec = importlib.util.spec_from_file_location("full40_force_validator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_remainder_curator(repo: Path):
    path = repo / "scripts/curate_matched_start_full40_remainder.py"
    spec = importlib.util.spec_from_file_location("full40_remainder_contract", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_remainder_hdf(
    repo: Path, name: str, row: dict, path: Path, force_validator
) -> dict:
    raw_case = repo / "cfd/tandem_cylinders/cases" / name
    config = json.loads((raw_case / "case_config.json").read_text(encoding="utf-8"))
    checks = {
        "case": name,
        "split": row["split"],
        "phase_bin": row["phase_bin"],
        "source_restart_time": row["source_restart_time"],
        "source_state_sha256": row["source_state_sha256"],
        "action_target": row["action_target"],
        "action_points": row["action_points"],
        "start_time": row["run_window"][0],
        "end_time": row["run_window"][1],
        "analysis_window": row["analysis_window"],
    }
    if any(config.get(key) != value for key, value in checks.items()):
        raise ValueError(f"raw config differs from full40 predeclaration: {name}")
    start, end = float(config["start_time"]), float(config["end_time"])
    cases_root = raw_case.parent
    front, _ = force_validator.assembled_force(
        cases_root, raw_case, config, "forceFront", start, end
    )
    rear, _ = force_validator.assembled_force(
        cases_root, raw_case, config, "forceRear", start, end
    )
    expected_time = np.linspace(start, end, 801)
    action = np.asarray(config["action_points"], dtype=np.float64)
    max_pressure_mean = 0.0
    coverage = []
    with h5py.File(path, "r") as handle:
        shapes = {
            "state": (801, 3, 128, 256),
            "mask": (801, 1, 128, 256),
            "omega": (801, 1),
            "force": (801, 4),
            "time": (801, 1),
        }
        if any(key not in handle or handle[key].shape != shape for key, shape in shapes.items()):
            raise ValueError(f"HDF5 schema differs: {name}")
        validate_coordinate_and_channel_contract(handle, name)
        if handle.attrs.get("case") != name or handle.attrs.get("split") != row["split"]:
            raise ValueError(f"HDF5 identity/split differs: {name}")
        if json.loads(handle.attrs["config_json"]) != config:
            raise ValueError(f"embedded config differs: {name}")
        time = handle["time"][:, 0].astype(np.float64)
        omega = handle["omega"][:, 0].astype(np.float64)
        force = handle["force"][:].astype(np.float64)
        if not np.allclose(time, expected_time, rtol=0, atol=2e-5):
            raise ValueError(f"HDF5 time grid differs: {name}")
        if not np.allclose(omega, np.interp(time, action[:, 0], action[:, 1]), atol=1e-6):
            raise ValueError(f"HDF5 action differs: {name}")
        expected_force = np.stack(
            [
                np.interp(time, front[:, 0], front[:, 1]),
                np.interp(time, front[:, 0], front[:, 2]),
                np.interp(time, rear[:, 0], rear[:, 1]),
                np.interp(time, rear[:, 0], rear[:, 2]),
            ],
            axis=1,
        )
        if not np.allclose(force, expected_force, rtol=2e-6, atol=2e-6):
            raise ValueError(f"HDF5 force differs: {name}")
        for index in range(801):
            state = handle["state"][index]
            raw_mask = handle["mask"][index, 0]
            if not np.isin(raw_mask, (0, 1)).all():
                raise ValueError(f"HDF5 mask is not binary: {name}/{index}")
            mask = raw_mask.astype(bool)
            if not np.isfinite(state).all() or not mask.any():
                raise ValueError(f"HDF5 state/mask differs: {name}/{index}")
            coverage.append(float(mask.mean()))
            max_pressure_mean = max(
                max_pressure_mean,
                abs(float(state[2][mask].mean(dtype=np.float64))),
            )
    if max_pressure_mean > 1e-6 or not (0.8 < min(coverage) <= max(coverage) < 1.0):
        raise ValueError(f"HDF5 gauge/coverage differs: {name}")
    raw_receipt = (
        repo
        / "artifacts/matched_start_full40_extension/transfer_verified"
        / f"{name}.json"
    )
    vtk_receipt = (
        repo / "artifacts/matched_start_full40_extension/vtk_ready" / f"{name}.json"
    )
    raw = json.loads(raw_receipt.read_text(encoding="utf-8"))
    vtk = json.loads(vtk_receipt.read_text(encoding="utf-8"))
    if (
        raw.get("status") != "FULL40_RAW_TRANSFER_VERIFIED"
        or raw.get("case") != name
        or vtk.get("status") != "VTK_READY"
        or vtk.get("case") != name
        or vtk.get("raw_transfer_receipt_sha256") != sha256(raw_receipt)
    ):
        raise ValueError(f"raw/VTK receipt lineage differs: {name}")
    return {
        "case": name,
        "split": row["split"],
        "hdf5_sha256": sha256(path),
        "frames": 801,
        "max_abs_pressure_mean": max_pressure_mean,
        "coverage_range": [min(coverage), max(coverage)],
        "raw_transfer_receipt_sha256": sha256(raw_receipt),
        "vtk_receipt_sha256": sha256(vtk_receipt),
    }


def validate_coordinate_and_channel_contract(handle, name: str) -> None:
    expected_x = np.linspace(8.0, 25.0, 256, dtype=np.float32)
    expected_y = np.linspace(4.0, 11.0, 128, dtype=np.float32)
    if (
        "x" not in handle
        or "y" not in handle
        or handle["x"].shape != (256,)
        or handle["y"].shape != (128,)
        or not np.array_equal(handle["x"][:], expected_x)
        or not np.array_equal(handle["y"][:], expected_y)
    ):
        raise ValueError(f"HDF5 coordinate contract differs: {name}")
    try:
        state_channels = json.loads(handle.attrs["state_channels"])
        force_channels = json.loads(handle.attrs["force_channels"])
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"HDF5 channel metadata is absent or invalid: {name}") from error
    if state_channels != STATE_CHANNELS or force_channels != FORCE_CHANNELS:
        raise ValueError(f"HDF5 channel metadata differs: {name}")


def compute_train_normalization(paths: list[Path]) -> dict:
    if len(paths) != 20:
        raise ValueError("normalization requires the explicit 20-file train manifest")
    state_sum = np.zeros(3, dtype=np.float64)
    state_sq = np.zeros(3, dtype=np.float64)
    state_count = np.zeros(3, dtype=np.int64)
    state_min = np.full(3, np.inf, dtype=np.float64)
    state_max = np.full(3, -np.inf, dtype=np.float64)
    force_sum = np.zeros(4, dtype=np.float64)
    force_sq = np.zeros(4, dtype=np.float64)
    force_count = 0
    for path in paths:
        with h5py.File(path, "r") as handle:
            if handle.attrs.get("split") != "train":
                raise ValueError(f"non-train file passed to normalization: {path}")
            state = handle["state"][:]
            valid = handle["mask"][:, 0].astype(bool)
            for channel in range(3):
                values = state[:, channel][valid]
                state_sum[channel] += values.sum(dtype=np.float64)
                state_sq[channel] += np.square(values, dtype=np.float64).sum()
                state_count[channel] += len(values)
                state_min[channel] = min(state_min[channel], values.min())
                state_max[channel] = max(state_max[channel], values.max())
            force = handle["force"][:].astype(np.float64)
            force_sum += force.sum(axis=0)
            force_sq += np.square(force).sum(axis=0)
            force_count += len(force)
    state_mean = state_sum / state_count
    force_mean = force_sum / force_count
    state_std = np.sqrt(
        np.maximum(state_sq / state_count - state_mean**2, 1e-12)
    )
    force_std = np.sqrt(
        np.maximum(force_sq / force_count - force_mean**2, 1e-12)
    )
    state_abs_normalized_max = np.maximum(
        np.abs(state_min - state_mean), np.abs(state_max - state_mean)
    ) / state_std
    return {
        "computed_from": "train split only",
        "source_scope": "explicit 20-case train manifest only",
        "state_channels": STATE_CHANNELS,
        "state_mean": state_mean.tolist(),
        "state_std": state_std.tolist(),
        "state_abs_normalized_channel_max_train": state_abs_normalized_max.tolist(),
        "state_abs_normalized_max_train": float(state_abs_normalized_max.max()),
        "state_support_computed_from": "exact train split valid cells",
        "force_channels": FORCE_CHANNELS[2:],
        "force_mean": force_mean[2:].tolist(),
        "force_std": force_std[2:].tolist(),
        "all_force_channels": FORCE_CHANNELS,
        "all_force_mean": force_mean.tolist(),
        "all_force_std": force_std.tolist(),
    }


def readiness(repo: Path = REPO) -> dict:
    matrix = load_matrix(repo)
    manifests = split_manifests(matrix)
    blockers = []
    try:
        curator = load_remainder_curator(repo)
        curator.validate_raw_aggregate(repo, matrix)
    except (FileNotFoundError, ValueError):
        blockers.append("REMAINDER_31_RAW_AGGREGATE_NOT_READY")
    try:
        nine = validate_nine_reuse(repo, matrix)
    except (FileNotFoundError, ValueError):
        nine = {}
        blockers.append("NINE_FINAL_QC_OR_HDF_REUSE_NOT_READY")
    try:
        remainder = validate_remainder_layout(repo, matrix)
    except ValueError:
        remainder = {}
        blockers.append("REMAINDER_31_ATOMIC_HDF_NOT_READY")
    return {
        "status": "FULL40_FINALIZATION_READY" if not blockers else "FULL40_FINALIZATION_BLOCKED",
        "profile": PROFILE,
        "blockers": blockers,
        "split_manifests": manifests,
        "verified_nine_reuse_count": len(nine),
        "remainder_staging_count": len(remainder),
        "normalization_inputs": manifests["train"],
        "frozen_seal_inputs": manifests["frozen_test"],
        "execution_reviewed": EXECUTION_REVIEWED,
    }


def write_json(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def finalize(repo: Path) -> None:
    matrix = load_matrix(repo)
    manifests = split_manifests(matrix)
    curator = load_remainder_curator(repo)
    raw_aggregate_sha256 = curator.validate_raw_aggregate(repo, matrix)
    sources = validate_nine_reuse(repo, matrix)
    remainder = validate_remainder_layout(repo, matrix)
    force_validator = load_existing_finalizer(repo)
    audits = {}
    for name, path in remainder.items():
        raw_receipt_sha256 = curator.validate_raw_receipt_files(
            repo, name, matrix[name]
        )
        curator.validate_vtk_receipt(
            repo, name, raw_receipt_sha256, matrix[name]["split"]
        )
        audits[name] = validate_remainder_hdf(
            repo, name, matrix[name], path, force_validator
        )
    nine_root = repo / NINE_FINAL.relative_to(REPO)
    nine_source = {
        "manifest": str((nine_root / "manifest.json").relative_to(repo)),
        "manifest_sha256": sha256(nine_root / "manifest.json"),
        "commissioning_qc": str(
            (nine_root / "commissioning_qc.json").relative_to(repo)
        ),
        "commissioning_qc_sha256": sha256(nine_root / "commissioning_qc.json"),
    }
    sources.update(remainder)
    final = repo / FINAL.relative_to(REPO)
    temporary = final.with_name(f".{final.name}.tmp")
    if final.exists() or temporary.exists():
        raise FileExistsError("refusing existing full40 final/temporary dataset")
    for split, names in manifests.items():
        (temporary / split).mkdir(parents=True, exist_ok=True)
        for name in names:
            os.link(sources[name], temporary / split / f"{name}.h5")
    (temporary / "splits").mkdir()
    for split, names in manifests.items():
        split_path = temporary / "splits" / f"{split}.json"
        write_json(
            split_path,
            {
                "split": split,
                "cases": names,
                "hdf5_sha256": {
                    name: sha256(temporary / split / f"{name}.h5") for name in names
                },
            },
        )
    normalization = compute_train_normalization(
        [temporary / "train" / f"{name}.h5" for name in manifests["train"]]
    )
    normalization["train_split_manifest_sha256"] = sha256(
        temporary / "splits/train.json"
    )
    write_json(temporary / "normalization.json", normalization)
    write_json(
        temporary / "frozen_test_seal.json",
        {
            "status": "FROZEN_TEST_SEALED_NO_MODEL_ACCESS",
            "cases": manifests["frozen_test"],
            "split_manifest_sha256": sha256(
                temporary / "splits/frozen_test.json"
            ),
            "forbidden": ["normalization", "selection", "ranking", "reporting"],
        },
    )
    write_json(
        temporary / "manifest.json",
        {
            "schema_version": 1,
            "profile": PROFILE,
            "full40_predeclaration_sha256": PREDECLARATION_SHA256,
            "trajectory_counts": EXPECTED_COUNTS,
            "frames_per_trajectory": 801,
            "pairs_per_trajectory": 800,
            "grid": {"nx": 256, "ny": 128, "x_range": [8, 25], "y_range": [4, 11]},
            "fields": [
                "u",
                "v",
                "gauge_pressure",
                "valid_mask",
                "rear_omega",
                "front_cd_cl",
                "rear_cd_cl",
            ],
            "pressure_preprocessing": (
                "subtract valid-domain spatial mean independently at every frame"
            ),
            "max_abs_omega": MAX_ABS_OMEGA,
            "normalization": "normalization.json; explicit train manifest only",
            "normalization_sha256": sha256(temporary / "normalization.json"),
            "frozen_test_seal_sha256": sha256(temporary / "frozen_test_seal.json"),
            "training_use": "train only; validation for model selection; frozen_test sealed",
            "curator_pipeline": [
                "PhysicsNeMo Curator Source[VTKSource + Mesh/BVH]",
                "NumericalQualityFilter",
                "atomic TrajectoryHDF5Sink",
            ],
            "split_manifests": {
                split: {
                    "path": f"splits/{split}.json",
                    "sha256": sha256(temporary / "splits" / f"{split}.json"),
                }
                for split in SPLITS
            },
            "nine_reused_after_final_qc": sorted(set(sources) - set(remainder)),
            "nine_reuse_source": nine_source,
            "remainder_hdf_qc": audits,
            "remainder_31_raw_aggregate_sha256": raw_aggregate_sha256,
        },
    )
    os.replace(temporary, final)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approval-token")
    args = parser.parse_args()
    repo = args.repo.resolve()
    report = readiness(repo)
    if not args.execute:
        print(json.dumps(report, indent=2))
        return
    if report["status"] != "FULL40_FINALIZATION_READY":
        parser.error(f"finalization prerequisites are incomplete: {report['blockers']}")
    if not EXECUTION_REVIEWED or args.approval_token != EXECUTION_TOKEN:
        parser.error("full40 finalization execution is not reviewed/enabled")
    finalize(repo)


if __name__ == "__main__":
    main()
