#!/usr/bin/env python3
"""Frozen-blind preflight for official PhysicsNeMo training on immutable dev30."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
DATA = Path("data/curated/tandem_cylinders_matched_start_full40_dev30_v1")
PREDECLARATION_SHA256 = "d7ff174ef10194a8739357376335ca13ff9b45c8079970846bb71f15a715d24b"
IMAGE = "fluid-control-physicsnemo:2.2.2"
IMAGE_ID = "sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
FORCE_CHANNELS = ["front_cd", "front_cl", "rear_cd", "rear_cl"]
EXPECTED_PHASES = {"train": {0, 2, 4, 6}, "validation": {1, 5}}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def valid_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def validate_normalization(stats: dict, train_manifest_sha: str) -> None:
    if (
        stats.get("computed_from") != "train split only"
        or stats.get("source_scope") != "explicit 20-case train manifest only"
        or stats.get("train_split_manifest_sha256") != train_manifest_sha
        or stats.get("state_channels") != ["u", "v", "gauge_pressure"]
        or stats.get("all_force_channels") != FORCE_CHANNELS
        or stats.get("force_channels") != ["rear_cd", "rear_cl"]
    ):
        raise ValueError("train-only normalization contract differs")
    for key, count in (("state_mean", 3), ("state_std", 3), ("all_force_mean", 4), ("all_force_std", 4)):
        values = stats.get(key)
        if not isinstance(values, list) or len(values) != count:
            raise ValueError(f"normalization shape differs: {key}")
        converted = [float(value) for value in values]
        if not all(math.isfinite(value) for value in converted):
            raise ValueError(f"normalization is non-finite: {key}")
        if key.endswith("_std") and not all(value > 0 for value in converted):
            raise ValueError(f"normalization std differs: {key}")
    if stats.get("force_mean") != stats["all_force_mean"][2:4] or stats.get("force_std") != stats["all_force_std"][2:4]:
        raise ValueError("rear-force aliases differ")
    channel_max = stats.get("state_abs_normalized_channel_max_train")
    maximum = stats.get("state_abs_normalized_max_train")
    if (
        not isinstance(channel_max, list)
        or len(channel_max) != 3
        or not all(float(value) > 0 and math.isfinite(float(value)) for value in channel_max)
        or not isinstance(maximum, (int, float))
        or not math.isclose(float(maximum), max(map(float, channel_max)))
    ):
        raise ValueError("train state-support contract differs")


def build_plan(repo: Path = REPO) -> dict:
    blockers = []
    root = repo / DATA
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return {"status": "FULL40_DEV30_FNO_RETRAIN_BLOCKED", "blockers": ["DEV30_MANIFEST_MISSING"]}
    try:
        manifest = load_json(manifest_path)
    except (OSError, TypeError, ValueError):
        return {"status": "FULL40_DEV30_FNO_RETRAIN_BLOCKED", "blockers": ["DEV30_MANIFEST_INVALID"]}
    if (
        manifest.get("profile") != "matched_start_full40_v1"
        or manifest.get("release_kind") != "immutable_development_train20_validation10"
        or manifest.get("full40_predeclaration_sha256") != PREDECLARATION_SHA256
        or manifest.get("declared_trajectory_counts") != {"train": 20, "validation": 10, "frozen_test": 10}
        or manifest.get("materialized_trajectory_counts") != {"train": 20, "validation": 10}
        or manifest.get("frozen_test_materialized") is not False
        or manifest.get("frozen_test_directory_present") is not False
        or not math.isclose(float(manifest.get("max_abs_omega", -1)), 0.75)
    ):
        blockers.append("DEV30_MANIFEST_CONTRACT_MISMATCH")
    if (root / "frozen_test").exists():
        blockers.append("FROZEN_DIRECTORY_MUST_BE_ABSENT")
    seal_path = root / "frozen_test_seal.json"
    if not seal_path.is_file() or sha256(seal_path) != manifest.get("frozen_test_seal_sha256"):
        blockers.append("FROZEN_DECLARATION_SEAL_MISMATCH")
    else:
        seal = load_json(seal_path)
        if (
            seal.get("status") != "FROZEN_TEST_DECLARED_NOT_MATERIALIZED_NO_MODEL_ACCESS"
            or len(seal.get("cases", [])) != 10
            or seal.get("phase_bins") != [3, 7]
            or seal.get("materialized_in_release") is not False
        ):
            blockers.append("FROZEN_DECLARATION_SEAL_CONTRACT_MISMATCH")

    split_hashes = {}
    for split, count in (("train", 20), ("validation", 10)):
        declared = manifest.get("split_manifests", {}).get(split, {})
        relative = Path(declared.get("path", ""))
        path = root / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not path.is_file()
            or sha256(path) != declared.get("sha256")
        ):
            blockers.append(f"{split.upper()}_MANIFEST_MISMATCH")
            continue
        document = load_json(path)
        names = document.get("cases")
        hashes = document.get("hdf5_sha256")
        if (
            document.get("split") != split
            or not isinstance(names, list)
            or len(names) != count
            or names != sorted(names)
            or not isinstance(hashes, dict)
            or set(hashes) != set(names)
            or not all(valid_sha(value) for value in hashes.values())
        ):
            blockers.append(f"{split.upper()}_MANIFEST_CONTRACT_MISMATCH")
            continue
        actual_names = sorted(path.stem for path in (root / split).glob("*.h5"))
        if actual_names != names:
            blockers.append(f"{split.upper()}_HDF_LAYOUT_MISMATCH")
        elif any(sha256(root / split / f"{name}.h5") != hashes[name] for name in names):
            blockers.append(f"{split.upper()}_HDF_SHA256_MISMATCH")
        split_hashes[split] = declared["sha256"]

    normalization_path = root / "normalization.json"
    if not normalization_path.is_file() or sha256(normalization_path) != manifest.get("normalization_sha256"):
        blockers.append("TRAIN20_NORMALIZATION_SHA256_MISMATCH")
    elif "train" in split_hashes:
        try:
            validate_normalization(load_json(normalization_path), split_hashes["train"])
        except (TypeError, ValueError):
            blockers.append("TRAIN20_NORMALIZATION_CONTRACT_MISMATCH")

    for config_name, expected_fraction in (("tandem_fno_full40_onestep.yaml", 0.20), ("tandem_fno_full40_h20.yaml", 0.15)):
        path = repo / "conf" / config_name
        try:
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            training = config["training"]
            if float(training["gpu_memory_fraction"]) != expected_fraction:
                raise ValueError
        except (FileNotFoundError, KeyError, TypeError, ValueError, yaml.YAMLError):
            blockers.append(f"CONFIG_CONTRACT_MISMATCH:{config_name}")
    return {
        "status": "FULL40_DEV30_FNO_RETRAIN_READY" if not blockers else "FULL40_DEV30_FNO_RETRAIN_BLOCKED",
        "blockers": blockers,
        "data_release": str(DATA),
        "manifest_sha256": sha256(manifest_path),
        "development_split_manifest_sha256": split_hashes,
        "frozen_policy": "declared seal only; no frozen directory exists or is mounted",
        "training_container_mount_policy": "readonly scripts/src/conf/dev30 plus one dedicated writable output",
        "image": IMAGE,
        "image_id": IMAGE_ID,
        "minimum_mem_available_gib": 20,
    }


if __name__ == "__main__":
    print(json.dumps(build_plan(), indent=2))
