#!/usr/bin/env python3
"""Verify byte-identical dev inputs before formal full40 Gate/PPO use."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEV30 = REPO / "data/curated/tandem_cylinders_matched_start_full40_dev30_v1"
FULL40 = REPO / "data/curated/tandem_cylinders_matched_start_full40_v1"
PREDECLARATION = (
    REPO / "artifacts/tandem_cylinders/matched_start_full40_predeclared_20261003.json"
)
PREDECLARATION_SHA256 = "d7ff174ef10194a8739357376335ca13ff9b45c8079970846bb71f15a715d24b"
EXPECTED_PHASES = {"train": {0, 2, 4, 6}, "validation": {1, 5}}
EXPECTED_COUNTS = {"train": 20, "validation": 10}
NORMALIZATION_KEYS = {
    "computed_from",
    "source_scope",
    "state_channels",
    "state_mean",
    "state_std",
    "state_abs_normalized_channel_max_train",
    "state_abs_normalized_max_train",
    "state_support_computed_from",
    "force_channels",
    "force_mean",
    "force_std",
    "all_force_channels",
    "all_force_mean",
    "all_force_std",
    "train_split_manifest_sha256",
}


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


def split_document(root: Path, manifest: dict, split: str) -> tuple[dict, str]:
    declaration = manifest.get("split_manifests", {}).get(split, {})
    relative = Path(declaration.get("path", ""))
    path = root / relative
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not path.is_file()
        or sha256(path) != declaration.get("sha256")
    ):
        raise ValueError(f"{root.name}/{split} split manifest differs")
    return load_json(path), declaration["sha256"]


def validate_release_manifest(dev: dict, formal: dict) -> None:
    common = {
        "profile": "matched_start_full40_v1",
        "full40_predeclaration_sha256": PREDECLARATION_SHA256,
        "max_abs_omega": 0.75,
    }
    for key, value in common.items():
        if dev.get(key) != value or formal.get(key) != value:
            raise ValueError(f"release manifest differs: {key}")
    if (
        dev.get("release_kind") != "immutable_development_train20_validation10"
        or dev.get("declared_trajectory_counts")
        != {"train": 20, "validation": 10, "frozen_test": 10}
        or dev.get("materialized_trajectory_counts") != EXPECTED_COUNTS
        or dev.get("frozen_test_materialized") is not False
        or dev.get("frozen_test_directory_present") is not False
        or formal.get("trajectory_counts")
        != {"train": 20, "validation": 10, "frozen_test": 10}
    ):
        raise ValueError("release materialization/count contract differs")


def verify(
    dev_root: Path,
    formal_root: Path,
    predeclaration_path: Path,
) -> dict:
    errors = []
    details: dict = {}
    if sha256(predeclaration_path) != PREDECLARATION_SHA256:
        return {
            "status": "DEV30_FULL40_PROMOTION_FAIL",
            "errors": ["PREDECLARATION_SHA256_MISMATCH"],
            "frozen_hdf_opened_or_enumerated": False,
        }
    predeclaration = load_json(predeclaration_path)
    cases = predeclaration.get("cases", {})
    try:
        dev_manifest = load_json(dev_root / "manifest.json")
        formal_manifest = load_json(formal_root / "manifest.json")
        validate_release_manifest(dev_manifest, formal_manifest)
    except (FileNotFoundError, TypeError, ValueError) as error:
        return {
            "status": "DEV30_FULL40_PROMOTION_FAIL",
            "errors": [f"RELEASE_MANIFEST_MISMATCH:{error}"],
            "frozen_hdf_opened_or_enumerated": False,
        }

    if (dev_root / "frozen_test").exists():
        errors.append("DEV30_FROZEN_DIRECTORY_PRESENT")
    try:
        dev_seal_path = dev_root / "frozen_test_seal.json"
        formal_seal_path = formal_root / "frozen_test_seal.json"
        if (
            sha256(dev_seal_path) != dev_manifest.get("frozen_test_seal_sha256")
            or sha256(formal_seal_path)
            != formal_manifest.get("frozen_test_seal_sha256")
        ):
            raise ValueError("seal digest differs")
        dev_seal = load_json(dev_seal_path)
        formal_seal = load_json(formal_seal_path)
        if (
            dev_seal.get("status")
            != "FROZEN_TEST_DECLARED_NOT_MATERIALIZED_NO_MODEL_ACCESS"
            or dev_seal.get("materialized_in_release") is not False
            or formal_seal.get("status") != "FROZEN_TEST_SEALED_NO_MODEL_ACCESS"
            or dev_seal.get("cases") != formal_seal.get("cases")
        ):
            raise ValueError("seal contract differs")
        details["frozen_case_declaration_count"] = len(dev_seal["cases"])
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        errors.append(f"FROZEN_SEAL_METADATA_MISMATCH:{error}")

    split_details = {}
    for split, expected_count in EXPECTED_COUNTS.items():
        try:
            dev_split, dev_split_sha = split_document(
                dev_root, dev_manifest, split
            )
            formal_split, formal_split_sha = split_document(
                formal_root, formal_manifest, split
            )
            dev_names = dev_split.get("cases")
            formal_names = formal_split.get("cases")
            if (
                dev_split.get("split") != split
                or formal_split.get("split") != split
                or dev_names != formal_names
                or not isinstance(dev_names, list)
                or len(dev_names) != expected_count
                or dev_names != sorted(dev_names)
            ):
                raise ValueError("case list differs")
            phases = {int(cases[name]["phase_bin"]) for name in dev_names}
            if phases != EXPECTED_PHASES[split]:
                raise ValueError("phase set differs")
            dev_hashes = dev_split.get("hdf5_sha256")
            formal_hashes = formal_split.get("hdf5_sha256")
            if dev_hashes != formal_hashes or set(dev_hashes or {}) != set(dev_names):
                raise ValueError("declared HDF SHA map differs")
            for name in dev_names:
                expected = dev_hashes[name]
                if (
                    sha256(dev_root / split / f"{name}.h5") != expected
                    or sha256(formal_root / split / f"{name}.h5") != expected
                ):
                    raise ValueError(f"actual HDF SHA differs: {name}")
            if dev_split_sha != formal_split_sha:
                raise ValueError("split manifest byte SHA differs")
            split_details[split] = {
                "cases": expected_count,
                "phase_bins": sorted(phases),
                "split_manifest_sha256": dev_split_sha,
                "hdf_sha_map_identical": True,
            }
        except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
            errors.append(f"{split.upper()}_PROMOTION_MISMATCH:{error}")
    details["splits"] = split_details

    try:
        dev_normalization_path = dev_root / "normalization.json"
        formal_normalization_path = formal_root / "normalization.json"
        dev_normalization_sha = sha256(dev_normalization_path)
        formal_normalization_sha = sha256(formal_normalization_path)
        if (
            dev_normalization_sha != dev_manifest.get("normalization_sha256")
            or formal_normalization_sha
            != formal_manifest.get("normalization_sha256")
            or dev_normalization_sha != formal_normalization_sha
        ):
            raise ValueError("normalization file SHA differs")
        dev_normalization = load_json(dev_normalization_path)
        formal_normalization = load_json(formal_normalization_path)
        if set(dev_normalization) != NORMALIZATION_KEYS:
            raise ValueError("dev30 normalization key set differs")
        if set(formal_normalization) != NORMALIZATION_KEYS:
            raise ValueError("formal normalization key set differs; review new definition")
        if dev_normalization != formal_normalization:
            raise ValueError("normalization definitions or values differ")
        if (
            dev_normalization.get("computed_from") != "train split only"
            or dev_normalization.get("source_scope")
            != "explicit 20-case train manifest only"
            or dev_normalization.get("train_split_manifest_sha256")
            != split_details.get("train", {}).get("split_manifest_sha256")
        ):
            raise ValueError("train-only normalization provenance differs")
        details["normalization"] = {
            "sha256": dev_normalization_sha,
            "definitions_and_values_identical": True,
            "key_set": sorted(NORMALIZATION_KEYS),
        }
    except (FileNotFoundError, TypeError, ValueError) as error:
        errors.append(f"NORMALIZATION_PROMOTION_MISMATCH:{error}")

    passed = not errors
    return {
        "status": (
            "DEV30_FULL40_PROMOTION_PASS"
            if passed
            else "DEV30_FULL40_PROMOTION_FAIL"
        ),
        "errors": errors,
        "formal_gate_input_authorized": passed,
        "ppo_identity_prerequisite_passed": passed,
        "ppo_authorized": False,
        "ppo_policy": (
            "promotion identity PASS is necessary but not sufficient; the formal full40 validation gate must independently pass"
        ),
        "dev30_root": str(dev_root),
        "full40_root": str(formal_root),
        "predeclaration_sha256": PREDECLARATION_SHA256,
        "frozen_hdf_opened_or_enumerated": False,
        "scope": (
            "promotion identity only; no frozen HDF metric, ranking, or model evaluation"
        ),
        "details": details,
    }


def write_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(payload, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev30", type=Path, default=DEV30)
    parser.add_argument("--full40", type=Path, default=FULL40)
    parser.add_argument("--predeclaration", type=Path, default=PREDECLARATION)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    report = verify(
        args.dev30.resolve(),
        args.full40.resolve(),
        args.predeclaration.resolve(),
    )
    write_exclusive(args.output.resolve(), report)
    print(json.dumps(report, indent=2))
    if report["status"] != "DEV30_FULL40_PROMOTION_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
