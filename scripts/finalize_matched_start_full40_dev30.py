#!/usr/bin/env python3
"""Atomically publish immutable full40 train20+validation10 development data."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PREDECLARATION = Path(
    "artifacts/tandem_cylinders/matched_start_full40_predeclared_20261003.json"
)
PREDECLARATION_SHA256 = "d7ff174ef10194a8739357376335ca13ff9b45c8079970846bb71f15a715d24b"
NINE_FINAL = Path("data/curated/tandem_cylinders_matched_start_commissioning_train9_v1")
REMAINDER_STAGING = Path("data/curated/.staging/matched_start_full40_v1")
FINAL = Path("data/curated/tandem_cylinders_matched_start_full40_dev30_v1")
PROFILE = "matched_start_full40_v1"
RELEASE_KIND = "immutable_development_train20_validation10"
EXPECTED_COUNTS = {"train": 20, "validation": 10, "frozen_test": 10}
EXPECTED_PHASES = {"train": {0, 2, 4, 6}, "validation": {1, 5}, "frozen_test": {3, 7}}
MAX_ABS_OMEGA = 0.75
EXECUTION_TOKEN = "FINALIZE_REVIEWED_MATCHED_START_FULL40_DEV30"


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


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def case_contract(predeclaration: dict) -> dict[str, list[str]]:
    cases = predeclaration.get("cases", {})
    result = {
        split: sorted(name for name, row in cases.items() if row.get("split") == split)
        for split in EXPECTED_COUNTS
    }
    if len(cases) != 40 or {key: len(value) for key, value in result.items()} != EXPECTED_COUNTS:
        raise ValueError("full40 predeclared case counts differ")
    for split, names in result.items():
        if {int(cases[name]["phase_bin"]) for name in names} != EXPECTED_PHASES[split]:
            raise ValueError(f"full40 predeclared phase isolation differs: {split}")
    return result


def frozen_seal(predeclaration: dict, names: list[str]) -> dict:
    cases = predeclaration["cases"]
    return {
        "status": "FROZEN_TEST_DECLARED_NOT_MATERIALIZED_NO_MODEL_ACCESS",
        "full40_predeclaration_sha256": PREDECLARATION_SHA256,
        "cases": names,
        "phase_bins": sorted({int(cases[name]["phase_bin"]) for name in names}),
        "source_state_sha256": {
            name: cases[name]["source_state_sha256"] for name in names
        },
        "materialized_in_release": False,
        "forbidden": ["filesystem_mount", "enumeration", "normalization", "selection", "ranking", "reporting"],
    }


def development_sources(repo: Path, predeclaration: dict, contract: dict[str, list[str]]) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    nine_root = repo / NINE_FINAL
    nine_manifest = load_json(nine_root / "manifest.json")
    nine_qc = load_json(nine_root / "commissioning_qc.json")
    if (
        nine_manifest.get("training_use") != "FORBIDDEN"
        or nine_qc.get("status") != "MATCHED_START_COMMISSIONING_NINE_CASE_QC_OK"
        or nine_qc.get("training_use") != "FORBIDDEN_COMMISSIONING_ONLY"
    ):
        raise ValueError("nine-case commissioning provenance differs")
    nine_hashes = nine_manifest.get("hdf5_sha256", {})
    expected_nine = {
        name
        for name, row in predeclaration["cases"].items()
        if row.get("disposition") == "existing_nine_case_commissioning"
    }
    if len(expected_nine) != 9 or set(nine_hashes) != expected_nine:
        raise ValueError("nine-case reuse set differs")
    for name in expected_nine:
        path = nine_root / "train" / f"{name}.h5"
        if not path.is_file() or sha256(path) != nine_hashes[name]:
            raise ValueError(f"nine-case HDF differs: {name}")
        sources[name] = path

    expected_development = set(contract["train"] + contract["validation"])
    for name in sorted(expected_development - expected_nine):
        split = predeclaration["cases"][name]["split"]
        path = repo / REMAINDER_STAGING / name / split / f"{name}.h5"
        if not path.is_file() or path.with_suffix(".h5.tmp").exists():
            raise FileNotFoundError(f"atomic development HDF is not ready: {name}")
        sources[name] = path
    if set(sources) != expected_development:
        raise ValueError("development source set differs")
    return sources


def readiness(repo: Path = REPO) -> dict:
    blockers = []
    path = repo / PREDECLARATION
    if not path.is_file() or sha256(path) != PREDECLARATION_SHA256:
        return {"status": "FULL40_DEV30_RELEASE_BLOCKED", "blockers": ["PREDECLARATION_MISMATCH"]}
    predeclaration = load_json(path)
    try:
        contract = case_contract(predeclaration)
    except (TypeError, ValueError):
        return {"status": "FULL40_DEV30_RELEASE_BLOCKED", "blockers": ["CASE_CONTRACT_MISMATCH"]}
    try:
        sources = development_sources(repo, predeclaration, contract)
    except (FileNotFoundError, TypeError, ValueError):
        sources = {}
        blockers.append("DEVELOPMENT_30_HDF_OR_QC_NOT_READY")
    final = repo / FINAL
    temporary = final.with_name(f".{final.name}.tmp")
    if final.exists() or temporary.exists():
        blockers.append("DEV30_FINAL_OR_TEMPORARY_ALREADY_EXISTS")
    return {
        "status": "FULL40_DEV30_RELEASE_READY" if not blockers else "FULL40_DEV30_RELEASE_BLOCKED",
        "blockers": blockers,
        "profile": PROFILE,
        "release_kind": RELEASE_KIND,
        "materialized_counts": {"train": 20, "validation": 10},
        "declared_frozen_count": 10,
        "verified_source_count": len(sources),
        "frozen_hdf_opened_or_enumerated": False,
        "output": str(FINAL),
    }


def write_json(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def copy_exclusive(source: Path, target: Path) -> None:
    with source.open("rb") as incoming, target.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, length=8 << 20)
        outgoing.flush()
        os.fsync(outgoing.fileno())


def make_read_only(root: Path) -> None:
    for path in root.rglob("*"):
        os.chmod(path, 0o555 if path.is_dir() else 0o444)
    os.chmod(root, 0o555)


def finalize(repo: Path) -> None:
    predeclaration_path = repo / PREDECLARATION
    if sha256(predeclaration_path) != PREDECLARATION_SHA256:
        raise ValueError("full40 predeclaration SHA differs")
    predeclaration = load_json(predeclaration_path)
    contract = case_contract(predeclaration)
    sources = development_sources(repo, predeclaration, contract)
    full = load_module(repo / "scripts/finalize_matched_start_full40.py", "dev30_full_validator")
    curator = load_module(repo / "scripts/curate_matched_start_full40_remainder.py", "dev30_curator_validator")
    force_validator = full.load_existing_finalizer(repo)
    audits = {}
    nine_names = set(load_json(repo / NINE_FINAL / "manifest.json")["hdf5_sha256"])
    for name in sorted(set(sources) - nine_names):
        row = predeclaration["cases"][name]
        receipt_sha = curator.validate_raw_receipt_files(repo, name, row)
        curator.validate_vtk_receipt(repo, name, receipt_sha, row["split"])
        audits[name] = full.validate_remainder_hdf(
            repo, name, row, sources[name], force_validator
        )

    final = repo / FINAL
    temporary = final.with_name(f".{final.name}.tmp")
    if final.exists() or temporary.exists():
        raise FileExistsError("refusing existing dev30 final/temporary release")
    for split in ("train", "validation"):
        (temporary / split).mkdir(parents=True, exist_ok=False)
        for name in contract[split]:
            copy_exclusive(sources[name], temporary / split / f"{name}.h5")
    (temporary / "splits").mkdir()
    split_documents = {}
    for split in ("train", "validation"):
        document = {
            "split": split,
            "cases": contract[split],
            "hdf5_sha256": {
                name: sha256(temporary / split / f"{name}.h5")
                for name in contract[split]
            },
        }
        path = temporary / "splits" / f"{split}.json"
        write_json(path, document)
        split_documents[split] = {
            "path": f"splits/{split}.json",
            "sha256": sha256(path),
        }
    normalization = full.compute_train_normalization(
        [temporary / "train" / f"{name}.h5" for name in contract["train"]]
    )
    normalization["train_split_manifest_sha256"] = split_documents["train"]["sha256"]
    write_json(temporary / "normalization.json", normalization)
    seal = frozen_seal(predeclaration, contract["frozen_test"])
    write_json(temporary / "frozen_test_seal.json", seal)
    nine_root = repo / NINE_FINAL
    manifest = {
        "schema_version": 1,
        "profile": PROFILE,
        "release_kind": RELEASE_KIND,
        "full40_predeclaration_sha256": PREDECLARATION_SHA256,
        "declared_trajectory_counts": EXPECTED_COUNTS,
        "materialized_trajectory_counts": {"train": 20, "validation": 10},
        "frames_per_trajectory": 801,
        "pairs_per_trajectory": 800,
        "grid": {"nx": 256, "ny": 128, "x_range": [8, 25], "y_range": [4, 11]},
        "max_abs_omega": MAX_ABS_OMEGA,
        "normalization": "train20 only",
        "normalization_sha256": sha256(temporary / "normalization.json"),
        "split_manifests": split_documents,
        "frozen_test_seal_sha256": sha256(temporary / "frozen_test_seal.json"),
        "frozen_test_materialized": False,
        "frozen_test_directory_present": False,
        "training_use": "train only; validation selection; frozen unavailable and sealed",
        "nine_reuse_source": {
            "manifest_sha256": sha256(nine_root / "manifest.json"),
            "commissioning_qc_sha256": sha256(nine_root / "commissioning_qc.json"),
        },
        "remainder_development_hdf_qc": audits,
        "future_full40_promotion_contract": (
            "train/validation case lists and every HDF SHA256 must match this release byte-for-byte"
        ),
    }
    write_json(temporary / "manifest.json", manifest)
    make_read_only(temporary)
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
    if report["status"] != "FULL40_DEV30_RELEASE_READY":
        parser.error(f"dev30 release prerequisites incomplete: {report['blockers']}")
    if args.approval_token != EXECUTION_TOKEN:
        parser.error("reviewed dev30 finalization approval token is required")
    finalize(repo)


if __name__ == "__main__":
    main()
