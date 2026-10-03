from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_dev30_full40_promotion.py"
if not SCRIPT.exists():
    SCRIPT = ROOT / "verify_dev30_full40_promotion.py"
SPEC = importlib.util.spec_from_file_location("promotion", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def fixture(tmp_path: Path, monkeypatch):
    cases = {}
    for split, phases in MODULE.EXPECTED_PHASES.items():
        per_phase = MODULE.EXPECTED_COUNTS[split] // len(phases)
        for phase in phases:
            for index in range(per_phase):
                name = f"case_{split}_b{phase:02d}_{index}"
                cases[name] = {"split": split, "phase_bin": phase}
    for phase in (3, 7):
        for index in range(5):
            name = f"case_frozen_test_b{phase:02d}_{index}"
            cases[name] = {"split": "frozen_test", "phase_bin": phase}
    predeclaration = tmp_path / "predeclaration.json"
    write(predeclaration, {"cases": cases})
    predeclaration_sha = digest(predeclaration)
    monkeypatch.setattr(MODULE, "PREDECLARATION_SHA256", predeclaration_sha)

    dev = tmp_path / "dev30"
    formal = tmp_path / "full40"
    split_declarations = {}
    for split in MODULE.EXPECTED_COUNTS:
        names = sorted(name for name, row in cases.items() if row["split"] == split)
        hashes = {}
        for name in names:
            content = f"real-{split}-{name}".encode()
            roots = (formal,) if split == "frozen_test" else (dev, formal)
            for root in roots:
                path = root / split / f"{name}.h5"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            hashes[name] = hashlib.sha256(content).hexdigest()
        document = {"split": split, "cases": names, "hdf5_sha256": hashes}
        roots = (formal,) if split == "frozen_test" else (dev, formal)
        for root in roots:
            path = root / "splits" / f"{split}.json"
            write(path, document)
        split_manifest_root = formal if split == "frozen_test" else dev
        split_declarations[split] = {
            "path": f"splits/{split}.json",
            "sha256": digest(split_manifest_root / "splits" / f"{split}.json"),
        }

    normalization = {
        "computed_from": "train split only",
        "source_scope": "explicit 20-case train manifest only",
        "state_channels": ["u", "v", "gauge_pressure"],
        "state_mean": [0.0, 0.0, 0.0],
        "state_std": [1.0, 1.0, 1.0],
        "state_abs_normalized_channel_max_train": [2.0, 3.0, 4.0],
        "state_abs_normalized_max_train": 4.0,
        "state_support_computed_from": "exact train split valid cells",
        "force_channels": ["rear_cd", "rear_cl"],
        "force_mean": [0.0, 0.0],
        "force_std": [1.0, 1.0],
        "all_force_channels": ["front_cd", "front_cl", "rear_cd", "rear_cl"],
        "all_force_mean": [0.0] * 4,
        "all_force_std": [1.0] * 4,
        "train_split_manifest_sha256": split_declarations["train"]["sha256"],
    }
    for root in (dev, formal):
        write(root / "normalization.json", normalization)

    frozen_names = sorted(name for name, row in cases.items() if row["split"] == "frozen_test")
    dev_seal = {
        "status": "FROZEN_TEST_DECLARED_NOT_MATERIALIZED_NO_MODEL_ACCESS",
        "cases": frozen_names,
        "materialized_in_release": False,
    }
    formal_seal = {
        "status": "FROZEN_TEST_SEALED_NO_MODEL_ACCESS",
        "cases": frozen_names,
    }
    write(dev / "frozen_test_seal.json", dev_seal)
    write(formal / "frozen_test_seal.json", formal_seal)
    common = {
        "profile": "matched_start_full40_v1",
        "full40_predeclaration_sha256": predeclaration_sha,
        "max_abs_omega": 0.75,
        "normalization_sha256": digest(dev / "normalization.json"),
        "split_manifests": {key: value for key, value in split_declarations.items() if key != "frozen_test"},
    }
    write(
        dev / "manifest.json",
        {
            **common,
            "release_kind": "immutable_development_train20_validation10",
            "declared_trajectory_counts": {"train": 20, "validation": 10, "frozen_test": 10},
            "materialized_trajectory_counts": {"train": 20, "validation": 10},
            "frozen_test_materialized": False,
            "frozen_test_directory_present": False,
            "frozen_test_seal_sha256": digest(dev / "frozen_test_seal.json"),
        },
    )
    write(
        formal / "manifest.json",
        {
            **common,
            "trajectory_counts": {"train": 20, "validation": 10, "frozen_test": 10},
            "frozen_test_seal_sha256": digest(formal / "frozen_test_seal.json"),
        },
    )
    return dev, formal, predeclaration


def test_identical_development_inputs_pass_without_frozen_hdf_access(tmp_path, monkeypatch) -> None:
    dev, formal, predeclaration = fixture(tmp_path, monkeypatch)
    report = MODULE.verify(dev, formal, predeclaration)
    assert report["status"] == "DEV30_FULL40_PROMOTION_PASS"
    assert report["formal_gate_input_authorized"] is True
    assert report["ppo_identity_prerequisite_passed"] is True
    assert report["ppo_authorized"] is False
    assert report["frozen_hdf_opened_or_enumerated"] is False


def test_changed_validation_hdf_fails_closed(tmp_path, monkeypatch) -> None:
    dev, formal, predeclaration = fixture(tmp_path, monkeypatch)
    target = next((formal / "validation").glob("*.h5"))
    target.write_bytes(b"changed")
    report = MODULE.verify(dev, formal, predeclaration)
    assert report["status"] == "DEV30_FULL40_PROMOTION_FAIL"
    assert any("VALIDATION_PROMOTION_MISMATCH" in error for error in report["errors"])


def test_new_normalization_field_requires_review(tmp_path, monkeypatch) -> None:
    dev, formal, predeclaration = fixture(tmp_path, monkeypatch)
    normalization_path = formal / "normalization.json"
    normalization = json.loads(normalization_path.read_text())
    normalization["new_definition"] = "not silently accepted"
    write(normalization_path, normalization)
    formal_manifest_path = formal / "manifest.json"
    manifest = json.loads(formal_manifest_path.read_text())
    manifest["normalization_sha256"] = digest(normalization_path)
    write(formal_manifest_path, manifest)
    report = MODULE.verify(dev, formal, predeclaration)
    assert report["status"] == "DEV30_FULL40_PROMOTION_FAIL"
    assert any("NORMALIZATION_PROMOTION_MISMATCH" in error for error in report["errors"])


def test_dev_frozen_directory_presence_fails_closed(tmp_path, monkeypatch) -> None:
    dev, formal, predeclaration = fixture(tmp_path, monkeypatch)
    (dev / "frozen_test").mkdir()
    report = MODULE.verify(dev, formal, predeclaration)
    assert report["status"] == "DEV30_FULL40_PROMOTION_FAIL"
    assert "DEV30_FROZEN_DIRECTORY_PRESENT" in report["errors"]
