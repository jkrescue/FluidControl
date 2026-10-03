#!/usr/bin/env python3
"""Read-only, frozen-safe preflight for official PhysicsNeMo full40 retraining."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import yaml

PROFILE = "matched_start_full40_v1"
DATA = Path("data/curated/tandem_cylinders_matched_start_full40_v1")
PREDECLARATION = Path(
    "artifacts/tandem_cylinders/matched_start_full40_predeclared_20261003.json"
)
PREDECLARATION_SHA256 = "d7ff174ef10194a8739357376335ca13ff9b45c8079970846bb71f15a715d24b"
IMAGE = "fluid-control-physicsnemo:2.2.2"
IMAGE_ID = "sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
EXPECTED_COUNTS = {"train": 20, "validation": 10, "frozen_test": 10}
EXPECTED_PHASES = {"train": {0, 2, 4, 6}, "validation": {1, 5}, "frozen_test": {3, 7}}
FORCE_CHANNELS = ["front_cd", "front_cl", "rear_cd", "rear_cl"]
CONFIGS = (
    Path("conf/tandem_fno_full40_onestep.yaml"),
    Path("conf/tandem_fno_full40_h20.yaml"),
)
RESOURCE_EVIDENCE = {
    Path("artifacts/tandem_fno_total_drag_batch16_smoke_20261002/resolved_config.yaml"): "8bfdaead3359bb759665561a80633dead2f32f7c7b16d6c4c57aada5f573972e",
    Path("artifacts/tandem_fno_total_drag_batch16_smoke_20261002/runtime_metadata.json"): "629899a34b593e8cad800387c881e76989e04a229e56187fe81bae55c476351f",
    Path("artifacts/tandem_fno_total_drag_batch16_smoke_20261002/training_history.json"): "cb2e03a4d5b7af439f6fc11b7e222136e0f4085fbd1c15ce80170707845d445d",
    Path("artifacts/tandem_fno_control_gap_v4_h20_rear_drag_seed20261003_5epoch/resolved_config.yaml"): "8f5096e4ceee0873a16c292e3f1bbf91557589b728ac2d1cc049d8eed24b06e9",
    Path("artifacts/tandem_fno_control_gap_v4_h20_rear_drag_seed20261003_5epoch/runtime_metadata.json"): "94184844a3e2b8c1a29701fccc2626e15ac4f3443a997a628b455a4f3ca961bc",
    Path("artifacts/tandem_fno_control_gap_v4_h20_rear_drag_seed20261003_5epoch/training_history.json"): "7a923bb1f9574566793fe1c9930958aa5a1ed77c980265b7acb3996eef71d0bf",
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


def expected_cases(predeclaration: dict) -> dict[str, list[str]]:
    cases = predeclaration.get("cases", {})
    result = {
        split: sorted(name for name, row in cases.items() if row.get("split") == split)
        for split in EXPECTED_COUNTS
    }
    if {split: len(names) for split, names in result.items()} != EXPECTED_COUNTS:
        raise ValueError("predeclared full40 split counts differ")
    for split, names in result.items():
        phases = {int(cases[name]["phase_bin"]) for name in names}
        if phases != EXPECTED_PHASES[split]:
            raise ValueError(f"predeclared phase isolation differs: {split}")
    return result


def validate_normalization(stats: dict) -> None:
    if stats.get("computed_from") != "train split only":
        raise ValueError("normalization is not declared train-only")
    if stats.get("state_channels") != ["u", "v", "gauge_pressure"]:
        raise ValueError("state channel order differs")
    if stats.get("all_force_channels") != FORCE_CHANNELS:
        raise ValueError("all-force channel order is absent or differs")
    for key, count in (
        ("state_mean", 3),
        ("state_std", 3),
        ("all_force_mean", 4),
        ("all_force_std", 4),
    ):
        values = stats.get(key)
        if not isinstance(values, list) or len(values) != count:
            raise ValueError(f"normalization field differs: {key}")
        numbers = [float(value) for value in values]
        if not all(math.isfinite(value) for value in numbers):
            raise ValueError(f"normalization field is non-finite: {key}")
        if key.endswith("_std") and not all(value > 0 for value in numbers):
            raise ValueError(f"normalization standard deviation is invalid: {key}")
    channel_max = stats.get("state_abs_normalized_channel_max_train")
    state_max = stats.get("state_abs_normalized_max_train")
    if (
        not isinstance(channel_max, list)
        or len(channel_max) != 3
        or not all(math.isfinite(float(value)) and float(value) > 0 for value in channel_max)
        or not isinstance(state_max, (int, float))
        or not math.isfinite(float(state_max))
        or not math.isclose(float(state_max), max(map(float, channel_max)))
    ):
        raise ValueError("train state-support bound differs")
    if stats.get("force_channels") != ["rear_cd", "rear_cl"]:
        raise ValueError("rear-force alias channel order differs")
    if stats.get("force_mean") != stats["all_force_mean"][2:4] or stats.get(
        "force_std"
    ) != stats["all_force_std"][2:4]:
        raise ValueError("rear-force normalization aliases differ")


def validate_config(repo: Path, relative: Path) -> None:
    config = yaml.safe_load((repo / relative).read_text(encoding="utf-8"))
    if config.get("data", {}).get("root") != str(DATA):
        raise ValueError(f"config data root differs: {relative}")
    training = config.get("training", {})
    maximum_fraction = 0.15 if relative.name.endswith("h20.yaml") else 0.20
    if float(training.get("gpu_memory_fraction", 1.0)) > maximum_fraction:
        raise ValueError(f"GPU allocator cap exceeds evidence: {relative}")
    if relative.name.endswith("h20.yaml"):
        if training.get("rollout_steps") != 20:
            raise ValueError("H20 config does not use 20 rollout steps")
        if training.get("checkpoint_interval") != 1 or training.get(
            "checkpoint_keep_last"
        ) != training.get("epochs"):
            raise ValueError("H20 config must retain every epoch for validation selection")
    elif training.get("batch_size") != 16:
        raise ValueError("one-step batch differs from measured batch16 smoke")


def validate_resource_evidence(repo: Path) -> dict:
    for relative, expected in RESOURCE_EVIDENCE.items():
        path = repo / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"resource evidence differs: {relative}")
    one_step = yaml.safe_load(
        (repo / next(iter(RESOURCE_EVIDENCE))).read_text(encoding="utf-8")
    )
    one_training = one_step["training"]
    if (
        one_training.get("batch_size") != 16
        or one_training.get("gpu_memory_fraction") != 0.20
        or one_training.get("smoke") is not True
    ):
        raise ValueError("one-step resource evidence contract differs")
    h20_path = next(
        path for path in RESOURCE_EVIDENCE if "control_gap_v4_h20" in str(path) and path.name == "resolved_config.yaml"
    )
    h20 = yaml.safe_load((repo / h20_path).read_text(encoding="utf-8"))
    h20_training = h20["training"]
    if (
        h20_training.get("batch_size") != 4
        or h20_training.get("gpu_memory_fraction") != 0.15
        or h20_training.get("rollout_steps") != 20
    ):
        raise ValueError("H20 resource evidence contract differs")
    return {
        "one_step": "batch16 completed one smoke epoch at allocator 0.20; not measured at 0.15",
        "h20": "batch4 H20 completed five full epochs at allocator 0.15",
    }


def build_plan(repo: Path) -> dict:
    blockers = []
    predeclared_path = repo / PREDECLARATION
    if not predeclared_path.is_file() or sha256(predeclared_path) != PREDECLARATION_SHA256:
        return {"status": "FULL40_FNO_RETRAIN_BLOCKED", "blockers": ["PREDECLARATION_MISMATCH"]}
    predeclared = load_json(predeclared_path)
    expected = expected_cases(predeclared)
    data = repo / DATA
    manifest_path = data / "manifest.json"
    normalization_path = data / "normalization.json"
    if not manifest_path.is_file():
        blockers.append("FINAL_MANIFEST_NOT_READY")
    if not normalization_path.is_file():
        blockers.append("TRAIN20_NORMALIZATION_NOT_READY")
    manifest = load_json(manifest_path) if manifest_path.is_file() else {}
    if manifest:
        if manifest.get("profile") != PROFILE or manifest.get("trajectory_counts") != EXPECTED_COUNTS:
            blockers.append("FINAL_PROFILE_OR_COUNTS_DIFFER")
        if not math.isclose(float(manifest.get("max_abs_omega", -1)), 0.75):
            blockers.append("ACTION_SCALE_0P75_NOT_DECLARED")
    if normalization_path.is_file():
        try:
            validate_normalization(load_json(normalization_path))
        except (TypeError, ValueError):
            blockers.append("TRAIN20_NORMALIZATION_SCHEMA_DIFFER")

    # Training and validation manifests may be read for model development.
    # Frozen HDF5 and its split manifest are deliberately never opened here.
    split_contract = {}
    for split in ("train", "validation"):
        declared = manifest.get("split_manifests", {}).get(split, {})
        relative = Path(declared.get("path", ""))
        path = data / relative
        valid_path = (
            bool(relative.parts)
            and not relative.is_absolute()
            and ".." not in relative.parts
            and path.is_file()
        )
        if not valid_path or sha256(path) != declared.get("sha256"):
            blockers.append(f"{split.upper()}_SPLIT_MANIFEST_MISMATCH")
            continue
        document = load_json(path)
        names = document.get("cases")
        if document.get("split") != split or names != expected[split]:
            blockers.append(f"{split.upper()}_CASE_LIST_MISMATCH")
            continue
        files = sorted(path.stem for path in (data / split).glob("*.h5"))
        declared_hashes = document.get("hdf5_sha256", {})
        if files != names or set(declared_hashes) != set(names):
            blockers.append(f"{split.upper()}_HDF_LAYOUT_MISMATCH")
        elif any(
            sha256(data / split / f"{name}.h5") != declared_hashes[name]
            for name in names
        ):
            blockers.append(f"{split.upper()}_HDF_SHA256_MISMATCH")
        split_contract[split] = names

    seal_path = data / "frozen_test_seal.json"
    seal = load_json(seal_path) if seal_path.is_file() else {}
    if (
        seal.get("status") != "FROZEN_TEST_SEALED_NO_MODEL_ACCESS"
        or seal.get("cases") != expected["frozen_test"]
        or set(seal.get("forbidden", []))
        != {"normalization", "selection", "ranking", "reporting"}
    ):
        blockers.append("FROZEN_TEST_SEAL_MISMATCH")
    for config in CONFIGS:
        try:
            validate_config(repo, config)
        except (FileNotFoundError, TypeError, ValueError):
            blockers.append(f"CONFIG_MISMATCH:{config}")
    try:
        resource_evidence = validate_resource_evidence(repo)
    except (FileNotFoundError, StopIteration, TypeError, ValueError):
        resource_evidence = None
        blockers.append("RESOURCE_EVIDENCE_MISMATCH")
    return {
        "status": "FULL40_FNO_RETRAIN_READY_FOR_REVIEWED_RUN" if not blockers else "FULL40_FNO_RETRAIN_BLOCKED",
        "blockers": blockers,
        "data_profile": PROFILE,
        "split_counts": EXPECTED_COUNTS,
        "development_inputs": split_contract,
        "frozen_policy": "seal metadata checked; frozen split manifest and HDF5 never opened",
        "normalization_policy": "train split only; exact 20 cases enforced by split manifest and HDF hashes",
        "normalization_compatibility": "four-force and rear-force aliases plus train state-support bounds",
        "model": "official PhysicsNeMo 2.2.2 FNO; existing project architecture; no custom network",
        "stages": ["one-step 30 epochs from random initialization", "H20 rollout 10 epochs from one-step best"],
        "validation_gate": {
            "split": "validation",
            "horizons": [1, 10, 50, 100],
            "selection_metric": "pooled H100 endpoint total-Cd NRMSE",
            "maximum": 0.10,
            "secondary_metrics": ["macro/worst H100 NRMSE", "front/rear Cl MAE", "strict start0 matched-action delta MAE and sign ranking"],
        },
        "image": IMAGE,
        "image_id": IMAGE_ID,
        "minimum_mem_available_gib": 20,
        "resource_evidence": resource_evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(build_plan(args.repo.resolve()), indent=2))


if __name__ == "__main__":
    main()
