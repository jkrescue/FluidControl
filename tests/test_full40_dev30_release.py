from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, relative: str):
    path = ROOT / relative
    if not path.exists():
        path = ROOT / path.name
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FINALIZE = load("dev30_finalize", "scripts/finalize_matched_start_full40_dev30.py")
PLAN = load("dev30_plan", "scripts/plan_full40_dev30_fno_retrain.py")


def matrix() -> dict:
    cases = {}
    splits = (("train", (0, 2, 4, 6), 5), ("validation", (1, 5), 5), ("frozen_test", (3, 7), 5))
    for split, phases, actions in splits:
        for phase in phases:
            for action in range(actions):
                name = f"case_{split}_b{phase:02d}_{action}"
                cases[name] = {
                    "split": split,
                    "phase_bin": phase,
                    "source_state_sha256": {"U": f"{phase:064x}", "p": f"{action:064x}"},
                }
    return {"cases": cases}


def test_case_contract_and_frozen_seal_are_declaration_only() -> None:
    document = matrix()
    contract = FINALIZE.case_contract(document)
    assert {key: len(value) for key, value in contract.items()} == FINALIZE.EXPECTED_COUNTS
    seal = FINALIZE.frozen_seal(document, contract["frozen_test"])
    assert seal["phase_bins"] == [3, 7]
    assert seal["materialized_in_release"] is False
    assert "enumeration" in seal["forbidden"]


def test_case_contract_rejects_phase_leakage() -> None:
    document = matrix()
    first = next(name for name, row in document["cases"].items() if row["split"] == "validation")
    document["cases"][first]["phase_bin"] = 0
    with pytest.raises(ValueError, match="phase isolation"):
        FINALIZE.case_contract(document)


def test_preflight_fails_closed_without_release(tmp_path: Path) -> None:
    result = PLAN.build_plan(tmp_path)
    assert result == {
        "status": "FULL40_DEV30_FNO_RETRAIN_BLOCKED",
        "blockers": ["DEV30_MANIFEST_MISSING"],
    }


def test_normalization_requires_exact_train_manifest_and_aliases() -> None:
    stats = {
        "computed_from": "train split only",
        "source_scope": "explicit 20-case train manifest only",
        "train_split_manifest_sha256": "a" * 64,
        "state_channels": ["u", "v", "gauge_pressure"],
        "state_mean": [0.0] * 3,
        "state_std": [1.0] * 3,
        "state_abs_normalized_channel_max_train": [2.0, 3.0, 4.0],
        "state_abs_normalized_max_train": 4.0,
        "force_channels": ["rear_cd", "rear_cl"],
        "force_mean": [0.0, 0.0],
        "force_std": [1.0, 1.0],
        "all_force_channels": PLAN.FORCE_CHANNELS,
        "all_force_mean": [0.0] * 4,
        "all_force_std": [1.0] * 4,
    }
    PLAN.validate_normalization(stats, "a" * 64)
    with pytest.raises(ValueError, match="train-only"):
        PLAN.validate_normalization({**stats, "computed_from": "all splits"}, "a" * 64)


def test_runner_mounts_no_repository_or_frozen_path() -> None:
    path = ROOT / "scripts/run_full40_dev30_fno_retrain_spark.sh"
    if not path.exists():
        path = ROOT / path.name
    text = path.read_text(encoding="utf-8")
    assert 'src=$root,dst=/workspace' not in text
    assert "src=$root/scripts,dst=/workspace/scripts,readonly" in text
    assert "src=$root/src,dst=/workspace/src,readonly" in text
    assert "src=$root/conf,dst=/workspace/conf,readonly" in text
    assert "src=$data_host,dst=/workspace/devdata,readonly" in text
    assert "frozen_test" not in text
    assert "--min-free-gib 20" in text
    assert "--gpus device=0" in text
    assert "refusing existing output" in text


def test_finalizer_source_does_not_construct_frozen_hdf_paths() -> None:
    path = ROOT / "scripts/finalize_matched_start_full40_dev30.py"
    if not path.exists():
        path = ROOT / path.name
    text = path.read_text(encoding="utf-8")
    assert 'contract["train"] + contract["validation"]' in text
    assert 'contract["train"] + contract["validation"] + contract["frozen_test"]' not in text
    assert "compute_train_normalization" in text
