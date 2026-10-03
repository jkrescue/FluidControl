from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PATH = REPO / "scripts/plan_matched_start_full40_curator.py"
SPEC = importlib.util.spec_from_file_location("full40_curator_plan", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def matrix() -> dict:
    return MODULE.load_matrix()


def test_exact_phase_split_prevents_action_leakage() -> None:
    cases = matrix()
    MODULE.validate_split_contract(cases)
    for phase_bin in range(8):
        phase = [row for row in cases.values() if row["phase_bin"] == phase_bin]
        assert len({row["split"] for row in phase}) == 1
        assert {row["action_target"] for row in phase} == MODULE.EXPECTED_ACTIONS


def test_partition_is_nine_reuse_plus_31_new_curation() -> None:
    result = MODULE.classify(matrix())
    assert len(result["reuse_nine_hdf"]) == 9
    assert len(result["curate_remainder"]) == 31
    assert not set(result["reuse_nine_hdf"]) & set(result["curate_remainder"])


def test_phase_split_mutation_is_rejected() -> None:
    cases = json.loads(json.dumps(matrix()))
    name = next(name for name, row in cases.items() if row["phase_bin"] == 3)
    cases[name]["split"] = "train"
    with pytest.raises(ValueError, match="split"):
        MODULE.validate_split_contract(cases)


def test_plan_blocks_without_complete_31_raw_qc_and_nine_hdf(tmp_path) -> None:
    predecl = (
        tmp_path
        / "artifacts/tandem_cylinders/matched_start_full40_predeclared_20261003.json"
    )
    predecl.parent.mkdir(parents=True)
    predecl.write_bytes(MODULE.PREDECLARATION.read_bytes())
    plan = MODULE.build_plan(tmp_path)
    assert plan["status"] == "FULL40_CURATOR_BLOCKED"
    assert "REMAINDER_31_RAW_AGGREGATE_NOT_READY" in plan["blockers"]
    frozen = plan["normalization_contract"]["frozen_test_excluded_from_fit_and_selection"]
    assert len(frozen) == 10
    assert not set(frozen) & set(plan["normalization_contract"]["fit_inputs"])


def test_official_api_and_resource_contract_are_explicit() -> None:
    plan = MODULE.build_plan()
    assert "physicsnemo_curator.run.run_pipeline" in plan["official_api_contract"]
    assert plan["resource_contract"]["maximum_parallel_curator_cases"] == 3
    assert plan["resource_contract"]["spark_start_free_disk_gib_at_least"] == 250
