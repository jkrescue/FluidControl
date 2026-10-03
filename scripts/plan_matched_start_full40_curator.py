#!/usr/bin/env python3
"""Read-only readiness plan for the isolated matched-start full40 Curator profile."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PREDECLARATION = (
    REPO / "artifacts/tandem_cylinders/matched_start_full40_predeclared_20261003.json"
)
PREDECLARATION_SHA256 = "d7ff174ef10194a8739357376335ca13ff9b45c8079970846bb71f15a715d24b"
NINE_AGGREGATE = REPO / "artifacts/matched_start_acquisition/aggregate_qc/result.json"
REMAINDER_AGGREGATE = (
    REPO / "artifacts/matched_start_full40_extension/aggregate_qc/result.json"
)
NINE_FINAL = (
    REPO / "data/curated/tandem_cylinders_matched_start_commissioning_train9_v1"
)
PROFILE = "matched_start_full40_v1"
STAGING = REPO / "data/curated/.staging" / PROFILE
FINAL = REPO / "data/curated/tandem_cylinders_matched_start_full40_v1"
EXPECTED_SPLIT_BINS = {
    "train": {0, 2, 4, 6},
    "validation": {1, 5},
    "frozen_test": {3, 7},
}
EXPECTED_ACTIONS = {-0.75, -0.375, 0.0, 0.375, 0.75}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_matrix(path: Path = PREDECLARATION) -> dict[str, dict]:
    if path == PREDECLARATION and sha256(path) != PREDECLARATION_SHA256:
        raise ValueError("full40 predeclaration SHA-256 differs")
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data.get("cases", {})
    if len(cases) != 40:
        raise ValueError("full40 predeclaration must contain exactly 40 cases")
    validate_split_contract(cases)
    return cases


def validate_split_contract(cases: dict[str, dict]) -> None:
    split_counts = Counter(row.get("split") for row in cases.values())
    if split_counts != {"train": 20, "validation": 10, "frozen_test": 10}:
        raise ValueError(f"full40 split counts differ: {split_counts}")
    by_phase = defaultdict(list)
    for name, row in cases.items():
        phase = row.get("phase_bin")
        split = row.get("split")
        action = float(row.get("action_target"))
        if phase not in range(8) or action not in EXPECTED_ACTIONS:
            raise ValueError(f"case is outside phase/action matrix: {name}")
        if phase not in EXPECTED_SPLIT_BINS[split]:
            raise ValueError(f"phase split leakage: {name}")
        if row.get("analysis_window")[1] - row.get("analysis_window")[0] != 60.0:
            raise ValueError(f"analysis window differs from fixed 60D/U: {name}")
        by_phase[phase].append(row)
    for phase in range(8):
        rows = by_phase[phase]
        if len(rows) != 5 or {float(row["action_target"]) for row in rows} != EXPECTED_ACTIONS:
            raise ValueError(f"phase b{phase:02d} does not contain five exact actions")
        if len({row["split"] for row in rows}) != 1:
            raise ValueError(f"phase b{phase:02d} crosses data splits")
        if len(
            {json.dumps(row["source_state_sha256"], sort_keys=True) for row in rows}
        ) != 1:
            raise ValueError(f"phase b{phase:02d} branches are not matched-start")


def classify(cases: dict[str, dict]) -> dict:
    reuse = sorted(
        name
        for name, row in cases.items()
        if row["disposition"] == "existing_nine_case_commissioning"
    )
    curate = sorted(
        name
        for name, row in cases.items()
        if row["disposition"] == "planned_new_remainder_case"
    )
    if len(reuse) != 9 or len(curate) != 31 or set(reuse) & set(curate):
        raise ValueError("nine-case reuse / 31-case curation partition differs")
    return {"reuse_nine_hdf": reuse, "curate_remainder": curate}


def aggregate_status(
    path: Path, expected_status: str, expected_case_count: int
) -> tuple[bool, str | None]:
    if not path.is_file():
        return False, None
    data = json.loads(path.read_text(encoding="utf-8"))
    case_count = data.get("case_count", len(data.get("cases", [])))
    return (
        data.get("status") == expected_status and case_count == expected_case_count,
        sha256(path),
    )


def build_plan(repo: Path = REPO) -> dict:
    cases = load_matrix(
        repo / "artifacts/tandem_cylinders/matched_start_full40_predeclared_20261003.json"
    )
    groups = classify(cases)
    blockers = []
    nine_ok, nine_sha = aggregate_status(
        repo / NINE_AGGREGATE.relative_to(REPO),
        "MATCHED_START_9_CASE_COMMISSIONING_QC_PASS",
        9,
    )
    if not nine_ok:
        blockers.append("NINE_CASE_RAW_AGGREGATE_NOT_READY")
    remainder_ok, remainder_sha = aggregate_status(
        repo / REMAINDER_AGGREGATE.relative_to(REPO),
        "MATCHED_START_FULL40_EXTENSION_31_CASE_RAW_QC_PASS",
        31,
    )
    if not remainder_ok:
        blockers.append("REMAINDER_31_RAW_AGGREGATE_NOT_READY")
    nine_final = repo / NINE_FINAL.relative_to(REPO)
    nine_manifest = nine_final / "manifest.json"
    nine_qc = nine_final / "commissioning_qc.json"
    nine_curated_ok = nine_manifest.is_file() and nine_qc.is_file()
    if nine_curated_ok:
        manifest = json.loads(nine_manifest.read_text(encoding="utf-8"))
        qc = json.loads(nine_qc.read_text(encoding="utf-8"))
        nine_curated_ok = (
            manifest.get("profile") == "matched_start_commissioning_train9_v1"
            and manifest.get("trajectory_counts")
            == {"train": 9, "validation": 0, "test": 0}
            and qc.get("status") == "MATCHED_START_COMMISSIONING_NINE_CASE_QC_OK"
            and len(qc.get("cases", [])) == 9
        )
    if not nine_curated_ok:
        blockers.append("NINE_CASE_CURATED_PROFILE_NOT_FINALIZED")

    split_cases = {
        split: sorted(name for name, row in cases.items() if row["split"] == split)
        for split in EXPECTED_SPLIT_BINS
    }
    train_inputs = split_cases["train"]
    return {
        "status": "FULL40_CURATOR_READY" if not blockers else "FULL40_CURATOR_BLOCKED",
        "profile": PROFILE,
        "blockers": blockers,
        "upstream": {
            "full40_predeclaration_sha256": PREDECLARATION_SHA256,
            "nine_case_raw_aggregate_sha256": nine_sha,
            "remainder_31_raw_aggregate_sha256": remainder_sha,
        },
        "case_partition": groups,
        "split_cases": split_cases,
        "staging_root": str(STAGING.relative_to(REPO)),
        "final_root": str(FINAL.relative_to(REPO)),
        "official_api_contract": [
            "physicsnemo_curator.core.base.Source",
            "physicsnemo_curator.core.base.Filter",
            "physicsnemo_curator.core.base.Sink",
            "physicsnemo_curator.domains.mesh.sources.vtk.VTKSource",
            "physicsnemo.mesh.spatial.BVH",
            "physicsnemo_curator.run.run_pipeline",
        ],
        "normalization_contract": {
            "fit_inputs": train_inputs,
            "validation_excluded_from_fit": split_cases["validation"],
            "frozen_test_excluded_from_fit_and_selection": split_cases["frozen_test"],
        },
        "frozen_test_guard": {
            "curation_allowed": "deterministic conversion and integrity hashes only",
            "forbidden_before_final_evaluation_authorization": [
                "normalization fitting",
                "model selection",
                "reward or drag ranking",
                "aggregate physical outcome reporting",
            ],
            "loader_policy": "explicit train/validation manifests; never glob dataset root",
        },
        "resource_contract": {
            "maximum_parallel_curator_cases": 3,
            "spark_start_mem_available_gib_at_least": 64,
            "spark_running_mem_available_gib_at_least": 40,
            "spark_start_free_disk_gib_at_least": 250,
            "spark_running_free_disk_gib_at_least": 150,
            "estimated_remainder_vtk_gib": 70,
            "estimated_remainder_hdf_gib": 9,
        },
        "interpretation_guard": (
            "deterministic CFD-to-HDF preparation only; not model training, "
            "generalization evidence, or closed-loop control"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    args = parser.parse_args()
    print(json.dumps(build_plan(args.repo.resolve()), indent=2))


if __name__ == "__main__":
    main()
