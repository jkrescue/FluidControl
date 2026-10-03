#!/usr/bin/env python3
"""Isolated, dry-run-first Curator entry point for the full40 remainder.

The live nine-case Curator implementation is imported but never modified.  An
execution can only be enabled after a separate review changes EXECUTION_REVIEWED.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PREDECLARATION = (
    REPO / "artifacts/tandem_cylinders/matched_start_full40_predeclared_20261003.json"
)
PREDECLARATION_SHA256 = "d7ff174ef10194a8739357376335ca13ff9b45c8079970846bb71f15a715d24b"
AUTHORIZATION_SHA256 = "da8bccaf18a86666ac78804e775c1608d8fc390bfca1484cbd1eaabe93c17151"
RAW_AGGREGATE = REPO / "artifacts/matched_start_full40_extension/aggregate_qc/result.json"
RAW_RECEIPTS = REPO / "artifacts/matched_start_full40_extension/transfer_verified"
VTK_RECEIPTS = REPO / "artifacts/matched_start_full40_extension/vtk_ready"
STAGING = REPO / "data/curated/.staging/matched_start_full40_v1"
BASE_CURATOR = REPO / "scripts/curate_low_action_phase94_validation.py"
EXECUTION_REVIEWED = True
EXECUTION_TOKEN = "EXECUTE_REVIEWED_FULL40_CURATOR_REMAINDER"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_matrix(repo: Path = REPO) -> dict[str, dict]:
    path = repo / PREDECLARATION.relative_to(REPO)
    if sha256(path) != PREDECLARATION_SHA256:
        raise ValueError("full40 predeclaration SHA-256 differs")
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data.get("cases", {})
    if len(cases) != 40:
        raise ValueError("full40 predeclaration does not contain 40 cases")
    return cases


def validate_raw_aggregate(repo: Path, matrix: dict[str, dict]) -> str:
    path = repo / RAW_AGGREGATE.relative_to(REPO)
    if not path.is_file():
        raise FileNotFoundError("31-case raw aggregate QC is absent")
    data = json.loads(path.read_text(encoding="utf-8"))
    remainder = {
        name: row
        for name, row in matrix.items()
        if row["disposition"] == "planned_new_remainder_case"
    }
    if (
        data.get("status") != "MATCHED_START_FULL40_EXTENSION_31_CASE_RAW_QC_PASS"
        or data.get("case_count") != 31
        or data.get("split_counts")
        != {"train": 11, "validation": 10, "frozen_test": 10}
        or data.get("full40_predeclaration_sha256") != PREDECLARATION_SHA256
        or data.get("full40_extension_authorization_sha256")
        != AUTHORIZATION_SHA256
        or len(remainder) != 31
    ):
        raise ValueError("31-case raw aggregate QC contract differs")
    for name, row in remainder.items():
        receipt_path = repo / RAW_RECEIPTS.relative_to(REPO) / f"{name}.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        required = {
            "status": "FULL40_RAW_TRANSFER_VERIFIED",
            "case": name,
            "split": row["split"],
            "phase_bin": row["phase_bin"],
            "action_target": row["action_target"],
            "full40_predeclaration_sha256": PREDECLARATION_SHA256,
            "full40_extension_authorization_sha256": AUTHORIZATION_SHA256,
            "source_state_sha256": row["source_state_sha256"],
            "coverage": "raw OpenFOAM solver files only; VTK is not included",
            "next_required_stage": "SPARK_PINNED_FOAMTOVTK_801_FRAMES",
        }
        if any(receipt.get(key) != value for key, value in required.items()):
            raise ValueError(f"raw receipt differs: {name}")
    receipt_root = repo / RAW_RECEIPTS.relative_to(REPO)
    actual_receipts = {item.stem for item in receipt_root.glob("*.json")}
    if actual_receipts != set(remainder):
        raise ValueError("raw receipt set is incomplete or contains unplanned cases")
    return sha256(path)


def validate_raw_receipt_files(repo: Path, name: str, expected: dict) -> str:
    """Re-hash every transferred raw file consumed downstream for one case."""

    receipt_path = repo / RAW_RECEIPTS.relative_to(REPO) / f"{name}.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    required = {
        "status": "FULL40_RAW_TRANSFER_VERIFIED",
        "case": name,
        "split": expected["split"],
        "phase_bin": expected["phase_bin"],
        "action_target": expected["action_target"],
        "full40_predeclaration_sha256": PREDECLARATION_SHA256,
        "full40_extension_authorization_sha256": AUTHORIZATION_SHA256,
        "source_state_sha256": expected["source_state_sha256"],
        "coverage": "raw OpenFOAM solver files only; VTK is not included",
        "next_required_stage": "SPARK_PINNED_FOAMTOVTK_801_FRAMES",
    }
    if any(receipt.get(key) != value for key, value in required.items()):
        raise ValueError(f"raw receipt contract differs: {name}")
    if receipt.get("baseline_force_source_sha256") != json.loads(
        (repo / "cfd/tandem_cylinders/cases" / name / "case_config.json").read_text(
            encoding="utf-8"
        )
    ).get("source_force_sha256"):
        raise ValueError(f"raw receipt baseline-force provenance differs: {name}")
    manifest_relative = Path(receipt.get("worker_raw_manifest", ""))
    exact_relative = Path(
        "artifacts/matched_start_full40_extension/worker_transfer_manifests"
    ) / f"{name}.sha256"
    if (
        manifest_relative != exact_relative
        or manifest_relative.is_absolute()
        or ".." in manifest_relative.parts
    ):
        raise ValueError(f"worker raw manifest path differs: {name}")
    manifest = repo / manifest_relative
    if not manifest.is_file() or sha256(manifest) != receipt.get(
        "worker_raw_manifest_sha256"
    ):
        raise ValueError(f"worker raw manifest SHA differs: {name}")
    rows: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match or match.group(2) in rows:
            raise ValueError(f"worker raw manifest row differs: {name}")
        rows[match.group(2)] = match.group(1)
    if not rows or len(rows) != receipt.get("raw_file_count"):
        raise ValueError(f"worker raw manifest file count differs: {name}")
    cases_root = repo / "cfd/tandem_cylinders/cases"
    for relative, digest in rows.items():
        candidate = Path(relative)
        if (
            candidate.is_absolute()
            or ".." in candidate.parts
            or not candidate.parts
            or candidate.parts[0] != name
        ):
            raise ValueError(f"worker raw manifest path differs: {name}")
        actual = cases_root / candidate
        if not actual.is_file() or sha256(actual) != digest:
            raise ValueError(f"transferred raw file SHA differs: {name}/{relative}")
    return sha256(receipt_path)


def validate_case_config(name: str, config: dict, expected: dict) -> None:
    checks = {
        "case": name,
        "split": expected["split"],
        "phase_bin": expected["phase_bin"],
        "source_restart_time": expected["source_restart_time"],
        "source_state_sha256": expected["source_state_sha256"],
        "action_target": expected["action_target"],
        "action_points": expected["action_points"],
        "start_time": expected["run_window"][0],
        "end_time": expected["run_window"][1],
        "analysis_window": expected["analysis_window"],
    }
    for key, value in checks.items():
        if config.get(key) != value:
            raise ValueError(f"case config differs: {name}/{key}")
    if config.get("expected_field_frames") != 801:
        raise ValueError(f"case config differs: {name}/expected_field_frames")


def validate_source_force(repo: Path, name: str, config: dict) -> None:
    declared = config.get("source_force_sha256")
    if not isinstance(declared, dict) or set(declared) != {"forceFront", "forceRear"}:
        raise ValueError(f"source-force declaration differs: {name}")
    source_name = config.get("source_restart_case")
    if source_name != "tandem_backward_dt005":
        raise ValueError(f"source restart differs: {name}")
    for object_name, expected in declared.items():
        path = (
            repo
            / "cfd/tandem_cylinders/cases"
            / source_name
            / "postProcessing"
            / object_name
            / "0/coefficient.dat"
        )
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"baseline source-force SHA differs: {name}/{object_name}")


def validate_vtk_receipt(
    repo: Path, name: str, raw_receipt_sha256: str, expected_split: str
) -> dict:
    path = repo / VTK_RECEIPTS.relative_to(REPO) / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"VTK_READY receipt is absent: {name}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        data.get("status") != "VTK_READY"
        or data.get("case") != name
        or data.get("split") != expected_split
    ):
        raise ValueError(f"VTK_READY receipt differs: {name}")
    if (
        data.get("frames") != 801
        or data.get("raw_transfer_receipt_sha256") != raw_receipt_sha256
        or data.get("openfoam_image")
        != (
            "opencfd/openfoam-default@sha256:"
            "33fb575aa9980d2bc42fd58c75ae698c489293ba30c991380fe3f899c622f319"
        )
        or data.get("overwrite_policy") != "REFUSE"
    ):
        raise ValueError(f"VTK_READY frame count differs: {name}")
    manifest_relative = Path(data.get("vtk_manifest", ""))
    if manifest_relative.is_absolute() or ".." in manifest_relative.parts:
        raise ValueError(f"VTK manifest path is unsafe: {name}")
    manifest = repo / manifest_relative
    if not manifest.is_file() or sha256(manifest) != data.get("vtk_manifest_sha256"):
        raise ValueError(f"VTK manifest SHA differs: {name}")
    rows = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match or match.group(2) in rows:
            raise ValueError(f"VTK manifest row differs: {name}")
        rows[match.group(2)] = match.group(1)
    if len(rows) != 801:
        raise ValueError(f"VTK manifest must contain exactly 801 files: {name}")
    cases_root = repo / "cfd/tandem_cylinders/cases"
    expected_paths = {
        str(path.relative_to(cases_root))
        for path in (cases_root / name / "VTK_curator").glob("*/internal.vtu")
    }
    if set(rows) != expected_paths:
        raise ValueError(f"VTK manifest does not cover the exact Curator input set: {name}")
    for relative, expected in rows.items():
        candidate = Path(relative)
        if (
            candidate.is_absolute()
            or ".." in candidate.parts
            or len(candidate.parts) != 4
            or candidate.parts[0] != name
            or candidate.parts[1] != "VTK_curator"
            or candidate.parts[3] != "internal.vtu"
        ):
            raise ValueError(f"VTK manifest path differs: {name}")
        actual = cases_root / candidate
        if not actual.is_file() or sha256(actual) != expected:
            raise ValueError(f"VTK file SHA differs: {name}/{relative}")
    return data


def case_plan(repo: Path, name: str) -> dict:
    matrix = load_matrix(repo)
    expected = matrix.get(name)
    if expected is None or expected.get("disposition") != "planned_new_remainder_case":
        raise ValueError("case is not in the 31-case remainder")
    case_root = repo / "cfd/tandem_cylinders/cases" / name
    config = json.loads((case_root / "case_config.json").read_text(encoding="utf-8"))
    validate_case_config(name, config, expected)
    validate_source_force(repo, name, config)
    raw_receipt = repo / RAW_RECEIPTS.relative_to(REPO) / f"{name}.json"
    raw_receipt_sha256 = validate_raw_receipt_files(repo, name, expected)
    vtk = validate_vtk_receipt(repo, name, raw_receipt_sha256, expected["split"])
    output = repo / STAGING.relative_to(REPO) / name
    target = output / expected["split"] / f"{name}.h5"
    if target.exists() or target.with_suffix(".h5.tmp").exists():
        raise FileExistsError(f"refusing existing full40 HDF5 output: {name}")
    return {
        "status": "FULL40_REMAINDER_CASE_READY_FOR_CURATOR",
        "case": name,
        "split": expected["split"],
        "raw_aggregate_gate": "DEFERRED_TO_FINALIZER_AFTER_ALL_31_CASES",
        "raw_receipt": str(RAW_RECEIPTS.relative_to(REPO) / f"{name}.json"),
        "vtk_receipt": str(VTK_RECEIPTS.relative_to(REPO) / f"{name}.json"),
        "vtk_manifest_sha256": vtk.get("vtk_manifest_sha256"),
        "output": str(output.relative_to(repo)),
        "target": str(target.relative_to(repo)),
        "frozen_guard": (
            "integrity conversion only; no statistics, ranking, or model access"
            if expected["split"] == "frozen_test"
            else None
        ),
        "official_pipeline": [
            "physicsnemo_curator Source/Filter/Sink",
            "VTKSource + PhysicsNeMo Mesh/BVH",
            "run_pipeline",
        ],
    }


def load_base_curator(repo: Path):
    path = repo / BASE_CURATOR.relative_to(REPO)
    spec = importlib.util.spec_from_file_location("full40_reused_curator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_source(base, cases_root: Path, name: str, config: dict, nx: int, ny: int):
    """Construct one official-API Source while retaining its declared split."""

    class Full40Source(base.TandemTrajectorySource):
        def __init__(self) -> None:
            self.cases_root = cases_root
            # Reuse the already validated matched-start t0-force path in __getitem__.
            self.profile = base.MATCHED_START_PROFILE
            base.validate_matched_start_source_force(cases_root, name, config)
            self.records = [{"name": name, "split": config["split"], "config": config}]
            self.nx = nx
            self.ny = ny
            self.x = base.np.linspace(8.0, 25.0, nx, dtype=base.np.float32)
            self.y = base.np.linspace(4.0, 11.0, ny, dtype=base.np.float32)
            yy, xx = base.torch.meshgrid(
                base.torch.from_numpy(self.y),
                base.torch.from_numpy(self.x),
                indexing="ij",
            )
            self.query_points = base.torch.stack(
                [xx.reshape(-1), yy.reshape(-1), base.torch.full((nx * ny,), 0.05)],
                dim=1,
            )
            self.vtk_sources = [
                base.VTKSource(
                    str(cases_root / name / "VTK_curator"),
                    file_pattern="*/internal.vtu",
                    manifold_dim=3,
                    point_source="vertices",
                    backend="pyvista",
                    key_filters=[
                        {
                            "path_pattern": "**/internal.vtu",
                            "mode": "include",
                            "keys": ["U", "p"],
                        }
                    ],
                )
            ]
            if len(self.vtk_sources[0]) != 801:
                raise ValueError(f"{name}: expected 801 VTK frames")

    return Full40Source()


def execute(repo: Path, plan: dict, nx: int, ny: int) -> None:
    base = load_base_curator(repo)
    case_root = repo / "cfd/tandem_cylinders/cases" / plan["case"]
    config = json.loads((case_root / "case_config.json").read_text(encoding="utf-8"))
    source = make_source(
        base,
        repo / "cfd/tandem_cylinders/cases",
        plan["case"],
        config,
        nx,
        ny,
    )
    output = repo / plan["output"]
    pipeline = source.filter(base.NumericalQualityFilter(0.75)).write(
        base.TrajectoryHDF5Sink(output, atomic_tmp=True)
    )
    result = base.run_pipeline(
        pipeline, n_jobs=1, backend="sequential", indices=None, use_tui=False
    )
    if len(result) != 1 or not result[0]:
        raise RuntimeError("official Curator pipeline returned no HDF5 output")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--case", required=True)
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approval-token")
    args = parser.parse_args()
    repo = args.repo.resolve()
    plan = case_plan(repo, args.case)
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    if not EXECUTION_REVIEWED or args.approval_token != EXECUTION_TOKEN:
        parser.error("full40 Curator execution is not reviewed/enabled")
    execute(repo, plan, args.nx, args.ny)


if __name__ == "__main__":
    main()
