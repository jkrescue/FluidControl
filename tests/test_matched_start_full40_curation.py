from __future__ import annotations

import importlib.util
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CURATE = load(
    "full40_remainder_curator",
    "scripts/curate_matched_start_full40_remainder.py",
)
FINALIZE = load("full40_finalizer", "scripts/finalize_matched_start_full40.py")
VTK = load("full40_vtk", "scripts/prepare_matched_start_full40_vtk.py")


def copy_predeclaration(root: Path) -> dict:
    target = (
        root
        / "artifacts/tandem_cylinders/matched_start_full40_predeclared_20261003.json"
    )
    target.parent.mkdir(parents=True)
    target.write_bytes(CURATE.PREDECLARATION.read_bytes())
    return json.loads(target.read_text(encoding="utf-8"))["cases"]


def build_ready_case(root: Path, split: str = "validation") -> str:
    matrix = copy_predeclaration(root)
    remainder = {
        name: row
        for name, row in matrix.items()
        if row["disposition"] == "planned_new_remainder_case"
    }
    name = next(name for name, row in remainder.items() if row["split"] == split)
    aggregate = root / CURATE.RAW_AGGREGATE.relative_to(CURATE.REPO)
    aggregate.parent.mkdir(parents=True)
    aggregate.write_text(
        json.dumps(
            {
                "status": "MATCHED_START_FULL40_EXTENSION_31_CASE_RAW_QC_PASS",
                "case_count": 31,
                "split_counts": {
                    "train": 11,
                    "validation": 10,
                    "frozen_test": 10,
                },
                "full40_predeclaration_sha256": CURATE.PREDECLARATION_SHA256,
                "full40_extension_authorization_sha256": (
                    CURATE.AUTHORIZATION_SHA256
                ),
            }
        )
    )
    for case, row in remainder.items():
        raw = root / "cfd/tandem_cylinders/cases" / case / "raw_marker.bin"
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(case.encode())
        manifest = (
            root
            / "artifacts/matched_start_full40_extension/worker_transfer_manifests"
            / f"{case}.sha256"
        )
        manifest.parent.mkdir(parents=True, exist_ok=True)
        relative = raw.relative_to(root / "cfd/tandem_cylinders/cases")
        manifest.write_text(
            f"{hashlib.sha256(raw.read_bytes()).hexdigest()}  {relative}\n"
        )
        receipt = root / CURATE.RAW_RECEIPTS.relative_to(CURATE.REPO) / f"{case}.json"
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(
            json.dumps(
                {
                    "status": "FULL40_RAW_TRANSFER_VERIFIED",
                    "case": case,
                    "split": row["split"],
                    "phase_bin": row["phase_bin"],
                    "action_target": row["action_target"],
                    "full40_predeclaration_sha256": CURATE.PREDECLARATION_SHA256,
                    "full40_extension_authorization_sha256": (
                        CURATE.AUTHORIZATION_SHA256
                    ),
                    "worker_raw_manifest": str(manifest.relative_to(root)),
                    "worker_raw_manifest_sha256": hashlib.sha256(
                        manifest.read_bytes()
                    ).hexdigest(),
                    "raw_file_count": 1,
                    "coverage": "raw OpenFOAM solver files only; VTK is not included",
                    "next_required_stage": "SPARK_PINNED_FOAMTOVTK_801_FRAMES",
                    "source_state_sha256": row["source_state_sha256"],
                }
            )
        )
    row = remainder[name]
    case = root / "cfd/tandem_cylinders/cases" / name
    case.mkdir(parents=True, exist_ok=True)
    config = {
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
        "expected_field_frames": 801,
        "source_restart_case": "tandem_backward_dt005",
    }
    source_hashes = {}
    for object_name in ("forceFront", "forceRear"):
        source = (
            root
            / "cfd/tandem_cylinders/cases/tandem_backward_dt005/postProcessing"
            / object_name
            / "0/coefficient.dat"
        )
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("# Time Cd Cl\n0 1 0\n1 1 0\n")
        source_hashes[object_name] = hashlib.sha256(source.read_bytes()).hexdigest()
    config["source_force_sha256"] = source_hashes
    (case / "case_config.json").write_text(json.dumps(config))
    selected_receipt = (
        root / CURATE.RAW_RECEIPTS.relative_to(CURATE.REPO) / f"{name}.json"
    )
    selected_payload = json.loads(selected_receipt.read_text(encoding="utf-8"))
    selected_payload["baseline_force_source_sha256"] = source_hashes
    selected_receipt.write_text(json.dumps(selected_payload))
    vtk = root / CURATE.VTK_RECEIPTS.relative_to(CURATE.REPO) / f"{name}.json"
    vtk.parent.mkdir(parents=True)
    manifest = root / "artifacts/matched_start_full40_extension/vtk_manifests" / f"{name}.sha256"
    manifest.parent.mkdir(parents=True)
    rows = []
    for index in range(801):
        frame = (
            root
            / "cfd/tandem_cylinders/cases"
            / name
            / "VTK_curator"
            / str(index)
            / "internal.vtu"
        )
        frame.parent.mkdir(parents=True)
        frame.write_bytes(f"frame-{index}".encode())
        digest = hashlib.sha256(frame.read_bytes()).hexdigest()
        rows.append(
            f"{digest}  {frame.relative_to(root / 'cfd/tandem_cylinders/cases')}"
        )
    manifest.write_text("\n".join(rows) + "\n")
    raw_receipt = root / CURATE.RAW_RECEIPTS.relative_to(CURATE.REPO) / f"{name}.json"
    vtk.write_text(
        json.dumps(
            {
                "status": "VTK_READY",
                "case": name,
                "split": row["split"],
                "frames": 801,
                "raw_transfer_receipt_sha256": hashlib.sha256(
                    raw_receipt.read_bytes()
                ).hexdigest(),
                "vtk_manifest": str(manifest.relative_to(root)),
                "vtk_manifest_sha256": hashlib.sha256(
                    manifest.read_bytes()
                ).hexdigest(),
                "openfoam_image": VTK.OPENFOAM_IMAGE,
                "overwrite_policy": "REFUSE",
            }
        )
    )
    return name


def test_remainder_case_plan_requires_strict_case_raw_qc_and_preserves_split(tmp_path):
    name = build_ready_case(tmp_path, "validation")
    plan = CURATE.case_plan(tmp_path, name)
    assert plan["status"] == "FULL40_REMAINDER_CASE_READY_FOR_CURATOR"
    assert plan["split"] == "validation"
    assert plan["target"].endswith(f"/{name}.h5")
    assert CURATE.EXECUTION_REVIEWED is True


def test_frozen_case_plan_is_explicitly_sealed(tmp_path):
    name = build_ready_case(tmp_path, "frozen_test")
    plan = CURATE.case_plan(tmp_path, name)
    assert plan["split"] == "frozen_test"
    assert "no statistics" in plan["frozen_guard"]


def test_per_case_conversion_can_precede_global_aggregate(tmp_path):
    name = build_ready_case(tmp_path, "train")
    (tmp_path / CURATE.RAW_AGGREGATE.relative_to(CURATE.REPO)).unlink()
    plan = CURATE.case_plan(tmp_path, name)
    assert plan["raw_aggregate_gate"] == "DEFERRED_TO_FINALIZER_AFTER_ALL_31_CASES"


def test_per_case_receipt_requires_exact_authorization(tmp_path):
    name = build_ready_case(tmp_path, "train")
    receipt = tmp_path / CURATE.RAW_RECEIPTS.relative_to(CURATE.REPO) / f"{name}.json"
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["full40_extension_authorization_sha256"] = "0" * 64
    receipt.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="raw receipt contract"):
        CURATE.case_plan(tmp_path, name)


def test_final_aggregate_requires_exact_authorization_and_all_receipts(tmp_path):
    build_ready_case(tmp_path, "train")
    matrix = CURATE.load_matrix(tmp_path)
    aggregate = tmp_path / CURATE.RAW_AGGREGATE.relative_to(CURATE.REPO)
    payload = json.loads(aggregate.read_text(encoding="utf-8"))
    payload["full40_extension_authorization_sha256"] = "0" * 64
    aggregate.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="aggregate QC contract"):
        CURATE.validate_raw_aggregate(tmp_path, matrix)


def test_raw_manifest_mutation_fails_before_vtk_or_hdf(tmp_path) -> None:
    name = build_ready_case(tmp_path, "train")
    raw = tmp_path / "cfd/tandem_cylinders/cases" / name / "raw_marker.bin"
    raw.write_bytes(b"mutated-after-transfer")
    with pytest.raises(ValueError, match="transferred raw file SHA"):
        CURATE.case_plan(tmp_path, name)


def test_vtk_manifest_must_equal_exact_curator_input_set(tmp_path) -> None:
    name = build_ready_case(tmp_path, "validation")
    receipt_path = (
        tmp_path / CURATE.VTK_RECEIPTS.relative_to(CURATE.REPO) / f"{name}.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    manifest = tmp_path / receipt["vtk_manifest"]
    unrelated = tmp_path / "cfd/tandem_cylinders/cases" / name / "raw_marker.bin"
    lines = manifest.read_text(encoding="utf-8").splitlines()
    lines[0] = (
        f"{hashlib.sha256(unrelated.read_bytes()).hexdigest()}  "
        f"{unrelated.relative_to(tmp_path / 'cfd/tandem_cylinders/cases')}"
    )
    manifest.write_text("\n".join(lines) + "\n")
    receipt["vtk_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt))
    raw_receipt = (
        tmp_path / CURATE.RAW_RECEIPTS.relative_to(CURATE.REPO) / f"{name}.json"
    )
    with pytest.raises(ValueError, match="exact Curator input set"):
        CURATE.validate_vtk_receipt(
            tmp_path,
            name,
            hashlib.sha256(raw_receipt.read_bytes()).hexdigest(),
            "validation",
        )


def test_vtk_receipt_is_atomic_and_binds_801_frame_hashes(tmp_path) -> None:
    name = "matched_start_acquisition_validation_b01_m075"
    vtk_root = tmp_path / "cfd/tandem_cylinders/cases" / name / "VTK_curator"
    for index in range(801):
        path = vtk_root / str(index) / "internal.vtu"
        path.parent.mkdir(parents=True)
        path.write_bytes(f"vtk-{index}".encode())
    item = {
        "case": name,
        "split": "validation",
        "raw_transfer_receipt_sha256": "b" * 64,
        "command": ["bash", "export", name],
    }
    receipt = VTK.verify_and_record(tmp_path, item)
    assert receipt["status"] == "VTK_READY"
    assert receipt["frames"] == 801
    manifest = tmp_path / receipt["vtk_manifest"]
    assert len(manifest.read_text().splitlines()) == 801
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == receipt[
        "vtk_manifest_sha256"
    ]


def test_explicit_final_manifests_are_20_10_10_and_phase_disjoint() -> None:
    matrix = FINALIZE.load_matrix()
    manifests = FINALIZE.split_manifests(matrix)
    assert {split: len(names) for split, names in manifests.items()} == {
        "train": 20,
        "validation": 10,
        "frozen_test": 10,
    }
    phase_sets = {
        split: {matrix[name]["phase_bin"] for name in names}
        for split, names in manifests.items()
    }
    assert phase_sets == {
        "train": {0, 2, 4, 6},
        "validation": {1, 5},
        "frozen_test": {3, 7},
    }


def test_nine_cases_are_hash_verified_reuse_only(tmp_path) -> None:
    matrix = copy_predeclaration(tmp_path)
    names = sorted(
        name
        for name, row in matrix.items()
        if row["disposition"] == "existing_nine_case_commissioning"
    )
    root = tmp_path / FINALIZE.NINE_FINAL.relative_to(FINALIZE.REPO)
    hashes = {}
    qc_rows = []
    for name in names:
        path = root / "train" / f"{name}.h5"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        qc_rows.append(
            {
                "case": name,
                "split": "train",
                "phase_bin": f"b{matrix[name]['phase_bin']:02d}",
                "source_state_sha256": matrix[name]["source_state_sha256"],
                "hdf5_sha256": hashes[name],
            }
        )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "profile": "matched_start_commissioning_train9_v1",
                "trajectory_counts": {"train": 9, "validation": 0, "test": 0},
                "hdf5_sha256": hashes,
            }
        )
    )
    (root / "commissioning_qc.json").write_text(
        json.dumps(
            {
                "status": "MATCHED_START_COMMISSIONING_NINE_CASE_QC_OK",
                "cases": qc_rows,
            }
        )
    )
    assert len(FINALIZE.validate_nine_reuse(tmp_path, matrix)) == 9
    (root / "train" / f"{names[0]}.h5").write_bytes(b"mutated")
    with pytest.raises(ValueError, match="HDF5 hash differs"):
        FINALIZE.validate_nine_reuse(tmp_path, matrix)


def test_nine_case_qc_provenance_must_match_hdf_manifest(tmp_path) -> None:
    matrix = copy_predeclaration(tmp_path)
    names = sorted(
        name
        for name, row in matrix.items()
        if row["disposition"] == "existing_nine_case_commissioning"
    )
    root = tmp_path / FINALIZE.NINE_FINAL.relative_to(FINALIZE.REPO)
    hashes = {}
    rows = []
    for name in names:
        path = root / "train" / f"{name}.h5"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(
            {
                "case": name,
                "split": "train",
                "phase_bin": f"b{matrix[name]['phase_bin']:02d}",
                "source_state_sha256": matrix[name]["source_state_sha256"],
                "hdf5_sha256": hashes[name],
            }
        )
    rows[0]["hdf5_sha256"] = "0" * 64
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "profile": "matched_start_commissioning_train9_v1",
                "trajectory_counts": {"train": 9, "validation": 0, "test": 0},
                "hdf5_sha256": hashes,
            }
        )
    )
    (root / "commissioning_qc.json").write_text(
        json.dumps(
            {
                "status": "MATCHED_START_COMMISSIONING_NINE_CASE_QC_OK",
                "cases": rows,
            }
        )
    )
    with pytest.raises(ValueError, match="final QC provenance"):
        FINALIZE.validate_nine_reuse(tmp_path, matrix)


def write_tiny_hdf(path: Path, split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["split"] = split
        handle.create_dataset("state", data=np.ones((1, 3, 2, 2), np.float32))
        handle.create_dataset("mask", data=np.ones((1, 1, 2, 2), np.uint8))
        handle.create_dataset("force", data=np.ones((1, 4), np.float32))


def test_normalization_accepts_only_explicit_20_train_files(tmp_path):
    paths = []
    for index in range(20):
        path = tmp_path / f"train_{index}.h5"
        write_tiny_hdf(path, "train")
        paths.append(path)
    result = FINALIZE.compute_train_normalization(paths)
    assert result["computed_from"] == "train split only"
    assert result["source_scope"] == "explicit 20-case train manifest only"
    assert result["all_force_channels"] == [
        "front_cd",
        "front_cl",
        "rear_cd",
        "rear_cl",
    ]
    assert result["force_channels"] == ["rear_cd", "rear_cl"]
    assert FINALIZE.MAX_ABS_OMEGA == 0.75
    write_tiny_hdf(paths[-1], "validation")
    with pytest.raises(ValueError, match="non-train"):
        FINALIZE.compute_train_normalization(paths)
    with pytest.raises(ValueError, match="20-file"):
        FINALIZE.compute_train_normalization(paths[:-1])


def test_hdf_coordinate_and_channel_metadata_are_fail_closed(tmp_path) -> None:
    path = tmp_path / "contract.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "x", data=np.linspace(8.0, 25.0, 256, dtype=np.float32)
        )
        handle.create_dataset(
            "y", data=np.linspace(4.0, 11.0, 128, dtype=np.float32)
        )
        handle.attrs["state_channels"] = json.dumps(FINALIZE.STATE_CHANNELS)
        handle.attrs["force_channels"] = json.dumps(FINALIZE.FORCE_CHANNELS)
    with h5py.File(path, "r") as handle:
        FINALIZE.validate_coordinate_and_channel_contract(handle, "case")
    with h5py.File(path, "r+") as handle:
        handle["x"][0] = -999.0
    with h5py.File(path, "r") as handle:
        with pytest.raises(ValueError, match="coordinate contract"):
            FINALIZE.validate_coordinate_and_channel_contract(handle, "case")


def test_current_finalization_is_dry_run_only(tmp_path) -> None:
    assert VTK.EXECUTION_REVIEWED is True
    assert CURATE.EXECUTION_REVIEWED is True
    assert FINALIZE.EXECUTION_REVIEWED is False
    copy_predeclaration(tmp_path)
    report = FINALIZE.readiness(tmp_path)
    assert report["status"] == "FULL40_FINALIZATION_BLOCKED"
    assert len(report["normalization_inputs"]) == 20
    assert len(report["frozen_seal_inputs"]) == 10


def test_remainder_curator_cli_help_smoke() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/curate_matched_start_full40_remainder.py"),
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--repo" in result.stdout
