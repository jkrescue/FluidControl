#!/usr/bin/env python3
"""Fail-closed official PhysicsNeMo Curator entry point for dynamic train8."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PREDECL = Path("artifacts/tandem_cylinders/dynamic_train8_predeclared_20261003.json")
PREDECL_SHA = "4cf4e7c9b9da27b71e58db2e94b0750736b7f79f09a2aebc3ffa97729e882c5a"
RAW_QC = Path("artifacts/tandem_cylinders/dynamic_train8_real_cfd_qc_20261003.json")
AUTH = Path("artifacts/tandem_cylinders/dynamic_train8_curation_authorization_20261003.json")
VTK_RECEIPTS = Path("artifacts/tandem_cylinders/dynamic_train8_vtk_ready")
OUTPUT = Path("data/curated/tandem_cylinders_dynamic_train8_v1")
FULL40 = Path("data/curated/tandem_cylinders_matched_start_full40_v1")
BASE = Path("scripts/curate_low_action_phase94_validation.py")
FORCE_SHA = {
    "forceFront": "bce88443ce3d19a6411c31b9266af16a3dd4e992f7adb1ddabc659238a1e88d1",
    "forceRear": "654c5bbf64b758505bdfe4d379e153f460862902309e96d2b01974e45536680b",
}
EXPECTED = {
    f"dynamic_train8_b{phase:02d}_{profile}"
    for phase in (0, 2, 4, 6)
    for profile in ("prbs", "multisine")
}
EXECUTION_REVIEWED = False
TOKEN = "EXECUTE_REVIEWED_DYNAMIC_TRAIN8_CURATOR"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError(f"refusing overwrite: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def omega_table(path: Path) -> list[list[float]]:
    text = path.read_text(encoding="utf-8")
    patch = re.search(r"rearCylinder\s*\{(?P<body>.*?)\n\s*\}", text, re.S)
    table = patch and re.search(r"omega\s+table\s*\((?P<rows>.*?)\)\s*;", patch.group("body"), re.S)
    if table is None:
        raise ValueError("rear-cylinder omega table absent")
    return [[float(a), float(b)] for a, b in re.findall(
        r"\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)", table.group("rows"))]


def make_authorization(repo: Path) -> dict:
    if len(PREDECL_SHA) != 64 or sha256(repo / PREDECL) != PREDECL_SHA:
        raise ValueError("reviewed predeclaration SHA differs")
    predecl, aggregate = load(repo / PREDECL), load(repo / RAW_QC)
    if (
        predecl.get("status") != "DYNAMIC_TRAIN8_PREDECLARED_NOT_EXECUTED"
        or set(predecl.get("cases", {})) != EXPECTED
        or aggregate.get("status") != "DYNAMIC_TRAIN8_REAL_OPENFOAM_QC_COMPLETE"
        or aggregate.get("predeclaration_sha256") != PREDECL_SHA
    ):
        raise ValueError("raw aggregate/predeclaration contract differs")
    baseline = repo / "cfd/tandem_cylinders/cases/tandem_backward_dt005/postProcessing"
    for force, digest in FORCE_SHA.items():
        if sha256(baseline / force / "0/coefficient.dat") != digest:
            raise ValueError(f"baseline force source differs: {force}")
    rows = {}
    for name, expected in predecl["cases"].items():
        if expected.get("split") != "train" or expected.get("phase_bin") not in (0, 2, 4, 6):
            raise ValueError(f"non-train phase in train8: {name}")
        case = repo / "cfd/tandem_cylinders/cases" / name
        config_path = case / "case_config.json"
        marker_path = case / "solver_complete.dynamic_train8.json"
        qc_path = case / "solver_log_qc.dynamic_train8.json"
        log_path = case / "log.pimpleFoam.dynamic_train8"
        config, marker, qc = load(config_path), load(marker_path), load(qc_path)
        if any(config.get(key) != value for key, value in expected.items()):
            raise ValueError(f"case config differs: {name}")
        if (
            marker.get("case") != name
            or marker.get("predeclaration_sha256") != PREDECL_SHA
            or marker.get("solver_log_sha256") != sha256(log_path)
            or marker.get("solver_qc_sha256") != sha256(qc_path)
            or qc.get("steps") != 4000
        ):
            raise ValueError(f"solver provenance differs: {name}")
        source = case / "source_restart_provenance"
        for field, digest in expected["source_state_sha256"].items():
            if sha256(source / field) != digest:
                raise ValueError(f"source state differs: {name}/{field}")
        start = expected["source_restart_time"]
        if omega_table(case / f"{start:g}" / "U") != expected["action_points"]:
            raise ValueError(f"actual action table differs: {name}")
        rows[name] = {
            "split": "train",
            "phase_bin": expected["phase_bin"],
            "profile": expected["profile"],
            "run_window": expected["run_window"],
            "action_points": expected["action_points"],
            "source_state_sha256": expected["source_state_sha256"],
            "case_config_sha256": sha256(config_path),
            "solver_marker_sha256": sha256(marker_path),
            "solver_log_sha256": sha256(log_path),
            "solver_qc_sha256": sha256(qc_path),
        }
    return {
        "status": "DYNAMIC_TRAIN8_CURATION_AUTHORIZED_TRAIN_ONLY",
        "predeclaration_sha256": PREDECL_SHA,
        "raw_aggregate_qc_sha256": sha256(repo / RAW_QC),
        "baseline_force_source_sha256": FORCE_SHA,
        "expected_frames_per_case": 201,
        "force_channels": ["front_cd", "front_cl", "rear_cd", "rear_cl"],
        "normalization_policy": (
            "preserve immutable full40 train20 transform for warm-start checkpoint continuity; "
            "optional augmented-train descriptive stats are not a transform"
        ),
        "validation_or_frozen_accessed": False,
        "cases": rows,
    }


def checked_authorization(repo: Path) -> dict:
    saved = load(repo / AUTH)
    if saved != make_authorization(repo):
        raise ValueError("saved train8 authorization is stale")
    return saved


def base_module(repo: Path):
    spec = importlib.util.spec_from_file_location("dynamic_train8_base", repo / BASE)
    if spec is None or spec.loader is None:
        raise ImportError(repo / BASE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_vtk_receipt(repo: Path, name: str) -> dict:
    auth = checked_authorization(repo)
    if name not in auth["cases"]:
        raise ValueError("case not authorized")
    case = repo / "cfd/tandem_cylinders/cases" / name
    files = sorted((case / "VTK_dynamic_train8").glob("*/internal.vtu"))
    if len(files) != 201:
        raise ValueError(f"{name}: expected exactly 201 VTK frames")
    base = base_module(repo)
    source = base.VTKSource(
        str(case / "VTK_dynamic_train8"), file_pattern="*/internal.vtu",
        manifold_dim=3, point_source="vertices", backend="pyvista",
    )
    expected = np.linspace(*auth["cases"][name]["run_window"], 201)
    actual = []
    for index in range(len(source)):
        mesh = next(source[index])
        actual.append(float(mesh.global_data["TimeValue"].reshape(-1)[0].item()))
    if not np.allclose(actual, expected, rtol=0, atol=1e-5):
        raise ValueError(f"{name}: VTK time sequence differs")
    log = case / "log.foamToVTK.dynamic_train8"
    nonblank = [line.strip() for line in log.read_text().splitlines() if line.strip()]
    if not nonblank or re.fullmatch(r"End:\s+\d+(?:\.\d+)? s, \d+ kB \(peak\)", nonblank[-1]) is None:
        raise ValueError(f"{name}: foamToVTK log incomplete")
    return {
        "status": "DYNAMIC_TRAIN8_VTK_READY",
        "case": name,
        "split": "train",
        "frames": 201,
        "solver_marker_sha256": sha256(case / "solver_complete.dynamic_train8.json"),
        "curation_authorization_sha256": sha256(repo / AUTH),
        "foam_to_vtk_log_sha256": sha256(log),
        "vtk_file_sha256": {str(path.relative_to(repo)): sha256(path) for path in files},
    }


def checked_vtk_receipt(repo: Path, name: str) -> None:
    saved = load(repo / VTK_RECEIPTS / f"{name}.json")
    if saved != make_vtk_receipt(repo, name):
        raise ValueError(f"stale VTK receipt: {name}")


def source(base, repo: Path, name: str, config: dict):
    class DynamicTrainSource(base.TandemTrajectorySource):
        def __init__(self):
            self.cases_root = repo / "cfd/tandem_cylinders/cases"
            self.profile = base.MATCHED_START_PROFILE
            self.records = [{"name": name, "split": "train", "config": config}]
            self.nx, self.ny = 256, 128
            self.x = base.np.linspace(8, 25, self.nx, dtype=base.np.float32)
            self.y = base.np.linspace(4, 11, self.ny, dtype=base.np.float32)
            yy, xx = base.torch.meshgrid(
                base.torch.from_numpy(self.y), base.torch.from_numpy(self.x), indexing="ij"
            )
            self.query_points = base.torch.stack(
                [xx.reshape(-1), yy.reshape(-1), base.torch.full((self.nx * self.ny,), 0.05)], dim=1
            )
            self.vtk_sources = [base.VTKSource(
                str(self.cases_root / name / "VTK_dynamic_train8"),
                file_pattern="*/internal.vtu", manifold_dim=3,
                point_source="vertices", backend="pyvista",
                key_filters=[{"path_pattern": "**/internal.vtu", "mode": "include", "keys": ["U", "p"]}],
            )]
            if len(self.vtk_sources[0]) != 201:
                raise ValueError(f"{name}: expected 201 VTK frames")
    return DynamicTrainSource()


def curate(repo: Path, name: str) -> None:
    auth = checked_authorization(repo)
    if name not in auth["cases"]:
        raise ValueError("case not authorized")
    checked_vtk_receipt(repo, name)
    config = load(repo / "cfd/tandem_cylinders/cases" / name / "case_config.json")
    config.update(source_force_sha256=FORCE_SHA, expected_frames=201)
    base = base_module(repo)
    base.validate_matched_start_source_force(repo / "cfd/tandem_cylinders/cases", name, config)
    pipe = (
        source(base, repo, name, config)
        .filter(base.NumericalQualityFilter(0.75))
        .write(base.TrajectoryHDF5Sink(repo / OUTPUT, atomic_tmp=True))
    )
    result = base.run_pipeline(pipe, n_jobs=1, backend="sequential", indices=None, use_tui=False)
    if len(result) != 1 or not result[0]:
        raise RuntimeError("PhysicsNeMo Curator returned no HDF")


def finalize(repo: Path) -> dict:
    import h5py

    auth = checked_authorization(repo)
    root = repo / OUTPUT
    paths = sorted((root / "train").glob("*.h5"))
    if {path.stem for path in paths} != EXPECTED:
        raise ValueError("train8 HDF set differs")
    digests = {}
    for path in paths:
        with h5py.File(path) as handle:
            if handle["state"].shape != (201, 3, 128, 256) or handle["force"].shape != (201, 4):
                raise ValueError(f"shape differs: {path.stem}")
            start, end = auth["cases"][path.stem]["run_window"]
            times = np.linspace(start, end, 201)
            table = np.asarray(auth["cases"][path.stem]["action_points"])
            actions = np.interp(times, table[:, 0], table[:, 1])
            if not np.allclose(handle["time"][:, 0], times, rtol=0, atol=1e-5):
                raise ValueError(f"time differs: {path.stem}")
            if not np.allclose(handle["omega"][:, 0], actions, rtol=0, atol=1e-5):
                raise ValueError(f"action differs: {path.stem}")
            for key in ("state", "force"):
                dataset = handle[key]
                for begin in range(0, len(dataset), 16):
                    if not np.isfinite(dataset[begin : begin + 16]).all():
                        raise ValueError(f"nonfinite HDF: {path.stem}/{key}")
        digests[path.name] = sha256(path)
    source_normalization = repo / FULL40 / "normalization.json"
    target_normalization = root / "normalization.json"
    if target_normalization.exists():
        raise FileExistsError(f"refusing overwrite: {target_normalization}")
    with tempfile.NamedTemporaryFile(dir=root, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        shutil.copyfile(source_normalization, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, target_normalization)
    finally:
        temporary.unlink(missing_ok=True)
    if sha256(target_normalization) != sha256(source_normalization):
        raise ValueError("train20 normalization copy differs")
    manifest = {
        "schema_version": 1,
        "profile": "dynamic_train8_v1",
        "status": "DYNAMIC_TRAIN8_TRAIN_ONLY_CURATED",
        "trajectory_counts": {"train": 8, "validation": 0, "frozen_test": 0},
        "frames_per_trajectory": 201,
        "max_abs_omega": 0.75,
        "force_channels": ["front_cd", "front_cl", "rear_cd", "rear_cl"],
        "curation_authorization_sha256": sha256(repo / AUTH),
        "normalization_status": "reuse full40 train20 normalization; no refit",
        "normalization_reference": str(FULL40),
        "normalization_sha256": sha256(target_normalization),
        "validation_or_frozen_accessed": False,
        "hdf_sha256": digests,
        "official_pipeline": ["PhysicsNeMo Curator Source/Filter/Sink", "VTKSource", "run_pipeline"],
    }
    exclusive(root / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("authorize", "vtk-receipt", "curate", "finalize"))
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--case")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approval-token")
    args = parser.parse_args()
    repo = args.repo.resolve()
    if not EXECUTION_REVIEWED or len(PREDECL_SHA) != 64:
        raise RuntimeError("train8 curation execution is not reviewed/hard-bound")
    if not args.execute or args.approval_token != TOKEN:
        raise RuntimeError("explicit reviewed execution token required")
    if args.mode == "authorize":
        exclusive(repo / AUTH, make_authorization(repo))
    elif args.mode == "vtk-receipt":
        if args.case not in EXPECTED:
            raise ValueError("one declared case required")
        exclusive(repo / VTK_RECEIPTS / f"{args.case}.json", make_vtk_receipt(repo, args.case))
    elif args.mode == "curate":
        if args.case not in EXPECTED:
            raise ValueError("one declared case required")
        curate(repo, args.case)
    else:
        print(json.dumps(finalize(repo), indent=2))


if __name__ == "__main__":
    main()
