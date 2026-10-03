#!/usr/bin/env python3
"""Fail-closed Curator pipeline for historical direct-PPO train trajectories."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PREDECL = Path("artifacts/tandem_cylinders/directppo_train16_predeclared_20261004.json")
PREDECL_SHA = "7d9fc2a71ebe4bb0e817b1ce41f5e9a42310348bc5faf530bc1b3ad6a5229736"
AUTH = Path("artifacts/tandem_cylinders/directppo_train16_curation_authorization_20261004.json")
VTK_RECEIPTS = Path("artifacts/tandem_cylinders/directppo_train16_vtk_ready")
OUTPUT = Path("data/curated/tandem_cylinders_directppo_train16_v1")
FULL40 = Path("data/curated/tandem_cylinders_matched_start_full40_dev30_v1")
BASE = Path("scripts/curate_low_action_phase94_validation.py")
PREDECL_MODULE = Path("cfd/tandem_cylinders/predeclare_directppo_train16.py")
FORCE_SHA = {
    "forceFront": "bce88443ce3d19a6411c31b9266af16a3dd4e992f7adb1ddabc659238a1e88d1",
    "forceRear": "654c5bbf64b758505bdfe4d379e153f460862902309e96d2b01974e45536680b",
}
TOKEN = "EXECUTE_REVIEWED_DIRECTPPO_TRAIN16_CURATOR"
EXECUTION_REVIEWED = True


def module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def last_nonblank(path: Path) -> str:
    rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[-1] if rows else ""


def omega_table(path: Path) -> list[list[float]]:
    text = path.read_text(encoding="utf-8")
    patch = re.search(r"rearCylinder\s*\{(?P<body>.*?)\n\s*\}", text, re.S)
    table = patch and re.search(r"omega\s+table\s*\((?P<rows>.*?)\)\s*;", patch.group("body"), re.S)
    if table is None:
        raise ValueError(f"rear omega table absent: {path}")
    return [[float(a), float(b)] for a, b in re.findall(
        r"\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)", table.group("rows"))]


def make_authorization(repo: Path) -> dict:
    pre = module(repo / PREDECL_MODULE, "directppo_train16_predecl")
    path = repo / PREDECL
    if pre.sha256(path) != PREDECL_SHA:
        raise ValueError("reviewed predeclaration SHA differs")
    saved = load(path)
    if saved != pre.build(repo):
        raise ValueError("predeclaration is stale against raw inputs")
    base = module(repo / BASE, "directppo_train16_base")
    cases_root = repo / saved["cases_root"]
    rows = {}
    for name, expected in sorted(saved["cases"].items()):
        case = cases_root / name
        logs = sorted(case.glob("log.pimpleFoam.direct_cfd_*"))
        if len(logs) != 128 or any(last_nonblank(log) != "End" for log in logs):
            raise ValueError(f"incomplete 128-segment solver logs: {name}")
        start, end = map(float, expected["run_window"])
        actions = expected["action_points"]
        for index in range(128):
            begin, old = actions[index]
            finish, new = actions[index + 1]
            actual = omega_table(case / f"{begin:g}" / "U")
            if not np.allclose(actual, [[begin, old], [finish, new]], rtol=0, atol=1e-8):
                raise ValueError(f"boundary action differs: {name}/{index + 1}")
        force_sha = {}
        for object_name in ("forceFront", "forceRear"):
            paths = sorted(case.glob(f"postProcessing/{object_name}/*/coefficient.dat"))
            if len(paths) != 128:
                raise ValueError(f"force segment count differs: {name}/{object_name}")
            raw = base.load_merged_coefficients(paths)
            selected = raw[(raw[:, 0] >= start + 0.005 - 1e-9) & (raw[:, 0] <= end + 1e-9)]
            times = start + np.arange(1, 2561) * 0.005
            if selected.shape != (2560, 3) or not np.isfinite(selected).all() or not np.allclose(selected[:, 0], times, rtol=0, atol=1e-8):
                raise ValueError(f"force time grid differs: {name}/{object_name}")
            force_sha[object_name] = {
                str(item.relative_to(repo)): pre.sha256(item) for item in paths
            }
        rows[name] = {
            **expected,
            "solver_log_sha256": {str(item.relative_to(repo)): pre.sha256(item) for item in logs},
            "force_file_sha256": force_sha,
        }
    return {
        "status": "DIRECTPPO_TRAIN16_CURATION_AUTHORIZED_TRAIN_ONLY",
        "predeclaration_sha256": PREDECL_SHA,
        "expected_frames_per_case": 129,
        "expected_case_count": 16,
        "normalization_policy": "reuse byte-identical immutable full40 train20 transform",
        "validation_or_frozen_accessed": False,
        "cases": rows,
    }


def checked_auth(repo: Path) -> dict:
    saved = load(repo / AUTH)
    if saved != make_authorization(repo):
        raise ValueError("saved curation authorization is stale")
    return saved


def make_vtk_receipt(repo: Path, name: str) -> dict:
    auth = checked_auth(repo)
    if name not in auth["cases"]:
        raise ValueError("case outside authorization")
    pre = module(repo / PREDECL_MODULE, "directppo_train16_predecl_vtk")
    base = module(repo / BASE, "directppo_train16_base_vtk")
    case = repo / "cfd/tandem_cylinders/cases" / name
    files = sorted((case / "VTK_directppo_train16").glob("*/internal.vtu"))
    if len(files) != 129:
        raise ValueError(f"{name}: expected 129 VTK files")
    source = base.VTKSource(str(case / "VTK_directppo_train16"), file_pattern="*/internal.vtu", manifold_dim=3, point_source="vertices", backend="pyvista")
    actual = [float(next(source[i]).global_data["TimeValue"].reshape(-1)[0].item()) for i in range(len(source))]
    start, end = auth["cases"][name]["run_window"]
    # VTK TimeValue is float32 (observed quantization <=6.11e-6 near t=148).
    # The raw OpenFOAM directory/journal contract remains checked at 2e-6.
    if not np.allclose(actual, np.linspace(start, end, 129), rtol=0, atol=1e-5):
        raise ValueError(f"{name}: VTK times differ")
    log = case / "log.foamToVTK.directppo_train16"
    if not re.fullmatch(r"End:\s+\d+(?:\.\d+)? s, \d+ kB \(peak\)", last_nonblank(log)):
        raise ValueError(f"{name}: foamToVTK log incomplete")
    return {
        "status": "DIRECTPPO_TRAIN16_VTK_READY",
        "case": name,
        "split": "train",
        "frames": 129,
        "curation_authorization_sha256": pre.sha256(repo / AUTH),
        "foam_to_vtk_log_sha256": pre.sha256(log),
        "vtk_file_sha256": {str(item.relative_to(repo)): pre.sha256(item) for item in files},
    }


def canonical_time(actual: float, expected: float) -> float:
    """Snap reviewed float32 VTK metadata to its exact saved CFD endpoint."""
    if not math.isfinite(actual) or abs(actual - expected) > 1e-5:
        raise ValueError(
            f"VTK time cannot be matched to the reviewed endpoint: {actual} vs {expected}"
        )
    return float(expected)


def source(base, repo: Path, name: str, config: dict):
    class CanonicalTimeVTK:
        """Preserve the mesh while replacing only verified TimeValue metadata."""

        def __init__(self, wrapped, expected_times):
            self.wrapped = wrapped
            self.expected_times = list(map(float, expected_times))
            if len(wrapped) != len(self.expected_times):
                raise ValueError("VTK frame count differs before time canonicalization")

        def __len__(self):
            return len(self.wrapped)

        def relative_path(self, index):
            return self.wrapped.relative_path(index)

        def __getitem__(self, index):
            def generate():
                for mesh in self.wrapped[index]:
                    actual = float(mesh.global_data["TimeValue"].reshape(-1)[0].item())
                    exact = canonical_time(actual, self.expected_times[index])
                    mesh.global_data["TimeValue"] = base.torch.tensor(
                        [exact], dtype=base.torch.float64, device=mesh.points.device
                    )
                    yield mesh

            return generate()

    class DirectPPOSource(base.TandemTrajectorySource):
        def __init__(self):
            self.cases_root = repo / "cfd/tandem_cylinders/cases"
            self.profile = base.MATCHED_START_PROFILE
            self.records = [{"name": name, "split": "train", "config": config}]
            self.nx, self.ny = 256, 128
            self.x = base.np.linspace(8, 25, self.nx, dtype=base.np.float32)
            self.y = base.np.linspace(4, 11, self.ny, dtype=base.np.float32)
            yy, xx = base.torch.meshgrid(base.torch.from_numpy(self.y), base.torch.from_numpy(self.x), indexing="ij")
            self.query_points = base.torch.stack([xx.reshape(-1), yy.reshape(-1), base.torch.full((self.nx * self.ny,), 0.05)], dim=1)
            vtk = base.VTKSource(str(self.cases_root / name / "VTK_directppo_train16"), file_pattern="*/internal.vtu", manifold_dim=3, point_source="vertices", backend="pyvista", key_filters=[{"path_pattern": "**/internal.vtu", "mode": "include", "keys": ["U", "p"]}])
            if len(vtk) != 129:
                raise ValueError(f"{name}: expected 129 VTK frames")
            expected_times = base.np.linspace(
                float(config["start_time"]), float(config["end_time"]), 129
            )
            self.vtk_sources = [CanonicalTimeVTK(vtk, expected_times)]
    return DirectPPOSource()


def curate(repo: Path, name: str) -> None:
    auth = checked_auth(repo)
    if name not in auth["cases"] or load(repo / VTK_RECEIPTS / f"{name}.json") != make_vtk_receipt(repo, name):
        raise ValueError("case VTK receipt is absent or stale")
    case = repo / "cfd/tandem_cylinders/cases" / name
    config = load(case / "case_config.json")
    expected = auth["cases"][name]
    config.update(
        start_time=expected["run_window"][0], end_time=expected["run_window"][1],
        expected_frames=129, action_points=expected["action_points"],
        source_force_sha256=FORCE_SHA,
    )
    base = module(repo / BASE, "directppo_train16_base_curate")
    base.validate_matched_start_source_force(repo / "cfd/tandem_cylinders/cases", name, config)
    pipe = source(base, repo, name, config).filter(base.NumericalQualityFilter(0.75)).write(base.TrajectoryHDF5Sink(repo / OUTPUT, atomic_tmp=True))
    result = base.run_pipeline(pipe, n_jobs=1, backend="sequential", indices=None, use_tui=False)
    if len(result) != 1 or not result[0]:
        raise RuntimeError("official Curator pipeline returned no HDF")


def finalize(repo: Path) -> dict:
    import h5py
    pre = module(repo / PREDECL_MODULE, "directppo_train16_predecl_final")
    auth = checked_auth(repo)
    root = repo / OUTPUT
    paths = sorted((root / "train").glob("*.h5"))
    if {path.stem for path in paths} != set(auth["cases"]):
        raise ValueError("directppo train16 HDF set differs")
    digests = {}
    for path in paths:
        with h5py.File(path) as handle:
            if handle["state"].shape != (129, 3, 128, 256) or handle["force"].shape != (129, 4):
                raise ValueError(f"HDF shape differs: {path.stem}")
            expected = auth["cases"][path.stem]
            start, end = expected["run_window"]
            times = np.linspace(start, end, 129)
            actions = np.asarray(expected["action_points"])[:, 1]
            if not np.allclose(handle["time"][:, 0], times, rtol=0, atol=1e-5) or not np.allclose(handle["omega"][:, 0], actions, rtol=0, atol=1e-6):
                raise ValueError(f"HDF time/action differs: {path.stem}")
            for key in ("state", "force"):
                for begin in range(0, 129, 16):
                    if not np.isfinite(handle[key][begin:begin + 16]).all():
                        raise ValueError(f"nonfinite HDF: {path.stem}/{key}")
        digests[path.name] = pre.sha256(path)
    src, dst = repo / FULL40 / "normalization.json", root / "normalization.json"
    if dst.exists():
        raise FileExistsError(dst)
    with tempfile.NamedTemporaryFile(dir=root, delete=False) as stream:
        temp = Path(stream.name)
    try:
        shutil.copyfile(src, temp)
        os.link(temp, dst)
    finally:
        temp.unlink(missing_ok=True)
    if pre.sha256(dst) != "f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1":
        raise ValueError("normalization copy differs")
    payload = {
        "schema_version": 1,
        "profile": "directppo_train16_v1",
        "status": "DIRECTPPO_TRAIN16_TRAIN_ONLY_CURATED",
        "trajectory_counts": {"train": 16, "validation": 0, "frozen_test": 0},
        "frames_per_trajectory": 129,
        "max_abs_omega": 0.75,
        "normalization_sha256": pre.sha256(dst),
        "curation_authorization_sha256": pre.sha256(repo / AUTH),
        "validation_or_frozen_accessed": False,
        "source_scope": "historical changing exploratory PPO policies; not final-policy on-policy samples",
        "hdf_sha256": digests,
        "official_pipeline": ["PhysicsNeMo Curator Source/Filter/Sink", "VTKSource", "run_pipeline"],
    }
    pre.exclusive(root / "manifest.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("authorize", "vtk-receipt", "curate", "finalize"))
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--case")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approval-token")
    args = parser.parse_args()
    repo = args.repo.resolve()
    if not EXECUTION_REVIEWED or not args.execute or args.approval_token != TOKEN:
        raise RuntimeError("explicit reviewed train16 Curator token required")
    pre = module(repo / PREDECL_MODULE, "directppo_train16_predecl_main")
    if args.mode == "authorize":
        pre.exclusive(repo / AUTH, make_authorization(repo))
    elif args.mode == "vtk-receipt":
        pre.exclusive(repo / VTK_RECEIPTS / f"{args.case}.json", make_vtk_receipt(repo, args.case))
    elif args.mode == "curate":
        curate(repo, args.case)
    else:
        print(json.dumps(finalize(repo), indent=2))


if __name__ == "__main__":
    main()
