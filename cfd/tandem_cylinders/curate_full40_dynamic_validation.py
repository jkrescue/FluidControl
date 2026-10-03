#!/usr/bin/env python3
"""Fail-closed validation-only Curator entry point for dynamic6."""

from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, tempfile
from pathlib import Path
import h5py, numpy as np

REPO = Path(__file__).resolve().parents[2]
PREDECL = Path(
    "artifacts/tandem_cylinders/full40_dynamic_validation_predeclared_20261003.json"
)
PREDECL_SHA = "0478c8532bd2ded504ccd5f89303001eb8359f69b3036f296e31428a085d1272"
RAW_QC = Path(
    "artifacts/tandem_cylinders/full40_dynamic_validation_real_cfd_qc_20261003.json"
)
AUTH = Path(
    "artifacts/tandem_cylinders/full40_dynamic_validation_curation_authorization_20261003.json"
)
OUTPUT = Path("data/curated/tandem_cylinders_full40_dynamic_validation_v1")
FULL40 = Path("data/curated/tandem_cylinders_matched_start_full40_v1")
BASE = Path("scripts/curate_low_action_phase94_validation.py")
FORCE_SHA = {
    "forceFront": "bce88443ce3d19a6411c31b9266af16a3dd4e992f7adb1ddabc659238a1e88d1",
    "forceRear": "654c5bbf64b758505bdfe4d379e153f460862902309e96d2b01974e45536680b",
}
EXPECTED = {
    f"full40_dynamic_validation_b{b:02d}_{p}"
    for b in (1, 5)
    for p in ("minus", "zero", "plus")
}
EXECUTION_REVIEWED = False
TOKEN = "EXECUTE_REVIEWED_DYNAMIC6_CURATOR"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def exclusive(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
        json.dump(payload, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(tmp, path)
    except FileExistsError as e:
        raise FileExistsError(f"refusing overwrite: {path}") from e
    finally:
        tmp.unlink(missing_ok=True)


def make_authorization(repo):
    pre = repo / PREDECL
    qcpath = repo / RAW_QC
    if sha(pre) != PREDECL_SHA:
        raise ValueError("predeclaration SHA differs")
    p, q = load(pre), load(qcpath)
    if (
        q.get("status") != "FULL40_DYNAMIC_VALIDATION_REAL_OPENFOAM_QC_COMPLETE"
        or q.get("predeclaration_sha256") != PREDECL_SHA
    ):
        raise ValueError("real-CFD aggregate QC differs")
    if set(p["cases"]) != EXPECTED:
        raise ValueError("case matrix differs")
    baseline = repo / "cfd/tandem_cylinders/cases/tandem_backward_dt005/postProcessing"
    for obj, digest in FORCE_SHA.items():
        if sha(baseline / obj / "0/coefficient.dat") != digest:
            raise ValueError(f"baseline force differs: {obj}")
    rows = {}
    for name in sorted(EXPECTED):
        case = repo / "cfd/tandem_cylinders/cases" / name
        expected = p["cases"][name]
        cp = case / "case_config.json"
        mp = case / "solver_complete.full40_dynamic_validation.json"
        sp = case / "solver_log_qc.full40_dynamic_validation.json"
        c, m, s = load(cp), load(mp), load(sp)
        if any(c.get(k) != v for k, v in expected.items()):
            raise ValueError(f"case contract differs: {name}")
        if (
            m.get("case") != name
            or m.get("predeclaration_sha256") != PREDECL_SHA
            or m.get("solver_qc_sha256") != sha(sp)
            or s.get("steps") != 4000
        ):
            raise ValueError(f"solver provenance differs: {name}")
        rows[name] = {
            "split": "validation",
            "phase_bin": expected["phase_bin"],
            "profile": expected["profile"],
            "source_state_sha256": expected["source_state_sha256"],
            "action_points": expected["action_points"],
            "run_window": expected["run_window"],
            "case_config_sha256": sha(cp),
            "solver_marker_sha256": sha(mp),
            "solver_qc_sha256": sha(sp),
        }
    return {
        "status": "DYNAMIC6_CURATION_AUTHORIZED_VALIDATION_ONLY",
        "predeclaration_sha256": PREDECL_SHA,
        "real_cfd_aggregate_qc": str(RAW_QC),
        "real_cfd_aggregate_qc_sha256": sha(qcpath),
        "baseline_force_source_sha256": FORCE_SHA,
        "expected_frames_per_case": 201,
        "force_channels": ["front_cd", "front_cl", "rear_cd", "rear_cl"],
        "normalization_policy": "reference full40 train20 only; do not compute from dynamic6",
        "training_access": "FORBIDDEN",
        "frozen_test_accessed": False,
        "cases": rows,
    }


def checked_auth(repo):
    saved = load(repo / AUTH)
    if saved != make_authorization(repo):
        raise ValueError("saved authorization is stale")
    return saved


def base_module(repo):
    spec = importlib.util.spec_from_file_location("dynamic6_base", repo / BASE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def source(base, repo, name, config):
    class DynamicSource(base.TandemTrajectorySource):
        def __init__(self):
            self.cases_root = repo / "cfd/tandem_cylinders/cases"
            self.profile = base.MATCHED_START_PROFILE
            self.records = [{"name": name, "split": "validation", "config": config}]
            self.nx, self.ny = 256, 128
            self.x = base.np.linspace(8, 25, self.nx, dtype=base.np.float32)
            self.y = base.np.linspace(4, 11, self.ny, dtype=base.np.float32)
            yy, xx = base.torch.meshgrid(
                base.torch.from_numpy(self.y),
                base.torch.from_numpy(self.x),
                indexing="ij",
            )
            self.query_points = base.torch.stack(
                [
                    xx.reshape(-1),
                    yy.reshape(-1),
                    base.torch.full((self.nx * self.ny,), 0.05),
                ],
                dim=1,
            )
            self.vtk_sources = [
                base.VTKSource(
                    str(self.cases_root / name / "VTK_curator"),
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
            if len(self.vtk_sources[0]) != 201:
                raise ValueError(f"{name}: expected 201 VTK frames")

    return DynamicSource()


def curate(repo, name):
    auth = checked_auth(repo)
    if name not in auth["cases"]:
        raise ValueError("case not authorized")
    config = dict(load(repo / "cfd/tandem_cylinders/cases" / name / "case_config.json"))
    config.update(source_force_sha256=FORCE_SHA, expected_frames=201)
    base = base_module(repo)
    base.validate_matched_start_source_force(
        repo / "cfd/tandem_cylinders/cases", name, config
    )
    pipe = (
        source(base, repo, name, config)
        .filter(base.NumericalQualityFilter(0.75))
        .write(base.TrajectoryHDF5Sink(repo / OUTPUT, atomic_tmp=True))
    )
    result = base.run_pipeline(
        pipe, n_jobs=1, backend="sequential", indices=None, use_tui=False
    )
    if len(result) != 1 or not result[0]:
        raise RuntimeError("Curator returned no HDF")


def finalize(repo):
    auth = checked_auth(repo)
    root = repo / OUTPUT
    paths = sorted((root / "validation").glob("*.h5"))
    if {p.stem for p in paths} != EXPECTED:
        raise ValueError("HDF set differs")
    digests = {}
    for path in paths:
        with h5py.File(path) as h:
            if h["state"].shape != (201, 3, 128, 256) or h["force"].shape != (201, 4):
                raise ValueError(f"shape differs: {path.stem}")
            start, end = auth["cases"][path.stem]["run_window"]
            times = np.linspace(start, end, 201)
            table = np.asarray(auth["cases"][path.stem]["action_points"])
            action = np.interp(times, table[:, 0], table[:, 1])
            if not np.allclose(
                h["time"][:, 0], times, rtol=0, atol=2e-6
            ) or not np.allclose(h["omega"][:, 0], action, rtol=0, atol=2e-6):
                raise ValueError(f"time/action differs: {path.stem}")
            if (
                not np.isfinite(h["state"][:]).all()
                or not np.isfinite(h["force"][:]).all()
            ):
                raise ValueError(f"nonfinite: {path.stem}")
        digests[path.name] = sha(path)
    full = repo / FULL40
    manifest = {
        "schema_version": 1,
        "profile": "full40_dynamic_validation_v1",
        "status": "DYNAMIC6_VALIDATION_ONLY_CURATED",
        "trajectory_counts": {"train": 0, "validation": 6, "frozen_test": 0},
        "frames_per_trajectory": 201,
        "pairs_per_trajectory": 200,
        "max_abs_omega": 0.75,
        "force_channels": ["front_cd", "front_cl", "rear_cd", "rear_cl"],
        "curation_authorization_sha256": sha(repo / AUTH),
        "normalization_reference": str(FULL40),
        "normalization_sha256": sha(full / "normalization.json"),
        "full40_manifest_sha256": sha(full / "manifest.json"),
        "training_access": "FORBIDDEN",
        "frozen_test_accessed": False,
        "hdf_sha256": digests,
        "official_pipeline": [
            "physicsnemo_curator Source/Filter/Sink",
            "VTKSource + PhysicsNeMo Mesh/BVH",
            "run_pipeline",
        ],
    }
    exclusive(root / "manifest.json", manifest)
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("authorize", "curate", "finalize"))
    p.add_argument("--repo", type=Path, default=REPO)
    p.add_argument("--case")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--approval-token")
    a = p.parse_args()
    repo = a.repo.resolve()
    if a.mode == "authorize":
        payload = make_authorization(repo)
        if a.execute:
            exclusive(repo / AUTH, payload)
        print(json.dumps(payload, indent=2))
        return
    if not a.execute:
        print(
            json.dumps(
                {
                    "status": "READY",
                    "authorization_sha256": sha(repo / AUTH),
                    "mode": a.mode,
                    "case": a.case,
                },
                indent=2,
            )
        )
        checked_auth(repo)
        return
    if not EXECUTION_REVIEWED or a.approval_token != TOKEN:
        p.error("execution not reviewed")
    if a.mode == "curate":
        if not a.case:
            p.error("--case required")
        curate(repo, a.case)
    else:
        print(json.dumps(finalize(repo), indent=2))


if __name__ == "__main__":
    main()
