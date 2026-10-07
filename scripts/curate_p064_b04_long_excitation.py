#!/usr/bin/env python3
"""Case-local B04 VTK -> official Curator HDF adapter.

Preparation is safe by default.  Execution requires an approved spec and
``--execute``.  The raw OpenFOAM case is mounted read-only; foamToVTK writes
only into an isolated view.  The existing official Source/Filter/Sink classes
perform sampling and HDF writing.  This module never trains a model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def exclusive_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def validate_spec(spec_path: Path, self_path: Path) -> tuple[dict, dict, dict]:
    spec = read(spec_path)
    if spec["status"] not in {"PREPARATION_ONLY", "EXECUTION_APPROVED"}:
        raise ValueError("invalid status")
    if sha256(self_path) != spec["driver_sha256"]:
        raise ValueError("driver SHA mismatch")
    for item in spec["inputs"].values():
        if sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"input SHA mismatch: {item['path']}")
    for item in spec["baseline_force_sources"].values():
        if sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"baseline force SHA mismatch: {item['path']}")
    raw_result = read(Path(spec["inputs"]["raw_result"]["path"]))
    if (
        raw_result["status"] != "P064_B04_LONG_EXCITATION_OPENFOAM_COMPLETE_PENDING_CURATOR"
        or raw_result["saved_frames"] != 801
        or raw_result["first_time"] != 120.0
        or raw_result["last_time"] != 200.0
        or raw_result["curator_executed"]
        or raw_result["training_executed"]
    ):
        raise ValueError("raw result contract mismatch")
    helper = load_module("p064_b04_action_contract", Path(spec["inputs"]["action_helper"]["path"]))
    contract = helper.build_contract(Path(spec["repo"]))
    if contract["action"]["points_sha256"] != raw_result["action_points_sha256"]:
        raise ValueError("action contract differs from raw result")
    return spec, raw_result, contract


def numeric_times(raw_case: Path) -> list[tuple[float, Path]]:
    rows = []
    for path in raw_case.iterdir():
        try:
            time = float(path.name)
        except ValueError:
            continue
        if 120.0 <= time <= 200.0:
            rows.append((time, path))
    rows.sort()
    expected = 120.0 + 0.1 * np.arange(801)
    actual = np.asarray([row[0] for row in rows], dtype=np.float64)
    if actual.shape != (801,) or not np.allclose(actual, expected, rtol=0, atol=1e-8):
        raise ValueError("raw 801-frame clock mismatch")
    return rows


def make_view(raw_case: Path, view: Path, case_name: str, action_points: list[list[float]]) -> None:
    if view.exists():
        raise FileExistsError(view)
    view.mkdir(parents=True)
    # Docker mounts raw_case at the identical absolute path, so these links are
    # valid both on the host and inside the container.  Raw remains read-only.
    for name in ("constant", "system", "postProcessing"):
        (view / name).symlink_to(raw_case / name, target_is_directory=True)
    for _, source in numeric_times(raw_case):
        (view / source.name).symlink_to(source, target_is_directory=True)
    config = {
        "case": case_name,
        "split": "train",
        "start_time": 120.0,
        "end_time": 200.0,
        "expected_frames": 801,
        "expected_field_frames": 801,
        "action_points": action_points,
        "source_restart_case": str(Path(
            "/workspace/fluid_control/"
            "cfd/tandem_cylinders/cases/tandem_backward_dt005"
        )),
        "source_force_sha256": {
            "forceFront": "bce88443ce3d19a6411c31b9266af16a3dd4e992f7adb1ddabc659238a1e88d1",
            "forceRear": "654c5bbf64b758505bdfe4d379e153f460862902309e96d2b01974e45536680b",
        },
    }
    (view / "case_config.json").write_text(json.dumps(config, indent=2) + "\n")


def run_foam_to_vtk(spec: dict, raw_case: Path, view: Path) -> None:
    image = spec["foam_to_vtk"]["image"]
    name = spec["foam_to_vtk"]["container_name"]
    if subprocess.check_output(["docker", "ps", "-aq", "--filter", f"name=^{name}$"], text=True).strip():
        raise RuntimeError("refusing pre-existing same-name container")
    cidfile = view / "foam.cid"
    command = [
        "docker", "run", "--rm", "--cidfile", str(cidfile), "--name", name,
        "--network", "none", "--read-only", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges", "--user", f"{os.getuid()}:{os.getgid()}",
        "--memory", "8g", "--memory-swap", "8g", "--cpus", "1",
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=1g",
        "--mount", f"type=bind,src={raw_case},dst={raw_case},readonly",
        "--mount", f"type=bind,src={view},dst={view}", "-w", str(view), image,
        "timeout", "-k", "10s", "900s", "foamToVTK", "-case", str(view),
        "-fields", "(U p)", "-no-boundary", "-name", "VTK_curator",
    ]
    def available_gib() -> float:
        text = Path("/proc/meminfo").read_text()
        kb = int(next(line.split()[1] for line in text.splitlines() if line.startswith("MemAvailable:")))
        return kb / 2**20
    if available_gib() < 50 or shutil.disk_usage(view).free / 2**30 < 20:
        raise RuntimeError("startup memory/disk guard failed")
    log_path = view / "log.foamToVTK"
    log_stream = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(command, text=True, stdout=log_stream, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + 920
    old_handlers = {}
    def interrupted(signum, _frame):
        raise InterruptedError(f"received signal {signum}")
    for signum in (signal.SIGTERM, signal.SIGINT):
        old_handlers[signum] = signal.signal(signum, interrupted)
    try:
        while process.poll() is None:
            if time.monotonic() > deadline or available_gib() < 22 or shutil.disk_usage(view).free / 2**30 < 20:
                raise RuntimeError("foamToVTK deadline/resource guard failed")
            time.sleep(.5)
        if process.returncode:
            raise RuntimeError(f"foamToVTK failed: {process.returncode}")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(10)
            except subprocess.TimeoutExpired:
                process.kill()
        log_stream.flush()
        os.fsync(log_stream.fileno())
        log_stream.close()
        if cidfile.is_file():
            cid = cidfile.read_text().strip()
            if cid:
                subprocess.run(["docker", "rm", "-f", cid], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


def validate_vtk(spec: dict, view: Path) -> dict:
    paths = sorted((view / "VTK_curator").glob("*/internal.vtu"))
    if len(paths) != 801:
        raise ValueError(f"expected 801 VTK files, found {len(paths)}")
    base = load_module("p064_b04_vtk_receipt_base", Path(spec["inputs"]["curator_base"]["path"]))
    source = base.VTKSource(
        str(view / "VTK_curator"), file_pattern="*/internal.vtu", manifold_dim=3,
        point_source="vertices", backend="pyvista",
    )
    times = []
    for index in range(len(source)):
        mesh = next(source[index])
        times.append(float(mesh.global_data["TimeValue"].reshape(-1)[0].item()))
    if not np.allclose(times, 120 + .1 * np.arange(801), rtol=0, atol=2e-5):
        raise ValueError("VTK directory clock mismatch")
    return {
        "frames": 801,
        "first_time": times[0],
        "last_time": times[-1],
        "actual_vtk_times_float64": times,
        "actual_vtk_times_float64_sha256": hashlib.sha256(np.asarray(times, np.float64).tobytes()).hexdigest(),
        "vtk_inventory_sha256": hashlib.sha256(
            json.dumps({str(p.relative_to(view)): sha256(p) for p in paths}, sort_keys=True).encode()
        ).hexdigest(),
    }


def official_source(base, cases_root: Path, name: str, config: dict):
    class B04Source(base.TandemTrajectorySource):
        def __init__(self):
            self.cases_root = cases_root
            self.profile = base.MATCHED_START_PROFILE
            self.records = [{"name": name, "split": "train", "config": config}]
            self.nx, self.ny = 256, 128
            self.x = base.np.linspace(8, 25, self.nx, dtype=base.np.float32)
            self.y = base.np.linspace(4, 11, self.ny, dtype=base.np.float32)
            yy, xx = base.torch.meshgrid(
                base.torch.from_numpy(self.y), base.torch.from_numpy(self.x), indexing="ij"
            )
            self.query_points = base.torch.stack([
                xx.reshape(-1), yy.reshape(-1),
                base.torch.full((self.nx * self.ny,), 0.05),
            ], dim=1)
            vtk = base.VTKSource(
                str(cases_root / name / "VTK_curator"), file_pattern="*/internal.vtu",
                manifold_dim=3, point_source="vertices", backend="pyvista",
                key_filters=[{"path_pattern": "**/internal.vtu", "mode": "include", "keys": ["U", "p"]}],
            )
            if len(vtk) != 801:
                raise ValueError(f"expected 801 VTK frames, found {len(vtk)}")
            self.vtk_sources = [vtk]
    return B04Source()


def curate(spec: dict, raw_case: Path, view: Path, output: Path) -> Path:
    base = load_module("p064_b04_curator_base", Path(spec["inputs"]["curator_base"]["path"]))
    config = read(view / "case_config.json")
    # The original validator intentionally requires a relative legacy name;
    # this adapter instead binds both absolute baseline force files in spec and
    # lets the inherited __getitem__ resolve the absolute source_restart_case.
    for body in ("forceFront", "forceRear"):
        item = spec["baseline_force_sources"][body]
        if sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"baseline force differs: {body}")
    pipeline = (
        official_source(base, view.parent, view.name, config)
        .filter(base.NumericalQualityFilter(0.75))
        .write(base.TrajectoryHDF5Sink(output, atomic_tmp=True))
    )
    result = base.run_pipeline(pipeline, n_jobs=1, backend="sequential", indices=None, use_tui=False)
    if len(result) != 1 or not result[0]:
        raise RuntimeError("official Curator pipeline returned no HDF")
    paths = sorted((output / "train").glob("*.h5"))
    if len(paths) != 1:
        raise ValueError("expected one train HDF")
    return paths[0]


def expected_force(spec: dict, raw_case: Path, times: np.ndarray) -> np.ndarray:
    rows = []
    for body in ("forceFront", "forceRear"):
        paths = sorted((raw_case / "postProcessing" / body).glob("*/coefficient.dat"))
        samples = np.concatenate([np.loadtxt(path, ndmin=2)[:, [0, 1, 4]] for path in paths])
        samples = samples[np.argsort(samples[:, 0])]
        baseline = np.loadtxt(spec["baseline_force_sources"][body]["path"], ndmin=2)
        initial = baseline[np.isclose(baseline[:, 0], 120, rtol=0, atol=1e-8)]
        if initial.shape != (1, 13):
            raise ValueError(f"no unique t120 baseline force: {body}")
        samples = np.concatenate([initial[:, [0, 1, 4]], samples])
        rows.extend([np.interp(times, samples[:, 0], samples[:, column]) for column in (1, 2)])
    return np.stack(rows, axis=1).astype(np.float32)


def validate_hdf(spec: dict, raw_case: Path, hdf: Path, output: Path) -> dict:
    norm_source = Path(spec["inputs"]["normalization"]["path"])
    norm_target = output / "normalization.json"
    if norm_target.exists():
        raise FileExistsError(norm_target)
    shutil.copyfile(norm_source, norm_target)
    if sha256(norm_target) != spec["inputs"]["normalization"]["sha256"]:
        raise ValueError("normalization bytes changed")
    with h5py.File(hdf) as handle:
        expected = {"state", "mask", "omega", "force", "time", "x", "y"}
        if not expected.issubset(handle.keys()):
            raise ValueError("HDF schema mismatch")
        if handle["state"].shape != (801, 3, 128, 256) or handle["force"].shape != (801, 4):
            raise ValueError("HDF shape mismatch")
        times = handle["time"][:, 0]
        vtk_times = np.asarray(read(Path(spec["view"]) / "vtk_result.json")["actual_vtk_times_float64"], np.float64)
        if vtk_times.shape != (801,) or not np.allclose(vtk_times, 120 + 0.1 * np.arange(801), rtol=0, atol=2e-5):
            raise ValueError("bound VTK clock mismatch")
        if not np.array_equal(times, vtk_times.astype(np.float32)):
            raise ValueError("HDF clock mismatch")
        actions = np.asarray(read(Path(spec["view"]) / "case_config.json")["action_points"])
        expected_omega = np.interp(vtk_times, actions[:, 0], actions[:, 1]).astype(np.float32)
        if not np.array_equal(handle["omega"][:, 0], expected_omega):
            raise ValueError("HDF omega differs from bound action table")
        if not np.array_equal(handle["force"][:], expected_force(spec, raw_case, vtk_times)):
            raise ValueError("HDF force differs from raw clock and t120 baseline")
        for key in ("state", "force", "omega"):
            ds = handle[key]
            for begin in range(0, len(ds), 16):
                if not np.isfinite(ds[begin:begin + 16]).all():
                    raise ValueError(f"nonfinite {key}")
        first_last = {
            key: hashlib.sha256(np.ascontiguousarray(np.stack([handle[key][0], handle[key][-1]])).tobytes()).hexdigest()
            for key in ("state", "mask", "omega", "force", "time")
        }
    manifest = {
        "schema_version": 1,
        "profile": "p064_b04_long_excitation_train_only",
        "trajectory_counts": {"train": 1, "validation": 0, "test": 0},
        "frames_per_trajectory": 801,
        "pairs_per_trajectory": 800,
        "max_abs_omega": 0.75,
        "normalization_status": "byte-exact reuse; no refit",
        "normalization_sha256": spec["inputs"]["normalization"]["sha256"],
        "validation_or_frozen_accessed": False,
        "hdf_sha256": sha256(hdf),
    }
    exclusive_json(output / "manifest.json", manifest)
    verifier = load_module("p064_b04_h100_verifier", Path(spec["inputs"]["h100_verifier"]["path"]))
    reader_evidence = verifier.verify_b04_h100(output, hdf)
    return {
        "hdf_sha256": sha256(hdf),
        "first_last_content_sha256": first_last,
        "official_reader_h100_evidence": reader_evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--mode", choices=("vtk", "curate"), required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if sha256(args.spec) != args.spec_sha256:
        raise ValueError("spec SHA mismatch")
    spec, raw_result, contract = validate_spec(args.spec, Path(__file__).resolve())
    if args.mode != spec["authorized_mode"]:
        raise PermissionError("requested mode is outside this spec")
    if not args.execute:
        print(f"P064_B04_{args.mode.upper()}_PREPARATION_ONLY_NO_RAW_READ")
        return
    if not spec["execution_authorized"]:
        raise PermissionError("execution is not authorized")
    raw_case = Path(spec["raw_case"])
    view, output = Path(spec["view"]), Path(spec["planned_output"])
    if args.mode == "vtk":
        if view.exists() or output.exists():
            raise FileExistsError("view/output already exists")
        make_view(raw_case, view, spec["case_name"], contract["action"]["points"])
        run_foam_to_vtk(spec, raw_case, view)
        evidence = validate_vtk(spec, view)
        exclusive_json(view / "vtk_result.json", {
            "status": "P064_B04_LONG_EXCITATION_VTK_READY_NOT_CURATED",
            "spec_sha256": args.spec_sha256,
            "driver_sha256": spec["driver_sha256"],
            "raw_result_sha256": spec["inputs"]["raw_result"]["sha256"],
            "curator_executed": False,
            "training_executed": False,
            **evidence,
        })
        return
    if not view.is_dir() or not (view / "vtk_result.json").is_file():
        raise FileNotFoundError("reviewed VTK result is absent")
    validate_vtk(spec, view)
    if output.exists():
        raise FileExistsError(output)
    hdf = curate(spec, raw_case, view, output)
    evidence = validate_hdf(spec, raw_case, hdf, output)
    result = {
        "status": "P064_B04_LONG_EXCITATION_CURATED_TRAIN_ONLY",
        "spec_sha256": args.spec_sha256,
        "driver_sha256": spec["driver_sha256"],
        "raw_result_sha256": spec["inputs"]["raw_result"]["sha256"],
        "raw_result_status": raw_result["status"],
        "frames": 801,
        "pipeline": ["PhysicsNeMo Curator Source/Filter/Sink APIs", "official VTKSource", "project NumericalQualityFilter", "project TrajectoryHDF5Sink", "official run_pipeline"],
        "normalization_reused_byte_exact": True,
        "validation_or_frozen_accessed": False,
        "training_executed": False,
        **evidence,
    }
    exclusive_json(output / "result.json", result)


if __name__ == "__main__":
    main()
