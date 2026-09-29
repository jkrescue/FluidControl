#!/usr/bin/env python3
"""Curate OpenFOAM VTK trajectories into split HDF5 files.

This script deliberately uses PhysicsNeMo Curator's Source -> Filter -> Sink
pipeline.  It does not train a model.  Raw VTK fields are sampled onto one
fixed Cartesian crop, pressure gauge is removed per frame, force coefficients
are aligned to the saved field times, and train-only normalization statistics
are recorded.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Any, ClassVar

import h5py
import numpy as np
import torch
from physicsnemo_curator.core.base import Filter, Param, Sink, Source
from physicsnemo_curator.domains.mesh.sources.vtk import VTKSource
from physicsnemo_curator.run import run_pipeline


CONSTANT_SPLITS = {
    "control_small_m100": "train",
    "control_small_z000": "train",
    "control_small_p100": "train",
    "control_small_m050": "validation",
    "control_small_p050": "test",
}


def load_coefficients(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) >= 5:
            rows.append((float(fields[0]), float(fields[1]), float(fields[4])))
    result = np.asarray(rows, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3 or len(result) < 2:
        raise ValueError(f"invalid force coefficient file: {path}")
    return result


def case_records(cases_root: Path) -> list[dict[str, Any]]:
    records = []
    for case in sorted(cases_root.glob("dynamic_*")):
        config = json.loads((case / "case_config.json").read_text(encoding="utf-8"))
        records.append({"name": case.name, "split": config["split"], "config": config})
    for name, split in CONSTANT_SPLITS.items():
        case = cases_root / name
        config = json.loads((case / "case_config.json").read_text(encoding="utf-8"))
        records.append({"name": name, "split": split, "config": config})
    expected = {"train": 15, "validation": 3, "test": 3}
    actual = {split: sum(record["split"] == split for record in records) for split in expected}
    if actual != expected:
        raise ValueError(f"unexpected trajectory split counts: {actual}, expected {expected}")
    return records


def action_at(times: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    if "action_points" in config:
        table = np.asarray(config["action_points"], dtype=np.float64)
        return np.interp(times, table[:, 0], table[:, 1]).astype(np.float32)
    return np.full(times.shape, config["rear_angular_velocity_rad_per_s"], dtype=np.float32)


class TandemTrajectorySource(Source[dict[str, Any]]):
    name: ClassVar[str] = "Tandem-cylinder VTK trajectories"
    description: ClassVar[str] = "Read OpenFOAM VTK sequences and sample a fixed Cartesian crop"

    @classmethod
    def params(cls) -> list[Param]:
        return [Param(name="cases_root", description="OpenFOAM cases directory", type=str)]

    def __init__(self, cases_root: Path, nx: int, ny: int) -> None:
        self.cases_root = cases_root
        self.records = case_records(cases_root)
        self.nx = nx
        self.ny = ny
        self.x = np.linspace(8.0, 25.0, nx, dtype=np.float32)
        self.y = np.linspace(4.0, 11.0, ny, dtype=np.float32)
        yy, xx = torch.meshgrid(torch.from_numpy(self.y), torch.from_numpy(self.x), indexing="ij")
        self.query_points = torch.stack(
            [xx.reshape(-1), yy.reshape(-1), torch.full((nx * ny,), 0.05)], dim=1
        )
        self.vtk_sources = [
            VTKSource(
                str(cases_root / record["name"] / "VTK_curator"),
                file_pattern="*/internal.vtu",
                manifold_dim=3,
                point_source="vertices",
                backend="pyvista",
                key_filters=[
                    {"path_pattern": "**/internal.vtu", "mode": "include", "keys": ["U", "p"]}
                ],
            )
            for record in self.records
        ]
        for record, vtk_source in zip(self.records, self.vtk_sources, strict=True):
            if len(vtk_source) != 801:
                raise ValueError(
                    f"{record['name']}: expected 801 VTK files, found {len(vtk_source)}"
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Generator[dict[str, Any], None, None]:
        record = self.records[index]
        case = self.cases_root / record["name"]
        vtk_source = self.vtk_sources[index]

        frames = []
        masks = []
        times = []
        for frame_index in range(len(vtk_source)):
            mesh = next(vtk_source[frame_index])
            if "TimeValue" not in mesh.global_data.keys():
                raise ValueError(
                    f"{record['name']}:{vtk_source.relative_path(frame_index)} has no TimeValue"
                )
            time = float(mesh.global_data["TimeValue"].reshape(-1)[0].item())
            query_points = self.query_points.to(dtype=mesh.points.dtype, device=mesh.points.device)
            sampled = mesh.sample_data_at_points(query_points, data_source="points")
            if "U" not in sampled.keys() or "p" not in sampled.keys():
                raise ValueError(
                    f"{record['name']}:{vtk_source.relative_path(frame_index)} has no sampled U/p"
                )
            velocity = sampled["U"].detach().cpu().numpy().astype(np.float32, copy=False)
            pressure = sampled["p"].detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
            valid_flat = np.isfinite(velocity).all(axis=1) & np.isfinite(pressure)
            valid = valid_flat.astype(np.uint8).reshape(self.ny, self.nx)
            velocity = np.nan_to_num(velocity, copy=False)
            pressure = np.nan_to_num(pressure, copy=False)
            state = np.stack(
                [
                    velocity[:, 0].reshape(self.ny, self.nx),
                    velocity[:, 1].reshape(self.ny, self.nx),
                    pressure.reshape(self.ny, self.nx),
                ]
            )
            state[:, valid == 0] = 0.0
            frames.append(state)
            masks.append(valid[None])
            times.append(time)
            completed = frame_index + 1
            if completed == 1 or completed % 100 == 0 or completed == len(vtk_source):
                print(
                    f"{record['name']}: sampled {completed}/{len(vtk_source)} "
                    f"VTK frames with Curator VTKSource + PhysicsNeMo Mesh",
                    flush=True,
                )

        order = np.argsort(times)
        times_array = np.asarray(times, dtype=np.float64)[order]
        state_array = np.asarray(frames, dtype=np.float32)[order]
        mask_array = np.asarray(masks, dtype=np.uint8)[order]
        if not np.allclose(times_array, np.linspace(80.0, 160.0, 801), atol=2e-6):
            raise ValueError(f"{record['name']}: unexpected field time sequence")

        for frame, mask in zip(state_array, mask_array, strict=True):
            valid = mask[0].astype(bool)
            frame[2, valid] -= frame[2, valid].mean(dtype=np.float64)

        force_root = "80" if record["name"].startswith("dynamic_") else "0"
        aligned = []
        for object_name in ("forceFront", "forceRear"):
            raw = load_coefficients(case / "postProcessing" / object_name / force_root / "coefficient.dat")
            aligned.extend([np.interp(times_array, raw[:, 0], raw[:, column]) for column in (1, 2)])
        force = np.stack(aligned, axis=1).astype(np.float32)

        yield {
            "case": record["name"],
            "split": record["split"],
            "time": times_array.astype(np.float32)[:, None],
            "state": state_array,
            "mask": mask_array,
            "omega": action_at(times_array, record["config"])[:, None],
            "force": force,
            "x": self.x,
            "y": self.y,
            "config": record["config"],
        }


class NumericalQualityFilter(Filter[dict[str, Any]]):
    name: ClassVar[str] = "Tandem CFD numerical quality"
    description: ClassVar[str] = "Reject incomplete, non-finite, or empty curated trajectories"

    @classmethod
    def params(cls) -> list[Param]:
        return []

    def __call__(self, items: Generator[dict[str, Any], None, None]) -> Generator[dict[str, Any], None, None]:
        for item in items:
            if item["state"].shape[0] != 801 or item["state"].shape[1] != 3:
                raise ValueError(f"{item['case']}: invalid state shape {item['state'].shape}")
            if not np.isfinite(item["state"]).all() or not np.isfinite(item["force"]).all():
                raise ValueError(f"{item['case']}: non-finite curated value")
            coverage = item["mask"].mean()
            if not 0.8 < coverage < 1.0:
                raise ValueError(f"{item['case']}: unexpected valid-grid coverage {coverage}")
            if np.max(np.abs(item["omega"])) > 1.000001:
                raise ValueError(f"{item['case']}: action outside first-stage range")
            yield item


class TrajectoryHDF5Sink(Sink[dict[str, Any]]):
    name: ClassVar[str] = "Tandem trajectory HDF5 writer"
    description: ClassVar[str] = "Write one compressed, indexed HDF5 file per trajectory"

    @classmethod
    def params(cls) -> list[Param]:
        return [Param(name="output_dir", description="Curated dataset root", type=str)]

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def __call__(self, items: Iterator[dict[str, Any]], index: int) -> list[str]:
        paths = []
        for item in items:
            target = self.output_dir / item["split"] / f"{item['case']}.h5"
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(f"refusing to overwrite {target}")
            with h5py.File(target, "w") as handle:
                handle.create_dataset("state", data=item["state"], chunks=(1, 3, item["state"].shape[2], item["state"].shape[3]), compression="gzip", compression_opts=4)
                handle.create_dataset("mask", data=item["mask"], chunks=(1, 1, item["mask"].shape[2], item["mask"].shape[3]), compression="gzip", compression_opts=1)
                handle.create_dataset("omega", data=item["omega"], compression="gzip", compression_opts=1)
                handle.create_dataset("force", data=item["force"], compression="gzip", compression_opts=4)
                handle.create_dataset("time", data=item["time"])
                handle.create_dataset("x", data=item["x"])
                handle.create_dataset("y", data=item["y"])
                handle.attrs["case"] = item["case"]
                handle.attrs["split"] = item["split"]
                handle.attrs["state_channels"] = json.dumps(["u", "v", "gauge_pressure"])
                handle.attrs["force_channels"] = json.dumps(["front_cd", "front_cl", "rear_cd", "rear_cl"])
                handle.attrs["config_json"] = json.dumps(item["config"])
            paths.append(str(target))
        return paths


def normalization(output_dir: Path) -> dict[str, Any]:
    state_sum = np.zeros(3, dtype=np.float64)
    state_sq = np.zeros(3, dtype=np.float64)
    state_count = np.zeros(3, dtype=np.int64)
    force_sum = np.zeros(2, dtype=np.float64)
    force_sq = np.zeros(2, dtype=np.float64)
    force_count = 0
    for path in sorted((output_dir / "train").glob("*.h5")):
        with h5py.File(path, "r") as handle:
            state = handle["state"][:]
            valid = handle["mask"][:, 0].astype(bool)
            for channel in range(3):
                values = state[:, channel][valid]
                state_sum[channel] += values.sum(dtype=np.float64)
                state_sq[channel] += np.square(values, dtype=np.float64).sum(dtype=np.float64)
                state_count[channel] += len(values)
            force = handle["force"][:, 2:4]
            force_sum += force.sum(axis=0, dtype=np.float64)
            force_sq += np.square(force, dtype=np.float64).sum(axis=0, dtype=np.float64)
            force_count += len(force)
    state_mean = state_sum / state_count
    force_mean = force_sum / force_count
    state_std = np.sqrt(np.maximum(state_sq / state_count - state_mean**2, 1e-12))
    force_std = np.sqrt(np.maximum(force_sq / force_count - force_mean**2, 1e-12))
    return {
        "state_channels": ["u", "v", "gauge_pressure"],
        "state_mean": state_mean.tolist(),
        "state_std": state_std.tolist(),
        "force_channels": ["rear_cd", "rear_cl"],
        "force_mean": force_mean.tolist(),
        "force_std": force_std.tolist(),
        "computed_from": "train split only",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-root", type=Path, default=Path("cfd/tandem_cylinders/cases"))
    parser.add_argument("--output", type=Path, default=Path("data/curated/tandem_cylinders"))
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {args.output}")

    source = TandemTrajectorySource(args.cases_root.resolve(), args.nx, args.ny)
    pipeline = source.filter(NumericalQualityFilter()).write(TrajectoryHDF5Sink(args.output.resolve()))
    indices = range(min(args.limit, len(source))) if args.limit else None
    results = run_pipeline(
        pipeline, n_jobs=1, backend="sequential", indices=indices, use_tui=False
    )
    if args.limit:
        print(f"Curator smoke run wrote {len(results)} trajectory file(s); normalization deferred")
        return

    stats = normalization(args.output)
    (args.output / "normalization.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "trajectory_counts": {split: len(list((args.output / split).glob("*.h5"))) for split in ("train", "validation", "test")},
        "frames_per_trajectory": 801,
        "pairs_per_trajectory": 800,
        "grid": {"nx": args.nx, "ny": args.ny, "x_range": [8, 25], "y_range": [4, 11]},
        "fields": ["u", "v", "gauge_pressure", "valid_mask", "rear_omega", "front_cd_cl", "rear_cd_cl"],
        "pressure_preprocessing": "subtract valid-domain spatial mean independently at every frame",
        "normalization": "normalization.json, train split only",
        "curator_pipeline": [
            "TandemTrajectorySource[VTKSource + Mesh.sample_data_at_points]",
            "NumericalQualityFilter",
            "TrajectoryHDF5Sink",
        ],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
