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
import hashlib
import json
import os
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Any, ClassVar

import h5py
import numpy as np
import torch
from physicsnemo.mesh.spatial import BVH
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

PROFILE_COUNTS = {
    "stage1": {"train": 15, "validation": 3, "test": 3},
    "expanded_v1": {"train": 24, "validation": 4, "test": 4},
    "expanded_with_replacements": {"train": 24, "validation": 5, "test": 5},
    "expanded_independent_v2": {"train": 24, "validation": 4, "test": 4},
    "gate_b_aug_v3": {"train": 26, "validation": 4, "test": 5},
    "control_gap_v4": {"train": 28, "validation": 4, "test": 5},
    "phase_v1": {"train": 2, "validation": 1, "test": 1},
    "low_action_phase94_validation_v1": {"train": 0, "validation": 2, "test": 0},
    "matched_start_commissioning_train9_v1": {"train": 9, "validation": 0, "test": 0},
}
PROFILE_ACTION_LIMITS = {
    "stage1": 1.0,
    "expanded_v1": 5.0,
    "expanded_with_replacements": 5.0,
    "expanded_independent_v2": 5.0,
    "gate_b_aug_v3": 5.0,
    "control_gap_v4": 5.0,
    "phase_v1": 5.0,
    "low_action_phase94_validation_v1": 0.75,
    "matched_start_commissioning_train9_v1": 0.75,
}
PROFILE_FRAME_COUNTS = {
    "stage1": 801,
    "expanded_v1": 801,
    "expanded_with_replacements": 801,
    "expanded_independent_v2": 801,
    "gate_b_aug_v3": 801,
    "control_gap_v4": 801,
    "phase_v1": 241,
    "low_action_phase94_validation_v1": 801,
    "matched_start_commissioning_train9_v1": 801,
}
MATCHED_START_PROFILE = "matched_start_commissioning_train9_v1"
MATCHED_START_FORCE_OBJECTS = ("forceFront", "forceRear")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_matched_start_source_force(
    cases_root: Path, case_name: str, config: dict[str, Any]
) -> dict[str, str]:
    """Validate exact baseline force files before any VTK sampling or t0 read."""
    declared = config.get("source_force_sha256")
    if not isinstance(declared, dict) or set(declared) != set(
        MATCHED_START_FORCE_OBJECTS
    ):
        raise ValueError(
            f"{case_name}: source_force_sha256 must contain exact front/rear keys"
        )
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in declared.values()
    ):
        raise ValueError(f"{case_name}: invalid source-force SHA")
    source_name = config.get("source_restart_case")
    if source_name != "tandem_backward_dt005":
        raise ValueError(f"{case_name}: source_restart_case mismatch")
    for object_name in MATCHED_START_FORCE_OBJECTS:
        path = (
            cases_root
            / source_name
            / "postProcessing"
            / object_name
            / "0/coefficient.dat"
        )
        if not path.is_file() or sha256(path) != declared[object_name]:
            raise ValueError(
                f"{case_name}: baseline source-force SHA mismatch: {object_name}"
            )
    return declared


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


def load_merged_coefficients(paths: list[Path]) -> np.ndarray:
    """Merge restart-split force files, rejecting conflicting duplicate times."""
    if not paths:
        raise FileNotFoundError("no force coefficient files")
    samples: dict[float, np.ndarray] = {}
    for path in paths:
        for row in load_coefficients(path):
            key = round(float(row[0]), 8)
            if key in samples and not np.allclose(
                samples[key], row, rtol=0.0, atol=1.0e-10
            ):
                raise ValueError(f"conflicting force restart row at t={key}: {path}")
            samples[key] = row
    return np.asarray([samples[key] for key in sorted(samples)], dtype=np.float64)


def case_records(
    cases_root: Path, profile: str, selected_names: set[str] | None = None
) -> list[dict[str, Any]]:
    if profile == MATCHED_START_PROFILE:
        if not selected_names or len(selected_names) != 1:
            raise ValueError(
                "commissioning curation requires exactly one explicit --cases name"
            )
        name = next(iter(selected_names))
        case = cases_root / name
        config = json.loads((case / "case_config.json").read_text(encoding="utf-8"))
        validate_matched_start_source_force(cases_root, name, config)
        if config.get("split") != "train":
            raise ValueError(f"{name}: commissioning case must have split=train")
        if int(config.get("expected_field_frames", -1)) != 801:
            raise ValueError(f"{name}: expected_field_frames must equal 801")
        return [{"name": name, "split": "train", "config": config}]
    records = []
    if profile == "stage1":
        pattern = "dynamic_*"
    elif profile == "phase_v1":
        pattern = "phase_*"
    elif profile == "low_action_phase94_validation_v1":
        pattern = "validation_signed_low_pulse_*_phase94_v1_20261003"
    else:
        pattern = "expanded_*"
    augmentation = {"expanded_train_24", "expanded_train_25", "expanded_test_05"}
    excluded = {
        "expanded_v1": {"expanded_validation_04", "expanded_test_04"} | augmentation,
        "expanded_with_replacements": augmentation,
        "expanded_independent_v2": {"expanded_validation_03", "expanded_test_03"} | augmentation,
        "gate_b_aug_v3": {"expanded_validation_03", "expanded_test_03"},
        "control_gap_v4": {"expanded_validation_03", "expanded_test_03"},
    }.get(profile, set())
    for case in sorted(cases_root.glob(pattern)):
        if case.name in excluded:
            continue
        config = json.loads((case / "case_config.json").read_text(encoding="utf-8"))
        records.append({"name": case.name, "split": config["split"], "config": config})
    if profile == "control_gap_v4":
        for name in ("train_signed_pulse_p_v4_20261003", "train_signed_pulse_m_v4_20261003"):
            case = cases_root / name
            config = json.loads((case / "case_config.json").read_text(encoding="utf-8"))
            if config["split"] != "train" or config["dataset"] != "tandem_control_gap_targeted_train_v4":
                raise ValueError(f"{name}: not a train-only targeted CFD case")
            records.append({"name": name, "split": "train", "config": config})
    if profile == "stage1":
        for name, split in CONSTANT_SPLITS.items():
            case = cases_root / name
            config = json.loads((case / "case_config.json").read_text(encoding="utf-8"))
            records.append({"name": name, "split": split, "config": config})
    expected = PROFILE_COUNTS[profile]
    actual = {
        split: sum(record["split"] == split for record in records) for split in expected
    }
    if actual != expected:
        raise ValueError(
            f"unexpected trajectory split counts: {actual}, expected {expected}"
        )
    if selected_names is not None:
        available = {record["name"] for record in records}
        missing = selected_names - available
        if missing:
            raise ValueError(
                f"selected cases are not part of {profile}: {sorted(missing)}"
            )
        records = [record for record in records if record["name"] in selected_names]
    return records


def action_at(times: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    if "action_points" in config:
        table = np.asarray(config["action_points"], dtype=np.float64)
        return np.interp(times, table[:, 0], table[:, 1]).astype(np.float32)
    return np.full(
        times.shape, config["rear_angular_velocity_rad_per_s"], dtype=np.float32
    )


class TandemTrajectorySource(Source[dict[str, Any]]):
    name: ClassVar[str] = "Tandem-cylinder VTK trajectories"
    description: ClassVar[str] = (
        "Read OpenFOAM VTK sequences and sample a fixed Cartesian crop"
    )

    @classmethod
    def params(cls) -> list[Param]:
        return [
            Param(name="cases_root", description="OpenFOAM cases directory", type=str)
        ]

    def __init__(
        self,
        cases_root: Path,
        nx: int,
        ny: int,
        profile: str = "stage1",
        selected_names: set[str] | None = None,
    ) -> None:
        self.cases_root = cases_root
        self.profile = profile
        self.records = case_records(cases_root, profile, selected_names)
        self.nx = nx
        self.ny = ny
        self.x = np.linspace(8.0, 25.0, nx, dtype=np.float32)
        self.y = np.linspace(4.0, 11.0, ny, dtype=np.float32)
        yy, xx = torch.meshgrid(
            torch.from_numpy(self.y), torch.from_numpy(self.x), indexing="ij"
        )
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
                    {
                        "path_pattern": "**/internal.vtu",
                        "mode": "include",
                        "keys": ["U", "p"],
                    }
                ],
            )
            for record in self.records
        ]
        for record, vtk_source in zip(self.records, self.vtk_sources, strict=True):
            expected_frames = int(
                record["config"].get(
                    "expected_frames", PROFILE_FRAME_COUNTS[self.profile]
                )
            )
            if len(vtk_source) != expected_frames:
                raise ValueError(
                    f"{record['name']}: expected {expected_frames} VTK files, "
                    f"found {len(vtk_source)}"
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
        bvh = None
        reference_points = None
        reference_cells = None
        for frame_index in range(len(vtk_source)):
            mesh = next(vtk_source[frame_index])
            if "TimeValue" not in mesh.global_data:
                raise ValueError(
                    f"{record['name']}:{vtk_source.relative_path(frame_index)} has no TimeValue"
                )
            time = float(mesh.global_data["TimeValue"].reshape(-1)[0].item())
            query_points = self.query_points.to(
                dtype=mesh.points.dtype, device=mesh.points.device
            )
            if bvh is None:
                reference_points = mesh.points
                reference_cells = mesh.cells
                bvh = BVH.from_mesh(mesh)
            elif not (
                torch.equal(mesh.points, reference_points)
                and torch.equal(mesh.cells, reference_cells)
            ):
                raise ValueError(
                    f"{record['name']}: moving or reordered VTK mesh at frame {frame_index}"
                )
            sampled = mesh.sample_data_at_points(
                query_points, data_source="points", bvh=bvh
            )
            if "U" not in sampled or "p" not in sampled:
                raise ValueError(
                    f"{record['name']}:{vtk_source.relative_path(frame_index)} has no sampled U/p"
                )
            velocity = (
                sampled["U"].detach().cpu().numpy().astype(np.float32, copy=False)
            )
            pressure = (
                sampled["p"]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
                .reshape(-1)
            )
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
        expected_frames = int(
            record["config"].get(
                "expected_frames", PROFILE_FRAME_COUNTS[self.profile]
            )
        )
        start_time = float(record["config"].get("start_time", 80.0))
        end_time = float(record["config"].get("end_time", 160.0))
        expected_times = np.linspace(start_time, end_time, expected_frames)
        if not np.allclose(times_array, expected_times, atol=2e-6):
            raise ValueError(f"{record['name']}: unexpected field time sequence")

        for frame, mask in zip(state_array, mask_array, strict=True):
            valid = mask[0].astype(bool)
            frame[2, valid] -= frame[2, valid].mean(dtype=np.float64)

        force_root = f"{start_time:g}" if "source_restart_case" in record["config"] else "0"
        aligned = []
        for object_name in ("forceFront", "forceRear"):
            if self.profile == MATCHED_START_PROFILE:
                raw = load_merged_coefficients(
                    sorted(
                        case.glob(
                            f"postProcessing/{object_name}/*/coefficient.dat"
                        )
                    )
                )
            else:
                raw = load_coefficients(
                    case / "postProcessing" / object_name / force_root / "coefficient.dat"
                )
            if times_array[0] < raw[0, 0] - 1e-8:
                source_name = record["config"].get("source_restart_case")
                if not source_name:
                    raise ValueError(f"{record['name']}: missing exact force restart provenance")
                source = load_coefficients(
                    case.parent / source_name / "postProcessing"
                    / object_name / "0" / "coefficient.dat"
                )
                matches = np.flatnonzero(
                    np.isclose(source[:, 0], times_array[0], rtol=0, atol=1e-8)
                )
                if len(matches) != 1:
                    raise ValueError(f"{record['name']}: no unique source force at restart")
                raw = np.concatenate((source[matches], raw), axis=0)
            if times_array[0] < raw[0, 0] - 1e-8 or times_array[-1] > raw[-1, 0] + 1e-8:
                raise ValueError(f"{record['name']}: force series does not cover field times")
            aligned.extend(
                [np.interp(times_array, raw[:, 0], raw[:, column]) for column in (1, 2)]
            )
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
    description: ClassVar[str] = (
        "Reject incomplete, non-finite, or empty curated trajectories"
    )

    @classmethod
    def params(cls) -> list[Param]:
        return []

    def __init__(self, max_abs_omega: float = 1.0) -> None:
        self.max_abs_omega = float(max_abs_omega)

    def __call__(
        self, items: Generator[dict[str, Any], None, None]
    ) -> Generator[dict[str, Any], None, None]:
        for item in items:
            expected_frames = int(item["config"].get("expected_frames", 801))
            if item["state"].shape[0] != expected_frames or item["state"].shape[1] != 3:
                raise ValueError(
                    f"{item['case']}: invalid state shape {item['state'].shape}"
                )
            if (
                not np.isfinite(item["state"]).all()
                or not np.isfinite(item["force"]).all()
            ):
                raise ValueError(f"{item['case']}: non-finite curated value")
            coverage = item["mask"].mean()
            if not 0.8 < coverage < 1.0:
                raise ValueError(
                    f"{item['case']}: unexpected valid-grid coverage {coverage}"
                )
            if np.max(np.abs(item["omega"])) > self.max_abs_omega + 1.0e-6:
                raise ValueError(
                    f"{item['case']}: action outside profile range +/-{self.max_abs_omega}"
                )
            yield item


class TrajectoryHDF5Sink(Sink[dict[str, Any]]):
    name: ClassVar[str] = "Tandem trajectory HDF5 writer"
    description: ClassVar[str] = (
        "Write one compressed, indexed HDF5 file per trajectory"
    )

    @classmethod
    def params(cls) -> list[Param]:
        return [Param(name="output_dir", description="Curated dataset root", type=str)]

    def __init__(self, output_dir: Path, atomic_tmp: bool = False) -> None:
        self.output_dir = output_dir
        self.atomic_tmp = bool(atomic_tmp)

    def __call__(self, items: Iterator[dict[str, Any]], index: int) -> list[str]:
        paths = []
        for item in items:
            target = self.output_dir / item["split"] / f"{item['case']}.h5"
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".h5.tmp") if self.atomic_tmp else target
            if target.exists() or temporary.exists():
                raise FileExistsError(f"refusing to overwrite {target}")
            with h5py.File(temporary, "w") as handle:
                handle.create_dataset(
                    "state",
                    data=item["state"],
                    chunks=(1, 3, item["state"].shape[2], item["state"].shape[3]),
                    compression="gzip",
                    compression_opts=4,
                )
                handle.create_dataset(
                    "mask",
                    data=item["mask"],
                    chunks=(1, 1, item["mask"].shape[2], item["mask"].shape[3]),
                    compression="gzip",
                    compression_opts=1,
                )
                handle.create_dataset(
                    "omega", data=item["omega"], compression="gzip", compression_opts=1
                )
                handle.create_dataset(
                    "force", data=item["force"], compression="gzip", compression_opts=4
                )
                handle.create_dataset("time", data=item["time"])
                handle.create_dataset("x", data=item["x"])
                handle.create_dataset("y", data=item["y"])
                handle.attrs["case"] = item["case"]
                handle.attrs["split"] = item["split"]
                handle.attrs["state_channels"] = json.dumps(
                    ["u", "v", "gauge_pressure"]
                )
                handle.attrs["force_channels"] = json.dumps(
                    ["front_cd", "front_cl", "rear_cd", "rear_cl"]
                )
                handle.attrs["config_json"] = json.dumps(item["config"])
                handle.flush()
            if self.atomic_tmp:
                # link() is an atomic create-if-absent operation. Unlike
                # replace(), it cannot silently overwrite a concurrently
                # created final HDF5 path.
                os.link(temporary, target)
                temporary.unlink()
            paths.append(str(target))
        return paths


def normalization(output_dir: Path) -> dict[str, Any]:
    state_sum = np.zeros(3, dtype=np.float64)
    state_sq = np.zeros(3, dtype=np.float64)
    state_count = np.zeros(3, dtype=np.int64)
    state_min = np.full(3, np.inf, dtype=np.float64)
    state_max = np.full(3, -np.inf, dtype=np.float64)
    all_force_sum = np.zeros(4, dtype=np.float64)
    all_force_sq = np.zeros(4, dtype=np.float64)
    force_count = 0
    for path in sorted((output_dir / "train").glob("*.h5")):
        with h5py.File(path, "r") as handle:
            state = handle["state"][:]
            valid = handle["mask"][:, 0].astype(bool)
            for channel in range(3):
                values = state[:, channel][valid]
                state_sum[channel] += values.sum(dtype=np.float64)
                state_sq[channel] += np.square(values, dtype=np.float64).sum(
                    dtype=np.float64
                )
                state_count[channel] += len(values)
                state_min[channel] = min(state_min[channel], values.min())
                state_max[channel] = max(state_max[channel], values.max())
            force = handle["force"][:]
            all_force_sum += force.sum(axis=0, dtype=np.float64)
            all_force_sq += np.square(force, dtype=np.float64).sum(axis=0, dtype=np.float64)
            force_count += len(force)
    state_mean = state_sum / state_count
    all_force_mean = all_force_sum / force_count
    state_std = np.sqrt(np.maximum(state_sq / state_count - state_mean**2, 1e-12))
    all_force_std = np.sqrt(
        np.maximum(all_force_sq / force_count - all_force_mean**2, 1e-12)
    )
    state_abs_normalized_max = np.maximum(
        np.abs(state_min - state_mean), np.abs(state_max - state_mean)
    ) / state_std
    return {
        "state_channels": ["u", "v", "gauge_pressure"],
        "state_mean": state_mean.tolist(),
        "state_std": state_std.tolist(),
        "state_abs_normalized_channel_max_train": state_abs_normalized_max.tolist(),
        "state_abs_normalized_max_train": float(state_abs_normalized_max.max()),
        "state_support_computed_from": "exact train split valid cells",
        "force_channels": ["rear_cd", "rear_cl"],
        "force_mean": all_force_mean[2:4].tolist(),
        "force_std": all_force_std[2:4].tolist(),
        "all_force_channels": ["front_cd", "front_cl", "rear_cd", "rear_cl"],
        "all_force_mean": all_force_mean.tolist(),
        "all_force_std": all_force_std.tolist(),
        "computed_from": "train split only",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases-root", type=Path, default=Path("cfd/tandem_cylinders/cases")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/curated/tandem_cylinders")
    )
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--profile", choices=sorted(PROFILE_COUNTS), default="stage1")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Curator workers; use process_pool when greater than one",
    )
    parser.add_argument(
        "--backend", choices=("sequential", "process_pool"), default="sequential"
    )
    parser.add_argument("--cases", nargs="+", help="curate only these profile cases")
    parser.add_argument(
        "--defer-finalize",
        action="store_true",
        help="write selected HDF5 files without normalization or manifest",
    )
    parser.add_argument(
        "--atomic-hdf5",
        action="store_true",
        help="write case.h5.tmp and atomically rename only after HDF5 close",
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="write normalization and manifest after all HDF5 files exist",
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    if args.profile == "matched_start_commissioning_train9_v1":
        if args.finalize_only:
            parser.error(
                "commissioning profile requires finalize_matched_start_commissioning.py"
            )
        if (
            not args.cases or len(args.cases) != 1
            or not args.defer_finalize or not args.atomic_hdf5
        ):
            parser.error(
                "commissioning requires one --cases name, --defer-finalize, and --atomic-hdf5"
            )
    if args.backend == "sequential" and args.jobs != 1:
        parser.error("the sequential backend requires --jobs 1")
    if args.finalize_only and (args.cases or args.limit or args.defer_finalize):
        parser.error(
            "--finalize-only cannot be combined with --cases, --limit, or --defer-finalize"
        )
    incremental = bool(args.cases and args.defer_finalize)
    if (
        args.output.exists()
        and any(args.output.iterdir())
        and not (incremental or args.finalize_only)
    ):
        raise FileExistsError(
            f"refusing to overwrite non-empty output directory: {args.output}"
        )

    if args.finalize_only:
        expected = PROFILE_COUNTS[args.profile]
        actual = {
            split: len(list((args.output / split).glob("*.h5")))
            for split in ("train", "validation", "test")
        }
        if actual != expected:
            raise ValueError(
                f"cannot finalize incomplete dataset: {actual}, expected {expected}"
            )
        stats = normalization(args.output)
        (args.output / "normalization.json").write_text(
            json.dumps(stats, indent=2) + "\n", encoding="utf-8"
        )
        manifest = {
            "schema_version": 1,
            "profile": args.profile,
            "trajectory_counts": actual,
            "frames_per_trajectory": PROFILE_FRAME_COUNTS[args.profile],
            "pairs_per_trajectory": PROFILE_FRAME_COUNTS[args.profile] - 1,
            "grid": {
                "nx": args.nx,
                "ny": args.ny,
                "x_range": [8, 25],
                "y_range": [4, 11],
            },
            "fields": [
                "u",
                "v",
                "gauge_pressure",
                "valid_mask",
                "rear_omega",
                "front_cd_cl",
                "rear_cd_cl",
            ],
            "pressure_preprocessing": "subtract valid-domain spatial mean independently at every frame",
            "normalization": "normalization.json, train split only",
            "max_abs_omega": PROFILE_ACTION_LIMITS[args.profile],
            "curator_pipeline": [
                "TandemTrajectorySource[VTKSource + Mesh.sample_data_at_points]",
                "NumericalQualityFilter",
                "TrajectoryHDF5Sink",
            ],
        }
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(manifest, indent=2))
        return

    source = TandemTrajectorySource(
        args.cases_root.resolve(),
        args.nx,
        args.ny,
        args.profile,
        set(args.cases) if args.cases else None,
    )
    pipeline = source.filter(
        NumericalQualityFilter(PROFILE_ACTION_LIMITS[args.profile])
    ).write(TrajectoryHDF5Sink(args.output.resolve(), atomic_tmp=args.atomic_hdf5))
    indices = range(min(args.limit, len(source))) if args.limit else None
    results = run_pipeline(
        pipeline, n_jobs=args.jobs, backend=args.backend, indices=indices, use_tui=False
    )
    if any(not paths for paths in results):
        raise RuntimeError("Curator pipeline returned one or more empty sink results")
    if args.limit or args.defer_finalize:
        print(
            f"Curator wrote {len(results)} trajectory file(s); normalization deferred"
        )
        return

    stats = normalization(args.output)
    (args.output / "normalization.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "profile": args.profile,
        "trajectory_counts": {
            split: len(list((args.output / split).glob("*.h5")))
            for split in ("train", "validation", "test")
        },
        "frames_per_trajectory": PROFILE_FRAME_COUNTS[args.profile],
        "pairs_per_trajectory": PROFILE_FRAME_COUNTS[args.profile] - 1,
        "grid": {"nx": args.nx, "ny": args.ny, "x_range": [8, 25], "y_range": [4, 11]},
        "fields": [
            "u",
            "v",
            "gauge_pressure",
            "valid_mask",
            "rear_omega",
            "front_cd_cl",
            "rear_cd_cl",
        ],
        "pressure_preprocessing": "subtract valid-domain spatial mean independently at every frame",
        "normalization": "normalization.json, train split only",
        "max_abs_omega": PROFILE_ACTION_LIMITS[args.profile],
        "curator_pipeline": [
            "TandemTrajectorySource[VTKSource + Mesh.sample_data_at_points]",
            "NumericalQualityFilter",
            "TrajectoryHDF5Sink",
        ],
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
