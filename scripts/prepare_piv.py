"""Convert published PIV displacement fields to nondimensional snapshot arrays.

Raw Zenodo ZIP is never modified. Missing PIV vectors stay explicitly masked;
zero-filled placeholders are not treated as valid flow observations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat

RESOLUTION_M_PER_PX = 0.29076921e-3
DIAMETER_M = 0.03
U_INFINITY_M_PER_S = 0.31
SAMPLING_HZ = 120.0
VELOCITY_SCALE = RESOLUTION_M_PER_PX * SAMPLING_HZ / U_INFINITY_M_PER_S


def resize(values: np.ndarray, size: tuple[int, int], mode: str) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))
    if mode == "nearest":
        result = F.interpolate(tensor, size=size, mode=mode)
    else:
        result = F.interpolate(tensor, size=size, mode=mode, align_corners=False)
    return result.numpy()


def force_map(path: Path) -> dict[float, dict]:
    mat = loadmat(path)
    p = np.asarray(mat["p"], dtype=np.float64).ravel()
    cd = np.asarray(mat["CD"], dtype=np.float64).ravel()
    uncertainty = np.asarray(mat["Uncertainty"], dtype=np.float64).ravel()
    if not (len(p) == len(cd) == len(uncertainty)):
        raise ValueError("Force table column lengths do not match")
    values: dict[float, dict[str, float]] = {}
    for control in np.unique(np.round(p, 5)):
        indices = np.flatnonzero(np.isclose(p, control, atol=1e-5))
        finite_uncertainty = uncertainty[indices][np.isfinite(uncertainty[indices])]
        values[float(control)] = {
            "cd": float(np.mean(cd[indices])),
            "uncertainty": float(np.mean(finite_uncertainty)) if len(finite_uncertainty) else None,
            "replicates": len(indices),
        }
    return values


def process_case(
    h5_path: Path,
    destination: Path,
    forces: dict[float, dict],
    stride: int,
    size: tuple[int, int],
) -> dict:
    with h5py.File(h5_path, "r") as h5:
        if not all(name in h5 for name in ("U", "V", "X", "Y")):
            raise ValueError(f"Missing required PIV arrays in {h5_path}")
        control = float(np.asarray(h5.attrs["p"]).squeeze())
        stored = int(np.asarray(h5.attrs["Nsnapshots"]).squeeze())
        raw_shape = tuple(h5["U"].shape)
        if h5["U"].shape != h5["V"].shape:
            raise ValueError(f"PIV shape/snapshot attribute mismatch in {h5_path}")
        if raw_shape[0] == stored:
            frame_count = raw_shape[0]
            u = np.asarray(h5["U"][::stride, :, :], dtype=np.float32)
            v = np.asarray(h5["V"][::stride, :, :], dtype=np.float32)
        elif raw_shape[2] == stored:
            frame_count = raw_shape[2]
            u = np.moveaxis(np.asarray(h5["U"][:, :, ::stride], dtype=np.float32), -1, 0)
            v = np.moveaxis(np.asarray(h5["V"][:, :, ::stride], dtype=np.float32), -1, 0)
        else:
            raise ValueError(f"No HDF5 axis matches Nsnapshots={stored} in {h5_path}: {raw_shape}")
        x = np.asarray(h5["X"], dtype=np.float32)
        y = np.asarray(h5["Y"], dtype=np.float32)

    if x.shape != u.shape[1:] or y.shape != x.shape:
        raise ValueError(f"Coordinate and velocity grids differ in {h5_path}")
    x_axis0 = float(np.nanmean(np.abs(np.diff(x, axis=0))))
    x_axis1 = float(np.nanmean(np.abs(np.diff(x, axis=1))))
    if x_axis0 > x_axis1:
        u, v = u.transpose(0, 2, 1), v.transpose(0, 2, 1)
        x, y = x.T, y.T
    elif x_axis1 <= 0:
        raise ValueError(f"X grid does not vary along either axis in {h5_path}")

    key = min(forces, key=lambda item: abs(item - control))
    if abs(key - control) > 0.051:
        raise ValueError(f"No measured drag matching p={control} in {h5_path}")
    valid = np.isfinite(u) & np.isfinite(v)
    missing_fraction = float(1.0 - valid.mean())
    if missing_fraction > 0.2:
        raise ValueError(f"More than 20% missing PIV vectors in {h5_path}: {missing_fraction:.1%}")
    velocity = np.stack([np.where(valid, u, 0), np.where(valid, v, 0)], axis=1) * VELOCITY_SCALE
    valid = valid[:, None].astype(np.float32)
    velocity = resize(velocity, size, "bilinear")
    valid = resize(valid, size, "nearest").astype(bool)
    xy = np.stack([x, y], axis=0)[None]
    xy = resize(xy, size, "bilinear")[0]
    xy *= RESOLUTION_M_PER_PX / DIAMETER_M
    xy[1] -= float(np.nanmean(y) * RESOLUTION_M_PER_PX / DIAMETER_M)
    output = destination / f"{h5_path.stem}.npz"
    np.savez_compressed(
        output,
        velocity=velocity.astype(np.float32),
        valid=valid,
        xy=xy.astype(np.float32),
        p=np.float32(control),
        cd=np.float32(forces[key]["cd"]),
        cd_uncertainty=np.float32(forces[key]["uncertainty"] if forces[key]["uncertainty"] is not None else np.nan),
    )
    return {
        "source": h5_path.name,
        "processed": output.name,
        "p": control,
        "cd": forces[key]["cd"],
        "cd_uncertainty": forces[key]["uncertainty"],
        "original_snapshots": frame_count,
        "retained_snapshots": int(velocity.shape[0]),
        "original_hdf5_shape": list(raw_shape),
        "oriented_spatial_shape": list(u.shape[1:]),
        "processed_shape": list(velocity.shape),
        "missing_fraction_before_resize": missing_fraction,
        "x_nd_range": [float(np.nanmin(xy[0])), float(np.nanmax(xy[0]))],
        "y_nd_range": [float(np.nanmin(xy[1])), float(np.nanmax(xy[1]))],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path("data/raw/zenodo_20794709/ExperimentalDataset.zip"))
    parser.add_argument("--extracted", type=Path, default=Path("data/raw/zenodo_20794709/extracted"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/piv"))
    parser.add_argument("--stride", type=int, default=6, help="Retain every nth 120 Hz snapshot")
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=64)
    args = parser.parse_args()
    if args.stride < 1 or args.height < 16 or args.width < 16:
        raise ValueError("Invalid temporal stride or spatial size")
    args.extracted.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.archive) as archive:
        members = archive.namelist()
        mat_members = [name for name in members if name.endswith("AerodynamicForces.mat")]
        all_h5_members = [name for name in members if name.endswith(".h5")]
        # The repository README says *_zero* files were acquired without
        # actuation before each case. Their HDF5 p attribute still stores the
        # associated later actuation value, so using it as a label is wrong.
        reference_members = [name for name in all_h5_members if "_zero" in Path(name).stem.lower()]
        h5_members = [name for name in all_h5_members if name not in reference_members]
        if len(mat_members) != 1 or not h5_members:
            raise ValueError("Unexpected experimental archive layout")
        mat_path = Path(archive.extract(mat_members[0], args.extracted))
        forces = force_map(mat_path)
        cases = []
        for index, member in enumerate(h5_members, start=1):
            print(f"PIV case {index}/{len(h5_members)}: {member}", flush=True)
            h5_path = Path(archive.extract(member, args.extracted))
            cases.append(process_case(h5_path, args.out, forces, args.stride, (args.height, args.width)))
    report = {
        "source": "https://doi.org/10.5281/zenodo.20794709",
        "source_type": "experimental_time_resolved_PIV_and_force",
        "velocity_conversion": "raw pixel displacement * resolution_m_per_px * sampling_hz / U_infinity_m_per_s",
        "resolution_m_per_px": RESOLUTION_M_PER_PX,
        "diameter_m": DIAMETER_M,
        "u_infinity_m_per_s": U_INFINITY_M_PER_S,
        "sampling_hz": SAMPLING_HZ,
        "temporal_stride": args.stride,
        "retained_hz": SAMPLING_HZ / args.stride,
        "size_h_w": [args.height, args.width],
        "force_table": {str(key): value for key, value in forces.items()},
        "excluded_unactuated_reference_files": reference_members,
        "cases": cases,
    }
    (args.out / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Processed {len(cases)} cases into {args.out}", flush=True)


if __name__ == "__main__":
    main()
