#!/usr/bin/env python3
"""Sample one OpenFOAM VTU frame with Curator VTKSource and PhysicsNeMo Mesh."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from physicsnemo_curator.domains.mesh.sources.vtk import VTKSource


def sample_frame(vtk_root: Path, output: Path, nx: int = 256, ny: int = 128) -> None:
    """Project adapter: fresh official source/mesh each call; no persistent mesh cache."""
    source = VTKSource(
        str(vtk_root),
        file_pattern="*/internal.vtu",
        manifold_dim=3,
        point_source="vertices",
        backend="pyvista",
        key_filters=[
            {"path_pattern": "**/internal.vtu", "mode": "include", "keys": ["U", "p"]}
        ],
    )
    if len(source) != 1:
        raise ValueError(f"expected one VTU frame in {vtk_root}, found {len(source)}")
    mesh = next(source[0])
    if "TimeValue" not in mesh.global_data.keys():
        raise ValueError("VTU frame has no TimeValue")

    x = np.linspace(8.0, 25.0, nx, dtype=np.float32)
    y = np.linspace(4.0, 11.0, ny, dtype=np.float32)
    yy, xx = torch.meshgrid(torch.from_numpy(y), torch.from_numpy(x), indexing="ij")
    query = torch.stack(
        [xx.reshape(-1), yy.reshape(-1), torch.full((nx * ny,), 0.05)], dim=1
    ).to(dtype=mesh.points.dtype, device=mesh.points.device)
    sampled = mesh.sample_data_at_points(query, data_source="points")
    velocity = sampled["U"].detach().cpu().numpy().astype(np.float32, copy=False)
    pressure = sampled["p"].detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
    valid_flat = np.isfinite(velocity).all(axis=1) & np.isfinite(pressure)
    mask = valid_flat.astype(np.uint8).reshape(1, ny, nx)
    velocity = np.nan_to_num(velocity, copy=False)
    pressure = np.nan_to_num(pressure, copy=False)
    state = np.stack(
        [
            velocity[:, 0].reshape(ny, nx),
            velocity[:, 1].reshape(ny, nx),
            pressure.reshape(ny, nx),
        ]
    )
    valid = mask[0].astype(bool)
    state[:, ~valid] = 0.0
    state[2, valid] -= state[2, valid].mean(dtype=np.float64)
    time_value = float(mesh.global_data["TimeValue"].reshape(-1)[0].item())

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, state=state.astype(np.float32), mask=mask, time=np.asarray([time_value]),
        x=x, y=y,
    )
    print(
        f"CURATOR_FRAME_OK time={time_value:.10g} valid_coverage={mask.mean():.8f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vtk_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--ny", type=int, default=128)
    args = parser.parse_args()
    sample_frame(args.vtk_root, args.output, args.nx, args.ny)


if __name__ == "__main__":
    main()
