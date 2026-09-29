#!/usr/bin/env python3
"""Plot an actual OpenFOAM velocity snapshot exported with foamToVTK.

Run with ParaView's pvpython, which provides VTK and Matplotlib:
    pvpython plot_flow_snapshot.py input.vtu output.png
"""

import argparse
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.patches import Circle
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkFiltersCore import vtkCellCenters
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vtu", help="OpenFOAM foamToVTK internal.vtu")
    parser.add_argument("png", help="Output PNG")
    args = parser.parse_args()

    reader = vtkXMLUnstructuredGridReader()
    reader.SetFileName(args.vtu)
    reader.Update()
    mesh = reader.GetOutput()
    if mesh.GetNumberOfCells() == 0:
        raise ValueError("VTU contains no cells")
    velocity_array = mesh.GetCellData().GetArray("U")
    if velocity_array is None:
        raise ValueError("VTU has no cell-centered U field")

    centers = vtkCellCenters()
    centers.SetInputData(mesh)
    centers.Update()
    xy = vtk_to_numpy(centers.GetOutput().GetPoints().GetData())[:, :2]
    u = vtk_to_numpy(velocity_array)
    speed = np.linalg.norm(u[:, :2], axis=1)

    triangles = mtri.Triangulation(xy[:, 0], xy[:, 1])
    vertices = xy[triangles.triangles]
    edge_lengths = np.linalg.norm(vertices - np.roll(vertices, 1, axis=1), axis=2)
    centroids = vertices.mean(axis=1)
    mask = edge_lengths.max(axis=1) > 0.40
    for xc in (10.0, 15.0):
        mask |= np.linalg.norm(centroids - [xc, 7.5], axis=1) < 0.52
    triangles.set_mask(mask)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8.6), constrained_layout=True)
    panels = (
        (speed, "Speed magnitude |U|", "viridis", np.linspace(0, 1.35, 55), "m/s"),
        (u[:, 1], "Transverse velocity Uy", "RdBu_r", np.linspace(-0.75, 0.75, 61), "m/s"),
    )
    for ax, (field, title, cmap, levels, unit) in zip(axes, panels):
        artist = ax.tricontourf(triangles, field, levels=levels, cmap=cmap, extend="both")
        for xc in (10.0, 15.0):
            ax.add_patch(Circle((xc, 7.5), 0.5, facecolor="white", edgecolor="black", lw=1.5, zorder=10))
        ax.annotate("flow  →", (8.25, 10.1), fontsize=10, color="black",
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
        ax.set(xlim=(8, 26), ylim=(4.4, 10.6), xlabel="x / D", ylabel="y / D", title=title)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color="white", alpha=0.15, linewidth=0.5)
        cbar = fig.colorbar(artist, ax=ax, pad=0.015, shrink=0.92)
        cbar.set_label(unit)

    fig.suptitle("Two fixed cylinders in tandem · CFD snapshot at tU∞/D = 160\n"
                 "Re = 100 · L/D = 5 · OpenFOAM pimpleFoam · rear cylinder not rotating",
                 fontsize=14)
    fig.savefig(args.png, dpi=210, facecolor="white")
    print(f"Saved {args.png}; {mesh.GetNumberOfCells()} CFD cells; speed range {speed.min():.3f}–{speed.max():.3f}")


if __name__ == "__main__":
    main()
