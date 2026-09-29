#!/usr/bin/env python3
"""Validate CFD and PhysicsNeMo ParaView export files by reading every dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyvista as pv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    cfd_files = sorted((args.root / "cfd").glob("*/*.vtu"))
    prediction_files = sorted((args.root / "prediction").glob("*/*.vtr"))
    if not cfd_files or not prediction_files:
        raise FileNotFoundError("expected both cfd/*.vtu and prediction/*.vtr files")

    for path in cfd_files:
        grid = pv.read(path)
        if grid.n_points == 0 or not {"U", "p"}.issubset(grid.array_names):
            raise ValueError(f"invalid CFD VTK file: {path}")

    required = {
        "valid_mask",
        "ground_truth_U",
        "ground_truth_p",
        "prediction_U",
        "prediction_p",
        "absolute_error_U",
        "absolute_error_p",
    }
    for path in prediction_files:
        grid = pv.read(path)
        missing = required - set(grid.array_names)
        if grid.n_points == 0 or missing:
            raise ValueError(f"invalid prediction VTK file {path}; missing={sorted(missing)}")
        for name in required - {"valid_mask"}:
            if not np.isfinite(grid[name]).all():
                raise ValueError(f"non-finite array {name} in {path}")

    report = {
        "cfd_vtu_files": len(cfd_files),
        "prediction_vtr_files": len(prediction_files),
        "cfd_cases": sorted({path.parent.name for path in cfd_files}),
        "prediction_cases": sorted({path.parent.name for path in prediction_files}),
        "prediction_arrays": sorted(required),
        "status": "PARAVIEW_VTK_EXPORT_OK",
    }
    (args.root / "validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
