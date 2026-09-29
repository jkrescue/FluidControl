#!/usr/bin/env python3
"""Create a medium-grid constant-rotation case for the spatial check."""

from __future__ import annotations

import json
from pathlib import Path

from make_baselines import ROOT, write_case


NAME = "control_grid_p100_medium"


def main() -> None:
    case = ROOT / "cases" / NAME
    if case.exists():
        raise FileExistsError(f"refusing to overwrite existing case: {case}")
    write_case(NAME, tandem=True, baseline=True, refinement=2, delta_t=0.005, time_scheme="backward")
    path = case / "0" / "U"
    text = path.read_text(encoding="utf-8")
    old = "rearCylinder { type noSlip; }"
    new = (
        "rearCylinder\n    {\n"
        "        type rotatingWallVelocity;\n"
        "        origin (15 7.5 0);\n"
        "        axis (0 0 1);\n"
        "        omega 1;\n"
        "        value uniform (0 0 0);\n"
        "    }"
    )
    if text.count(old) != 1:
        raise ValueError(f"unexpected rear-cylinder patch in {path}")
    path.write_text(text.replace(old, new), encoding="utf-8")
    metadata = {
        "case": NAME,
        "purpose": "representative rotating-wall coarse/medium spatial-grid comparison",
        "source": "OpenFOAM v2512 pimpleFoam CFD; generated, not experimental data",
        "reynolds_number": 100,
        "rear_angular_velocity": 1.0,
        "rear_surface_speed_ratio": 0.5,
        "mesh_cells": 77160,
        "rear_cylinder_wall_faces": 192,
        "delta_t": 0.005,
        "time_scheme": "backward",
        "end_time": 160,
        "field_write_interval": 2,
        "comparison_case": "control_small_p100",
    }
    (case / "case_config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(case)


if __name__ == "__main__":
    main()
