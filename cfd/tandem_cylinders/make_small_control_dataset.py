#!/usr/bin/env python3
"""Create a small, real-CFD constant-rotation dataset without overwriting cases.

Each case uses the same Re=100 tandem-cylinder geometry as the verified static
baseline. The rear cylinder rotates about its fixed centre; the front cylinder
remains stationary. These are constant-action trajectories, not the paper's
time-varying reinforcement-learning policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from make_baselines import ROOT, write_case


OMEGAS = {
    "m100": -1.0,
    "m050": -0.5,
    "z000": 0.0,
    "p050": 0.5,
    "p100": 1.0,
}


def generate(label: str) -> Path:
    omega = OMEGAS[label]
    name = f"control_small_{label}"
    case = ROOT / "cases" / name
    if case.exists():
        raise FileExistsError(f"refusing to overwrite existing case: {case}")
    write_case(name, tandem=True, baseline=True, delta_t=0.005, time_scheme="backward")

    velocity_file = case / "0/U"
    velocity = velocity_file.read_text(encoding="utf-8")
    old = "rearCylinder { type noSlip; }"
    if velocity.count(old) != 1:
        raise ValueError(f"unexpected rear-cylinder patch in {velocity_file}")
    new = ("rearCylinder\n    {\n"
           "        type rotatingWallVelocity;\n"
           "        origin (15 7.5 0);\n"
           "        axis (0 0 1);\n"
           f"        omega {omega:g};\n"
           "        value uniform (0 0 0);\n"
           "    }")
    velocity_file.write_text(velocity.replace(old, new), encoding="utf-8")

    control_file = case / "system/controlDict"
    control = control_file.read_text(encoding="utf-8")
    if control.count("writeInterval 2;") != 1:
        raise ValueError(f"unexpected output interval in {control_file}")
    control_file.write_text(control.replace("writeInterval 2;", "writeInterval 0.1;"), encoding="utf-8")

    metadata = {
        "case": name,
        "source": "OpenFOAM v2512 pimpleFoam CFD; generated, not experimental data",
        "geometry": "two stationary-centre tandem cylinders, diameter 1, centres (10,7.5) and (15,7.5)",
        "reynolds_number": 100,
        "kinematic_viscosity": 0.01,
        "inlet_velocity": [1, 0, 0],
        "rear_angular_velocity_rad_per_s": omega,
        "rear_surface_speed_ratio": omega / 2,
        "front_angular_velocity_rad_per_s": 0,
        "mesh_cells": 19290,
        "delta_t": 0.005,
        "time_scheme": "backward",
        "end_time": 160,
        "field_write_interval": 0.1,
        "discard_for_learning_before_time": 80,
        "fields": ["U", "p"],
        "force_series": ["front Cd/Cl", "rear Cd/Cl"],
        "wake_velocity_probes": 32,
        "limitations": "coarse grid; constant action only; numerical grid convergence must be checked separately",
    }
    (case / "case_config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", nargs="+", choices=tuple(OMEGAS))
    labels = parser.parse_args().labels
    for label in labels:
        print(generate(label))


if __name__ == "__main__":
    main()
