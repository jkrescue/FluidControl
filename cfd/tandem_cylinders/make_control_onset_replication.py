#!/usr/bin/env python3
"""Predeclare matched instant-onset CFD cases to resolve a lift discrepancy.

The completed long validation panel ramps omega over t=80..82. This
replication changes only the onset: omega=+/-1 immediately at t=80.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from make_control_landscape_long_panel import ACTION_INTERVAL, ANALYSIS_START, END_TIME, START_TIME
from make_control_landscape_panel import CASES, SOURCE_CASE, file_sha256
from make_expanded_control_dataset import replace_once, replace_rear_patch

PANEL = {
    "landscape_step_val_p100_20261003": 1.0,
    "landscape_step_val_m100_20261003": -1.0,
}


def action_points(omega: float) -> list[tuple[float, float]]:
    if omega not in (-1.0, 1.0):
        raise ValueError("only predeclared signed unit rotation is accepted")
    return [
        (round(START_TIME + i * ACTION_INTERVAL, 10), omega)
        for i in range(801)
    ]


def audit() -> dict:
    return {
        "panel": "tandem_rotation_onset_replication_validation_20261003",
        "status": "PREDECLARED_NO_CFD_RESULTS",
        "reason": "resolve ramp-vs-step lift-RMS discrepancy at matched restart and window",
        "source_restart": "tandem_backward_dt005/t=80",
        "analysis_window": [ANALYSIS_START, END_TIME],
        "changed_variable": "action onset only; instant unit rotation rather than two-time-unit ramp",
        "split": "validation_only_not_training_or_frozen_test",
        "cases": PANEL,
    }


def generate(name: str) -> Path:
    omega = PANEL[name]
    target = CASES / name
    if target.exists():
        raise FileExistsError(target)
    source_time = SOURCE_CASE / "80"
    for field in ("U", "p"):
        if not (source_time / field).is_file():
            raise FileNotFoundError(source_time / field)
    points = action_points(omega)
    with tempfile.TemporaryDirectory(prefix=f".{name}.", dir=CASES) as temporary:
        stage = Path(temporary) / name
        stage.mkdir()
        shutil.copytree(SOURCE_CASE / "constant", stage / "constant")
        shutil.copytree(SOURCE_CASE / "system", stage / "system")
        shutil.copytree(source_time, stage / "80")
        velocity = stage / "80/U"
        velocity.write_text(replace_rear_patch(velocity.read_text(), points))
        control_path = stage / "system/controlDict"
        control = control_path.read_text()
        control = replace_once(control, "startTime 0;", "startTime 80;", control_path)
        control = replace_once(control, "writeInterval 2;", "writeInterval 0.1;", control_path)
        if "endTime 160;" not in control:
            raise ValueError("unexpected source CFD endpoint")
        control_path.write_text(control)
        metadata = {
            "case": name,
            "panel": "tandem_rotation_onset_replication_validation_20261003",
            "split": "validation",
            "action_kind": "instant_constant",
            "omega_final": omega,
            "action_points": points,
            "source": "OpenFOAM v2512 pimpleFoam real CFD; not experimental data",
            "source_restart_case": SOURCE_CASE.name,
            "source_restart_time": START_TIME,
            "source_restart_u_sha256": file_sha256(source_time / "U"),
            "source_restart_p_sha256": file_sha256(source_time / "p"),
            "geometry": "two fixed-centre tandem cylinders D=1, centres (10,7.5) and (15,7.5)",
            "reynolds_number": 100,
            "kinematic_viscosity": 0.01,
            "mesh_cells": 19290,
            "delta_t": 0.005,
            "time_scheme": "backward",
            "start_time": START_TIME,
            "end_time": END_TIME,
            "analysis_window": [ANALYSIS_START, END_TIME],
            "field_write_interval": ACTION_INTERVAL,
            "expected_frames": 801,
            "force_series": ["front Cd/Cl", "rear Cd/Cl"],
            "limitations": "single onset comparison, coarse grid, no feedback or wall torque",
        }
        (stage / "case_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
        stage.rename(target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="default: both predeclared cases")
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    selected = args.names or list(PANEL)
    if any(name not in PANEL for name in selected) or len(selected) != len(set(selected)):
        parser.error("unknown or duplicate onset-replication name")
    if args.audit:
        if args.audit.exists():
            parser.error(f"refusing to overwrite {args.audit}")
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text(json.dumps(audit(), indent=2) + "\n")
    for name in selected:
        print(generate(name))


if __name__ == "__main__":
    main()
