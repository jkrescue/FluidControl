#!/usr/bin/env python3
"""Predeclare smooth zero-mean periodic-rotation CFD benchmarks."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from pathlib import Path

from make_control_landscape_long_panel import ACTION_INTERVAL, ANALYSIS_START, END_TIME, START_TIME
from make_control_landscape_panel import CASES, SOURCE_CASE, file_sha256
from make_expanded_control_dataset import replace_once, replace_rear_patch

PANEL = {
    "periodic_val_p10_20261003": 10.0,
    "periodic_val_p20_20261003": 20.0,
}
SHARPNESS = 3.0
OMEGA_LIMIT = 1.0
RATE_LIMIT = 2.0


def action_points(period: float) -> list[tuple[float, float]]:
    if period not in PANEL.values():
        raise ValueError("only predeclared periods are accepted")
    result = []
    for index in range(801):
        time = round(START_TIME + index * ACTION_INTERVAL, 10)
        phase = 2 * math.pi * (time - START_TIME) / period
        omega = math.tanh(SHARPNESS * math.sin(phase)) / math.tanh(SHARPNESS)
        result.append((time, round(omega, 12)))
    return result


def metrics(points: list[tuple[float, float]]) -> dict:
    if len(points) != 801 or points[0] != (80.0, 0.0) or points[-1][0] != END_TIME:
        raise ValueError("invalid periodic action grid")
    amplitude = max(abs(omega) for _, omega in points)
    rate = max(abs((b[1] - a[1]) / (b[0] - a[0])) for a, b in zip(points, points[1:]))
    if amplitude > OMEGA_LIMIT + 1e-9 or rate > RATE_LIMIT + 1e-9:
        raise ValueError("periodic schedule violates amplitude/rate bound")
    return {"max_abs_omega": amplitude, "max_abs_domega_dt": rate}


def audit() -> dict:
    return {
        "panel": "tandem_periodic_open_loop_benchmark_validation_20261003",
        "status": "PREDECLARED_NO_CFD_RESULTS",
        "split": "validation_only_not_training_or_frozen_test",
        "source_restart": "tandem_backward_dt005/t=80",
        "analysis_window": [ANALYSIS_START, END_TIME],
        "schedule": "omega=tanh(3*sin(2*pi*(t-80)/period))/tanh(3)",
        "acceptance_from_RESEARCH_OBJECTIVE_md": {
            "total_cd_reduction_min": 0.02,
            "rear_cl_fluctuation_rms_increase_max": 0.05,
            "abs_mean_rear_cl_over_zero_fluctuation_rms_max": 0.10,
        },
        "cases": {name: {"period": period, **metrics(action_points(period))}
                  for name, period in PANEL.items()},
    }


def generate(name: str) -> Path:
    period = PANEL[name]
    target = CASES / name
    if target.exists():
        raise FileExistsError(target)
    source_time = SOURCE_CASE / "80"
    for field in ("U", "p"):
        if not (source_time / field).is_file():
            raise FileNotFoundError(source_time / field)
    points = action_points(period)
    action_metrics = metrics(points)
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
            "panel": "tandem_periodic_open_loop_benchmark_validation_20261003",
            "split": "validation",
            "action_kind": "smooth_zero_mean_alternating",
            "period": period,
            "action_points": points,
            "action_metrics": action_metrics,
            "action_interpolation": "OpenFOAM Function1 table, linear",
            "source": "OpenFOAM v2512 pimpleFoam real CFD, not experimental data",
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
            "limitations": "coarse-grid one-phase open-loop baseline; not a learned closed-loop policy",
        }
        (stage / "case_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
        stage.rename(target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="default: both predeclared periods")
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    selected = args.names or list(PANEL)
    if any(name not in PANEL for name in selected) or len(selected) != len(set(selected)):
        parser.error("unknown or duplicate periodic case")
    if args.audit:
        if args.audit.exists():
            parser.error(f"refusing to overwrite {args.audit}")
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text(json.dumps(audit(), indent=2) + "\n")
    for name in selected:
        print(generate(name))


if __name__ == "__main__":
    main()
