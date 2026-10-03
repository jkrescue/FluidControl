#!/usr/bin/env python3
"""Freeze long matched-start constant-rotation CFD cases before solving.

The 36-unit exploratory panel gave a short-window force response different
from the existing 80-unit constant-rotation panel. These three real-CFD
cases resolve whether that difference is a control transient.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from make_control_landscape_panel import CASES, SOURCE_CASE, file_sha256
from make_expanded_control_dataset import replace_once, replace_rear_patch

START_TIME = 80.0
END_TIME = 160.0
ANALYSIS_START = 120.0
ACTION_INTERVAL = 0.1


@dataclass(frozen=True)
class LongCase:
    name: str
    omega_final: float


LONG_PANEL = (
    LongCase("landscape_long_val_zero_20261003", 0.0),
    LongCase("landscape_long_val_p100_20261003", 1.0),
    LongCase("landscape_long_val_m100_20261003", -1.0),
)


def action_points(spec: LongCase) -> list[tuple[float, float]]:
    count = round((END_TIME - START_TIME) / ACTION_INTERVAL)
    return [
        (round(START_TIME + index * ACTION_INTERVAL, 10),
         spec.omega_final * min(index * ACTION_INTERVAL / 2.0, 1.0))
        for index in range(count + 1)
    ]


def validate_points(points: list[tuple[float, float]]) -> dict[str, float]:
    if len(points) != 801 or points[0][0] != START_TIME or points[-1][0] != END_TIME:
        raise ValueError("long action schedule has an invalid horizon")
    if points[0][1] != 0.0:
        raise ValueError("long action schedule must start from zero")
    rates = []
    for (t0, w0), (t1, w1) in zip(points, points[1:]):
        if t1 <= t0 or not all(math.isfinite(value) for value in (t0, t1, w0, w1)):
            raise ValueError("non-finite or non-monotone long action schedule")
        rates.append(abs((w1 - w0) / (t1 - t0)))
    peak = max(abs(omega) for _, omega in points)
    rate = max(rates)
    if peak > 1.0 + 1e-9 or rate > 0.5 + 1e-9:
        raise ValueError("long action schedule exceeds predeclared amplitude/rate")
    return {"max_abs_omega": peak, "max_abs_domega_dt": rate}


def audit() -> dict:
    return {
        "panel": "tandem_control_landscape_long_validation_20261003",
        "status": "PREDECLARED_ACTIONS_NO_CFD_RESULTS",
        "split": "validation_only_not_training_or_frozen_test",
        "source_restart": "tandem_backward_dt005/t=80",
        "time_window": [START_TIME, END_TIME],
        "analysis_window": [ANALYSIS_START, END_TIME],
        "cases": {spec.name: {"omega_final": spec.omega_final,
                              **validate_points(action_points(spec))}
                  for spec in LONG_PANEL},
    }


def generate(spec: LongCase) -> Path:
    source_time = SOURCE_CASE / "80"
    for name in ("U", "p"):
        if not (source_time / name).is_file():
            raise FileNotFoundError(source_time / name)
    target = CASES / spec.name
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    points = action_points(spec)
    metrics = validate_points(points)
    with tempfile.TemporaryDirectory(prefix=f".{spec.name}.", dir=CASES) as temporary:
        stage = Path(temporary) / spec.name
        stage.mkdir()
        shutil.copytree(SOURCE_CASE / "constant", stage / "constant")
        shutil.copytree(SOURCE_CASE / "system", stage / "system")
        shutil.copytree(source_time, stage / "80")
        velocity = stage / "80" / "U"
        velocity.write_text(replace_rear_patch(velocity.read_text(), points))
        control_path = stage / "system" / "controlDict"
        control = control_path.read_text()
        control = replace_once(control, "startTime 0;", "startTime 80;", control_path)
        control = replace_once(control, "writeInterval 2;", "writeInterval 0.1;", control_path)
        # The source endTime is already 160; assert it instead of assuming.
        if "endTime 160;" not in control:
            raise ValueError("source controlDict endTime is not 160")
        control_path.write_text(control)
        metadata = {
            "case": spec.name,
            "panel": "tandem_control_landscape_long_validation_20261003",
            "split": "validation",
            "action_kind": "long_zero" if spec.omega_final == 0 else "long_constant",
            "omega_final": spec.omega_final,
            "action_points": points,
            "action_metrics": metrics,
            "action_interpolation": "OpenFOAM Function1 scalar table, linear",
            "source": "generated OpenFOAM v2512 pimpleFoam CFD, not experimental data",
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
            "limitations": "coarse-grid, paired open-loop test; not a learned closed-loop policy",
        }
        (stage / "case_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
        stage.rename(target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="case names; default selects all")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    plan = audit()
    known = {spec.name: spec for spec in LONG_PANEL}
    unknown = set(args.names) - set(known)
    if unknown:
        parser.error(f"unknown case(s): {', '.join(sorted(unknown))}")
    selected = [known[name] for name in args.names] if args.names else list(LONG_PANEL)
    if args.audit:
        if args.audit.exists():
            parser.error(f"refusing to overwrite action audit: {args.audit}")
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text(json.dumps(plan, indent=2) + "\n")
    if args.list:
        for spec in selected:
            print(spec.name)
        return
    for spec in selected:
        print(generate(spec))


if __name__ == "__main__":
    main()
