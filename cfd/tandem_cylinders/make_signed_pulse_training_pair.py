#!/usr/bin/env python3
"""Freeze a paired, train-only real-CFD acquisition near signed unit rotation."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from pathlib import Path

from make_control_landscape_panel import CASES, SOURCE_CASE, file_sha256
from make_expanded_control_dataset import replace_once, replace_rear_patch

START_TIME = 82.0
END_TIME = 162.0
INTERVAL = 0.1
NAMES = {
    "train_signed_pulse_p_v4_20261003": 1.0,
    "train_signed_pulse_m_v4_20261003": -1.0,
}


def pulse_amplitude(cycle: int) -> float:
    return 0.75 if cycle % 2 == 0 else 1.25


def unsigned_action(time: float) -> float:
    tau = time - START_TIME
    if not 0 <= tau <= END_TIME - START_TIME + 1e-9:
        raise ValueError("action time outside frozen trajectory")
    if math.isclose(tau, END_TIME - START_TIME, abs_tol=1e-9):
        return 0.0
    cycle = int(tau // 20.0)
    within = tau - 20.0 * cycle
    amplitude = pulse_amplitude(cycle)
    if within < 2.0:
        return amplitude * within / 2.0
    if within < 10.0:
        return amplitude
    if within < 12.0:
        return amplitude * (12.0 - within) / 2.0
    return 0.0


def action_points(sign: float) -> list[tuple[float, float]]:
    if sign not in (-1.0, 1.0):
        raise ValueError("only predeclared mirror signs are permitted")
    return [
        (round(START_TIME + index * INTERVAL, 10),
         round(sign * unsigned_action(START_TIME + index * INTERVAL), 12))
        for index in range(801)
    ]


def metrics(points: list[tuple[float, float]]) -> dict:
    if len(points) != 801 or points[0] != (START_TIME, 0.0) or points[-1] != (END_TIME, 0.0):
        raise ValueError("invalid paired training schedule")
    peak = max(abs(omega) for _, omega in points)
    rate = max(abs((b[1] - a[1]) / (b[0] - a[0])) for a, b in zip(points, points[1:]))
    if peak > 1.25 + 1e-9 or rate > 0.625 + 1e-9:
        raise ValueError("training action exceeds declared amplitude/rate")
    return {"max_abs_omega": peak, "max_abs_domega_dt": rate}


def audit() -> dict:
    positive = action_points(1.0)
    negative = action_points(-1.0)
    if any(t0 != t1 or abs(w0 + w1) > 1e-10
           for (t0, w0), (t1, w1) in zip(positive, negative)):
        raise ValueError("train-only schedules are not exact signed mirrors")
    return {
        "dataset": "tandem_control_gap_targeted_train_v4",
        "status": "PREDECLARED_TRAIN_ONLY_ACTIONS_NO_CFD_LABELS",
        "split": "train_only",
        "source_restart": "tandem_backward_dt005/t=82",
        "time_window": [START_TIME, END_TIME],
        "action_design": "paired signed pulses, amplitudes 0.75/1.25 alternate every 20 units; two-unit ramps and eight-unit holds",
        "selection_reason": "validation action-ranking ambiguity near signed unit rotation; this is targeted acquisition, not a proven active-learning gain",
        "cases": {name: {"sign": sign, **metrics(action_points(sign))} for name, sign in NAMES.items()},
        "leakage_guard": "distinct train phase t=82; do not import t=80 validation panel or frozen test labels",
    }


def generate(name: str) -> Path:
    sign = NAMES[name]
    target = CASES / name
    if target.exists():
        raise FileExistsError(target)
    source_time = SOURCE_CASE / "82"
    for field in ("U", "p"):
        if not (source_time / field).is_file():
            raise FileNotFoundError(source_time / field)
    points = action_points(sign)
    action_metrics = metrics(points)
    with tempfile.TemporaryDirectory(prefix=f".{name}.", dir=CASES) as temporary:
        stage = Path(temporary) / name
        stage.mkdir()
        shutil.copytree(SOURCE_CASE / "constant", stage / "constant")
        shutil.copytree(SOURCE_CASE / "system", stage / "system")
        shutil.copytree(source_time, stage / "82")
        velocity = stage / "82/U"
        velocity.write_text(replace_rear_patch(velocity.read_text(), points))
        control_path = stage / "system/controlDict"
        control = control_path.read_text()
        control = replace_once(control, "startTime 0;", "startTime 82;", control_path)
        control = replace_once(control, "endTime 160;", "endTime 162;", control_path)
        control = replace_once(control, "writeInterval 2;", "writeInterval 0.1;", control_path)
        control_path.write_text(control)
        metadata = {
            "case": name,
            "dataset": "tandem_control_gap_targeted_train_v4",
            "split": "train",
            "action_kind": "signed_low_amplitude_pulses",
            "sign": sign,
            "action_points": points,
            "action_metrics": action_metrics,
            "action_interpolation": "OpenFOAM Function1 table, linear",
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
            "field_write_interval": INTERVAL,
            "expected_frames": 801,
            "force_series": ["front Cd/Cl", "rear Cd/Cl"],
            "purpose": "train-only targeted real-CFD label acquisition for signed-action control ranking",
            "limitations": "one phase and two schedules; no demonstrated AL efficiency or control benefit",
        }
        (stage / "case_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
        stage.rename(target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="default: both mirror schedules")
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    selected = args.names or list(NAMES)
    if any(name not in NAMES for name in selected) or len(set(selected)) != len(selected):
        parser.error("unknown or duplicate training-pair name")
    if args.audit:
        if args.audit.exists():
            parser.error(f"refusing to overwrite {args.audit}")
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text(json.dumps(audit(), indent=2) + "\n")
    for name in selected:
        print(generate(name))


if __name__ == "__main__":
    main()
