#!/usr/bin/env python3
"""Freeze a paired OpenFOAM action panel for tandem-cylinder control value.

All four exploratory validation cases start from the same uncontrolled t=80
state.  No frozen test trajectory or model prediction is used to choose them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from make_expanded_control_dataset import replace_once, replace_rear_patch

ROOT = Path(__file__).resolve().parent
CASES = ROOT / "cases"
SOURCE_CASE = CASES / "tandem_backward_dt005"
START_TIME = 80.0
END_TIME = 116.0
ACTION_INTERVAL = 0.1
DELTA_T = 0.005
OMEGA_LIMIT = 5.0
RATE_LIMIT = 1.6
PERIOD = 6.154


@dataclass(frozen=True)
class PanelCase:
    name: str
    kind: str


PANEL = (
    PanelCase("landscape_val_zero_20261003", "zero"),
    PanelCase("landscape_val_p100_20261003", "positive_constant"),
    PanelCase("landscape_val_m100_20261003", "negative_constant"),
    PanelCase("landscape_val_sine_20261003", "shedding_period_sine"),
)


def action_points(spec: PanelCase) -> list[tuple[float, float]]:
    count = round((END_TIME - START_TIME) / ACTION_INTERVAL)
    points = []
    for index in range(count + 1):
        tau = index * ACTION_INTERVAL
        ramp = min(tau / 2.0, 1.0)
        if spec.kind == "zero":
            omega = 0.0
        elif spec.kind == "positive_constant":
            omega = ramp
        elif spec.kind == "negative_constant":
            omega = -ramp
        elif spec.kind == "shedding_period_sine":
            omega = ramp * math.sin(2.0 * math.pi * tau / PERIOD)
        else:
            raise ValueError(f"unsupported action kind: {spec.kind}")
        points.append((round(START_TIME + tau, 10), omega))
    return points


def action_metrics(points: list[tuple[float, float]]) -> dict[str, float]:
    if len(points) < 2 or points[0][0] != START_TIME or points[-1][0] != END_TIME:
        raise ValueError("action table does not span the fixed CFD interval")
    if abs(points[0][1]) > 1.0e-12:
        raise ValueError("action must start from zero at restart")
    rates = []
    for (t0, value0), (t1, value1) in zip(points, points[1:]):
        if t1 <= t0 or not all(math.isfinite(v) for v in (t0, t1, value0, value1)):
            raise ValueError("non-finite or non-monotone action table")
        rates.append(abs((value1 - value0) / (t1 - t0)))
    peak = max(abs(value) for _, value in points)
    rate = max(rates)
    if peak > OMEGA_LIMIT + 1.0e-9 or rate > RATE_LIMIT + 1.0e-9:
        raise ValueError("action exceeds already sampled amplitude/rate envelope")
    return {"max_abs_omega": peak, "max_abs_domega_dt": rate}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def panel_audit() -> dict:
    return {
        "panel": "tandem_control_landscape_validation_20261003",
        "status": "PREDECLARED_ACTIONS_NO_CFD_RESULTS",
        "split": "validation_only_not_training_or_frozen_test",
        "source_restart": "tandem_backward_dt005/t=80",
        "time_window": [START_TIME, END_TIME],
        "analysis_window": [92.0, END_TIME],
        "reynolds_number": 100,
        "centre_spacing_over_diameter": 5,
        "cases": {
            spec.name: {"kind": spec.kind, **action_metrics(action_points(spec))}
            for spec in PANEL
        },
        "interpretation": "exploratory matched-start open-loop control landscape; not learned closed-loop benefit",
    }


def generate(spec: PanelCase) -> Path:
    source_time = SOURCE_CASE / "80"
    for file in (source_time / "U", source_time / "p"):
        if not file.is_file():
            raise FileNotFoundError(file)
    target = CASES / spec.name
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    points = action_points(spec)
    metrics = action_metrics(points)
    with tempfile.TemporaryDirectory(prefix=f".{spec.name}.", dir=CASES) as temporary:
        stage = Path(temporary) / spec.name
        stage.mkdir()
        shutil.copytree(SOURCE_CASE / "constant", stage / "constant")
        shutil.copytree(SOURCE_CASE / "system", stage / "system")
        shutil.copytree(source_time, stage / "80")
        velocity = stage / "80" / "U"
        velocity.write_text(
            replace_rear_patch(velocity.read_text(encoding="utf-8"), points),
            encoding="utf-8",
        )
        control_path = stage / "system" / "controlDict"
        control = control_path.read_text(encoding="utf-8")
        control = replace_once(control, "startTime 0;", "startTime 80;", control_path)
        control = replace_once(control, "endTime 160;", "endTime 116;", control_path)
        control = replace_once(control, "writeInterval 2;", "writeInterval 0.1;", control_path)
        control_path.write_text(control, encoding="utf-8")
        metadata = {
            "case": spec.name,
            "panel": "tandem_control_landscape_validation_20261003",
            "split": "validation",
            "action_kind": spec.kind,
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
            "delta_t": DELTA_T,
            "time_scheme": "backward",
            "start_time": START_TIME,
            "end_time": END_TIME,
            "analysis_window": [92.0, END_TIME],
            "field_write_interval": ACTION_INTERVAL,
            "expected_frames": round((END_TIME - START_TIME) / ACTION_INTERVAL) + 1,
            "force_series": ["front Cd/Cl", "rear Cd/Cl"],
            "limitations": "coarse-grid, short exploratory open-loop panel; not policy validation",
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
    audit = panel_audit()
    known = {spec.name: spec for spec in PANEL}
    unknown = set(args.names) - set(known)
    if unknown:
        parser.error(f"unknown case(s): {', '.join(sorted(unknown))}")
    selected = [known[name] for name in args.names] if args.names else list(PANEL)
    if args.audit:
        if args.audit.exists():
            parser.error(f"refusing to overwrite action audit: {args.audit}")
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text(json.dumps(audit, indent=2) + "\n")
    if args.list:
        for spec in selected:
            print(spec.name)
        return
    for spec in selected:
        print(generate(spec))


if __name__ == "__main__":
    main()
