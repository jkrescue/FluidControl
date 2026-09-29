#!/usr/bin/env python3
"""Create restart-based time-varying rear-cylinder control trajectories.

The cases restart from the validated coarse-grid, second-order tandem baseline at
t=80.  Only the rear-cylinder angular velocity changes.  Every trajectory is
written to a new directory and the source case is never modified.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CASES = ROOT / "cases"
SOURCE_CASE = CASES / "tandem_backward_dt005"
START_TIME = 80.0
END_TIME = 160.0
DELTA_T = 0.005
FIELD_INTERVAL = 0.1


@dataclass(frozen=True)
class Schedule:
    name: str
    split: str
    kind: str
    seed: int


SCHEDULES = tuple(
    [Schedule(f"dynamic_train_{i:02d}", "train", "random_ramp", 2026092900 + i) for i in range(8)]
    + [Schedule(f"dynamic_train_{i:02d}", "train", "multisine", 2026092900 + i) for i in range(8, 12)]
    + [
        Schedule("dynamic_validation_00", "validation", "quarter_ramp", 2026093000),
        Schedule("dynamic_validation_01", "validation", "multisine", 2026093001),
        Schedule("dynamic_test_00", "test", "quarter_ramp", 2026093100),
        Schedule("dynamic_test_01", "test", "chirp", 2026093101),
    ]
)


def random_ramp(seed: int, levels: tuple[float, ...], spacing: float) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    points = [(START_TIME, 0.0)]
    time = START_TIME + spacing
    previous = 0.0
    while time < END_TIME:
        choices = [value for value in levels if value != previous]
        previous = rng.choice(choices)
        points.append((time, previous))
        time += spacing
    points.append((END_TIME, rng.choice([value for value in levels if value != previous])))
    return points


def multisine(seed: int) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    amplitude_1 = rng.uniform(0.45, 0.65)
    amplitude_2 = rng.uniform(0.18, 0.30)
    period_1 = rng.uniform(13.0, 21.0)
    period_2 = rng.uniform(5.0, 9.0)
    phase = rng.uniform(0.0, 2.0 * math.pi)
    points = []
    time = START_TIME
    while time < END_TIME + 1e-9:
        tau = time - START_TIME
        omega = amplitude_1 * math.sin(2 * math.pi * tau / period_1)
        omega += amplitude_2 * math.sin(2 * math.pi * tau / period_2 + phase)
        points.append((round(time, 10), max(-0.95, min(0.95, omega))))
        time += 0.5
    return points


def chirp(seed: int) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    phase = rng.uniform(0.0, 2.0 * math.pi)
    duration = END_TIME - START_TIME
    points = []
    time = START_TIME
    while time < END_TIME + 1e-9:
        tau = time - START_TIME
        cycles = tau / 22.0 + 0.5 * (1.0 / 7.0 - 1.0 / 22.0) * tau * tau / duration
        envelope = 0.55 + 0.35 * tau / duration
        points.append((round(time, 10), envelope * math.sin(2 * math.pi * cycles + phase)))
        time += 0.5
    return points


def schedule_points(spec: Schedule) -> list[tuple[float, float]]:
    if spec.kind == "random_ramp":
        return random_ramp(spec.seed, (-1.0, -0.5, 0.0, 0.5, 1.0), 4.0)
    if spec.kind == "quarter_ramp":
        return random_ramp(spec.seed, (-0.75, -0.25, 0.25, 0.75), 3.5)
    if spec.kind == "multisine":
        return multisine(spec.seed)
    if spec.kind == "chirp":
        return chirp(spec.seed)
    raise ValueError(spec.kind)


def foam_table(points: list[tuple[float, float]]) -> str:
    rows = "\n".join(f"            ({time:.10g} {omega:.10g})" for time, omega in points)
    return f"table\n        (\n{rows}\n        )"


def replace_rear_patch(text: str, points: list[tuple[float, float]]) -> str:
    pattern = r"(\n\s*rearCylinder\s*\n\s*\{).*?(\n\s*\}\s*\n\s*frontBack)"
    replacement = (
        "\n    rearCylinder\n    {\n"
        "        type rotatingWallVelocity;\n"
        "        origin (15 7.5 0);\n"
        "        axis (0 0 1);\n"
        f"        omega {foam_table(points)};\n"
        "        value uniform (0 0 0);\n"
        "    }\n    frontBack"
    )
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.S)
    if count != 1:
        raise ValueError("could not replace rearCylinder patch in restart U field")
    return updated


def replace_once(text: str, old: str, new: str, path: Path) -> str:
    if text.count(old) != 1:
        raise ValueError(f"expected one {old!r} in {path}, found {text.count(old)}")
    return text.replace(old, new)


def generate(spec: Schedule) -> Path:
    if not (SOURCE_CASE / "80" / "U").is_file():
        raise FileNotFoundError("validated source restart at t=80 is missing")
    case = CASES / spec.name
    if case.exists():
        raise FileExistsError(f"refusing to overwrite existing case: {case}")

    case.mkdir(parents=True)
    shutil.copytree(SOURCE_CASE / "constant", case / "constant")
    shutil.copytree(SOURCE_CASE / "system", case / "system")
    shutil.copytree(SOURCE_CASE / "80", case / "80")

    points = schedule_points(spec)
    velocity_path = case / "80" / "U"
    velocity_path.write_text(
        replace_rear_patch(velocity_path.read_text(encoding="utf-8"), points),
        encoding="utf-8",
    )

    control_path = case / "system" / "controlDict"
    control = control_path.read_text(encoding="utf-8")
    control = replace_once(control, "startTime 0;", "startTime 80;", control_path)
    control = replace_once(control, "writeInterval 2;", "writeInterval 0.1;", control_path)
    control_path.write_text(control, encoding="utf-8")

    metadata = {
        "case": spec.name,
        "split": spec.split,
        "schedule_kind": spec.kind,
        "seed": spec.seed,
        "source_restart_case": "tandem_backward_dt005",
        "source_restart_time": START_TIME,
        "source": "OpenFOAM v2512 pimpleFoam CFD; generated, not experimental data",
        "geometry": "two stationary-centre tandem cylinders, diameter 1, centres (10,7.5) and (15,7.5)",
        "reynolds_number": 100,
        "kinematic_viscosity": 0.01,
        "rear_angular_velocity_range": [min(value for _, value in points), max(value for _, value in points)],
        "rear_surface_speed_ratio_definition": "omega*D/(2*U_inf)=omega/2 for D=U_inf=1",
        "action_interpolation": "OpenFOAM Function1 scalar table, linear interpolation",
        "action_points": points,
        "mesh_cells": 19290,
        "delta_t": DELTA_T,
        "time_scheme": "backward",
        "start_time": START_TIME,
        "end_time": END_TIME,
        "field_write_interval": FIELD_INTERVAL,
        "fields": ["U", "p"],
        "force_series": ["front Cd/Cl", "rear Cd/Cl"],
        "wake_velocity_probes": 32,
        "limitations": "coarse-grid first-stage data; action range limited to |omega|<=1",
    }
    (case / "case_config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="case names; default creates all missing schedules")
    args = parser.parse_args()
    selected = [spec for spec in SCHEDULES if not args.names or spec.name in args.names]
    unknown = set(args.names) - {spec.name for spec in SCHEDULES}
    if unknown:
        parser.error(f"unknown schedule(s): {', '.join(sorted(unknown))}")
    for spec in selected:
        print(generate(spec))


if __name__ == "__main__":
    main()
