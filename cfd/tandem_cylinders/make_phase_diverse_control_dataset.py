#!/usr/bin/env python3
"""Create short OpenFOAM trajectories from independent shedding phases.

The existing expanded dataset always restarts from t=80.  These cases instead
restart from four later snapshots of the validated uncontrolled baseline and use
independent, rate-bounded rear-cylinder rotation histories.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from make_expanded_control_dataset import replace_once, replace_rear_patch

ROOT = Path(__file__).resolve().parent
CASES = ROOT / "cases"
SOURCE_CASE = CASES / "tandem_backward_dt005"
DURATION = 24.0
DELTA_T = 0.005
FIELD_INTERVAL = 0.1
OMEGA_LIMIT = 5.0
RATE_LIMIT = 1.6


@dataclass(frozen=True)
class PhaseCase:
    name: str
    split: str
    source_time: float
    schedule_kind: str
    seed: int


PHASE_CASES = (
    PhaseCase("phase_train_82_multisine", "train", 82.0, "multisine", 2026100282),
    PhaseCase("phase_train_84_ramp", "train", 84.0, "random_ramp", 2026100284),
    PhaseCase(
        "phase_validation_86_multisine", "validation", 86.0, "multisine", 2026100286
    ),
    PhaseCase("phase_test_88_ramp", "test", 88.0, "random_ramp", 2026100288),
)


def multisine(spec: PhaseCase) -> list[tuple[float, float]]:
    rng = random.Random(spec.seed)
    amplitude_1 = rng.uniform(2.4, 2.7)
    amplitude_2 = rng.uniform(0.5, 0.65)
    period_1 = rng.uniform(17.0, 21.0)
    period_2 = rng.uniform(10.0, 12.0)
    points = []
    time = spec.source_time
    while time <= spec.source_time + DURATION + 1.0e-9:
        tau = time - spec.source_time
        ramp = min(tau / 4.0, 1.0)
        omega = ramp * (
            amplitude_1 * math.sin(2.0 * math.pi * tau / period_1)
            + amplitude_2 * math.sin(2.0 * math.pi * tau / period_2)
        )
        points.append((round(time, 10), omega))
        time += 0.25
    return points


def random_ramp(spec: PhaseCase) -> list[tuple[float, float]]:
    rng = random.Random(spec.seed)
    levels = (-4.0, -2.0, 0.0, 2.0, 4.0)
    points = [(spec.source_time, 0.0)]
    previous = 0.0
    time = spec.source_time + 4.0
    end_time = spec.source_time + DURATION
    while time <= end_time + 1.0e-9:
        choices = [value for value in levels if value != previous and abs(value - previous) <= 4.0]
        previous = rng.choice(choices)
        points.append((min(time, end_time), previous))
        time += 4.0
    return points


def schedule_points(spec: PhaseCase) -> list[tuple[float, float]]:
    if spec.schedule_kind == "multisine":
        return multisine(spec)
    if spec.schedule_kind == "random_ramp":
        return random_ramp(spec)
    raise ValueError(f"unsupported schedule: {spec.schedule_kind}")


def schedule_metrics(points: list[tuple[float, float]]) -> dict[str, float]:
    rates = [
        abs((value_1 - value_0) / (time_1 - time_0))
        for (time_0, value_0), (time_1, value_1) in zip(points, points[1:])
    ]
    return {
        "omega_min": min(value for _, value in points),
        "omega_max": max(value for _, value in points),
        "max_abs_domega_dt": max(rates),
    }


def validate() -> dict:
    rows = {}
    for spec in PHASE_CASES:
        points = schedule_points(spec)
        expected_end = spec.source_time + DURATION
        if not math.isclose(points[0][0], spec.source_time) or not math.isclose(
            points[-1][0], expected_end
        ):
            raise ValueError(f"{spec.name}: action table does not span the case")
        if not math.isclose(points[0][1], 0.0, abs_tol=1.0e-12):
            raise ValueError(f"{spec.name}: restart action must be zero")
        if any(b[0] <= a[0] for a, b in zip(points, points[1:])):
            raise ValueError(f"{spec.name}: action times are not increasing")
        metrics = schedule_metrics(points)
        if max(abs(metrics["omega_min"]), abs(metrics["omega_max"])) > OMEGA_LIMIT:
            raise ValueError(f"{spec.name}: action exceeds omega support")
        if metrics["max_abs_domega_dt"] > RATE_LIMIT + 1.0e-9:
            raise ValueError(f"{spec.name}: action rate exceeds expanded-data support")
        rows[spec.name] = {
            "split": spec.split,
            "source_time": spec.source_time,
            "schedule_kind": spec.schedule_kind,
            **metrics,
        }
    return {
        "dataset": "tandem_phase_diverse_v1",
        "duration": DURATION,
        "cases": rows,
    }


def generate(spec: PhaseCase) -> Path:
    source_name = f"{spec.source_time:g}"
    source_time = SOURCE_CASE / source_name
    if not (source_time / "U").is_file() or not (source_time / "p").is_file():
        raise FileNotFoundError(f"validated source restart is missing: {source_time}")
    target = CASES / spec.name
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing case: {target}")

    target.mkdir(parents=True)
    shutil.copytree(SOURCE_CASE / "constant", target / "constant")
    shutil.copytree(SOURCE_CASE / "system", target / "system")
    shutil.copytree(source_time, target / source_name)

    points = schedule_points(spec)
    velocity_path = target / source_name / "U"
    velocity_path.write_text(
        replace_rear_patch(velocity_path.read_text(encoding="utf-8"), points),
        encoding="utf-8",
    )

    control_path = target / "system" / "controlDict"
    control = control_path.read_text(encoding="utf-8")
    control = replace_once(control, "startTime 0;", f"startTime {source_name};", control_path)
    control = replace_once(
        control, "endTime 160;", f"endTime {spec.source_time + DURATION:g};", control_path
    )
    control = replace_once(control, "writeInterval 2;", "writeInterval 0.1;", control_path)
    control_path.write_text(control, encoding="utf-8")

    metadata = {
        "case": spec.name,
        "dataset": "tandem_phase_diverse_v1",
        "split": spec.split,
        "schedule_kind": spec.schedule_kind,
        "seed": spec.seed,
        "source_restart_case": SOURCE_CASE.name,
        "source_restart_time": spec.source_time,
        "source_restart_control": "uncontrolled rear cylinder, omega=0",
        "source": "OpenFOAM v2512 pimpleFoam CFD; generated, not experimental data",
        "purpose": "independent shedding-phase generalization and rollout audit",
        "geometry": "two fixed-centre tandem cylinders, D=1, centres (10,7.5) and (15,7.5)",
        "reynolds_number": 100,
        "kinematic_viscosity": 0.01,
        "rear_angular_velocity_limit": [-OMEGA_LIMIT, OMEGA_LIMIT],
        "rear_angular_velocity_range": [
            min(value for _, value in points),
            max(value for _, value in points),
        ],
        "action_interpolation": "OpenFOAM Function1 scalar table, linear interpolation",
        "action_points": points,
        "action_metrics": schedule_metrics(points),
        "mesh_cells": 19290,
        "delta_t": DELTA_T,
        "time_scheme": "backward",
        "start_time": spec.source_time,
        "end_time": spec.source_time + DURATION,
        "duration": DURATION,
        "field_write_interval": FIELD_INTERVAL,
        "expected_frames": int(round(DURATION / FIELD_INTERVAL)) + 1,
        "force_series": ["front Cd/Cl", "rear Cd/Cl"],
        "wake_velocity_probes": 32,
        "limitations": "coarse grid; phase panel augments but does not replace policy-guided CFD",
    }
    (target / "case_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="case names; default selects all")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    audit = validate()
    if args.audit:
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    known = {spec.name: spec for spec in PHASE_CASES}
    unknown = set(args.names) - set(known)
    if unknown:
        parser.error(f"unknown case(s): {', '.join(sorted(unknown))}")
    selected = [known[name] for name in args.names] if args.names else list(PHASE_CASES)
    if args.list:
        for spec in selected:
            print(spec.name)
        return
    for spec in selected:
        print(generate(spec))


if __name__ == "__main__":
    main()
