#!/usr/bin/env python3
"""Create the expanded high-rotation dynamic CFD dataset.

All cases restart from the validated coarse-grid solution at t=80.  The rear
cylinder angular velocity is bounded by |omega|<=5, corresponding to
|q|<=2.5 for D=U_inf=1.  Trajectory membership is fixed before simulation.
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
OMEGA_LIMIT = 5.0


@dataclass(frozen=True)
class Schedule:
    name: str
    split: str
    kind: str
    seed: int


SCHEDULES = tuple(
    [
        Schedule(f"expanded_train_{i:02d}", "train", "random_ramp", 2026100100 + i)
        for i in range(10)
    ]
    + [
        Schedule(f"expanded_train_{i:02d}", "train", "multisine", 2026100100 + i)
        for i in range(10, 16)
    ]
    + [
        Schedule(f"expanded_train_{i:02d}", "train", "chirp", 2026100100 + i)
        for i in range(16, 20)
    ]
    + [
        Schedule(f"expanded_train_{i:02d}", "train", "edge_hold", 2026100100 + i)
        for i in range(20, 24)
    ]
    + [
        Schedule(
            "expanded_validation_00", "validation", "interstitial_ramp", 2026100200
        ),
        Schedule("expanded_validation_01", "validation", "multisine", 2026100201),
        Schedule("expanded_validation_02", "validation", "chirp", 2026100202),
        Schedule("expanded_validation_03", "validation", "edge_hold", 2026100203),
        Schedule("expanded_test_00", "test", "interstitial_ramp", 2026100300),
        Schedule("expanded_test_01", "test", "multisine", 2026100301),
        Schedule("expanded_test_02", "test", "chirp", 2026100302),
        Schedule("expanded_test_03", "test", "edge_hold", 2026100303),
    ]
)


def random_ramp(
    seed: int, levels: tuple[float, ...], spacing: float
) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    points = [(START_TIME, 0.0)]
    previous = 0.0
    time = START_TIME + spacing
    while time <= END_TIME + 1.0e-9:
        choices = [value for value in levels if value != previous]
        previous = rng.choice(choices)
        points.append((min(time, END_TIME), previous))
        time += spacing
    if points[-1][0] < END_TIME:
        points.append((END_TIME, previous))
    return points


def multisine(seed: int) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    amplitude_1 = rng.uniform(2.4, 3.4)
    amplitude_2 = rng.uniform(0.8, 1.4)
    period_1 = rng.uniform(13.0, 23.0)
    period_2 = rng.uniform(5.5, 10.0)
    phase_1 = rng.uniform(0.0, 2.0 * math.pi)
    phase_2 = rng.uniform(0.0, 2.0 * math.pi)
    points = []
    time = START_TIME
    while time <= END_TIME + 1.0e-9:
        tau = time - START_TIME
        omega = amplitude_1 * math.sin(2.0 * math.pi * tau / period_1 + phase_1)
        omega += amplitude_2 * math.sin(2.0 * math.pi * tau / period_2 + phase_2)
        omega *= min(tau / 2.0, 1.0)
        # Leave a small margin so the table interpolation never exceeds the limit.
        points.append((round(time, 10), max(-4.9, min(4.9, omega))))
        time += 0.25
    return points


def chirp(seed: int) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    phase = rng.uniform(0.0, 2.0 * math.pi)
    start_period = rng.uniform(20.0, 25.0)
    end_period = rng.uniform(4.8, 6.5)
    start_amplitude = rng.uniform(2.0, 2.8)
    end_amplitude = rng.uniform(4.4, 4.9)
    duration = END_TIME - START_TIME
    start_frequency = 1.0 / start_period
    end_frequency = 1.0 / end_period
    points = []
    time = START_TIME
    while time <= END_TIME + 1.0e-9:
        tau = time - START_TIME
        fraction = tau / duration
        cycles = (
            start_frequency * tau
            + 0.5 * (end_frequency - start_frequency) * tau * fraction
        )
        amplitude = start_amplitude + (end_amplitude - start_amplitude) * fraction
        omega = amplitude * math.sin(2.0 * math.pi * cycles + phase)
        omega *= min(tau / 2.0, 1.0)
        points.append((round(time, 10), omega))
        time += 0.25
    return points


EDGE_SEQUENCES = (
    (-5.0, 5.0, -4.0, 4.0, -2.0, 2.0, 0.0, 5.0, -5.0, 0.0),
    (5.0, -5.0, 3.0, -3.0, 1.0, -1.0, 4.0, -4.0, 0.0, 5.0),
    (-4.5, 4.5, -2.5, 2.5, -5.0, 5.0, -1.5, 1.5, 0.0, -4.5),
    (4.5, -4.5, 2.5, -2.5, 5.0, -5.0, 1.5, -1.5, 0.0, 4.5),
)


def edge_hold(seed: int, *, permuted: bool = False) -> list[tuple[float, float]]:
    sequence = list(EDGE_SEQUENCES[seed % len(EDGE_SEQUENCES)])
    if permuted:
        random.Random(seed).shuffle(sequence)
    block = (END_TIME - START_TIME) / len(sequence)
    transition = 1.5
    points = [(START_TIME, 0.0)]
    for index, target in enumerate(sequence):
        block_start = START_TIME + index * block
        transition_end = min(block_start + transition, END_TIME)
        if transition_end > points[-1][0]:
            points.append((transition_end, target))
        block_end = START_TIME + (index + 1) * block
        if block_end > points[-1][0]:
            points.append((block_end, target))
    return points


def schedule_points(spec: Schedule) -> list[tuple[float, float]]:
    if spec.kind == "random_ramp":
        return random_ramp(
            spec.seed, tuple(float(value) for value in range(-5, 6)), 5.0
        )
    if spec.kind == "interstitial_ramp":
        return random_ramp(
            spec.seed, (-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5), 5.0
        )
    if spec.kind == "multisine":
        return multisine(spec.seed)
    if spec.kind == "chirp":
        return chirp(spec.seed)
    if spec.kind == "edge_hold":
        return edge_hold(spec.seed)
    if spec.kind == "edge_hold_permuted":
        return edge_hold(spec.seed, permuted=True)
    raise ValueError(spec.kind)


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


def validate_schedules() -> dict:
    report: dict[str, dict] = {}
    split_rates: dict[str, list[float]] = {
        split: [] for split in ("train", "validation", "test")
    }
    for spec in SCHEDULES:
        points = schedule_points(spec)
        if points[0][0] != START_TIME or points[-1][0] != END_TIME:
            raise ValueError(f"{spec.name}: action table does not span the trajectory")
        if any(
            time_1 <= time_0 for (time_0, _), (time_1, _) in zip(points, points[1:])
        ):
            raise ValueError(f"{spec.name}: action times are not strictly increasing")
        metrics = schedule_metrics(points)
        if (
            max(abs(metrics["omega_min"]), abs(metrics["omega_max"]))
            > OMEGA_LIMIT + 1.0e-9
        ):
            raise ValueError(f"{spec.name}: action exceeds omega limit")
        report[spec.name] = {"split": spec.split, "kind": spec.kind, **metrics}
        split_rates[spec.split].append(metrics["max_abs_domega_dt"])
    train_rate = max(split_rates["train"])
    for split in ("validation", "test"):
        if max(split_rates[split]) > train_rate + 1.0e-9:
            raise ValueError(f"{split} action rate exceeds training coverage")
    return {
        "cases": report,
        "split_counts": {
            split: sum(spec.split == split for spec in SCHEDULES)
            for split in split_rates
        },
        "split_max_abs_domega_dt": {
            split: max(values) for split, values in split_rates.items()
        },
    }


def foam_table(points: list[tuple[float, float]]) -> str:
    rows = "\n".join(
        f"            ({time:.10g} {omega:.10g})" for time, omega in points
    )
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
    source_time = SOURCE_CASE / "80"
    if not (source_time / "U").is_file() or not (source_time / "p").is_file():
        raise FileNotFoundError(f"validated source restart is missing: {source_time}")
    case = CASES / spec.name
    if case.exists():
        raise FileExistsError(f"refusing to overwrite existing case: {case}")

    case.mkdir(parents=True)
    shutil.copytree(SOURCE_CASE / "constant", case / "constant")
    shutil.copytree(SOURCE_CASE / "system", case / "system")
    shutil.copytree(source_time, case / "80")

    points = schedule_points(spec)
    velocity_path = case / "80" / "U"
    velocity_path.write_text(
        replace_rear_patch(velocity_path.read_text(encoding="utf-8"), points),
        encoding="utf-8",
    )

    control_path = case / "system" / "controlDict"
    control = control_path.read_text(encoding="utf-8")
    control = replace_once(control, "startTime 0;", "startTime 80;", control_path)
    control = replace_once(
        control, "writeInterval 2;", "writeInterval 0.1;", control_path
    )
    control_path.write_text(control, encoding="utf-8")

    metadata = {
        "case": spec.name,
        "dataset": "tandem_cylinder_dynamic_rotation_expanded_v1",
        "split": spec.split,
        "schedule_kind": spec.kind,
        "seed": spec.seed,
        "source_restart_case": SOURCE_CASE.name,
        "source_restart_time": START_TIME,
        "source": "OpenFOAM v2512 pimpleFoam CFD; generated, not experimental data",
        "geometry": "two stationary-centre tandem cylinders, D=1, centres (10,7.5) and (15,7.5)",
        "reynolds_number": 100,
        "kinematic_viscosity": 0.01,
        "rear_angular_velocity_limit": [-OMEGA_LIMIT, OMEGA_LIMIT],
        "rear_angular_velocity_range": [
            min(value for _, value in points),
            max(value for _, value in points),
        ],
        "rear_surface_speed_ratio_limit": [-OMEGA_LIMIT / 2.0, OMEGA_LIMIT / 2.0],
        "rear_surface_speed_ratio_definition": "q=omega*D/(2*U_inf)=omega/2 for D=U_inf=1",
        "action_interpolation": "OpenFOAM Function1 scalar table, linear interpolation",
        "action_points": points,
        "action_metrics": schedule_metrics(points),
        "mesh_cells": 19290,
        "delta_t": DELTA_T,
        "time_scheme": "backward",
        "start_time": START_TIME,
        "end_time": END_TIME,
        "field_write_interval": FIELD_INTERVAL,
        "fields": ["U", "p"],
        "force_series": ["front Cd/Cl", "rear Cd/Cl"],
        "wake_velocity_probes": 32,
        "limitations": "coarse-grid training data; medium-grid high-rotation cases are independent validation",
    }
    (case / "case_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "names", nargs="*", help="case names; default creates all schedules"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print selected case names without creating cases",
    )
    parser.add_argument(
        "--audit", type=Path, help="write the deterministic pre-simulation action audit"
    )
    args = parser.parse_args()
    audit = validate_schedules()
    if args.audit:
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    known = {spec.name: spec for spec in SCHEDULES}
    unknown = set(args.names) - set(known)
    if unknown:
        parser.error(f"unknown schedule(s): {', '.join(sorted(unknown))}")
    selected = [known[name] for name in args.names] if args.names else list(SCHEDULES)
    if args.list:
        for spec in selected:
            print(spec.name)
        return
    for spec in selected:
        print(generate(spec))


if __name__ == "__main__":
    main()
