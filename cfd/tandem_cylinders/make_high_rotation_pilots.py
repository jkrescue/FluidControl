#!/usr/bin/env python3
"""Create short restart-based high-rotation CFD pilot cases."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CASES = ROOT / "cases"
START_TIME = 80.0
END_TIME = 100.0


@dataclass(frozen=True)
class Pilot:
    name: str
    omega: float
    delta_t: float = 0.005
    source_case: str = "tandem_backward_dt005"
    mesh_cells: int = 19290

    @property
    def q(self) -> float:
        return self.omega / 2.0


PILOTS = (
    Pilot("high_rotation_qm200_dt005", -4.0),
    Pilot("high_rotation_qp200_dt005", +4.0),
    Pilot("high_rotation_qm250_dt005", -5.0),
    Pilot("high_rotation_qp250_dt005", +5.0),
    Pilot("high_rotation_qm250_dt0025", -5.0, 0.0025),
    Pilot("high_rotation_qp250_dt0025", +5.0, 0.0025),
    Pilot(
        "high_rotation_qm250_medium_dt0025",
        -5.0,
        0.0025,
        "tandem_medium_backward_dt005",
        77160,
    ),
    Pilot(
        "high_rotation_qp250_medium_dt0025",
        +5.0,
        0.0025,
        "tandem_medium_backward_dt005",
        77160,
    ),
)


def replace_once(text: str, old: str, new: str, path: Path) -> str:
    if text.count(old) != 1:
        raise ValueError(f"expected one {old!r} in {path}, found {text.count(old)}")
    return text.replace(old, new)


def replace_rear_patch(text: str, omega: float) -> str:
    pattern = r"(\n\s*rearCylinder\s*\n\s*\{).*?(\n\s*\}\s*\n\s*frontBack)"
    replacement = (
        "\n    rearCylinder\n    {\n"
        "        type rotatingWallVelocity;\n"
        "        origin (15 7.5 0);\n"
        "        axis (0 0 1);\n"
        f"        omega {omega:g};\n"
        "        value uniform (0 0 0);\n"
        "    }\n    frontBack"
    )
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.S)
    if count != 1:
        raise ValueError("could not replace rearCylinder patch in restart U field")
    return updated


def generate(pilot: Pilot) -> Path:
    source_case = CASES / pilot.source_case
    source_time = source_case / "80"
    if not (source_time / "U").is_file() or not (source_time / "p").is_file():
        raise FileNotFoundError(f"validated restart fields are missing: {source_time}")
    case = CASES / pilot.name
    if case.exists():
        raise FileExistsError(f"refusing to overwrite existing case: {case}")

    case.mkdir(parents=True)
    shutil.copytree(source_case / "constant", case / "constant")
    shutil.copytree(source_case / "system", case / "system")
    shutil.copytree(source_time, case / "80")

    velocity_path = case / "80" / "U"
    velocity_path.write_text(
        replace_rear_patch(velocity_path.read_text(encoding="utf-8"), pilot.omega),
        encoding="utf-8",
    )

    control_path = case / "system" / "controlDict"
    control = control_path.read_text(encoding="utf-8")
    control = replace_once(control, "startTime 0;", "startTime 80;", control_path)
    control = replace_once(control, "endTime 160;", "endTime 100;", control_path)
    control = re.sub(r"deltaT\s+[^;]+;", f"deltaT {pilot.delta_t:g};", control, count=1)
    control = replace_once(control, "writeInterval 2;", "writeInterval 1;", control_path)
    control_path.write_text(control, encoding="utf-8")

    metadata = {
        "case": pilot.name,
        "purpose": "high-rotation stability pilot before expanded dataset generation",
        "source_restart_case": source_case.name,
        "source_restart_time": START_TIME,
        "reynolds_number": 100,
        "geometry": "two tandem cylinders, D=1, centres (10,7.5) and (15,7.5)",
        "inlet_velocity": 1.0,
        "rear_angular_velocity": pilot.omega,
        "rear_surface_speed_ratio": pilot.q,
        "surface_speed_ratio_definition": "q=omega*D/(2*U_inf)",
        "mesh_cells": pilot.mesh_cells,
        "delta_t": pilot.delta_t,
        "time_scheme": "backward",
        "start_time": START_TIME,
        "end_time": END_TIME,
        "field_write_interval": 1.0,
        "force_write_interval_steps": 1,
        "limitations": "short coarse-grid pilot; not part of the training dataset",
    }
    (case / "case_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="case names; default creates all pilots")
    args = parser.parse_args()
    known = {pilot.name: pilot for pilot in PILOTS}
    unknown = set(args.names) - set(known)
    if unknown:
        parser.error(f"unknown pilot(s): {', '.join(sorted(unknown))}")
    selected = [known[name] for name in args.names] if args.names else list(PILOTS)
    for pilot in selected:
        print(generate(pilot))


if __name__ == "__main__":
    main()
