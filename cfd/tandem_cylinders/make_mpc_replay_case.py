#!/usr/bin/env python3
"""Create an OpenFOAM replay case from an MPC action CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

from make_expanded_control_dataset import (
    CASES,
    SOURCE_CASE,
    replace_once,
    replace_rear_patch,
)


def load_actions(path: Path) -> list[float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(int(row["step"]) != index for index, row in enumerate(rows, 1)):
        raise ValueError("action CSV must contain consecutive steps beginning at 1")
    actions = [float(row["omega"]) for row in rows]
    if not all(math.isfinite(value) for value in actions):
        raise ValueError("action CSV contains a non-finite omega")
    return actions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actions", type=Path)
    parser.add_argument("case_name")
    parser.add_argument("--start-time", type=float, default=80.0)
    parser.add_argument("--control-interval", type=float, default=0.1)
    parser.add_argument("--initial-omega", type=float, default=0.0)
    args = parser.parse_args()

    if not args.case_name.startswith("mpc_replay_"):
        parser.error("case_name must begin with mpc_replay_")
    actions = load_actions(args.actions)
    points = [(args.start_time, args.initial_omega)] + [
        (args.start_time + index * args.control_interval, value)
        for index, value in enumerate(actions, 1)
    ]
    end_time = points[-1][0]

    source_time = SOURCE_CASE / f"{args.start_time:g}"
    if not (source_time / "U").is_file() or not (source_time / "p").is_file():
        raise FileNotFoundError(f"source restart is missing: {source_time}")
    target = CASES / args.case_name
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing case: {target}")

    target.mkdir(parents=True)
    shutil.copytree(SOURCE_CASE / "constant", target / "constant")
    shutil.copytree(SOURCE_CASE / "system", target / "system")
    shutil.copytree(source_time, target / f"{args.start_time:g}")

    velocity_path = target / f"{args.start_time:g}" / "U"
    velocity_path.write_text(
        replace_rear_patch(velocity_path.read_text(encoding="utf-8"), points),
        encoding="utf-8",
    )

    control_path = target / "system" / "controlDict"
    control = control_path.read_text(encoding="utf-8")
    control = replace_once(control, "startTime 0;", f"startTime {args.start_time:g};", control_path)
    control = replace_once(control, "endTime 160;", f"endTime {end_time:g};", control_path)
    control = replace_once(control, "writeInterval 2;", f"writeInterval {args.control_interval:g};", control_path)
    control_path.write_text(control, encoding="utf-8")

    metadata = {
        "case": args.case_name,
        "purpose": "OpenFOAM replay of a frozen surrogate MPC action sequence",
        "source_restart_case": SOURCE_CASE.name,
        "source_restart_time": args.start_time,
        "end_time": end_time,
        "control_interval": args.control_interval,
        "action_count": len(actions),
        "rear_angular_velocity_range": [min(actions), max(actions)],
        "max_abs_delta_omega": max(
            abs(second - first)
            for first, second in zip([args.initial_omega] + actions[:-1], actions)
        ),
        "action_points": points,
        "source": "OpenFOAM v2512 pimpleFoam CFD; generated, not experimental data",
        "reynolds_number": 100,
        "geometry": "tandem cylinders, D=1, L/D=5; rear cylinder rotates",
    }
    (target / "case_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(target)


if __name__ == "__main__":
    main()
