#!/usr/bin/env python3
"""Initialize an isolated segmented tandem OpenFOAM feedback case at t=80."""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import tempfile
from pathlib import Path

from make_expanded_control_dataset import CASES, SOURCE_CASE, replace_rear_patch


def substitute(path: Path, key: str, value: float) -> None:
    text = path.read_text(encoding="utf-8")
    updated, count = re.subn(
        rf"(?m)^{re.escape(key)}\s+[^;]+;",
        f"{key} {value:.10g};", text,
    )
    if count != 1:
        raise ValueError(f"expected one {key} entry in {path}, found {count}")
    path.write_text(updated, encoding="utf-8")


def initialize(name: str, steps: int, start_time: float = 80.0) -> Path:
    if not re.fullmatch(r"probe_feedback_[a-z0-9_]+", name):
        raise ValueError("case name must be probe_feedback_[a-z0-9_]+")
    if steps < 2 or steps > 1000:
        raise ValueError("steps must lie in 2..1000")
    if not math.isclose(start_time, 80.0, abs_tol=1e-9):
        raise ValueError("only the validated real CFD t=80 restart is supported")
    source_time = SOURCE_CASE / "80"
    if not all((source_time / field).is_file() for field in ("U", "p")):
        raise FileNotFoundError(f"missing original CFD restart: {source_time}")
    target = CASES / name
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing case: {target}")
    end_time = start_time + 0.1 * steps
    CASES.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".probe_feedback_stage_", dir=CASES) as outer:
        stage = Path(outer) / name
        stage.mkdir()
        shutil.copytree(SOURCE_CASE / "constant", stage / "constant")
        shutil.copytree(SOURCE_CASE / "system", stage / "system")
        shutil.copytree(source_time, stage / "80")
        initial_u = stage / "80" / "U"
        initial_u.write_text(
            replace_rear_patch(
                initial_u.read_text(encoding="utf-8"),
                [(start_time, 0.0), (end_time, 0.0)],
            ),
            encoding="utf-8",
        )
        control = stage / "system" / "controlDict"
        for key, value in (
            ("startTime", start_time),
            ("endTime", end_time),
            ("writeInterval", 0.1),
        ):
            substitute(control, key, value)
        metadata = {
            "case": name,
            "status": "initialized_not_solved",
            "purpose": "segmented real OpenFOAM state-feedback plumbing",
            "source": "OpenFOAM v2512 pimpleFoam CFD; generated, not experimental data",
            "source_restart_case": SOURCE_CASE.name,
            "source_restart_time": start_time,
            "reynolds_number": 100,
            "geometry": "tandem cylinders D=1, L/D=5, rear cylinder rotates",
            "mesh_cells": 19290,
            "delta_t": 0.005,
            "control_interval": 0.1,
            "steps": steps,
            "end_time": end_time,
            "omega_limit": 5.0,
            "delta_omega_limit": 0.5,
            "initial_omega": 0.0,
            "warmup_first_interval": "zero_request_to_obtain_raw_probe_observation",
            "policy_status": "none; not RL control",
        }
        (stage / "case_config.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        if target.exists():
            raise FileExistsError(f"case appeared during staging: {target}")
        stage.rename(target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    print(initialize(args.name, args.steps))


if __name__ == "__main__":
    main()
