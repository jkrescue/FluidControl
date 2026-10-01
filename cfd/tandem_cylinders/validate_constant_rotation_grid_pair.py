#!/usr/bin/env python3
"""Numerical QC for matched constant-omega coarse/medium OpenFOAM cases.

Passing this script is not evidence of spatial grid convergence.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from analyze_baseline import field_health, load_coefficients, log_health, probe_health


CASES = {
    "control_small_p100": {"cells": 19290, "snapshots": 1600, "wall_faces": 96},
    "control_grid_p100_medium": {"cells": 77160, "snapshots": 80, "wall_faces": 192},
}


def wall_speed(path: Path, expected_faces: int) -> dict:
    text = path.read_text(encoding="utf-8")
    patch = text.split("    rearCylinder\n", 1)[1].split("    frontBack\n", 1)[0]
    match = re.search(
        r"value\s+nonuniform List<vector>\s+(\d+)\s*\((.*?)\)\s*;",
        patch, flags=re.S,
    )
    if match is None or int(match.group(1)) != expected_faces:
        raise ValueError(f"missing {expected_faces} rear wall vectors: {path}")
    vectors = re.findall(
        r"\(([-+0-9.eE]+) ([-+0-9.eE]+) ([-+0-9.eE]+)\)",
        match.group(2),
    )
    if len(vectors) != expected_faces:
        raise ValueError(f"incomplete rear wall vectors: {path}")
    speeds = [math.hypot(float(vx), float(vy)) for vx, vy, _ in vectors]
    return {"min": min(speeds), "max": max(speeds), "faces": len(vectors)}


def validate_one(case: Path, expected: dict) -> dict:
    config = json.loads((case / "case_config.json").read_text(encoding="utf-8"))
    if config["mesh_cells"] != expected["cells"]:
        raise ValueError(f"mesh metadata mismatch: {case}")
    if "Mesh OK" not in (case / "log.checkMesh").read_text(encoding="utf-8"):
        raise ValueError(f"checkMesh failed: {case}")
    solver = log_health(case / "log.pimpleFoam")
    if not solver["solver_ended_cleanly"] or solver["steps"] != 32000:
        raise ValueError(f"incomplete solver: {case}: {solver}")
    if solver["max_courant"] >= 1 or solver["max_abs_global_continuity_per_step"] >= 1e-5:
        raise ValueError(f"numerical health failed: {case}: {solver}")
    fields = field_health(case)
    if fields["snapshot_count"] != expected["snapshots"] or abs(fields["last_time"] - 160) > 1e-8:
        raise ValueError(f"incomplete field series: {case}: {fields}")
    probes = probe_health(case / "postProcessing/wakeProbes/0/U")
    if probes["samples"] != 32000 or abs(probes["last_time"] - 160) > 1e-8:
        raise ValueError(f"incomplete wake probes: {case}")
    force_samples = {}
    for name in ("Front", "Rear"):
        rows = load_coefficients(
            case / "postProcessing" / f"force{name}" / "0" / "coefficient.dat"
        )
        if len(rows) != 32000 or abs(rows[-1][0] - 160) > 1e-8:
            raise ValueError(f"incomplete force coefficients: {case}/{name}")
        force_samples[name.lower()] = len(rows)
    wall = wall_speed(case / "160/U", expected["wall_faces"])
    if max(abs(wall["min"] - 0.5), abs(wall["max"] - 0.5)) > 0.002:
        raise ValueError(f"rear wall speed differs from omega*D/2: {case}: {wall}")
    return {
        "case": case.name,
        "mesh_cells": expected["cells"],
        "solver": solver,
        "fields": fields,
        "probe_samples": probes["samples"],
        "force_samples": force_samples,
        "rear_wall_speed_at_t160": wall,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [
        validate_one(args.cases_root / name, expected)
        for name, expected in CASES.items()
    ]
    report = {
        "status": "GRID_PAIR_NUMERICAL_QC_OK_NOT_CONVERGENCE",
        "source": "matched OpenFOAM v2512 pimpleFoam constant omega=+1",
        "time_window_for_comparison": [80, 160],
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("GRID_PAIR_NUMERICAL_QC_OK")


if __name__ == "__main__":
    main()
