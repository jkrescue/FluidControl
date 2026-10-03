#!/usr/bin/env python3
"""Validate one pinned OpenFOAM commissioning log before transfer staging."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


def validate_solver_log(path: Path, expected_steps: int, expected_end: float) -> dict:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    nonblank = [line for line in lines if line.strip()]
    times = []
    max_courant = 0.0
    max_continuity = 0.0
    for line in lines:
        if line.startswith("Time = "):
            times.append(float(line.split("=", 1)[1]))
        courant = re.search(r"Courant Number mean: \S+ max: (\S+)", line)
        if courant:
            max_courant = max(max_courant, float(courant.group(1)))
        continuity = re.search(
            r"time step continuity errors : .*global = ([-+0-9.eE]+)", line
        )
        if continuity:
            max_continuity = max(max_continuity, abs(float(continuity.group(1))))
    if not nonblank or nonblank[-1] != "End":
        raise ValueError("last nonblank solver-log line is not End")
    if len(times) != expected_steps:
        raise ValueError(f"expected {expected_steps} solver steps, got {len(times)}")
    if not times or not math.isclose(times[-1], expected_end, rel_tol=0, abs_tol=1e-8):
        raise ValueError(f"terminal solver time differs from {expected_end}")
    if max_courant >= 0.3:
        raise ValueError(f"maximum Courant number is not below 0.3: {max_courant}")
    if max_continuity >= 1e-9:
        raise ValueError(
            f"maximum absolute global continuity is not below 1e-9: {max_continuity}"
        )
    return {
        "steps": len(times),
        "terminal_time": times[-1],
        "max_courant": max_courant,
        "max_abs_global_continuity_per_step": max_continuity,
        "last_nonblank_line": nonblank[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--expected-steps", required=True, type=int)
    parser.add_argument("--expected-end", required=True, type=float)
    args = parser.parse_args()
    print(
        json.dumps(
            validate_solver_log(args.log, args.expected_steps, args.expected_end),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
