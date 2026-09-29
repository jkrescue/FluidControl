#!/usr/bin/env python3
"""Validate the five-case constant-rotation CFD pilot before calling it data."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from analyze_baseline import (ROOT, field_health, load_coefficients, log_health,
                              probe_health, window_stats)
from make_small_control_dataset import OMEGAS


def wall_speed(path: Path) -> tuple[float, float]:
    text = path.read_text(encoding="utf-8")
    patch = text.split("    rearCylinder\n", 1)[1].split("    frontBack\n", 1)[0]
    match = re.search(r"value\s+nonuniform List<vector>\s+96\s*\((.*?)\)\s*;", patch, re.S)
    if match is None:
        uniform = re.search(r"value\s+uniform\s+\(([^)]*)\)", patch)
        if uniform and all(abs(float(x)) < 1e-10 for x in uniform.group(1).split()):
            return 0.0, 0.0
        raise ValueError(f"cannot parse 96 rear wall vectors: {path}")
    vectors = re.findall(r"\(([-+0-9.eE]+) ([-+0-9.eE]+) ([-+0-9.eE]+)\)", match.group(1))
    if len(vectors) != 96:
        raise ValueError(f"expected 96 wall vectors, found {len(vectors)}: {path}")
    speeds = [math.hypot(float(x), float(y)) for x, y, _ in vectors]
    return min(speeds), max(speeds)


def validate(label: str) -> dict:
    case = ROOT / "cases" / f"control_small_{label}"
    config = json.loads((case / "case_config.json").read_text(encoding="utf-8"))
    if config["rear_angular_velocity_rad_per_s"] != OMEGAS[label]:
        raise ValueError(f"action metadata mismatch: {label}")
    solver = log_health(case / "log.pimpleFoam")
    if not solver["solver_ended_cleanly"] or solver["steps"] != 32000:
        raise ValueError(f"incomplete solver run: {label}: {solver}")
    if solver["max_courant"] >= 1 or solver["max_abs_global_continuity_per_step"] >= 1e-5:
        raise ValueError(f"numerical health failed: {label}: {solver}")
    fields = field_health(case)
    if fields["snapshot_count"] != 1600 or abs(fields["last_time"] - 160) > 1e-8:
        raise ValueError(f"incomplete field series: {label}: {fields}")
    probes = probe_health(case / "postProcessing/wakeProbes/0/U")
    if probes["samples"] != 32000 or abs(probes["last_time"] - 160) > 1e-8:
        raise ValueError(f"incomplete probes: {label}: {probes}")
    force_stats = {}
    for which in ("Front", "Rear"):
        rows = load_coefficients(case / f"postProcessing/force{which}/0/coefficient.dat")
        if len(rows) != 32000 or abs(rows[-1][0] - 160) > 1e-8:
            raise ValueError(f"incomplete force series: {label}/{which}")
        force_stats[which.lower()] = {
            "80_120": window_stats(rows, 80, 120),
            "120_160": window_stats(rows, 120, 160),
            "80_160": window_stats(rows, 80, 160),
        }
    wall = {str(t): wall_speed(case / str(t) / "U") for t in (0.1, 160)}
    target = abs(OMEGAS[label]) / 2
    for t, (minimum, maximum) in wall.items():
        if max(abs(minimum - target), abs(maximum - target)) > 0.002:
            raise ValueError(f"rear wall speed mismatch: {label}, t={t}, {minimum}:{maximum}, expected {target}")
    return {"case": case.name, "omega_star": OMEGAS[label], "solver": solver,
            "fields": fields, "probes": {"samples": probes["samples"], "probes_per_sample": 32},
            "force_window_80_160": force_stats, "rear_wall_speed_min_max": wall,
            "status": "passed_basic_numerical_QC_not_grid_convergence"}


def main() -> None:
    results = [validate(label) for label in OMEGAS]
    manifest = {
        "schema_version": 1,
        "dataset": "small_constant_rotation",
        "source": "independent OpenFOAM v2512 CFD, not PolyU experimental data",
        "raw_case_root": "cases/",
        "cases": results,
        "usable_time_window": [80, 160],
        "field_interval": 0.1,
        "split_by_whole_trajectory": {
            "train": ["control_small_m100", "control_small_z000", "control_small_p100"],
            "validation": ["control_small_m050"],
            "test": ["control_small_p050"],
        },
        "limitations": [
            "constant angular velocity per trajectory, not time-varying RL actions",
            "coarse mesh; basic numerical QC is not proof of grid convergence",
            "five trajectories are a pilot, not a statistically sufficient production training set",
        ],
        "model_training_started": False,
    }
    path = ROOT / "small_control_dataset_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Validated {len(results)} CFD trajectories; wrote {path}")


if __name__ == "__main__":
    main()
