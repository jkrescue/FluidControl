#!/usr/bin/env python3
"""Validate dynamic control trajectories and print or write a manifest."""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from analyze_baseline import ROOT, field_health, load_coefficients, log_health, probe_health
from make_dynamic_control_dataset import END_TIME, SCHEDULES, START_TIME


def validate_case(name: str) -> dict:
    case = ROOT / "cases" / name
    config = json.loads((case / "case_config.json").read_text(encoding="utf-8"))
    solver = log_health(case / "log.pimpleFoam")
    expected_steps = round((END_TIME - START_TIME) / config["delta_t"])
    if not solver["solver_ended_cleanly"] or solver["steps"] != expected_steps:
        raise ValueError(f"incomplete solver run: {name}: {solver}")
    if solver["max_courant"] >= 1 or solver["max_abs_global_continuity_per_step"] >= 1e-5:
        raise ValueError(f"numerical health failed: {name}: {solver}")

    fields = field_health(case)
    if fields["snapshot_count"] != 801 or not math.isclose(fields["last_time"], END_TIME):
        raise ValueError(f"incomplete fields: {name}: {fields}")

    probes = probe_health(case / "postProcessing" / "wakeProbes" / "80" / "U")
    if probes["samples"] != expected_steps or not math.isclose(probes["last_time"], END_TIME):
        raise ValueError(f"incomplete probes: {name}: {probes}")

    force_samples = {}
    for label in ("Front", "Rear"):
        rows = load_coefficients(case / "postProcessing" / f"force{label}" / "80" / "coefficient.dat")
        if len(rows) != expected_steps or not math.isclose(rows[-1][0], END_TIME):
            raise ValueError(f"incomplete force series: {name}/{label}")
        force_samples[label.lower()] = len(rows)

    actions = config["action_points"]
    if actions[0][0] != START_TIME or actions[-1][0] != END_TIME:
        raise ValueError(f"action table does not span trajectory: {name}")
    if max(abs(value) for _, value in actions) > 1.0000001:
        raise ValueError(f"action exceeds declared range: {name}")

    return {
        "case": name,
        "split": config["split"],
        "schedule_kind": config["schedule_kind"],
        "action_range": config["rear_angular_velocity_range"],
        "solver": solver,
        "fields": fields,
        "probe_samples": probes["samples"],
        "force_samples": force_samples,
        "status": "passed_basic_numerical_QC_on_validated_coarse_grid",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path, help="write JSON manifest after validation")
    parser.add_argument("--workers", type=int, default=4,
                        help="parallel read-only validators (default: 4)")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    names = [spec.name for spec in SCHEDULES]
    if args.workers == 1:
        cases = [validate_case(name) for name in names]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            cases = list(pool.map(validate_case, names))
    manifest = {
        "schema_version": 1,
        "dataset": "tandem_cylinder_dynamic_rotation_stage1",
        "source": "independent OpenFOAM v2512 CFD, not PolyU experimental data",
        "geometry": "two equal-diameter tandem cylinders at Re=100 and L/D=5; rear cylinder rotates",
        "time_window": [START_TIME, END_TIME],
        "field_interval": 0.1,
        "cases": cases,
        "splits": {
            split: [case["case"] for case in cases if case["split"] == split]
            for split in ("train", "validation", "test")
        },
        "model_training_started": False,
        "limitations": [
            "first-stage action range is limited to |omega|<=1",
            "all trajectories share Re=100, L/D=5, and one geometry",
            "validation establishes numerical health and representative grid sensitivity, not reproduction of the authors' LBM solver",
        ],
    }
    output = json.dumps(manifest, indent=2) + "\n"
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(output, encoding="utf-8")
        print(f"Validated {len(cases)} trajectories; wrote {args.write}")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
