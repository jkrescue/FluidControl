#!/usr/bin/env python3
"""Validate all expanded dynamic trajectories and write their manifest."""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from analyze_baseline import (
    ROOT,
    field_health,
    load_coefficients,
    log_health,
    probe_health,
)
from make_expanded_control_dataset import (
    END_TIME,
    OMEGA_LIMIT,
    SCHEDULES,
    START_TIME,
    schedule_metrics,
)


from make_expanded_edge_replacements import REPLACEMENTS
from make_gate_b_augmentation import CASES as GATE_B_AUGMENTATION_CASES


def validate_case(name: str) -> dict:
    case = ROOT / "cases" / name
    config = json.loads((case / "case_config.json").read_text(encoding="utf-8"))
    solver = log_health(case / "log.pimpleFoam")
    expected_steps = round((END_TIME - START_TIME) / config["delta_t"])
    if not solver["solver_ended_cleanly"] or solver["steps"] != expected_steps:
        raise ValueError(f"incomplete solver run: {name}: {solver}")
    if (
        solver["max_courant"] >= 1.0
        or solver["max_abs_global_continuity_per_step"] >= 1.0e-5
    ):
        raise ValueError(f"numerical health failed: {name}: {solver}")

    fields = field_health(case)
    if fields["snapshot_count"] != 801 or not math.isclose(
        fields["last_time"], END_TIME
    ):
        raise ValueError(f"incomplete fields: {name}: {fields}")

    probes = probe_health(case / "postProcessing" / "wakeProbes" / "80" / "U")
    if probes["samples"] != expected_steps or not math.isclose(
        probes["last_time"], END_TIME
    ):
        raise ValueError(f"incomplete probes: {name}: {probes}")

    force_samples = {}
    for label in ("Front", "Rear"):
        rows = load_coefficients(
            case / "postProcessing" / f"force{label}" / "80" / "coefficient.dat"
        )
        if len(rows) != expected_steps or not math.isclose(rows[-1][0], END_TIME):
            raise ValueError(f"incomplete force series: {name}/{label}")
        force_samples[label.lower()] = len(rows)

    points = [(float(time), float(value)) for time, value in config["action_points"]]
    if points[0][0] != START_TIME or points[-1][0] != END_TIME:
        raise ValueError(f"action table does not span trajectory: {name}")
    if max(abs(value) for _, value in points) > OMEGA_LIMIT + 1.0e-9:
        raise ValueError(f"action exceeds declared range: {name}")

    return {
        "case": name,
        "split": config["split"],
        "schedule_kind": config["schedule_kind"],
        "action": schedule_metrics(points),
        "solver": solver,
        "fields": fields,
        "probe_samples": probes["samples"],
        "force_samples": force_samples,
        "status": "passed_numerical_QC_on_validated_coarse_grid",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "names", nargs="*", help="validate selected cases; default validates all"
    )
    parser.add_argument(
        "--write", type=Path, help="write JSON manifest after validation"
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    known = {spec.name for spec in (*SCHEDULES, *REPLACEMENTS, *GATE_B_AUGMENTATION_CASES)}
    unknown = set(args.names) - known
    if unknown:
        parser.error(f"unknown expanded cases: {', '.join(sorted(unknown))}")
    names = args.names or [spec.name for spec in SCHEDULES]
    if args.workers == 1:
        cases = [validate_case(name) for name in names]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            cases = list(pool.map(validate_case, names))

    split_rates = {}
    for split in ("train", "validation", "test"):
        values = [
            case["action"]["max_abs_domega_dt"]
            for case in cases
            if case["split"] == split
        ]
        if values:
            split_rates[split] = max(values)
    if set(split_rates) == {"train", "validation", "test"} and (
        split_rates["validation"] > split_rates["train"] + 1.0e-9
        or split_rates["test"] > split_rates["train"] + 1.0e-9
    ):
        raise ValueError(
            f"validation/test action-rate coverage exceeds training: {split_rates}"
        )

    manifest = {
        "schema_version": 1,
        "dataset": "tandem_cylinder_dynamic_rotation_expanded_v1",
        "source": "independent OpenFOAM v2512 CFD",
        "geometry": "two equal-diameter tandem cylinders at Re=100 and L/D=5; rear cylinder rotates",
        "time_window": [START_TIME, END_TIME],
        "field_interval": 0.1,
        "omega_limit": [-OMEGA_LIMIT, OMEGA_LIMIT],
        "surface_speed_ratio_limit": [-OMEGA_LIMIT / 2.0, OMEGA_LIMIT / 2.0],
        "split_max_abs_domega_dt": split_rates,
        "cases": cases,
        "splits": {
            split: [case["case"] for case in cases if case["split"] == split]
            for split in ("train", "validation", "test")
        },
        "limitations": [
            "all training trajectories use the 19,290-cell coarse grid",
            "high-rotation medium-grid cases remain independent numerical validation",
            "all trajectories share Re=100, L/D=5, and one geometry",
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
