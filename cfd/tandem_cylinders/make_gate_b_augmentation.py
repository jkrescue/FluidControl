#!/usr/bin/env python3
"""Predeclare independent high-rotation CFD cases after the Gate-B diagnosis.

The two training trajectories cover sustained high absolute rotation. A new
test trajectory is held out because the previous test_04 informed this design.
Only action schedules are inspected for selection; no CFD labels are used.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
from pathlib import Path

from make_expanded_control_dataset import (
    END_TIME,
    OMEGA_LIMIT,
    SCHEDULES,
    START_TIME,
    Schedule,
    generate,
    schedule_metrics,
    schedule_points,
)
from make_expanded_edge_replacements import REPLACEMENTS


CASES = (
    Schedule("expanded_train_24", "train", "edge_hold_permuted", 2026100408),
    Schedule("expanded_train_25", "train", "edge_hold_permuted", 2026100428),
    Schedule("expanded_test_05", "test", "edge_hold_permuted", 2026100520),
)


def sampled_rms(points: list[tuple[float, float]]) -> float:
    times = [point[0] for point in points]
    values = [point[1] for point in points]
    samples = []
    for frame in range(round((END_TIME - START_TIME) / 0.1) + 1):
        time = START_TIME + frame * 0.1
        i = min(max(bisect.bisect_right(times, time) - 1, 0), len(times) - 2)
        t0, t1 = times[i : i + 2]
        v0, v1 = values[i : i + 2]
        samples.append(v0 + (v1 - v0) * (time - t0) / (t1 - t0))
    return math.sqrt(sum(value * value for value in samples) / len(samples))


def audit() -> dict:
    original = (*SCHEDULES, *REPLACEMENTS)
    all_specs = (*original, *CASES)
    signatures = [tuple(schedule_points(spec)) for spec in all_specs]
    # Original dataset has one known triplicate; new cases must be unique from
    # every prior case and one another.
    new_signatures = signatures[-len(CASES) :]
    if len(set(new_signatures)) != len(CASES) or any(
        signature in signatures[: -len(CASES)] for signature in new_signatures
    ):
        raise ValueError("new control schedules duplicate an existing trajectory")
    train_rate = max(
        schedule_metrics(schedule_points(spec))["max_abs_domega_dt"]
        for spec in SCHEDULES
        if spec.split == "train"
    )
    rows = []
    for spec in CASES:
        points = schedule_points(spec)
        metrics = schedule_metrics(points)
        rms = sampled_rms(points)
        if max(abs(metrics["omega_min"]), abs(metrics["omega_max"])) > OMEGA_LIMIT:
            raise ValueError(f"{spec.name}: angular-speed bound exceeded")
        if metrics["max_abs_domega_dt"] > train_rate + 1e-9:
            raise ValueError(f"{spec.name}: rate outside existing train support")
        if rms < 3.62:
            raise ValueError(f"{spec.name}: insufficient high-rotation coverage")
        rows.append(
            {"case": spec.name, "split": spec.split, "kind": spec.kind,
             "seed": spec.seed, "sampled_omega_rms": rms, **metrics}
        )
    return {
        "status": "GATE_B_AUGMENTATION_SCHEDULES_OK",
        "reason": "diagnostic test_04 has high long-horizon drag error in sustained high-rotation segments",
        "new_test_role": "fresh frozen holdout after adapting to test_04; do not train or tune on test_05",
        "preexisting_train_rate_limit": train_rate,
        "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    report = audit()
    known = {spec.name: spec for spec in CASES}
    if set(args.names) - set(known):
        parser.error("unknown augmentation case")
    selected = [known[name] for name in args.names] if args.names else list(CASES)
    if args.audit:
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.list:
        for spec in selected:
            print(spec.name)
        return
    for spec in selected:
        path = generate(spec)
        metadata_path = path / "case_config.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["dataset"] = "tandem_cylinder_dynamic_rotation_gate_b_aug_v3"
        metadata["selection_note"] = report["reason"]
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
