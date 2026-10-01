#!/usr/bin/env python3
"""Generate independent edge-hold CFD replacements for the duplicated holdouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from make_expanded_control_dataset import (
    OMEGA_LIMIT,
    SCHEDULES,
    Schedule,
    generate,
    schedule_metrics,
    schedule_points,
)

REPLACEMENTS = (
    Schedule("expanded_validation_04", "validation", "edge_hold_permuted", 2026100204),
    Schedule("expanded_test_04", "test", "edge_hold_permuted", 2026100304),
)


def audit() -> dict:
    old = {spec.name: schedule_points(spec) for spec in SCHEDULES}
    replacement = {spec.name: schedule_points(spec) for spec in REPLACEMENTS}
    if len({tuple(points) for points in [*old.values(), *replacement.values()]}) != len(
        old
    ) - 2 + len(replacement):
        # The original 32 cases intentionally contain one triplicate edge-hold action.
        raise ValueError("replacement actions are not unique from original schedules")
    train_max_rate = max(
        schedule_metrics(points)["max_abs_domega_dt"]
        for spec in SCHEDULES
        if spec.split == "train"
        for points in [old[spec.name]]
    )
    rows = {}
    for spec in REPLACEMENTS:
        metrics = schedule_metrics(replacement[spec.name])
        if max(abs(metrics["omega_min"]), abs(metrics["omega_max"])) > OMEGA_LIMIT:
            raise ValueError(f"{spec.name}: action exceeds omega limit")
        if metrics["max_abs_domega_dt"] > train_max_rate + 1.0e-9:
            raise ValueError(f"{spec.name}: action rate outside training coverage")
        rows[spec.name] = {
            "split": spec.split,
            "seed": spec.seed,
            "kind": spec.kind,
            **metrics,
        }
    return {
        "status": "EDGE_REPLACEMENTS_OK",
        "train_max_abs_domega_dt": train_max_rate,
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "names", nargs="*", help="replacement case names; defaults to both"
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    report = audit()
    known = {spec.name: spec for spec in REPLACEMENTS}
    unknown = set(args.names) - set(known)
    if unknown:
        parser.error(f"unknown replacement case(s): {', '.join(sorted(unknown))}")
    selected = (
        [known[name] for name in args.names] if args.names else list(REPLACEMENTS)
    )
    if args.audit:
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.list:
        for spec in selected:
            print(spec.name)
        return
    for spec in selected:
        print(generate(spec))


if __name__ == "__main__":
    main()
