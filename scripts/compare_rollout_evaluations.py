#!/usr/bin/env python3
"""Compare two tandem-cylinder rollout evaluation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = (
    "state_mae_physical_units",
    "rear_force_mae",
    "rear_cd_mae",
    "rear_cl_mae",
)


def comparison(baseline: dict, candidate: dict) -> dict:
    """Return matched aggregate and per-case error changes."""
    horizons = sorted(
        set(baseline["summary"]) & set(candidate["summary"]), key=int
    )
    result = {
        "baseline_checkpoint": baseline["checkpoint_dir"],
        "candidate_checkpoint": candidate["checkpoint_dir"],
        "horizons": {},
        "cases": {},
    }
    for horizon in horizons:
        result["horizons"][horizon] = {}
        for metric in METRICS:
            old = float(baseline["summary"][horizon][metric])
            new = float(candidate["summary"][horizon][metric])
            result["horizons"][horizon][metric] = {
                "baseline": old,
                "candidate": new,
                "relative_change_percent": 100.0 * (new - old) / old,
            }

    baseline_cases = {row["case"]: row for row in baseline["cases"]}
    candidate_cases = {row["case"]: row for row in candidate["cases"]}
    for case in sorted(set(baseline_cases) & set(candidate_cases)):
        result["cases"][case] = {}
        for horizon in horizons:
            result["cases"][case][horizon] = {}
            for metric in METRICS:
                old = float(baseline_cases[case]["horizons"][horizon][metric])
                new = float(candidate_cases[case]["horizons"][horizon][metric])
                result["cases"][case][horizon][metric] = {
                    "baseline": old,
                    "candidate": new,
                    "relative_change_percent": 100.0 * (new - old) / old,
                }
    return result


def markdown(report: dict) -> str:
    """Render the aggregate comparison as a compact Markdown table."""
    lines = [
        "| Horizon | Metric | Baseline | Candidate | Change |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    labels = {
        "state_mae_physical_units": "Field MAE",
        "rear_force_mae": "Rear force MAE",
        "rear_cd_mae": "Rear Cd MAE",
        "rear_cl_mae": "Rear Cl MAE",
    }
    for horizon, metrics in report["horizons"].items():
        for metric, values in metrics.items():
            lines.append(
                f"| {horizon} | {labels[metric]} | {values['baseline']:.8g} | "
                f"{values['candidate']:.8g} | {values['relative_change_percent']:+.2f}% |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    report = comparison(baseline, candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown(report), encoding="utf-8")
    print(markdown(report))


if __name__ == "__main__":
    main()
