#!/usr/bin/env python3
"""Summarize altered-action-input diagnostics against observed CFD actions."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    by_mode = {report["action_mode"]: report for report in reports}
    if "observed" not in by_mode:
        raise ValueError("an observed-action report is required")

    result = {"reference": "observed", "modes": {}}
    rows = []
    for mode, report in by_mode.items():
        result["modes"][mode] = {}
        for horizon, values in report["summary"].items():
            reference = by_mode["observed"]["summary"][horizon]
            result["modes"][mode][horizon] = {
                metric: {
                    "value": values[metric],
                    "relative_change": (
                        values[metric] / reference[metric] - 1.0
                        if reference[metric] != 0.0 else None
                    ),
                }
                for metric in METRICS
            }
            result["modes"][mode][horizon].update({
                "stable": values["stable"],
                "failed_segments": values["failed_segments"],
            })
            if mode != "observed":
                rows.append((
                    mode,
                    horizon,
                    values["state_mae_physical_units"],
                    values["state_mae_physical_units"]
                    / reference["state_mae_physical_units"] - 1.0,
                    values["rear_force_mae"],
                    values["rear_force_mae"] / reference["rear_force_mae"] - 1.0,
                ))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "| Action input | Horizon | Field MAE | Change vs observed | Rear force MAE | Change vs observed |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, horizon, field, field_change, force, force_change in rows:
        lines.append(
            f"| {mode} | {horizon} | {field:.8g} | {field_change:+.2%} | "
            f"{force:.8g} | {force_change:+.2%} |"
        )
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
