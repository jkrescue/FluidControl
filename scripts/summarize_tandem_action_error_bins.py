#!/usr/bin/env python3
"""Aggregate rollout errors by action magnitude and action-rate bins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


OMEGA_EDGES = (0.0, 1.0, 2.0, 3.0, 4.0, 5.000001)
RATE_EDGES = (0.0, 1.0, 2.0, 4.0, 6.0, 6.700001)
ERROR_KEYS = ("state_mae_physical_units", "rear_force_mae")


def aggregate(rows: list[dict], value_key: str, edges: tuple[float, ...]) -> list[dict]:
    result = []
    for low, high in zip(edges[:-1], edges[1:]):
        selected = [row for row in rows if low <= row[value_key] < high]
        result.append({
            "range": [low, high],
            "count": len(selected),
            **{
                key: float(np.mean([row[key] for row in selected])) if selected else None
                for key in ERROR_KEYS
            },
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("segments", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.segments.read_text(encoding="utf-8"))
    report = {"source": str(args.segments), "horizons": {}}
    lines = [
        "| Horizon | Statistic | Bin | Segments | Field MAE | Rear force MAE |",
        "| ---: | --- | --- | ---: | ---: | ---: |",
    ]
    horizons = sorted({int(row["horizon"]) for row in source["segments"]})
    for horizon in horizons:
        rows = [row for row in source["segments"] if int(row["horizon"]) == horizon]
        omega_bins = aggregate(rows, "max_abs_omega", OMEGA_EDGES)
        rate_bins = aggregate(rows, "max_abs_domega_dt", RATE_EDGES)
        report["horizons"][str(horizon)] = {
            "max_abs_omega_bins": omega_bins,
            "max_abs_domega_dt_bins": rate_bins,
        }
        for label, bins in (("max |omega|", omega_bins), ("max |domega/dt|", rate_bins)):
            for row in bins:
                low, high = row["range"]
                field = "-" if row[ERROR_KEYS[0]] is None else f"{row[ERROR_KEYS[0]]:.8g}"
                force = "-" if row[ERROR_KEYS[1]] is None else f"{row[ERROR_KEYS[1]]:.8g}"
                lines.append(
                    f"| {horizon} | {label} | [{low:g}, {high:g}) | {row['count']} | "
                    f"{field} | {force} |"
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
