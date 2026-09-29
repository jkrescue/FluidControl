#!/usr/bin/env python3
"""Compare matched coarse and medium high-rotation pilot statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = (
    "rear_cd_mean",
    "rear_cd_std",
    "rear_cl_mean",
    "rear_cl_abs_mean",
    "rear_cl_rms",
    "rear_cl_dominant_frequency",
)


def relative_difference(reference: float, candidate: float) -> float | None:
    if abs(reference) < 1.0e-12:
        return None
    return abs(candidate - reference) / abs(reference)


def symmetry(rows: dict[float, dict], metric: str) -> dict:
    negative = abs(float(rows[-2.5][metric]))
    positive = abs(float(rows[2.5][metric]))
    scale = 0.5 * (negative + positive)
    return {
        "negative_magnitude": negative,
        "positive_magnitude": positive,
        "relative_magnitude_difference": abs(positive - negative) / scale,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coarse", type=Path)
    parser.add_argument("medium", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    coarse = json.loads(args.coarse.read_text(encoding="utf-8"))
    medium = json.loads(args.medium.read_text(encoding="utf-8"))
    coarse_rows = {float(row["q"]): row for row in coarse["cases"]}
    medium_rows = {float(row["q"]): row for row in medium["cases"]}
    if set(coarse_rows) != {-2.5, 2.5} or set(medium_rows) != {-2.5, 2.5}:
        raise ValueError("expected matched q=-2.5 and q=+2.5 cases")

    comparison = {}
    for q in (-2.5, 2.5):
        comparison[str(q)] = {}
        for metric in METRICS:
            old = float(coarse_rows[q][metric])
            new = float(medium_rows[q][metric])
            comparison[str(q)][metric] = {
                "coarse": old,
                "medium": new,
                "absolute_difference": abs(new - old),
                "relative_difference": relative_difference(old, new),
            }

    cl_rms_differences = [
        comparison[str(q)]["rear_cl_rms"]["relative_difference"] for q in (-2.5, 2.5)
    ]
    report = {
        "coarse_report": str(args.coarse),
        "medium_report": str(args.medium),
        "force_window": coarse["cases"][0]["force_window"],
        "comparison": comparison,
        "mean_rear_cl_rms_relative_difference": sum(cl_rms_differences) / 2.0,
        "max_rear_cl_rms_relative_difference": max(cl_rms_differences),
        "coarse_symmetry": {
            "rear_cl_mean": symmetry(coarse_rows, "rear_cl_mean"),
            "rear_cl_rms": symmetry(coarse_rows, "rear_cl_rms"),
        },
        "medium_symmetry": {
            "rear_cl_mean": symmetry(medium_rows, "rear_cl_mean"),
            "rear_cl_rms": symmetry(medium_rows, "rear_cl_rms"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
