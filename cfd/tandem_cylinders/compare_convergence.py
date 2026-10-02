#!/usr/bin/env python3
"""Compare time-window force and sensor statistics of two stationary cases."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from analyze_baseline import load_coefficients, window_stats


ROOT = Path(__file__).resolve().parent / "cases"
WINDOW = (80.0, 160.0)
CASES = ("tandem_baseline", "tandem_dt005", "tandem_medium_dt005",
         "tandem_backward_dt005", "tandem_medium_backward_dt005",
         "control_small_p100", "control_grid_p100_medium")


def probe_moments(path: Path) -> dict:
    count = 0
    sums = [[0.0, 0.0] for _ in range(32)]
    squares = [[0.0, 0.0] for _ in range(32)]
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        time = float(line.split(maxsplit=1)[0])
        if not WINDOW[0] <= time <= WINDOW[1]:
            continue
        vectors = re.findall(r"\(([^()]*)\)", line)
        if len(vectors) != 32:
            raise ValueError(f"{path}: expected 32 probes at t={time}, found {len(vectors)}")
        for index, vector in enumerate(vectors):
            u, v, _ = map(float, vector.split())
            if not math.isfinite(u) or not math.isfinite(v):
                raise ValueError(f"{path}: non-finite probe at t={time}")
            for component, value in enumerate((u, v)):
                sums[index][component] += value
                squares[index][component] += value * value
        count += 1
    if count < 3:
        raise ValueError(f"{path}: too few probe samples")
    means = [[total / count for total in pair] for pair in sums]
    rms = [[math.sqrt(max(0, squares[i][j] / count - means[i][j] ** 2))
            for j in range(2)] for i in range(32)]
    return {"samples": count, "mean": means, "rms": rms}


def relative_difference(a: float, b: float) -> float:
    return abs(b - a) / max(abs(a), 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", choices=CASES)
    parser.add_argument("candidate", choices=CASES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.reference == args.candidate:
        parser.error("choose different cases")
    report = {"reference": args.reference, "candidate": args.candidate,
              "window": list(WINDOW), "force_relative_difference": {}}
    for label, folder in (("front", "forceFront"), ("rear", "forceRear")):
        statistics = []
        for name in (args.reference, args.candidate):
            path = ROOT / name / "postProcessing" / folder / "0" / "coefficient.dat"
            statistics.append(window_stats(load_coefficients(path), *WINDOW))
        report["force_relative_difference"][label] = {
            field: relative_difference(statistics[0][field], statistics[1][field])
            for field in ("cd_mean", "cl_rms", "st_zero_crossing")
        }
        report["force_relative_difference"][label]["values"] = {
            field: [statistics[0][field], statistics[1][field]]
            for field in ("cd_mean", "cl_rms", "st_zero_crossing")
        }
    probes = [probe_moments(ROOT / name / "postProcessing" / "wakeProbes" / "0" / "U")
              for name in (args.reference, args.candidate)]
    report["probe_max_absolute_difference"] = {
        quantity: [max(abs(probes[1][quantity][i][j] - probes[0][quantity][i][j])
                       for i in range(32)) for j in range(2)]
        for quantity in ("mean", "rms")
    }
    report["probe_samples"] = [p["samples"] for p in probes]
    payload = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
