#!/usr/bin/env python3
"""Read raw OpenFOAM force coefficients; report predeclared baseline windows.

Standard-library only. Results are numerical diagnostics, not proof of grid
convergence or agreement with Zhao et al. (2024).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WINDOWS = ((80.0, 120.0), (120.0, 160.0), (80.0, 160.0))


def load_coefficients(path: Path) -> list[tuple[float, float, float]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 5:
            continue
        t, cd, cl = float(fields[0]), float(fields[1]), float(fields[4])
        if not all(math.isfinite(value) for value in (t, cd, cl)):
            raise ValueError(f"non-finite coefficient at t={t}")
        rows.append((t, cd, cl))
    if any(b[0] <= a[0] for a, b in zip(rows, rows[1:])):
        raise ValueError(f"non-increasing or duplicate time in {path}")
    return rows


def rising_zero_crossings(times: list[float], values: list[float]) -> list[float]:
    centre = statistics.mean(values)
    zeroes = []
    for i in range(len(times) - 1):
        a, b = values[i] - centre, values[i + 1] - centre
        if a < 0 <= b and b > a:
            zeroes.append(times[i] - a * (times[i + 1] - times[i]) / (b - a))
    return zeroes


def window_stats(rows: list[tuple[float, float, float]], begin: float, end: float) -> dict:
    part = [row for row in rows if begin <= row[0] <= end]
    if len(part) < 3:
        return {"window": [begin, end], "status": "insufficient_samples", "samples": len(part)}
    times, drag, lift = map(list, zip(*part))
    cl_mean = statistics.mean(lift)
    cl_rms = math.sqrt(statistics.mean((v - cl_mean) ** 2 for v in lift))
    zeroes = rising_zero_crossings(times, lift)
    periods = [b - a for a, b in zip(zeroes, zeroes[1:])]
    return {
        "window": [begin, end],
        "samples": len(part),
        "cd_mean": statistics.mean(drag),
        "cl_mean": cl_mean,
        "cl_abs_mean": statistics.mean(abs(v) for v in lift),
        "cl_rms": cl_rms,
        "cl_min": min(lift),
        "cl_max": max(lift),
        "cl_abs_max": max(abs(v) for v in lift),
        "period_count": len(periods),
        "st_zero_crossing": 1 / statistics.median(periods) if periods else None,
    }


def log_health(path: Path) -> dict:
    maximum_co = 0.0
    maximum_global_continuity = 0.0
    number_of_steps = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        co = re.search(r"Courant Number mean: \S+ max: (\S+)", line)
        if co:
            maximum_co = max(maximum_co, float(co.group(1)))
        continuity = re.search(r"time step continuity errors : .*global = ([-+0-9.eE]+)", line)
        if continuity:
            maximum_global_continuity = max(maximum_global_continuity, abs(float(continuity.group(1))))
        if line.startswith("Time = "):
            number_of_steps += 1
    return {"steps": number_of_steps, "max_courant": maximum_co,
            "max_abs_global_continuity_per_step": maximum_global_continuity,
            "solver_ended_cleanly": path.read_text(encoding="utf-8", errors="replace").rstrip().endswith("End")}


def probe_health(path: Path) -> dict:
    samples = 0
    first_time = last_time = None
    min_u = min_v = float("inf")
    max_u = max_v = -float("inf")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        timestamp = float(line.split(maxsplit=1)[0])
        vectors = re.findall(r"\(([^()]*)\)", line)
        if len(vectors) != 32:
            raise ValueError(f"expected 32 probes, found {len(vectors)} at t={timestamp}")
        for vector in vectors:
            u, v, w = map(float, vector.split())
            if not all(map(math.isfinite, (u, v, w))):
                raise ValueError(f"non-finite probe value at t={timestamp}")
            min_u, max_u = min(min_u, u), max(max_u, u)
            min_v, max_v = min(min_v, v), max(max_v, v)
        first_time = timestamp if first_time is None else first_time
        last_time = timestamp
        samples += 1
    return {"samples": samples, "probes_per_sample": 32, "first_time": first_time,
            "last_time": last_time, "u_range": [min_u, max_u], "v_range": [min_v, max_v]}


def field_health(case: Path) -> dict:
    times = []
    for folder in case.iterdir():
        if not folder.is_dir():
            continue
        try:
            time = float(folder.name)
        except ValueError:
            continue
        if time <= 0:
            continue
        for field in ("U", "p"):
            path = folder / field
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"missing/empty {path}")
            if re.search(r"\b(?:nan|inf|infinity)\b", path.read_text(encoding="utf-8"), re.IGNORECASE):
                raise ValueError(f"non-finite text in {path}")
        times.append(time)
    return {"snapshot_count": len(times), "first_time": min(times) if times else None,
            "last_time": max(times) if times else None, "fields": ["U", "p"],
            "text_nonfinite_check": "passed"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("single_baseline", "tandem_baseline",
                                         "tandem_dt005", "tandem_medium_dt005",
                                         "tandem_backward_dt005", "tandem_medium_backward_dt005",
                                         "control_small_m100", "control_small_m050",
                                         "control_small_z000", "control_small_p050",
                                         "control_small_p100"))
    args = parser.parse_args()
    case = ROOT / "cases" / args.case
    report = {"case": args.case, "windows": {}, "solver": log_health(case / "log.pimpleFoam"),
              "fields": field_health(case)}
    for label, obj in (("front", "forceFront"), ("rear", "forceRear")):
        path = case / "postProcessing" / obj / "0" / "coefficient.dat"
        if path.exists():
            rows = load_coefficients(path)
            report["windows"][label] = [window_stats(rows, a, b) for a, b in WINDOWS]
            report["windows"][label].append({"last_time": rows[-1][0], "total_samples": len(rows)})
    probes = case / "postProcessing" / "wakeProbes" / "0" / "U"
    if probes.exists():
        report["probes"] = probe_health(probes)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
