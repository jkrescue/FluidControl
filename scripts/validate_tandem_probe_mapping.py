#!/usr/bin/env python3
"""Audit 32 tandem velocity probes against original OpenFOAM probe output."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np


def bilinear_probes(state: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    values = []
    for py in np.linspace(6.0, 9.0, 32):
        px = 17.0
        ix = int(np.searchsorted(x, px) - 1)
        iy = int(np.searchsorted(y, py) - 1)
        if not (0 <= ix < len(x)-1 and 0 <= iy < len(y)-1):
            raise ValueError("probe outside curated field")
        wx = (px - x[ix]) / (x[ix+1] - x[ix])
        wy = (py - y[iy]) / (y[iy+1] - y[iy])
        corners = state[:2, iy:iy+2, ix:ix+2]
        values.append(
            (1-wx)*(1-wy)*corners[:, 0, 0]
            + wx*(1-wy)*corners[:, 0, 1]
            + (1-wx)*wy*corners[:, 1, 0]
            + wx*wy*corners[:, 1, 1]
        )
    return np.asarray(values)


def raw_probe_rows(path: Path, times: set[float]) -> dict[float, np.ndarray]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            time = round(float(line.split(maxsplit=1)[0]), 6)
            if time not in times:
                continue
            vectors = re.findall(r"\(([^()]*)\)", line)
            if len(vectors) != 32:
                raise ValueError(f"expected 32 raw probes at time {time}")
            rows[time] = np.asarray([
                [float(value) for value in vector.split()[:2]]
                for vector in vectors
            ])
    if set(rows) != times:
        raise ValueError(f"missing raw probe times: {sorted(times - set(rows))}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cases-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[100, 400, 800])
    parser.add_argument("--max-abs-error", type=float, default=0.02)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.max_abs_error <= 0 or len(set(args.frames)) != len(args.frames):
        parser.error("positive error threshold and unique frames required")
    h5_path = args.data / args.split / f"{args.case}.h5"
    probe_path = args.cases_root / args.case / "postProcessing/wakeProbes/80/U"
    with h5py.File(h5_path, "r") as handle:
        count = len(handle["state"])
        if any(frame < 0 or frame >= count for frame in args.frames):
            parser.error(f"frames must lie within 0..{count-1}")
        x, y = handle["x"][:], handle["y"][:]
        times = {
            frame: round(float(handle["time"][frame, 0]), 6)
            for frame in args.frames
        }
        raw = raw_probe_rows(probe_path, set(times.values()))
        comparisons = []
        for frame in args.frames:
            interpolated = bilinear_probes(handle["state"][frame], x, y)
            errors = np.abs(interpolated - raw[times[frame]])
            comparisons.append({
                "frame": frame,
                "time": times[frame],
                "mean_abs_error": float(errors.mean()),
                "max_abs_error": float(errors.max()),
                "u_mean_abs_error": float(errors[:, 0].mean()),
                "v_mean_abs_error": float(errors[:, 1].mean()),
            })
    largest = max(row["max_abs_error"] for row in comparisons)
    report = {
        "status": "TANDEM_PROBE_MAPPING_OK" if largest <= args.max_abs_error else "TANDEM_PROBE_MAPPING_FAILED",
        "case": args.case,
        "split": args.split,
        "source": "real OpenFOAM wakeProbes vs Curator HDF5 bilinear interpolation",
        "max_abs_error_threshold": args.max_abs_error,
        "max_observed_abs_error": largest,
        "comparisons": comparisons,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "TANDEM_PROBE_MAPPING_OK":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
