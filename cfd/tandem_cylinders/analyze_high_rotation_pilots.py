#!/usr/bin/env python3
"""Audit high-rotation pilot logs and rear-cylinder force statistics."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
CASES = ROOT / "cases"
DEFAULT_CASES = (
    "high_rotation_qm200_dt005",
    "high_rotation_qp200_dt005",
    "high_rotation_qm250_dt005",
    "high_rotation_qp250_dt005",
)


def coefficient_files(case: Path) -> list[Path]:
    files = sorted(case.glob("postProcessing/forceRear/*/coefficient.dat"))
    if not files:
        raise FileNotFoundError(f"no rear coefficient files found in {case}")
    return files


def analyze(name: str, window_start: float, window_end: float | None) -> dict:
    case = CASES / name
    config = json.loads((case / "case_config.json").read_text(encoding="utf-8"))
    log_paths = sorted(case.glob("log.pimpleFoam*"))
    log_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace") for path in log_paths
    )
    if not re.search(r"^End\s*$", log_text, re.MULTILINE):
        raise ValueError(f"solver did not reach End: {name}")
    fatal = re.findall(
        r"FOAM FATAL|segmentation fault|core dumped|\bnan\b|\binf\b",
        log_text,
        re.IGNORECASE,
    )
    if fatal:
        raise ValueError(f"fatal/non-finite markers in {name}: {fatal[:5]}")
    times = [float(value) for value in re.findall(r"^Time = ([0-9.eE+-]+)", log_text, re.MULTILINE)]
    courant = [
        (float(mean), float(maximum))
        for mean, maximum in re.findall(
            r"Courant Number mean: ([0-9.eE+-]+) max: ([0-9.eE+-]+)", log_text
        )
    ]
    arrays = [np.loadtxt(path, comments="#") for path in coefficient_files(case)]
    data = np.concatenate([array.reshape(-1, array.shape[-1]) for array in arrays])
    data = data[np.argsort(data[:, 0])]
    data = data[np.concatenate(([True], np.diff(data[:, 0]) > 1.0e-12))]
    selected = data[data[:, 0] >= window_start]
    if window_end is not None:
        selected = selected[selected[:, 0] <= window_end]
    if len(selected) < 100:
        raise ValueError(f"too few force samples after t={window_start}: {name}")
    # OpenFOAM v2512 forceCoeffs columns: time, Cd, Cd(f), Cd(r), Cl, ...
    cd = selected[:, 1]
    cl = selected[:, 4]
    sample_dt = float(np.median(np.diff(selected[:, 0])))
    frequencies = np.fft.rfftfreq(len(cl), d=sample_dt)
    spectrum = np.abs(np.fft.rfft(cl - cl.mean()))
    dominant_frequency = float(frequencies[1 + np.argmax(spectrum[1:])])
    return {
        "case": name,
        "omega": config["rear_angular_velocity"],
        "q": config["rear_surface_speed_ratio"],
        "delta_t": config["delta_t"],
        "completed_time": max(times),
        "solver_steps": len(times),
        "courant_mean_max": max(value[0] for value in courant),
        "courant_absolute_max": max(value[1] for value in courant),
        "force_window": [window_start, window_end if window_end is not None else max(times)],
        "force_samples": len(selected),
        "rear_cd_mean": float(cd.mean()),
        "rear_cd_std": float(cd.std()),
        "rear_cl_mean": float(cl.mean()),
        "rear_cl_abs_mean": float(np.abs(cl).mean()),
        "rear_cl_rms": float(np.sqrt(np.mean(cl * cl))),
        "rear_cl_min": float(cl.min()),
        "rear_cl_max": float(cl.max()),
        "rear_cl_dominant_frequency": dominant_frequency,
        "finite": bool(np.isfinite(selected).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*", default=DEFAULT_CASES)
    parser.add_argument("--window-start", type=float, default=90.0)
    parser.add_argument("--window-end", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"window_start": args.window_start, "window_end": args.window_end, "cases": []}
    for name in args.cases:
        row = analyze(name, args.window_start, args.window_end)
        report["cases"].append(row)
        print(
            f"{name}: Co_max={row['courant_absolute_max']:.4g}, "
            f"Cd_mean={row['rear_cd_mean']:.6g}, Cl_mean={row['rear_cl_mean']:.6g}, "
            f"Cl_rms={row['rear_cl_rms']:.6g}"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("HIGH_ROTATION_PILOTS_OK")


if __name__ == "__main__":
    main()
