#!/usr/bin/env python3
"""Audit the 69-channel total-drag observation against original OpenFOAM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from fluid_control.openfoam_observation import total_drag_observation_at
from validate_tandem_probe_mapping import bilinear_probes


def compare(
    data: Path, cases_root: Path, split: str, case: str, frames: list[int]
) -> dict:
    if split not in {"train", "validation", "test"}:
        raise ValueError("invalid split")
    curated = data / split / f"{case}.h5"
    raw = cases_root / case
    if not curated.is_file() or not raw.is_dir():
        raise FileNotFoundError(f"missing curated or raw CFD case: {case}")
    config = json.loads((raw / "case_config.json").read_text(encoding="utf-8"))
    if config.get("case") != case:
        raise ValueError("OpenFOAM case metadata mismatch")
    points = np.asarray(config["action_points"], dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or not np.isfinite(points).all()
        or not np.all(np.diff(points[:, 0]) > 0)
    ):
        raise ValueError("invalid raw OpenFOAM action schedule")
    if not frames or len(set(frames)) != len(frames):
        raise ValueError("empty or duplicate frame request")
    rows = []
    with h5py.File(curated, "r") as handle:
        count = len(handle["state"])
        if any(not 0 <= frame < count for frame in frames):
            raise IndexError(f"requested frame outside 0..{count - 1}")
        x = np.asarray(handle["x"][:])
        y = np.asarray(handle["y"][:])
        for frame in frames:
            time = float(handle["time"][frame, 0])
            raw_case = raw
            if frame == 0 and "source_restart_case" in config:
                raw_case = cases_root / str(config["source_restart_case"])
            raw_omega = float(np.interp(time, points[:, 0], points[:, 1]))
            real, sources = total_drag_observation_at(raw_case, time, raw_omega)
            probes = bilinear_probes(handle["state"][frame], x, y).reshape(-1)
            estimated = np.concatenate(
                (
                    probes,
                    np.asarray(handle["force"][frame], dtype=np.float64),
                    np.asarray(handle["omega"][frame], dtype=np.float64),
                )
            )
            if real.shape != (69,) or estimated.shape != (69,):
                raise ValueError("expected exactly 69 observation channels")
            if not np.isfinite(real).all() or not np.isfinite(estimated).all():
                raise FloatingPointError("non-finite CFD observation")
            error = np.abs(real - estimated)
            rows.append(
                {
                    "frame": frame,
                    "time": time,
                    "probe_max_abs_error": float(error[:64].max()),
                    "front_force_max_abs_error": float(error[64:66].max()),
                    "rear_force_max_abs_error": float(error[66:68].max()),
                    "omega_abs_error": float(error[68]),
                    "raw_front_force_sources": sources["front_force_sources"],
                }
            )
    limits = {"probe": 0.02, "force": 1e-4, "omega": 1e-5}
    passed = all(
        row["probe_max_abs_error"] <= limits["probe"]
        and row["front_force_max_abs_error"] <= limits["force"]
        and row["rear_force_max_abs_error"] <= limits["force"]
        and row["omega_abs_error"] <= limits["omega"]
        for row in rows
    )
    return {
        "status": "TANDEM_69D_OBSERVATION_PARITY_OK" if passed else "TANDEM_69D_OBSERVATION_PARITY_FAILED",
        "scope": "raw_openfoam_vs_curated_stage_c_input_not_closed_loop_benefit",
        "case": case,
        "split": split,
        "channel_order": "32*(u,v),front_cd,front_cl,rear_cd,rear_cl,applied_omega",
        "thresholds": limits,
        "comparisons": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cases-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[100, 400, 800])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    report = compare(args.data, args.cases_root, args.split, args.case, args.frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "TANDEM_69D_OBSERVATION_PARITY_OK":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
