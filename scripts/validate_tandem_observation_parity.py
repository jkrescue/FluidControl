#!/usr/bin/env python3
"""Compare all 67 HydroGym observation channels with original OpenFOAM output."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from validate_tandem_probe_mapping import bilinear_probes, raw_probe_rows

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "cfd" / "tandem_cylinders"))
from analyze_baseline import load_coefficients  # noqa: E402


def exact_force(rows: list[tuple[float, float, float]], target: float) -> np.ndarray:
    matched = [(cd, cl) for time, cd, cl in rows if abs(time - target) <= 1e-6]
    if len(matched) != 1:
        raise ValueError(f"expected one rear-force row at t={target}, got {len(matched)}")
    return np.asarray(matched[0], dtype=np.float64)


def compare(data: Path, cases_root: Path, split: str, case: str, frames: list[int]) -> dict:
    if split not in {"train", "validation", "test"}:
        raise ValueError("invalid split")
    h5_path = data / split / f"{case}.h5"
    case_path = cases_root / case
    if not h5_path.is_file() or not case_path.is_dir():
        raise FileNotFoundError(f"missing curated or raw case: {case}")
    config = json.loads((case_path / "case_config.json").read_text(encoding="utf-8"))
    if config.get("case") != case:
        raise ValueError("raw case metadata mismatch")
    points = np.asarray(config["action_points"], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.diff(points[:, 0]) > 0):
        raise ValueError("invalid OpenFOAM action schedule")
    if not np.isfinite(points).all() or np.abs(points[:, 1]).max() > 5.0 + 1e-6:
        raise ValueError("non-finite or out-of-support action schedule")
    if not frames or len(set(frames)) != len(frames):
        raise ValueError("empty or duplicate requested frames")
    with h5py.File(h5_path, "r") as handle:
        if any(frame < 0 or frame >= len(handle["state"]) for frame in frames):
            raise IndexError("requested frame outside curated trajectory")
        times = [round(float(handle["time"][frame, 0]), 6) for frame in frames]
        if len(set(times)) != len(times):
            raise ValueError("duplicate CFD frame times")
        if min(times) < points[0, 0] - 1e-6 or max(times) > points[-1, 0] + 1e-6:
            raise ValueError("requested frame outside OpenFOAM action schedule")
        probes = raw_probe_rows(
            case_path / "postProcessing/wakeProbes/80/U", set(times)
        )
        force_rows = load_coefficients(
            case_path / "postProcessing/forceRear/80/coefficient.dat"
        )
        comparisons = []
        for frame, time in zip(frames, times):
            raw_force = exact_force(force_rows, time)
            raw_omega = float(np.interp(time, points[:, 0], points[:, 1]))
            raw_obs = np.concatenate((probes[time].reshape(-1), raw_force, [raw_omega]))
            curated_probe = bilinear_probes(
                handle["state"][frame], handle["x"][:], handle["y"][:]
            ).reshape(-1)
            curated_obs = np.concatenate((
                curated_probe,
                np.asarray(handle["force"][frame, 2:4], dtype=np.float64),
                np.asarray(handle["omega"][frame], dtype=np.float64),
            ))
            if raw_obs.shape != (67,) or curated_obs.shape != (67,):
                raise ValueError("expected 67 observation channels")
            if not np.isfinite(raw_obs).all() or not np.isfinite(curated_obs).all():
                raise FloatingPointError("non-finite observation")
            error = np.abs(raw_obs - curated_obs)
            comparisons.append({
                "frame": frame,
                "time": time,
                "probe_max_abs_error": float(error[:64].max()),
                "force_max_abs_error": float(error[64:66].max()),
                "omega_abs_error": float(error[66]),
                "overall_max_abs_error": float(error.max()),
            })
    probe_limit = 0.02
    force_limit = 1e-4
    omega_limit = 1e-5
    passed = all(
        row["probe_max_abs_error"] <= probe_limit
        and row["force_max_abs_error"] <= force_limit
        and row["omega_abs_error"] <= omega_limit
        for row in comparisons
    )
    return {
        "status": "TANDEM_67D_OBSERVATION_PARITY_OK" if passed else "TANDEM_67D_OBSERVATION_PARITY_FAILED",
        "scope": "raw_openfoam_vs_curated_67d_layout_not_full_hydrogym_runtime",
        "case": case,
        "split": split,
        "channel_order": "32*(u,v),rear_cd,rear_cl,applied_omega",
        "thresholds": {"probe": probe_limit, "force": force_limit, "omega": omega_limit},
        "comparisons": comparisons,
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
    if report["status"] != "TANDEM_67D_OBSERVATION_PARITY_OK":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
