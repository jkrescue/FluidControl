#!/usr/bin/env python3
"""Validate every curated tandem-cylinder trajectory and train-only statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


EXPECTED_SHAPES = {
    "state": (801, 3, 128, 256),
    "mask": (801, 1, 128, 256),
    "omega": (801, 1),
    "force": (801, 4),
    "time": (801, 1),
    "x": (256,),
    "y": (128,),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/curated/tandem_cylinders"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/tandem_cylinders/curated_validation.json"),
    )
    parser.add_argument("--max-abs-omega", type=float,
                        help="override the action bound recorded in manifest.json")
    args = parser.parse_args()

    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    recorded_stats = json.loads((args.data / "normalization.json").read_text(encoding="utf-8"))
    expected_counts = manifest["trajectory_counts"]
    assert set(expected_counts) == {"train", "validation", "test"}
    max_abs_omega = float(
        args.max_abs_omega if args.max_abs_omega is not None else manifest.get("max_abs_omega", 1.0)
    )
    assert recorded_stats["computed_from"] == "train split only"

    state_sum = np.zeros(3, dtype=np.float64)
    state_sq = np.zeros(3, dtype=np.float64)
    state_count = np.zeros(3, dtype=np.int64)
    force_sum = np.zeros(2, dtype=np.float64)
    force_sq = np.zeros(2, dtype=np.float64)
    force_count = 0
    cases = []

    for split, expected_count in expected_counts.items():
        paths = sorted((args.data / split).glob("*.h5"))
        assert len(paths) == expected_count, (split, len(paths), expected_count)
        for path in paths:
            with h5py.File(path, "r") as handle:
                for key, shape in EXPECTED_SHAPES.items():
                    assert handle[key].shape == shape, (path, key, handle[key].shape)
                assert handle.attrs["case"] == path.stem
                assert handle.attrs["split"] == split

                time = handle["time"][:, 0]
                assert np.allclose(time, np.linspace(80.0, 160.0, 801), atol=2e-6)
                assert np.all(np.diff(time) > 0)

                state = handle["state"][:]
                mask = handle["mask"][:, 0].astype(bool)
                omega = handle["omega"][:, 0]
                force = handle["force"][:]
                assert np.isfinite(state).all()
                assert np.isfinite(omega).all()
                assert np.isfinite(force).all()
                assert np.max(np.abs(omega)) <= max_abs_omega + 1.0e-6

                counts = mask.sum(axis=(1, 2))
                pressure_sums = (state[:, 2] * mask).sum(axis=(1, 2), dtype=np.float64)
                max_pressure_mean = float(np.max(np.abs(pressure_sums / counts)))
                max_invalid_abs = float(np.max(np.abs(state) * (~mask[:, None])))
                coverage = float(mask.mean())
                assert 0.8 < coverage < 1.0
                assert max_pressure_mean < 1e-5
                assert max_invalid_abs == 0.0

                if split == "train":
                    for channel in range(3):
                        values = state[:, channel][mask]
                        state_sum[channel] += values.sum(dtype=np.float64)
                        state_sq[channel] += np.square(values, dtype=np.float64).sum(dtype=np.float64)
                        state_count[channel] += len(values)
                    rear_force = force[:, 2:4]
                    force_sum += rear_force.sum(axis=0, dtype=np.float64)
                    force_sq += np.square(rear_force, dtype=np.float64).sum(axis=0, dtype=np.float64)
                    force_count += len(rear_force)

                row = {
                    "case": path.stem,
                    "split": split,
                    "coverage": coverage,
                    "omega_min": float(omega.min()),
                    "omega_max": float(omega.max()),
                    "max_abs_pressure_mean": max_pressure_mean,
                    "max_abs_invalid_state": max_invalid_abs,
                }
                cases.append(row)
                print(json.dumps(row), flush=True)

    state_mean = state_sum / state_count
    state_std = np.sqrt(np.maximum(state_sq / state_count - state_mean**2, 1e-12))
    force_mean = force_sum / force_count
    force_std = np.sqrt(np.maximum(force_sq / force_count - force_mean**2, 1e-12))
    recomputed = {
        "state_mean": state_mean.tolist(),
        "state_std": state_std.tolist(),
        "force_mean": force_mean.tolist(),
        "force_std": force_std.tolist(),
    }
    for key, values in recomputed.items():
        assert np.allclose(values, recorded_stats[key], rtol=1e-8, atol=1e-10), (
            key,
            values,
            recorded_stats[key],
        )

    report = {
        "status": "CURATED_DATASET_OK",
        "trajectory_counts": expected_counts,
        "trajectory_total": len(cases),
        "max_abs_omega": max_abs_omega,
        "recomputed_normalization": recomputed,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("CURATED_DATASET_OK", flush=True)


if __name__ == "__main__":
    main()
