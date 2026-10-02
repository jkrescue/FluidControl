#!/usr/bin/env python3
"""Add front+rear force normalization to an existing curated tandem dataset."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

CHANNELS = ("front_cd", "front_cl", "rear_cd", "rear_cl")


def compute(root: Path) -> tuple[np.ndarray, np.ndarray, int]:
    total = np.zeros(4, dtype=np.float64)
    square = np.zeros(4, dtype=np.float64)
    count = 0
    paths = sorted((root / "train").glob("*.h5"))
    if not paths:
        raise FileNotFoundError(f"no training HDF5 trajectories under {root}")
    for path in paths:
        with h5py.File(path, "r") as handle:
            force = np.asarray(handle["force"][:], dtype=np.float64)
        if force.ndim != 2 or force.shape[1] != 4 or not np.isfinite(force).all():
            raise ValueError(f"invalid four-channel force data: {path}")
        total += force.sum(axis=0, dtype=np.float64)
        square += np.square(force, dtype=np.float64).sum(axis=0, dtype=np.float64)
        count += len(force)
    mean = total / count
    std = np.sqrt(np.maximum(square / count - mean**2, 1e-12))
    return mean, std, count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    path = root / "normalization.json"
    stats = json.loads(path.read_text(encoding="utf-8"))
    mean, std, count = compute(root)
    if not np.allclose(mean[2:4], stats["force_mean"], rtol=1e-8, atol=1e-10):
        raise ValueError("existing rear force mean does not match recomputed train data")
    if not np.allclose(std[2:4], stats["force_std"], rtol=1e-8, atol=1e-10):
        raise ValueError("existing rear force std does not match recomputed train data")
    stats.update({
        "all_force_channels": list(CHANNELS),
        "all_force_mean": mean.tolist(),
        "all_force_std": std.tolist(),
        "all_force_normalization_samples": count,
        "all_force_computed_from": "train split only",
    })
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    print(json.dumps({
        "status": "ALL_FORCE_NORMALIZATION_ADDED",
        "path": str(path),
        "samples": count,
        "channels": list(CHANNELS),
        "mean": mean.tolist(),
        "std": std.tolist(),
    }, indent=2))


if __name__ == "__main__":
    main()
