#!/usr/bin/env python3
"""Validate selected expanded Curator HDF5 trajectories before raw cleanup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


EXPECTED = {
    "state": (801, 3, 128, 256),
    "mask": (801, 1, 128, 256),
    "omega": (801, 1),
    "force": (801, 4),
    "time": (801, 1),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("cases", nargs="+")
    args = parser.parse_args()
    rows = []
    for name in args.cases:
        matches = list(args.data.glob(f"*/{name}.h5"))
        if len(matches) != 1:
            raise FileNotFoundError(f"expected one curated file for {name}, found {matches}")
        path = matches[0]
        with h5py.File(path, "r") as handle:
            for key, shape in EXPECTED.items():
                if handle[key].shape != shape:
                    raise ValueError(f"{name}:{key} shape {handle[key].shape}, expected {shape}")
            time = np.asarray(handle["time"][:, 0])
            omega = np.asarray(handle["omega"][:, 0])
            force = np.asarray(handle["force"][:])
            if not np.allclose(time, np.linspace(80.0, 160.0, 801), atol=2.0e-6):
                raise ValueError(f"{name}: invalid time axis")
            if not np.isfinite(omega).all() or not np.isfinite(force).all():
                raise ValueError(f"{name}: non-finite omega or force")
            # Read state one frame at a time to keep validation memory bounded.
            max_invalid = 0.0
            for index in range(801):
                state = np.asarray(handle["state"][index])
                mask = np.asarray(handle["mask"][index, 0]).astype(bool)
                if not np.isfinite(state).all() or not 0.8 < float(mask.mean()) < 1.0:
                    raise ValueError(f"{name}: invalid state or mask at frame {index}")
                max_invalid = max(max_invalid, float(np.max(np.abs(state[:, ~mask]))))
            if max_invalid != 0.0:
                raise ValueError(f"{name}: nonzero invalid-grid state {max_invalid}")
        rows.append({"case": name, "path": str(path), "status": "CURATED_CASE_OK"})
        print(json.dumps(rows[-1]), flush=True)
    print(f"EXPANDED_CURATED_CASES_OK count={len(rows)}")


if __name__ == "__main__":
    main()
