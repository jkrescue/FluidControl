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


def verify_raw_labels(name: str, case: Path, time: np.ndarray,
                      omega: np.ndarray, force: np.ndarray,
                      allow_legacy_initial_force: bool = False) -> dict[str, float]:
    """Cross-check saved labels against the independent OpenFOAM case files."""
    config = json.loads((case / "case_config.json").read_text(encoding="utf-8"))
    table = np.asarray(config["action_points"], dtype=np.float64)
    expected_omega = np.interp(time, table[:, 0], table[:, 1])
    omega_error = float(np.max(np.abs(omega - expected_omega)))

    expected_force = []
    first_restart_force = []
    for object_name in ("forceFront", "forceRear"):
        path = case / "postProcessing" / object_name / "80" / "coefficient.dat"
        raw = np.loadtxt(path, usecols=(0, 1, 4))
        if raw.ndim != 2 or raw.shape[1] != 3 or np.any(np.diff(raw[:, 0]) <= 0):
            raise ValueError(f"{name}: invalid raw force time series in {path}")
        first_restart_force.extend(raw[0, column] for column in (1, 2))
        if time[0] < raw[0, 0] - 1.0e-8:
            source_name = config.get("source_restart_case")
            if not source_name:
                raise ValueError(f"{name}: missing source force at restart")
            source_path = case.parent / source_name / "postProcessing" / object_name / "0" / "coefficient.dat"
            source = np.loadtxt(source_path, usecols=(0, 1, 4))
            matches = np.flatnonzero(np.isclose(source[:, 0], time[0], rtol=0, atol=1.0e-8))
            if len(matches) != 1:
                raise ValueError(f"{name}: no unique source force at restart in {source_path}")
            raw = np.concatenate((source[matches], raw), axis=0)
        expected_force.extend(
            np.interp(time, raw[:, 0], raw[:, column]) for column in (1, 2)
        )
    difference = np.abs(force - np.stack(expected_force, axis=1))
    force_error = float(np.max(difference))
    legacy_initial_error = 0.0
    if allow_legacy_initial_force and force_error > 1.0e-5:
        legacy_initial_error = float(np.max(difference[0]))
        first_sample_error = float(np.max(np.abs(force[0] - first_restart_force)))
        if first_sample_error <= 1.0e-5 and float(np.max(difference[1:])) <= 1.0e-5:
            force_error = float(np.max(difference[1:]))
        else:
            raise ValueError(
                f"{name}: legacy initial force is not the first restart sample; "
                f"first_sample={first_sample_error}, later={float(np.max(difference[1:]))}"
            )
    if omega_error > 5.0e-6 or force_error > 1.0e-5:
        raise ValueError(
            f"{name}: raw label mismatch, omega={omega_error}, force={force_error}"
        )
    return {"max_abs_omega_error": omega_error, "max_abs_force_error": force_error,
            "legacy_initial_force_error": legacy_initial_error}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cases-root", type=Path, default=None,
                        help="also compare labels with source OpenFOAM cases")
    parser.add_argument("--legacy-initial-force-cases", default="",
                        help="comma-separated inherited cases whose t=80 force used the first restart sample")
    parser.add_argument("cases", nargs="+")
    args = parser.parse_args()
    legacy_cases = set(filter(None, args.legacy_initial_force_cases.split(",")))
    if legacy_cases - set(args.cases):
        parser.error("legacy-initial-force cases must be among selected cases")
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
        label_errors = (
            verify_raw_labels(name, args.cases_root / name, time, omega, force,
                              name in legacy_cases)
            if args.cases_root is not None else {}
        )
        rows.append({"case": name, "path": str(path), **label_errors,
                     "status": "CURATED_CASE_OK"})
        print(json.dumps(rows[-1]), flush=True)
    print(f"EXPANDED_CURATED_CASES_OK count={len(rows)}")


if __name__ == "__main__":
    main()
