#!/usr/bin/env python3
"""Read-only dynamic6 time/force/action reconstruction from raw CFD records."""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = args.repo / "data/curated/tandem_cylinders_full40_dynamic_validation_v1"
    manifest = json.loads((data / "manifest.json").read_text())
    cases_root = args.repo / "cfd/tandem_cylinders/cases"
    result = {"status": "TIME_ALIGNMENT_DIAGNOSTIC", "frozen_test_accessed": False,
              "manifest_sha256": digest(data / "manifest.json"), "cases": {}}
    for phase in ("b01", "b05"):
        for action in ("minus", "zero", "plus"):
            name = f"full40_dynamic_validation_{phase}_{action}"
            case = cases_root / name
            config_path = case / "case_config.json"
            config = json.loads(config_path.read_text())
            path = data / "validation" / f"{name}.h5"
            if digest(path) != manifest["hdf_sha256"][path.name]:
                raise ValueError(f"HDF SHA differs: {name}")
            with h5py.File(path, "r") as handle:
                times = handle["time"][:].reshape(-1)
                forces = handle["force"][:]
                omega = handle["omega"][:].reshape(-1)
            start = float(config["source_restart_time"])
            expected_time = start + np.arange(201, dtype=np.float64) * .1
            if not np.array_equal(times, expected_time.astype(np.float32)):
                raise ValueError(f"Time grid differs: {name}")
            reconstructed = []
            reconstructed_stored = []
            raw_hashes = {}
            for obj in ("forceFront", "forceRear"):
                force_path = case / "postProcessing" / obj / f"{start:g}" / "coefficient.dat"
                baseline_path = cases_root / config["source_restart_case"] / "postProcessing" / obj / "0/coefficient.dat"
                raw = np.loadtxt(force_path, comments="#")[:, [0, 1, 4]]
                baseline = np.loadtxt(baseline_path, comments="#")[:, [0, 1, 4]]
                restart = baseline[np.isclose(baseline[:, 0], start, atol=1e-8, rtol=0)]
                if len(restart) != 1 or not np.all(np.diff(raw[:, 0]) > 0):
                    raise ValueError(f"Nonunique restart or unordered raw forces: {name}")
                if raw[0, 0] > start:
                    raw = np.concatenate([restart, raw])
                if raw[0, 0] > expected_time[0] or raw[-1, 0] < expected_time[-1] - 1e-8:
                    raise ValueError(f"Force coverage differs: {name}")
                reconstructed.extend(np.interp(expected_time, raw[:, 0], raw[:, col]) for col in (1, 2))
                reconstructed_stored.extend(np.interp(times.astype(np.float64), raw[:, 0], raw[:, col]) for col in (1, 2))
                raw_hashes[str(force_path.relative_to(args.repo))] = digest(force_path)
                raw_hashes[str(baseline_path.relative_to(args.repo))] = digest(baseline_path)
            table = np.asarray(config["action_points"])
            expected_omega = np.interp(expected_time, table[:, 0], table[:, 1]).astype(np.float32)
            expected_force = np.stack(reconstructed, axis=1).astype(np.float32)
            stored_force = np.stack(reconstructed_stored, axis=1).astype(np.float32)
            stored_omega = np.interp(times.astype(np.float64), table[:, 0], table[:, 1]).astype(np.float32)
            if not np.isfinite(forces).all() or not np.isfinite(omega).all():
                raise ValueError(f"Non-finite values: {name}")
            result["cases"][name] = {
                "frames": len(times), "hdf_sha256": manifest["hdf_sha256"][path.name],
                "config_sha256": digest(config_path), "raw_source_sha256": raw_hashes,
                "force_max_abs_difference_by_channel": np.max(np.abs(forces - expected_force), axis=0).tolist(),
                "omega_max_abs_difference": float(np.max(np.abs(omega - expected_omega))),
                "force_stored_time_max_abs_difference_by_channel": np.max(np.abs(forces - stored_force), axis=0).tolist(),
                "omega_stored_time_max_abs_difference": float(np.max(np.abs(omega - stored_omega))),
                "stored_float32_time_max_roundoff": float(np.max(np.abs(times - expected_time))),
            }
    result["implementation_sha256"] = digest(Path(__file__))
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: {f: v for f, v in row.items() if "difference" in f}
                      for k, row in result["cases"].items()}, indent=2))


if __name__ == "__main__":
    main()
