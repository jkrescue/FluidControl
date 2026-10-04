#!/usr/bin/env python3
"""Audit action-history coverage without reading flow fields or frozen data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np

CHANGE_EPS = 1.0e-8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def summarize_case(path: Path) -> dict:
    with h5py.File(path, "r") as handle:
        omega = np.asarray(handle["omega"][:], dtype=np.float64).reshape(-1)
        time = np.asarray(handle["time"][:], dtype=np.float64).reshape(-1)
    if len(omega) < 2 or len(omega) != len(time):
        raise ValueError(f"invalid omega/time lengths: {path}")
    if not np.isfinite(omega).all() or not np.isfinite(time).all():
        raise ValueError(f"nonfinite omega/time: {path}")
    dt = np.diff(time)
    if not np.all(dt > 0):
        raise ValueError(f"time is not strictly increasing: {path}")
    delta = np.diff(omega)
    changed = np.abs(delta) > CHANGE_EPS
    edges = np.r_[0, np.flatnonzero(changed) + 1, len(omega)]
    sample_dt = float(np.median(dt))
    dwell = [float((stop - start) * sample_dt) for start, stop in zip(edges[:-1], edges[1:], strict=True)]
    nonzero_sign = np.sign(omega[np.abs(omega) > CHANGE_EPS])
    sign_crossings = int(np.sum(nonzero_sign[1:] != nonzero_sign[:-1]))
    return {
        "case": path.stem,
        "samples": len(omega),
        "duration_du": float(time[-1] - time[0]),
        "dt_du": sample_dt,
        "omega_min": float(omega.min()),
        "omega_max": float(omega.max()),
        "omega_rms": float(np.sqrt(np.mean(np.square(omega)))),
        "max_abs_delta_omega": float(np.max(np.abs(delta))),
        "change_fraction": float(np.mean(changed)),
        "total_variation": float(np.sum(np.abs(delta))),
        "longest_constant_dwell_du": max(dwell),
        "sign_crossings_zeros_removed": sign_crossings,
        "unique_omega_rounded_4dp": int(len(np.unique(np.round(omega, 4)))),
        "omega_sha256": array_sha256(omega),
        "time_sha256": array_sha256(time),
    }


def triplet(rows: list[dict], key: str) -> dict:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    return {"min": float(values.min()), "median": float(np.median(values)), "max": float(values.max())}


def summarize_cohort(name: str, root: Path, expected_manifest_sha: str) -> dict:
    resolved = root.resolve()
    if any("frozen" in part.lower() for part in resolved.parts):
        raise ValueError(f"frozen data are forbidden: {root}")
    manifest = root.parent / "manifest.json"
    if not manifest.is_file() or sha256(manifest) != expected_manifest_sha:
        raise ValueError(f"manifest identity differs: {manifest}")
    paths = sorted(root.glob("*.h5"))
    if not paths:
        raise ValueError(f"no HDF5 cases: {root}")
    rows = [summarize_case(path) for path in paths]
    metrics = [
        "omega_rms", "max_abs_delta_omega", "change_fraction", "total_variation",
        "longest_constant_dwell_du", "sign_crossings_zeros_removed",
        "unique_omega_rounded_4dp",
    ]
    return {
        "name": name,
        "root": str(root),
        "manifest": str(manifest),
        "manifest_sha256": expected_manifest_sha,
        "case_count": len(rows),
        "omega_global_min": min(row["omega_min"] for row in rows),
        "omega_global_max": max(row["omega_max"] for row in rows),
        "aggregate_min_median_max": {key: triplet(rows, key) for key in metrics},
        "cases": rows,
    }


def build(args: argparse.Namespace) -> dict:
    cohorts = [
        summarize_cohort("train20_static", args.train20, args.dev30_manifest_sha),
        summarize_cohort("train8_dynamic", args.train8, args.train8_manifest_sha),
        summarize_cohort("train16_ppo", args.train16, args.train16_manifest_sha),
        summarize_cohort("dynamic6_validation", args.dynamic6, args.dynamic6_manifest_sha),
    ]
    return {
        "status": "FC_P002_ACTION_HISTORY_COVERAGE_AUDIT_COMPLETE",
        "definitions": {
            "change": f"abs(omega[t+1]-omega[t]) > {CHANGE_EPS}",
            "change_fraction": "changed intervals / all adjacent intervals",
            "constant_dwell": "consecutive samples separated by no change, duration=sample_count*median_dt",
            "sign_crossing": "remove abs(omega)<=change epsilon, then count adjacent nonzero sign changes",
            "total_variation": "sum(abs(delta omega)) over the recorded trajectory",
        },
        "scope": {
            "read_datasets": ["omega", "time"],
            "frozen_test_accessed": False,
            "claim_limit": (
                "This audit can exclude obvious scalar amplitude/rate-bound mismatch only. "
                "It does not establish joint-distribution, phase, state, or action-history equivalence."
            ),
        },
        "cohorts": cohorts,
    }


def write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train20", "train8", "train16", "dynamic6"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--dev30-manifest-sha", required=True)
    parser.add_argument("--train8-manifest-sha", required=True)
    parser.add_argument("--train16-manifest-sha", required=True)
    parser.add_argument("--dynamic6-manifest-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args)
    write_atomic(args.output, result)
    print(json.dumps({"status": result["status"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
