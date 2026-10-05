"""Strict read-only loading of causal OpenFOAM force histories."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _coefficient_rows(path: Path) -> list[tuple[float, float, float]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 5:
            continue
        timestamp, cd, cl = float(fields[0]), float(fields[1]), float(fields[4])
        if not all(math.isfinite(value) for value in (timestamp, cd, cl)):
            raise ValueError(f"non-finite coefficient at t={timestamp}")
        rows.append((timestamp, cd, cl))
    if any(right[0] <= left[0] for left, right in zip(rows, rows[1:])):
        raise ValueError(f"non-increasing or duplicate time in {path}")
    return rows


def _force_rows(case: Path, object_name: str) -> dict[float, tuple[float, float]]:
    paths = sorted(case.glob(f"postProcessing/{object_name}/*/coefficient.dat"))
    if not paths:
        raise FileNotFoundError(f"missing {object_name} force history in {case}")
    rows: dict[float, tuple[float, float]] = {}
    for path in paths:
        for timestamp, cd, cl in _coefficient_rows(path):
            key = round(timestamp, 8)
            value = (cd, cl)
            if key in rows and not np.allclose(rows[key], value, rtol=0.0, atol=1e-8):
                raise ValueError(f"conflicting force restart sample at t={timestamp}")
            rows[key] = value
    return rows


def actual_causal_prehistory(
    reference: Path,
    restart_time: float,
    *,
    provenance_root: Path,
    control_dt: float = 0.1,
    sample_count: int = 62,
) -> tuple[list[float], list[list[float]], dict[str, list[dict[str, str]]]]:
    """Load exact causal endpoints ending at ``restart_time`` without interpolation."""
    reference = Path(reference).resolve()
    provenance_root = Path(provenance_root).resolve()
    if (
        not reference.is_dir()
        or not reference.is_relative_to(provenance_root)
        or not math.isfinite(restart_time)
        or not math.isfinite(control_dt)
        or control_dt <= 0.0
        or sample_count < 2
    ):
        raise ValueError("invalid causal force-history request")
    front = _force_rows(reference, "forceFront")
    rear = _force_rows(reference, "forceRear")
    first = restart_time - control_dt * (sample_count - 1)
    times = [round(first + control_dt * index, 8) for index in range(sample_count)]
    forces = []
    for timestamp in times:
        if timestamp not in front or timestamp not in rear:
            raise FileNotFoundError(f"missing causal force sample at t={timestamp}")
        forces.append([*front[timestamp], *rear[timestamp]])
    if times[-1] > restart_time + 1e-8 or not math.isclose(
        times[-1], restart_time, rel_tol=0.0, abs_tol=1e-8
    ):
        raise ValueError("causal force history does not end exactly at restart")
    sources = {}
    for object_name in ("forceFront", "forceRear"):
        paths = sorted(reference.glob(f"postProcessing/{object_name}/*/coefficient.dat"))
        sources[object_name] = [
            {
                "path": str(path.resolve().relative_to(provenance_root)),
                "sha256": sha256_file(path),
            }
            for path in paths
        ]
    return times, forces, sources
