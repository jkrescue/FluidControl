"""Read the 67-channel tandem policy observation from original OpenFOAM outputs.

This is a read-only data bridge, not a CFD solver or a learned dynamics model.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np


_VECTOR = re.compile(r"\(([^()]*)\)")
_HEADER = re.compile(r"^# Probe (\d+) \(([^()]*)\)$")
_TIME_TOL = 1e-6


def _probe_file(path: Path, target_time: float) -> np.ndarray | None:
    headers: dict[int, np.ndarray] = {}
    matching: list[np.ndarray] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("# Probe "):
                found = _HEADER.match(line.strip())
                if found is None:
                    raise ValueError(f"invalid probe header in {path}")
                index = int(found.group(1))
                if index in headers:
                    raise ValueError(f"duplicate probe index {index}: {path}")
                headers[index] = np.fromstring(found.group(2), sep=" ")
                continue
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            time = float(parts[0])
            if abs(time - target_time) > _TIME_TOL:
                continue
            vectors = _VECTOR.findall(parts[1])
            if len(vectors) != 32:
                raise ValueError(f"expected 32 probe vectors at t={time}: {path}")
            parsed = np.asarray([np.fromstring(vector, sep=" ") for vector in vectors])
            if parsed.shape != (32, 3) or not np.isfinite(parsed).all():
                raise ValueError(f"invalid probe vectors at t={time}: {path}")
            matching.append(parsed[:, :2].copy())
    if set(headers) != set(range(32)):
        raise ValueError(f"missing or duplicated 32-probe header: {path}")
    for index in range(32):
        expected = np.array([17.0, 6.0 + 3.0 * index / 31, 0.05])
        if headers[index].shape != (3,) or not np.allclose(
            headers[index], expected, rtol=0, atol=1e-7
        ):
            raise ValueError(f"probe coordinates/order mismatch at index {index}: {path}")
    if len(matching) > 1:
        raise ValueError(f"duplicate probe time {target_time}: {path}")
    return matching[0] if matching else None


def _force_file(path: Path, target_time: float) -> np.ndarray | None:
    matching: list[np.ndarray] = []
    saw_header = False
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("# Time"):
                fields = line[1:].split()
                if len(fields) < 5 or fields[0] != "Time" or fields[1] != "Cd" or fields[4] != "Cl":
                    raise ValueError(f"unexpected force coefficient columns: {path}")
                saw_header = True
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            time = float(fields[0])
            if abs(time - target_time) > _TIME_TOL:
                continue
            if len(fields) < 5:
                raise ValueError(f"incomplete force coefficients at t={time}: {path}")
            value = np.asarray([float(fields[1]), float(fields[4])])
            if not np.isfinite(value).all():
                raise ValueError(f"non-finite rear force at t={time}: {path}")
            matching.append(value)
    if not saw_header:
        raise ValueError(f"missing force coefficient header: {path}")
    if len(matching) > 1:
        raise ValueError(f"duplicate force time {target_time}: {path}")
    return matching[0] if matching else None


def _select(case: Path, pattern: str, reader, target_time: float) -> tuple[np.ndarray, list[str]]:
    matches: list[tuple[Path, np.ndarray]] = []
    for path in sorted(case.glob(pattern)):
        value = reader(path, target_time)
        if value is not None:
            matches.append((path, value))
    if not matches:
        raise FileNotFoundError(f"no OpenFOAM {pattern} sample at t={target_time}: {case}")
    reference = matches[0][1]
    for path, value in matches[1:]:
        if not np.allclose(reference, value, rtol=0, atol=1e-7):
            raise ValueError(f"conflicting restart samples at t={target_time}: {path}")
    return reference, [str(path) for path, _ in matches]


def observation_at(case: Path, target_time: float, applied_omega: float) -> tuple[np.ndarray, dict]:
    """Return exactly [u0,v0,...,u31,v31,rear Cd,rear Cl,applied omega]."""
    case = Path(case)
    if not case.is_dir():
        raise FileNotFoundError(case)
    if not all(math.isfinite(value) for value in (target_time, applied_omega)):
        raise ValueError("time and applied omega must be finite")
    if abs(applied_omega) > 5.0 + 1e-6:
        raise ValueError("angular speed outside declared CFD action support")
    probes, probe_sources = _select(
        case, "postProcessing/wakeProbes/*/U", _probe_file, target_time
    )
    force, force_sources = _select(
        case, "postProcessing/forceRear/*/coefficient.dat", _force_file, target_time
    )
    observation = np.concatenate((probes.reshape(-1), force, [applied_omega]))
    if observation.shape != (67,) or not np.isfinite(observation).all():
        raise ValueError("invalid 67-channel OpenFOAM observation")
    return observation.astype(np.float32), {
        "time": float(target_time),
        "probe_sources": probe_sources,
        "force_sources": force_sources,
        "channel_order": "32*(u,v),rear_cd,rear_cl,applied_omega",
    }


def total_drag_observation_at(
    case: Path, target_time: float, applied_omega: float
) -> tuple[np.ndarray, dict]:
    """Read the 69-channel Stage-C observation at one exact OpenFOAM time.

    The 67-channel legacy reader remains unchanged. Both cylinder forces are
    read from the same case/time, avoiding a restart-boundary label offset.
    """
    legacy, provenance = observation_at(case, target_time, applied_omega)
    front, front_sources = _select(
        Path(case),
        "postProcessing/forceFront/*/coefficient.dat",
        _force_file,
        target_time,
    )
    result = np.concatenate((legacy[:64], front, legacy[64:]))
    if result.shape != (69,) or not np.isfinite(result).all():
        raise ValueError("invalid 69-channel total-drag observation")
    return result.astype(np.float32), {
        **provenance,
        "front_force_sources": front_sources,
        "channel_order": "32*(u,v),front_cd,front_cl,rear_cd,rear_cl,applied_omega",
    }
