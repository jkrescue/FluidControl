"""Bounded helpers for the P064 coarse-ROI force diagnostic.

No model, optimizer, CFD, split selection, or bulk HDF read belongs here.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CASES = (
    "matched_start_acquisition_train_b00_m075",
    "matched_start_acquisition_train_b00_zero",
    "matched_start_acquisition_train_b00_p075",
)
FRAME_INDICES = tuple(range(0, 801, 100))
TIME_ATOL = 2.0e-5  # float32 time near t=228 has ULP below this bound.
COMPONENT_ATOL = 1.0e-10
HDF_TOTAL_ATOL = 5.0e-7
NTHETA = 128


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _named_block(text: str, name: str, start: int = 0) -> str:
    matches = list(re.finditer(r"(?m)^\s*" + re.escape(name) + r"\s*$", text[start:]))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one OpenFOAM block: {name}")
    match = matches[0]
    begin = start + match.end()
    opening = text.find("{", begin)
    if opening < 0:
        raise ValueError(f"missing opening brace: {name}")
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[opening + 1:index]
    raise ValueError(f"unclosed OpenFOAM block: {name}")


def parse_force_components(text: str) -> dict[str, dict[str, float]]:
    """Parse only the required integrated front/rear scalar values."""
    required = (
        "Cd", "CdPressure", "CdViscous",
        "Cl", "ClPressure", "ClViscous",
    )
    result = {}
    for body in ("forceFront", "forceRear"):
        outer = _named_block(text, body)
        scalar = _named_block(outer, "scalar")
        values: dict[str, float] = {}
        for line in scalar.splitlines():
            fields = line.strip().rstrip(";").split()
            if len(fields) == 2 and fields[0] in required:
                if fields[0] in values:
                    raise ValueError(f"duplicate {body} {fields[0]}")
                values[fields[0]] = float(fields[1])
        if set(values) != set(required) or not all(math.isfinite(x) for x in values.values()):
            raise ValueError(f"incomplete/nonfinite {body} force components")
        result[body] = values
    return result


def component_vectors(parsed: dict[str, dict[str, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total, pressure, viscous = [], [], []
    for body in ("forceFront", "forceRear"):
        for direction in ("Cd", "Cl"):
            values = parsed[body]
            total.append(values[direction])
            pressure.append(values[direction + "Pressure"])
            viscous.append(values[direction + "Viscous"])
    arrays = tuple(np.asarray(x, dtype=np.float64) for x in (total, pressure, viscous))
    if not all(x.shape == (4,) and np.isfinite(x).all() for x in arrays):
        raise ValueError("force vector contract")
    return arrays


def validate_alignment(
    hdf_time: float,
    expected_time: float,
    hdf_total: np.ndarray,
    parsed: dict[str, dict[str, float]],
) -> dict[str, float]:
    if not math.isclose(float(hdf_time), float(expected_time), rel_tol=0.0, abs_tol=TIME_ATOL):
        raise ValueError("HDF/directory time mismatch")
    total, pressure, viscous = component_vectors(parsed)
    hdf = np.asarray(hdf_total, dtype=np.float64)
    if hdf.shape != (4,) or not np.isfinite(hdf).all():
        raise ValueError("HDF total-force contract")
    component_error = float(np.max(np.abs(pressure + viscous - total)))
    hdf_error = float(np.max(np.abs(hdf - total)))
    if component_error > COMPONENT_ATOL:
        raise ValueError("pressure+viscous does not equal OpenFOAM total")
    if hdf_error > HDF_TOTAL_ATOL:
        raise ValueError("HDF total does not equal OpenFOAM total")
    return {"component_max_abs": component_error, "hdf_total_max_abs": hdf_error}


@dataclass(frozen=True)
class ForceGeometry:
    center_x: float
    center_y: float
    radius: float = 0.5
    rho: float = 1.0
    u_inf: float = 1.0
    span: float = 0.1
    area_ref: float = 0.1
    kinematic_pressure: bool = True

    def __post_init__(self) -> None:
        values = (self.center_x, self.center_y, self.radius, self.rho, self.u_inf, self.span, self.area_ref)
        if not all(math.isfinite(x) for x in values) or min(values[2:]) <= 0:
            raise ValueError("invalid force geometry")


def _bilinear_fluid(field, mask, x, y, xq, yq) -> np.ndarray:
    field = np.asarray(field, dtype=np.float64)
    mask = np.asarray(mask)
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if field.shape != (len(y), len(x)) or mask.shape != field.shape:
        raise ValueError("grid shape mismatch")
    if not (np.isfinite(field).all() and np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("nonfinite grid")
    if not (np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0)):
        raise ValueError("grid is not increasing")
    ix = np.searchsorted(x, xq, side="right") - 1
    iy = np.searchsorted(y, yq, side="right") - 1
    if np.any(ix < 0) or np.any(ix >= len(x)-1) or np.any(iy < 0) or np.any(iy >= len(y)-1):
        raise ValueError("probe outside grid")
    corners = np.stack((mask[iy, ix], mask[iy, ix+1], mask[iy+1, ix], mask[iy+1, ix+1]))
    if not np.all(corners == 1):
        raise ValueError("pressure probe stencil intersects a solid cell")
    tx = (xq - x[ix]) / (x[ix+1] - x[ix])
    ty = (yq - y[iy]) / (y[iy+1] - y[iy])
    return ((1-tx)*(1-ty)*field[iy, ix] + tx*(1-ty)*field[iy, ix+1]
            + (1-tx)*ty*field[iy+1, ix] + tx*ty*field[iy+1, ix+1])


def coarse_pressure_coefficients(field, mask, x, y, geometry: ForceGeometry) -> np.ndarray:
    """Fixed offset probe, but true-wall R*dtheta quadrature area."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    h = max(float(np.max(np.diff(x))), float(np.max(np.diff(y))))
    theta = np.arange(NTHETA, dtype=np.float64) * (2.0 * math.pi / NTHETA)
    probe_radius = geometry.radius + 2.0*h
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    sampled = _bilinear_fluid(
        field, mask, x, y,
        geometry.center_x + probe_radius*cos_t,
        geometry.center_y + probe_radius*sin_t,
    )
    pressure = sampled * (geometry.rho if geometry.kinematic_pressure else 1.0)
    dtheta = 2.0 * math.pi / NTHETA
    # Pressure force on the body is - integral p*n*dS.  Sampling stays at the
    # offset radius, while dS deliberately uses the physical wall radius.
    force = -geometry.span*geometry.radius*dtheta*np.asarray([
        np.sum(pressure*cos_t), np.sum(pressure*sin_t)
    ])
    dynamic_reference = 0.5*geometry.rho*geometry.u_inf**2*geometry.area_ref
    return force/dynamic_reference


def read_selected_frames(reader_class, path: Path, indices=FRAME_INDICES):
    """Call the official-reader interface for selected frames only."""
    reader = reader_class(path, fields=["state", "mask", "omega", "force", "time"])
    rows = []
    try:
        for index in indices:
            sample, metadata = reader[index]
            rows.append((index, sample, metadata))
    finally:
        close = getattr(reader, "close", None)
        if close is not None:
            close()
    return rows
