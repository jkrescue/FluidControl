#!/usr/bin/env python3
"""Observational spatial-spectrum and gradient diagnostics for FNO snapshots.

This module never filters or modifies a rollout.  Its finite rectangular ROI
spectrum is a diagnostic of windowed snapshots, not a homogeneous-turbulence
energy spectrum E(k), and it defines no scientific pass/fail threshold.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


DEFAULT_ROI = (17.0, 24.0, 5.0, 10.0)
DEFAULT_BAND_EDGES = (0.0, 0.5, 1.0, 2.0, 4.0, math.inf)


def _uniform_spacing(values: np.ndarray, name: str) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 3 or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite one-dimensional coordinate")
    differences = np.diff(values)
    if not np.all(differences > 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    spacing = float(differences.mean())
    # Curated x/y originated from a float32 VTK grid.  Accept only its measured
    # representation jitter, while still rejecting physically nonuniform grids.
    tolerance = max(1e-12, abs(spacing) * 2e-5)
    if not np.allclose(differences, spacing, rtol=0.0, atol=tolerance):
        raise ValueError(f"{name} must be uniformly spaced")
    return spacing


def select_fluid_roi(
    x,
    y,
    mask,
    bounds: tuple[float, float, float, float] = DEFAULT_ROI,
) -> dict:
    """Validate the grid and return indices for a solid-free rectangular ROI."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.asarray(mask)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.shape != (y.size, x.size):
        raise ValueError("mask must have shape (len(y), len(x))")
    dx = _uniform_spacing(x, "x")
    dy = _uniform_spacing(y, "y")
    x0, x1, y0, y1 = map(float, bounds)
    if not (x0 < x1 and y0 < y1 and all(map(math.isfinite, bounds))):
        raise ValueError("finite ordered ROI bounds are required")
    x_indices = np.flatnonzero((x >= x0) & (x <= x1))
    y_indices = np.flatnonzero((y >= y0) & (y <= y1))
    if x_indices.size < 4 or y_indices.size < 4:
        raise ValueError("ROI must contain at least 4x4 grid points")
    if not np.array_equal(x_indices, np.arange(x_indices[0], x_indices[-1] + 1)):
        raise ValueError("ROI x indices must be contiguous")
    if not np.array_equal(y_indices, np.arange(y_indices[0], y_indices[-1] + 1)):
        raise ValueError("ROI y indices must be contiguous")
    roi_mask = mask[np.ix_(y_indices, x_indices)]
    if not np.all(roi_mask == 1):
        raise ValueError("ROI must contain fluid cells only (mask == 1)")
    return {
        "x_indices": x_indices,
        "y_indices": y_indices,
        "x": x[x_indices],
        "y": y[y_indices],
        "dx": dx,
        "dy": dy,
        "shape_yx": (int(y_indices.size), int(x_indices.size)),
        "bounds_requested": [x0, x1, y0, y1],
        "bounds_sampled": [
            float(x[x_indices[0]]),
            float(x[x_indices[-1]]),
            float(y[y_indices[0]]),
            float(y[y_indices[-1]]),
        ],
    }


def _relative_energy(error_energy: float, reference_energy: float):
    if reference_energy == 0.0:
        return None
    return float(error_energy / reference_energy)


def _band_label(lower: float, upper: float) -> str:
    return f"[{lower:g},{upper:g})" if math.isfinite(upper) else f"[{lower:g},inf)"


def _spectrum(field: np.ndarray, window: np.ndarray) -> tuple[np.ndarray, dict]:
    centered = np.asarray(field, dtype=np.float64) - float(np.mean(field))
    windowed = centered * window
    spectrum = np.fft.fft2(windowed, norm="ortho")
    spectral_energy = np.square(np.abs(spectrum))
    spatial_energy = float(np.square(windowed).sum(dtype=np.float64))
    frequency_energy = float(spectral_energy.sum(dtype=np.float64))
    tolerance = max(1e-12, spatial_energy * 1e-10)
    if not math.isclose(
        spatial_energy, frequency_energy, rel_tol=1e-10, abs_tol=tolerance
    ):
        raise RuntimeError("FFT Parseval consistency check failed")
    return spectral_energy, {
        "windowed_spatial_energy": spatial_energy,
        "fft_energy": frequency_energy,
        "absolute_difference": abs(spatial_energy - frequency_energy),
    }


def _band_energies(
    energy: np.ndarray,
    radial_frequency: np.ndarray,
    band_edges: tuple[float, ...],
) -> list[dict]:
    total = float(energy.sum(dtype=np.float64))
    rows = []
    for lower, upper in zip(band_edges[:-1], band_edges[1:]):
        selected = radial_frequency >= lower
        if math.isfinite(upper):
            selected &= radial_frequency < upper
        value = float(energy[selected].sum(dtype=np.float64))
        rows.append(
            {
                "band_cycles_per_D": _band_label(lower, upper),
                "lower_inclusive": float(lower),
                "upper_exclusive": float(upper) if math.isfinite(upper) else None,
                "energy": value,
                "fraction_of_reported_energy": value / total if total else None,
            }
        )
    return rows


def _field_spectral_report(
    fields: np.ndarray,
    window: np.ndarray,
    radial_frequency: np.ndarray,
    band_edges: tuple[float, ...],
) -> dict:
    per_channel = [_spectrum(channel, window) for channel in fields]
    velocity_energy = per_channel[0][0] + per_channel[1][0]
    pressure_energy = per_channel[2][0]
    return {
        "velocity_uv": {
            "total_energy": float(velocity_energy.sum(dtype=np.float64)),
            "bands": _band_energies(velocity_energy, radial_frequency, band_edges),
            "parseval": {
                "windowed_spatial_energy": sum(
                    per_channel[i][1]["windowed_spatial_energy"] for i in (0, 1)
                ),
                "fft_energy": float(velocity_energy.sum(dtype=np.float64)),
            },
        },
        "pressure": {
            "total_energy": float(pressure_energy.sum(dtype=np.float64)),
            "bands": _band_energies(pressure_energy, radial_frequency, band_edges),
            "parseval": per_channel[2][1],
        },
    }


def _gradient_energy(fields: np.ndarray, dx: float, dy: float) -> dict:
    energies = []
    for channel in fields:
        grad_y, grad_x = np.gradient(channel, dy, dx, edge_order=2)
        interior = np.s_[1:-1, 1:-1]
        energies.append(
            float(
                (
                    np.square(grad_x[interior]) + np.square(grad_y[interior])
                ).sum(dtype=np.float64)
            )
        )
    return {"velocity_uv": energies[0] + energies[1], "pressure": energies[2]}


def spatial_snapshot_diagnostics(
    truth,
    prediction,
    x,
    y,
    mask,
    *,
    bounds: tuple[float, float, float, float] = DEFAULT_ROI,
    band_edges: tuple[float, ...] = DEFAULT_BAND_EDGES,
) -> dict:
    """Diagnose one aligned (u,v,p) truth/prediction snapshot without mutation."""
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if truth.shape != prediction.shape or truth.ndim != 3 or truth.shape[0] != 3:
        raise ValueError("truth and prediction must be aligned (3, ny, nx) arrays")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("truth and prediction must be finite")
    band_edges = tuple(float(value) for value in band_edges)
    if (
        len(band_edges) < 2
        or band_edges[0] != 0.0
        or not math.isinf(band_edges[-1])
        or any(b <= a for a, b in zip(band_edges[:-1], band_edges[1:]))
    ):
        raise ValueError("bands must increase from 0 to infinity")
    roi = select_fluid_roi(x, y, mask, bounds)
    yi, xi = roi["y_indices"], roi["x_indices"]
    truth_roi = truth[:, yi[:, None], xi]
    prediction_roi = prediction[:, yi[:, None], xi]
    error_roi = prediction_roi - truth_roi
    ny, nx = roi["shape_yx"]
    window = np.outer(np.hanning(ny), np.hanning(nx))
    fx = np.fft.fftfreq(nx, d=roi["dx"])
    fy = np.fft.fftfreq(ny, d=roi["dy"])
    radial_frequency = np.hypot(fx[None, :], fy[:, None])
    spectra = {
        "truth": _field_spectral_report(
            truth_roi, window, radial_frequency, band_edges
        ),
        "prediction": _field_spectral_report(
            prediction_roi, window, radial_frequency, band_edges
        ),
        "error": _field_spectral_report(
            error_roi, window, radial_frequency, band_edges
        ),
    }
    truth_gradient = _gradient_energy(truth_roi, roi["dx"], roi["dy"])
    error_gradient = _gradient_energy(error_roi, roi["dx"], roi["dy"])
    gradient = {}
    for name in ("velocity_uv", "pressure"):
        gradient[name] = {
            "truth_gradient_energy": truth_gradient[name],
            "error_gradient_energy": error_gradient[name],
            "error_over_truth_gradient_energy": _relative_energy(
                error_gradient[name], truth_gradient[name]
            ),
        }
    return {
        "status": "OBSERVATIONAL_FINITE_ROI_SPATIAL_DIAGNOSTIC",
        "scope": (
            "finite_roi_hann_windowed_snapshot_spectrum_not_homogeneous_"
            "turbulence_Ek_no_scientific_pass_threshold"
        ),
        "channels": ["u_over_Uinf", "v_over_Uinf", "p_over_rho_Uinf_squared"],
        "roi": {
            key: value
            for key, value in roi.items()
            if key not in ("x_indices", "y_indices", "x", "y")
        },
        "band_edges_cycles_per_D": [
            value if math.isfinite(value) else None for value in band_edges
        ],
        "spectral_energy": spectra,
        "central_gradient_energy": gradient,
    }


def inspect_hdf_roi(path: Path, frame: int = 0) -> dict:
    """Read only coordinate/mask metadata to validate a real curated HDF ROI."""
    import h5py

    with h5py.File(path, "r") as handle:
        x = handle["x"][:]
        y = handle["y"][:]
        masks = handle["mask"]
        if frame < 0 or frame >= masks.shape[0]:
            raise IndexError("frame is outside the HDF trajectory")
        roi = select_fluid_roi(x, y, masks[frame])
        return {
            "hdf": str(path),
            "frame": frame,
            "mask_shape": list(masks[frame].shape),
            "roi_shape_yx": list(roi["shape_yx"]),
            "dx": roi["dx"],
            "dy": roi["dy"],
            "bounds_requested": roi["bounds_requested"],
            "bounds_sampled": roi["bounds_sampled"],
            "mask_all_fluid": True,
            "coordinates_uniform": True,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect-hdf", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(inspect_hdf_roi(args.inspect_hdf, args.frame), indent=2))


if __name__ == "__main__":
    main()
