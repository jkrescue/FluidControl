from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/fno_spatial_diagnostics.py"
if not SCRIPT.exists():
    SCRIPT = Path("/tmp/fno_spatial_diagnostics.py")
SPEC = importlib.util.spec_from_file_location("fno_spatial_diagnostics", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def grid(nx=128, ny=96):
    x = np.arange(nx, dtype=np.float64) * 0.0625 + 16.0
    y = np.arange(ny, dtype=np.float64) * 0.0625 + 4.5
    mask = np.ones((ny, nx), dtype=np.uint8)
    return x, y, mask


class SpatialDiagnosticsTests(unittest.TestCase):
    def test_single_frequency_has_dominant_expected_band(self) -> None:
        x, y, mask = grid()
        xx, yy = np.meshgrid(x, y)
        frequency = 1.25
        wave = np.sin(2.0 * np.pi * frequency * xx)
        truth = np.stack((wave, np.zeros_like(wave), np.cos(2 * np.pi * frequency * yy)))
        result = MODULE.spatial_snapshot_diagnostics(truth, truth, x, y, mask)
        velocity_bands = result["spectral_energy"]["truth"]["velocity_uv"]["bands"]
        pressure_bands = result["spectral_energy"]["truth"]["pressure"]["bands"]
        self.assertEqual(
            max(velocity_bands, key=lambda row: row["energy"])["band_cycles_per_D"],
            "[1,2)",
        )
        self.assertEqual(
            max(pressure_bands, key=lambda row: row["energy"])["band_cycles_per_D"],
            "[1,2)",
        )

    def test_exact_prediction_has_zero_error_and_parseval(self) -> None:
        x, y, mask = grid()
        rng = np.random.default_rng(20261004)
        truth = rng.normal(size=(3, y.size, x.size))
        result = MODULE.spatial_snapshot_diagnostics(truth, truth.copy(), x, y, mask)
        for name in ("velocity_uv", "pressure"):
            self.assertEqual(result["spectral_energy"]["error"][name]["total_energy"], 0.0)
            parseval = result["spectral_energy"]["truth"][name]["parseval"]
            self.assertAlmostEqual(
                parseval["windowed_spatial_energy"], parseval["fft_energy"], places=10
            )
            self.assertEqual(
                result["central_gradient_energy"][name]["error_gradient_energy"],
                0.0,
            )

    def test_nonuniform_coordinates_and_solid_roi_are_rejected(self) -> None:
        x, y, mask = grid()
        field = np.zeros((3, y.size, x.size))
        bad_x = x.copy()
        bad_x[30] += 0.01
        with self.assertRaisesRegex(ValueError, "uniformly spaced"):
            MODULE.spatial_snapshot_diagnostics(field, field, bad_x, y, mask)
        solid = mask.copy()
        solid[np.argmin(abs(y - 7.0)), np.argmin(abs(x - 20.0))] = 0
        with self.assertRaisesRegex(ValueError, "fluid cells only"):
            MODULE.spatial_snapshot_diagnostics(field, field, x, y, solid)

    def test_float32_coordinate_representation_jitter_is_accepted(self) -> None:
        x, y, mask = grid()
        x = x.astype(np.float32).astype(np.float64)
        y = y.astype(np.float32).astype(np.float64)
        field = np.zeros((3, y.size, x.size))
        result = MODULE.spatial_snapshot_diagnostics(field, field, x, y, mask)
        self.assertEqual(result["roi"]["shape_yx"], (81, 112))

    def test_zero_reference_gradient_ratio_is_undefined(self) -> None:
        x, y, mask = grid()
        truth = np.zeros((3, y.size, x.size))
        prediction = np.ones_like(truth)
        result = MODULE.spatial_snapshot_diagnostics(truth, prediction, x, y, mask)
        for name in ("velocity_uv", "pressure"):
            self.assertIsNone(
                result["central_gradient_energy"][name][
                    "error_over_truth_gradient_energy"
                ]
            )

    def test_channel_and_band_contract_is_explicit(self) -> None:
        x, y, mask = grid()
        field = np.zeros((3, y.size, x.size))
        result = MODULE.spatial_snapshot_diagnostics(field, field, x, y, mask)
        self.assertIn("not_homogeneous_turbulence_Ek", result["scope"])
        self.assertEqual(result["band_edges_cycles_per_D"], [0.0, 0.5, 1.0, 2.0, 4.0, None])
        self.assertEqual(result["roi"]["bounds_requested"], [17.0, 24.0, 5.0, 10.0])


if __name__ == "__main__":
    unittest.main()
