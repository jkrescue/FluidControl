"""Parser-only fixtures; these are not CFD training or validation data."""
from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fluid_control.openfoam_observation import (  # noqa: E402
    _force_file,
    _probe_file,
    _select,
    observation_at,
    total_drag_observation_at,
)


def opened(text: str) -> MagicMock:
    path = MagicMock()
    path.open.return_value.__enter__.return_value = io.StringIO(text)
    return path


class OpenFOAMObservationParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.header = "".join(
            f"# Probe {i} (17 {6 + 3*i/31:.9f} 0.05)\n" for i in range(32)
        )
        self.row = "90 " + " ".join("(1 2 0)" for _ in range(32)) + "\n"

    def test_valid_probe_and_force_rows(self) -> None:
        probes = _probe_file(opened(self.header + self.row), 90.0)
        self.assertEqual(probes.shape, (32, 2))
        self.assertTrue(np.all(probes[:, 0] == 1))
        force = _force_file(
            opened("# Time Cd Cd(f) Cd(r) Cl\n90 0.8 0.3 0.5 -0.2\n"), 90.0
        )
        np.testing.assert_allclose(force, [0.8, -0.2])

    def test_missing_or_ambiguous_probe_rows_fail_closed(self) -> None:
        self.assertIsNone(_probe_file(opened(self.header + self.row), 91.0))
        with self.assertRaises(ValueError):
            _probe_file(opened(self.header + self.row + self.row), 90.0)
        with self.assertRaises(ValueError):
            _probe_file(opened(self.header + "# Probe 0 (17 6 0.05)\n" + self.row), 90.0)
        corrupted = self.header.replace("# Probe 1 (17 ", "# Probe 1 (18 ")
        with self.assertRaises(ValueError):
            _probe_file(opened(corrupted + self.row), 90.0)

    def test_force_column_order_and_action_bounds(self) -> None:
        with self.assertRaises(ValueError):
            _force_file(opened("# Time Cl Cd Cd(f) Cd(r)\n90 0 0 0 0\n"), 90.0)
        with self.assertRaises(ValueError):
            observation_at(Path("."), 90.0, 5.1)

    def test_observation_order(self) -> None:
        probes = np.repeat([[1.0, 2.0]], 32, axis=0)
        with patch(
            "fluid_control.openfoam_observation._select",
            side_effect=[(probes, ["probes"]), (np.array([0.8, -0.2]), ["force"])],
        ):
            values, metadata = observation_at(Path("."), 90.0, 1.5)
        self.assertEqual(values.shape, (67,))
        np.testing.assert_allclose(values[:4], [1, 2, 1, 2])
        np.testing.assert_allclose(values[-3:], [0.8, -0.2, 1.5])
        self.assertEqual(metadata["probe_sources"], ["probes"])

    def test_total_drag_observation_reads_both_forces_at_same_time(self) -> None:
        legacy = np.concatenate((np.ones(64), [0.8, -0.2, 1.5])).astype(np.float32)
        with patch(
            "fluid_control.openfoam_observation.observation_at",
            return_value=(legacy, {"time": 90.0}),
        ), patch(
            "fluid_control.openfoam_observation._select",
            return_value=(np.array([1.2, 0.3]), ["front-force"]),
        ) as selected:
            values, metadata = total_drag_observation_at(Path("."), 90.0, 1.5)
        self.assertEqual(values.shape, (69,))
        np.testing.assert_allclose(values[64:], [1.2, 0.3, 0.8, -0.2, 1.5])
        self.assertEqual(metadata["front_force_sources"], ["front-force"])
        self.assertIn("front_cd,front_cl,rear_cd,rear_cl", metadata["channel_order"])
        self.assertIn("forceFront", selected.call_args.args[1])

    def test_restart_samples_must_agree(self) -> None:
        case = MagicMock()
        case.glob.return_value = ["restart_80", "restart_90"]
        values = {
            "restart_80": np.array([1.0, 2.0]),
            "restart_90": np.array([1.0, 2.0]),
        }
        sample, sources = _select(case, "probe/*", lambda path, _: values[path], 90.0)
        np.testing.assert_allclose(sample, [1, 2])
        self.assertEqual(sources, ["restart_80", "restart_90"])
        values["restart_90"] = np.array([1.1, 2.0])
        with self.assertRaises(ValueError):
            _select(case, "probe/*", lambda path, _: values[path], 90.0)
        with self.assertRaises(FileNotFoundError):
            _select(case, "probe/*", lambda path, _: None, 90.0)


if __name__ == "__main__":
    unittest.main()
