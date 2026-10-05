from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from fluid_control.openfoam_force_history import actual_causal_prehistory


def write_force(path: Path, rows: list[tuple[float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Time Cd x x Cl\n"
        + "".join(f"{time:.8f} {cd} 0 0 {cl}\n" for time, cd, cl in rows),
        encoding="utf-8",
    )


class OpenFOAMForceHistoryTests(unittest.TestCase):
    def make_source(self, root: Path) -> Path:
        case = root / "source"
        times = [141.9 + 0.1 * index for index in range(62)]
        write_force(
            case / "postProcessing/forceFront/0/coefficient.dat",
            [(time, 1.0 + index, 2.0 + index) for index, time in enumerate(times)],
        )
        write_force(
            case / "postProcessing/forceRear/0/coefficient.dat",
            [(time, 3.0 + index, 4.0 + index) for index, time in enumerate(times)],
        )
        return case

    def test_exact_unique_causal_history_and_channel_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = self.make_source(root)
            times, forces, sources = actual_causal_prehistory(
                case, 148.0, provenance_root=root
            )
            self.assertEqual(len(times), 62)
            self.assertEqual(len(set(times)), 62)
            np.testing.assert_allclose(np.diff(times), 0.1, rtol=0.0, atol=1e-8)
            self.assertEqual(times[-1], 148.0)
            self.assertEqual(forces[0], [1.0, 2.0, 3.0, 4.0])
            self.assertEqual(forces[-1], [62.0, 63.0, 64.0, 65.0])
            self.assertEqual(set(sources), {"forceFront", "forceRear"})

    def test_identical_restart_duplicate_merges_but_conflict_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = self.make_source(root)
            duplicate = case / "postProcessing/forceFront/1/coefficient.dat"
            write_force(duplicate, [(148.0, 62.0, 63.0)])
            times, _, sources = actual_causal_prehistory(
                case, 148.0, provenance_root=root
            )
            self.assertEqual(len(times), 62)
            self.assertEqual(len(sources["forceFront"]), 2)
            write_force(duplicate, [(148.0, 62.0, 999.0)])
            with self.assertRaisesRegex(ValueError, "conflicting"):
                actual_causal_prehistory(case, 148.0, provenance_root=root)

    def test_missing_and_nonfinite_samples_reject(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = self.make_source(root)
            rear = case / "postProcessing/forceRear/0/coefficient.dat"
            rows = rear.read_text(encoding="utf-8").splitlines()
            rear.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(FileNotFoundError, "missing causal"):
                actual_causal_prehistory(case, 148.0, provenance_root=root)
            write_force(rear, [(148.0, float("nan"), 1.0)])
            with self.assertRaisesRegex(ValueError, "non-finite"):
                actual_causal_prehistory(case, 148.0, provenance_root=root)


if __name__ == "__main__":
    unittest.main()
