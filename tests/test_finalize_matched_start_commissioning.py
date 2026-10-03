from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


def load_module():
    path = Path(__file__).parents[1] / "scripts" / "finalize_matched_start_commissioning.py"
    spec = importlib.util.spec_from_file_location("matched_start_finalizer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MatchedStartFinalizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_incomplete_and_nontrain_staging_are_rejected(self) -> None:
        cases = [f"case_{index}" for index in range(9)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                self.module.validate_staging_layout(root, cases)
            staging = root / self.module.STAGING_ROOT
            for case in cases:
                path = staging / case / "train" / f"{case}.h5"
                path.parent.mkdir(parents=True)
                path.touch()
            self.module.validate_staging_layout(root, cases)
            forbidden = staging / cases[0] / "validation" / "bad.h5"
            forbidden.parent.mkdir()
            forbidden.touch()
            with self.assertRaisesRegex(ValueError, "validation"):
                self.module.validate_staging_layout(root, cases)

    def test_finalizer_uses_generator_scalar_contract(self) -> None:
        repo = Path(__file__).parents[1]
        orchestrator = self.module.load_orchestrator(repo)
        config = {
            "phase_bin": 4,
            "expected_field_frames": 801,
            "expected_aligned_force_samples_with_source_t0": 16001,
        }
        self.module.validate_generator_contract(
            config, "real_config", {"phase_bin": "b04"}, orchestrator
        )
        legacy = dict(config)
        del legacy["expected_aligned_force_samples_with_source_t0"]
        legacy["force_samples"] = 16001
        with self.assertRaisesRegex(ValueError, "expected_aligned_force"):
            self.module.validate_generator_contract(
                legacy, "legacy_config", {"phase_bin": "b04"}, orchestrator
            )

    def test_source_t0_is_prepended_without_interpolation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cases_root = Path(directory)
            case = cases_root / "branch"
            source = cases_root / "baseline"
            for root, rows in (
                (case, [(index * 0.005, 1.0 + index, -1.0 - index) for index in range(1, 16001)]),
                (source, [(0.0, 12.5, -3.25)]),
            ):
                for object_name in ("forceFront", "forceRear"):
                    path = root / "postProcessing" / object_name / "0" / "coefficient.dat"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(
                        "# Time Cd Cl\n"
                        + "".join(f"{time:.8f} {cd:.8f} {cl:.8f}\n" for time, cd, cl in rows),
                        encoding="utf-8",
                    )
            assembled, audit = self.module.assembled_force(
                cases_root,
                case,
                {"source_restart_case": "baseline"},
                "forceFront",
                0.0,
                80.0,
            )
            self.assertEqual(assembled.shape, (16001, 3))
            self.assertEqual(assembled[0].tolist(), [0.0, 12.5, -3.25])
            self.assertEqual(audit["raw_rows"], 16000)
            self.assertEqual(audit["assembled_rows"], 16001)


if __name__ == "__main__":
    unittest.main()
