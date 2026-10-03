from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_matched_start_phase_restarts.py"
if not SCRIPT.exists():
    SCRIPT = ROOT / "audit_matched_start_phase_restarts.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase_restart_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def _make_case(root: Path, *, include_states: bool = True) -> Path:
    case = root / "baseline"
    force = case / "postProcessing" / "forceRear" / "0" / "coefficient.dat"
    force.parent.mkdir(parents=True)
    time = np.arange(0.005, 160.0001, 0.005)
    rear_cl = np.sin(2.0 * np.pi * time / 16.0)
    columns = np.zeros((len(time), 13))
    columns[:, 0] = time
    columns[:, 4] = rear_cl
    np.savetxt(force, columns)
    for restart in range(100, 161, 2):
        directory = case / str(restart)
        directory.mkdir()
        if include_states:
            (directory / "U").write_text(f"U-{restart}\n", encoding="utf-8")
            (directory / "p").write_text(f"p-{restart}\n", encoding="utf-8")
    return case


class MatchedStartPhaseRestartTests(unittest.TestCase):
    def test_real_restart_matrix_covers_all_bins(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            case = _make_case(Path(directory))
            force_path = (
                case / "postProcessing" / "forceRear" / "0" / "coefficient.dat"
            )
            expected_force_sha256 = MODULE._sha256(force_path)
            result = MODULE.audit_baseline(case)
        self.assertEqual(result["decision"], "GO_9_CASE_COMMISSIONING")
        self.assertEqual(len(result["selections"]), 8)
        self.assertTrue(all(row["coverage_pass"] for row in result["selections"]))
        self.assertTrue(result["restart_qc"]["selected_restart_times_unique"])
        self.assertTrue(result["restart_qc"]["selected_state_hashes_unique"])
        self.assertEqual(result["commissioning"]["case_count"], 9)
        self.assertEqual(result["full_seed_matrix"]["authorization"], "not authorized by this audit")
        self.assertFalse(result["algorithm"]["state_interpolation"])
        self.assertEqual(
            result["signal_qc"]["force_sha256"],
            expected_force_sha256,
        )
        self.assertNotIn("rear_force_source_sha256", result["signal_qc"])

    def test_missing_real_states_is_no_go(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = MODULE.audit_baseline(
                _make_case(Path(directory), include_states=False)
            )
        self.assertEqual(result["decision"], "NO_GO_REGENERATE_BASELINE_RESTARTS")
        self.assertEqual(result["restart_qc"]["eligible_real_restart_count"], 0)

    def test_output_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "audit.json"
            output.write_text("existing", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                MODULE.write_audit({"new": "result"}, output)
            self.assertEqual(output.read_text(encoding="utf-8"), "existing")


if __name__ == "__main__":
    unittest.main()
