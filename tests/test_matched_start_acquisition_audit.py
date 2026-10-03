"""Pure fail-closed tests for matched-start transfer and time-grid QC."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load_module():
    path = (
        Path(__file__).parents[1]
        / "scripts/audit_matched_start_acquisition_commissioning.py"
    )
    spec = importlib.util.spec_from_file_location("matched_start_audit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = load_module()


def load_log_checker():
    path = (
        Path(__file__).parents[1]
        / "cfd/tandem_cylinders/check_matched_start_solver_log.py"
    )
    spec = importlib.util.spec_from_file_location("matched_start_log_checker", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LOG_CHECKER = load_log_checker()


def load_lock_helper():
    path = Path(__file__).parents[1] / "cfd/tandem_cylinders/matched_start_case_lock.py"
    spec = importlib.util.spec_from_file_location("matched_start_case_lock", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LOCK_HELPER = load_lock_helper()


def test_exact_time_grid_accepts_complete_force_grid() -> None:
    MODULE.exact_time_grid(np.linspace(106.0, 186.0, 16001), 106.0, 186.0, 0.005)


def test_exact_time_grid_rejects_missing_or_shifted_samples() -> None:
    time = np.linspace(106.0, 186.0, 16001)
    with pytest.raises(ValueError, match="expected 16001"):
        MODULE.exact_time_grid(np.delete(time, 100), 106.0, 186.0, 0.005)
    time[100] += 1e-4
    with pytest.raises(ValueError, match="complete fixed grid"):
        MODULE.exact_time_grid(time, 106.0, 186.0, 0.005)


def test_force_alignment_requires_one_real_source_t0_row() -> None:
    raw_time = np.linspace(106.005, 186.0, 16000)
    raw = np.column_stack((raw_time, np.ones((16000, 4))))
    source = np.asarray([[106.0, 1.0, 2.0, 3.0, 4.0]])
    aligned = MODULE.aligned_force_rows(raw, source, 106.0, 186.0)
    assert len(aligned) == 16001
    assert aligned[0].tolist() == source[0].tolist()
    with pytest.raises(ValueError, match="expected one source force row"):
        MODULE.aligned_force_rows(raw, np.empty((0, 5)), 106.0, 186.0)


def test_sha256_manifest_rejects_duplicate_path(tmp_path: Path) -> None:
    path = tmp_path / "bad.sha256"
    digest = "0" * 64
    path.write_text(f"{digest}  case/U\n{digest}  case/U\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        MODULE.parse_sha256_manifest(path)


def test_solver_log_accepts_end_with_trailing_blank_line(tmp_path: Path) -> None:
    path = tmp_path / "log"
    path.write_text(
        "Time = 0.005\n"
        "Courant Number mean: 0.03 max: 0.24\n"
        "time step continuity errors : sum local = 1e-12, global = -2e-13\n"
        "Time = 0.01\n"
        "End\n\n",
        encoding="utf-8",
    )
    result = LOG_CHECKER.validate_solver_log(path, 2, 0.01)
    assert result["last_nonblank_line"] == "End"


def test_solver_log_rejects_wrong_terminal_time_or_numerical_qc(tmp_path: Path) -> None:
    path = tmp_path / "log"
    path.write_text(
        "Time = 0.005\n"
        "Courant Number mean: 0.03 max: 0.31\n"
        "time step continuity errors : sum local = 1e-12, global = 2e-13\n"
        "End\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="terminal solver time"):
        LOG_CHECKER.validate_solver_log(path, 1, 0.01)
    with pytest.raises(ValueError, match="Courant"):
        LOG_CHECKER.validate_solver_log(path, 1, 0.005)


def test_case_lock_atomically_rejects_second_start_and_retains_terminal_state(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "case.lock"
    LOCK_HELPER.acquire(lock, "case_a")
    with pytest.raises(ValueError, match="already exists"):
        LOCK_HELPER.acquire(lock, "case_a")
    LOCK_HELPER.update(lock, "COMPLETED", 0)
    with pytest.raises(ValueError, match="already terminal"):
        LOCK_HELPER.update(lock, "FAILED", 1)
