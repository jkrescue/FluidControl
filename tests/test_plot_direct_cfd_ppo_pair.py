from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/plot_direct_cfd_ppo_pair.py"
SPEC = importlib.util.spec_from_file_location("plot_direct_cfd_ppo_pair", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_force(path: Path, rows: list[list[float]]) -> None:
    path.parent.mkdir(parents=True)
    np.savetxt(path, np.asarray(rows))


def test_paired_force_uses_raw_deduplicated_coefficients(tmp_path: Path) -> None:
    case = tmp_path / "case"
    write_force(
        case / "postProcessing/forceFront/0/coefficient.dat",
        [[1.0, 1.0, 0.0, 0.0, 0.1], [1.5, 2.0, 0.0, 0.0, 0.2]],
    )
    write_force(
        case / "postProcessing/forceFront/1/coefficient.dat",
        [[1.5, 9.0, 0.0, 0.0, 9.0], [2.0, 3.0, 0.0, 0.0, 0.3]],
    )
    write_force(
        case / "postProcessing/forceRear/0/coefficient.dat",
        [[1.0, 4.0, 0.0, 0.0, -0.1], [1.5, 5.0, 0.0, 0.0, -0.2]],
    )
    write_force(
        case / "postProcessing/forceRear/1/coefficient.dat",
        [[1.5, 8.0, 0.0, 0.0, 8.0], [2.0, 6.0, 0.0, 0.0, -0.3]],
    )

    values, sources = MODULE.paired_force(case, 1.0, 2.0)

    np.testing.assert_allclose(
        values,
        [[1.0, 5.0, -0.1], [1.5, 7.0, -0.2], [2.0, 9.0, -0.3]],
    )
    assert len(sources) == 4


def test_paired_force_rejects_different_time_grids(tmp_path: Path) -> None:
    case = tmp_path / "case"
    write_force(
        case / "postProcessing/forceFront/0/coefficient.dat",
        [[1.0, 1.0, 0.0, 0.0, 0.1], [2.0, 2.0, 0.0, 0.0, 0.2]],
    )
    write_force(
        case / "postProcessing/forceRear/0/coefficient.dat",
        [[1.0, 1.0, 0.0, 0.0, 0.1]],
    )

    with pytest.raises(ValueError, match="time grids differ"):
        MODULE.paired_force(case, 1.0, 2.0)


def test_action_endpoint_series_includes_zero_then_linear_ramp_endpoints() -> None:
    time, omega = MODULE.action_endpoint_series(
        [
            {"cfd_time": 148.1, "applied_omega": 0.1},
            {"cfd_time": 148.2, "applied_omega": -0.05},
        ],
        148.0,
    )

    np.testing.assert_allclose(time, [0.0, 0.1, 0.2])
    np.testing.assert_allclose(omega, [0.0, 0.1, -0.05])
