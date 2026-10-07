"""Pure recovery-contract checks for the E109 t=328 continuation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


RESTART_FILES = frozenset({"U", "U_0", "p", "phi", "phi_0", "uniform/time"})
START = 328.0
END = 408.0
STEPS = 800
SAVED_APPLIED_OMEGA = 0.19905773401260382


def require(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError(why)


def seed_from_e109(result: dict[str, Any]) -> tuple[np.ndarray, float]:
    """Return exact policy observation and limiter state from terminal E109."""
    require(result.get("cycles") == STEPS, "E109 must contain all 800 cycles")
    rows = result.get("rows")
    require(isinstance(rows, list) and len(rows) == STEPS, "E109 row count")
    row = rows[-1]
    require(row.get("step") == STEPS and row.get("start_time") == 327.9
            and row.get("end_time") == START, "E109 terminal interval")
    obs = np.asarray(row.get("output_observation"), dtype=np.float32)
    require(obs.shape == (69,) and np.isfinite(obs).all(), "terminal physical69")
    require(np.asarray(row.get("input_observation"), dtype=np.float32).shape == (69,),
            "terminal input physical69")
    previous = float(row.get("applied_omega"))
    require(previous == SAVED_APPLIED_OMEGA, "terminal double limiter state")
    require(float(obs[-1]) == np.float32(previous), "observation/limiter representation")
    require(float(np.asarray(row["input_observation"], np.float32)[-1]) != float(obs[-1]),
            "must not seed from t327.9 input observation")
    sources = row.get("observation_sources", {})
    require(sources.get("time") == START, "terminal observation time")
    for key in ("probe_sources", "force_sources", "front_force_sources"):
        values = sources.get(key)
        require(isinstance(values, list) and values
                and all("/327.9/" in value for value in values),
                "terminal source segment: " + key)
    return obs, previous


def require_restart_tree(path: Path, role: str) -> dict[str, str]:
    """Check only the structural fields required by the saved restart contract."""
    require(role in {"case_mpc", "case_zero"}, "branch role")
    files = {str(item.relative_to(path)) for item in path.rglob("*") if item.is_file()}
    require(RESTART_FILES.issubset(files), "missing backward restart fields")
    require("U_0_0" not in files and "phi_0_0" not in files,
            "unexpected unbound second-old-time convention")
    return {name: str(path / name) for name in sorted(RESTART_FILES)}


def prospective_windows() -> list[tuple[str, float, float]]:
    return [
        ("tail_block_1", 328.0, 348.0),
        ("tail_block_2", 348.0, 368.0),
        ("tail_block_3", 368.0, 388.0),
        ("tail_block_4", 388.0, 408.0),
        ("tail_full_80", 328.0, 408.0),
        ("joined_full_160", 248.0, 408.0),
        ("joined_post_transition_140", 268.0, 408.0),
    ]
