#!/usr/bin/env python3
"""Generate a frozen validation-only signed-pulse pair at restart phase t=94."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path

from make_expanded_control_dataset import replace_once, replace_rear_patch

ROOT = Path(__file__).resolve().parent
CASES = ROOT / "cases"
SOURCE = CASES / "tandem_backward_dt005"
START = 94.0
END = 174.0
INTERVAL = 0.1
PROFILE = "control_gap_low_action_phase94_validation_v1"
NAMES = {
    "validation_signed_low_pulse_p_phase94_v1_20261003": 1.0,
    "validation_signed_low_pulse_m_phase94_v1_20261003": -1.0,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unsigned_action(time: float) -> float:
    tau = time - START
    if not 0 <= tau <= END - START + 1.0e-9:
        raise ValueError("action time outside trajectory")
    if math.isclose(tau, END - START, abs_tol=1.0e-9):
        return 0.0
    within = tau % 20.0
    if within < 2.0:
        return 0.75 * within / 2.0
    if within < 10.0:
        return 0.75
    if within < 12.0:
        return 0.75 * (12.0 - within) / 2.0
    return 0.0


def action_points(sign: float) -> list[tuple[float, float]]:
    return [
        (
            round(START + index * INTERVAL, 10),
            round(sign * unsigned_action(START + index * INTERVAL), 12),
        )
        for index in range(801)
    ]


def action_metrics(points: list[tuple[float, float]]) -> dict[str, float]:
    peak = max(abs(omega) for _, omega in points)
    rate = max(
        abs((right[1] - left[1]) / (right[0] - left[0]))
        for left, right in zip(points, points[1:])
    )
    if peak > 1.0 or rate > 0.3750001:
        raise ValueError("low-action validation schedule exceeds frozen bounds")
    return {"max_abs_omega": peak, "max_abs_domega_dt": rate}


def metadata(name: str, sign: float) -> dict:
    source_time = SOURCE / "94"
    points = action_points(sign)
    return {
        "case": name,
        "dataset_profile": PROFILE,
        "split": "validation",
        "action_kind": "paired_signed_low_amplitude_dynamic_pulses",
        "sign": sign,
        "action_points": points,
        "action_metrics": action_metrics(points),
        "action_interpolation": "OpenFOAM Function1 table, linear",
        "source": "OpenFOAM v2512 pimpleFoam real CFD; not experimental data",
        "source_restart_case": SOURCE.name,
        "source_restart_time": START,
        "source_restart_u_sha256": sha256(source_time / "U"),
        "source_restart_p_sha256": sha256(source_time / "p"),
        "geometry": "two fixed-centre tandem cylinders D=1, centres (10,7.5) and (15,7.5)",
        "reynolds_number": 100,
        "kinematic_viscosity": 0.01,
        "mesh_cells": 19290,
        "delta_t": 0.005,
        "time_scheme": "backward",
        "start_time": START,
        "end_time": END,
        "field_write_interval": INTERVAL,
        "expected_frames": 801,
        "force_series": ["front Cd/Cl", "rear Cd/Cl"],
        "purpose": "validation-only diagnosis of low-action long-rollout error at an independent restart phase",
        "leakage_guard": "never merge into control_gap_v4 train/validation/test; store as a separate profile",
        "limitations": "one independent restart phase and a mirrored action pair; diagnostic, not a control-benefit claim",
    }


def generate(name: str, sign: float) -> Path:
    target = CASES / name
    if target.exists():
        raise FileExistsError(target)
    source_time = SOURCE / "94"
    for field in ("U", "p"):
        if not (source_time / field).is_file():
            raise FileNotFoundError(source_time / field)
    points = action_points(sign)
    with tempfile.TemporaryDirectory(prefix=f".{name}.", dir=CASES) as temporary:
        stage = Path(temporary) / name
        stage.mkdir()
        shutil.copytree(SOURCE / "constant", stage / "constant")
        shutil.copytree(SOURCE / "system", stage / "system")
        shutil.copytree(source_time, stage / "94")
        velocity = stage / "94/U"
        velocity.write_text(
            replace_rear_patch(velocity.read_text(encoding="utf-8"), points),
            encoding="utf-8",
        )
        control_path = stage / "system/controlDict"
        control = control_path.read_text(encoding="utf-8")
        control = replace_once(control, "startTime 0;", "startTime 94;", control_path)
        control = replace_once(control, "endTime 160;", "endTime 174;", control_path)
        control = replace_once(control, "writeInterval 2;", "writeInterval 0.1;", control_path)
        control_path.write_text(control, encoding="utf-8")
        (stage / "case_config.json").write_text(
            json.dumps(metadata(name, sign), indent=2) + "\n", encoding="utf-8"
        )
        stage.rename(target)
    return target


def main() -> None:
    positive = action_points(1.0)
    negative = action_points(-1.0)
    if any(
        left[0] != right[0] or abs(left[1] + right[1]) > 1.0e-12
        for left, right in zip(positive, negative)
    ):
        raise ValueError("signed schedules are not exact mirrors")
    audit = {
        "status": "PREDECLARED_VALIDATION_ONLY_ACTIONS_NO_CFD_LABELS",
        "dataset_profile": PROFILE,
        "split": "validation_only_separate_profile",
        "source_restart": "tandem_backward_dt005/t=94",
        "source_restart_u_sha256": sha256(SOURCE / "94/U"),
        "source_restart_p_sha256": sha256(SOURCE / "94/p"),
        "time_window": [START, END],
        "action_design": "paired +/-0.75 pulses every 20 units; two-unit ramps, eight-unit holds, two-unit ramp-downs",
        "selection_reason": "existing validation has no |omega|max<=1 windows at H100; acquire diagnostic labels without changing frozen v4 holdout",
        "cases": {
            name: {"sign": sign, **action_metrics(action_points(sign))}
            for name, sign in NAMES.items()
        },
        "leakage_guard": "never add these labels to control_gap_v4 or tune the already completed v4 model",
    }
    audit_path = CASES.parent / "low_action_phase94_validation_v1_predeclared.json"
    if audit_path.exists():
        raise FileExistsError(audit_path)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(audit_path)
    for name, sign in NAMES.items():
        print(generate(name, sign))


if __name__ == "__main__":
    main()
