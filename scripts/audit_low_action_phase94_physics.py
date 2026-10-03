#!/usr/bin/env python3
"""Predeclared last-60D/U physical audit against the same-phase zero case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

WINDOW = (114.0, 174.0)
ZERO = "alternating_t94_zero_20261003"
CONTROLLED = (
    "validation_signed_low_pulse_p_phase94_v1_20261003",
    "validation_signed_low_pulse_m_phase94_v1_20261003",
)


def load_force(path: Path) -> np.ndarray:
    rows = []
    header = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# Time"):
            header = line[1:].split()
            continue
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if header is None:
            raise ValueError(f"missing coefficient header: {path}")
        required = ("Time", "Cd", "Cl", "CmPitch")
        if any(name not in header for name in required):
            raise ValueError(f"missing coefficient column in {path}")
        rows.append(tuple(float(fields[header.index(name)]) for name in required))
    data = np.asarray(rows, dtype=np.float64)
    selected = data[(data[:, 0] >= WINDOW[0] - 1.0e-9) & (data[:, 0] <= WINDOW[1] + 1.0e-9)]
    # The predeclared closed interval [114, 174] contains both endpoints at
    # delta_t=0.005: (174 - 114) / 0.005 + 1 = 12001 samples.
    if len(selected) != 12001 or not np.isfinite(selected).all():
        raise ValueError(f"unexpected force window {path}: {len(selected)} rows")
    if not np.allclose(np.diff(selected[:, 0]), 0.005, rtol=0.0, atol=1.0e-9):
        raise ValueError(f"nonuniform force timestamps in {path}")
    return selected


def action_metrics(config: dict) -> dict:
    points = np.asarray(config.get("action_points", [[94.0, 0.0], [174.0, 0.0]]), dtype=np.float64)
    times = np.linspace(WINDOW[0], WINDOW[1], 12001)
    omega = np.interp(times, points[:, 0], points[:, 1])
    return {
        "mean_abs_omega": float(np.mean(np.abs(omega))),
        "rms_omega": float(np.sqrt(np.mean(np.square(omega)))),
        "max_abs_omega": float(np.max(np.abs(omega))),
    }


def metrics(cases_root: Path, name: str) -> tuple[dict, dict, np.ndarray]:
    case = cases_root / name
    config = json.loads((case / "case_config.json").read_text(encoding="utf-8"))
    front = load_force(case / "postProcessing/forceFront/94/coefficient.dat")
    rear = load_force(case / "postProcessing/forceRear/94/coefficient.dat")
    if not np.allclose(front[:, 0], rear[:, 0], rtol=0.0, atol=1.0e-9):
        raise ValueError(f"front/rear force times differ: {name}")
    total_cd = front[:, 1] + rear[:, 1]
    rear_cl = rear[:, 2]
    omega = np.interp(front[:, 0], np.asarray(config["action_points"])[:, 0], np.asarray(config["action_points"])[:, 1])
    result = {
        "samples": len(total_cd),
        "mean_total_cd": float(np.mean(total_cd)),
        "total_cd_fluctuation_rms": float(np.sqrt(np.mean(np.square(total_cd - np.mean(total_cd))))),
        "mean_rear_cd": float(np.mean(rear[:, 1])),
        "mean_rear_cl": float(np.mean(rear_cl)),
        "abs_mean_rear_cl": float(abs(np.mean(rear_cl))),
        "rear_cl_fluctuation_rms": float(np.sqrt(np.mean(np.square(rear_cl - np.mean(rear_cl))))),
        "action": action_metrics(config),
    }
    return config, result, np.column_stack((front[:, 0], omega, rear[:, 3]))


def canonical_comparison(row: dict, zero: dict) -> dict:
    drag_delta = row["mean_total_cd"] - zero["mean_total_cd"]
    lift_delta = row["rear_cl_fluctuation_rms"] - zero["rear_cl_fluctuation_rms"]
    drag_change = 100.0 * drag_delta / zero["mean_total_cd"]
    lift_ratio = row["rear_cl_fluctuation_rms"] / zero["rear_cl_fluctuation_rms"]
    mean_lift_ratio = row["abs_mean_rear_cl"] / zero["rear_cl_fluctuation_rms"]
    checks = {
        "drag_reduction_at_least_2_percent": drag_change <= -2.0,
        "rear_cl_fluctuation_within_1p05_zero": lift_ratio <= 1.05,
        "abs_mean_rear_cl_within_0p1_zero_cl_fluctuation_rms": mean_lift_ratio <= 0.1,
    }
    return {
        "mean_total_cd_delta": drag_delta,
        "mean_total_cd_change_percent": drag_change,
        "rear_cl_fluctuation_rms_delta": lift_delta,
        "rear_cl_fluctuation_rms_ratio": lift_ratio,
        "abs_mean_rear_cl_to_zero_cl_fluctuation_rms_ratio": mean_lift_ratio,
        **checks,
        "canonical_joint_gate": "PASS" if all(checks.values()) else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    configs = {}
    rows = {}
    torque_rows = {}
    for name in (ZERO, *CONTROLLED):
        configs[name], rows[name], torque_rows[name] = metrics(args.cases_root, name)
    provenance = {
        (
            config["source_restart_u_sha256"],
            config["source_restart_p_sha256"],
            config["source_restart_time"],
            config["start_time"],
            config["end_time"],
            config["delta_t"],
        )
        for config in configs.values()
    }
    if len(provenance) != 1:
        raise ValueError("cases do not share restart provenance, time window and dt")
    zero = rows[ZERO]
    comparisons = {}
    for name in CONTROLLED:
        row = rows[name]
        comparisons[name] = canonical_comparison(row, zero)
        actuator = -torque_rows[name][:, 1] * torque_rows[name][:, 2] / zero["mean_total_cd"]
        row["torque_work_proxy"] = {
            "mean_signed_minus_omega_cm_pitch_over_zero_cd": float(np.mean(actuator)),
            "mean_positive_only_minus_omega_cm_pitch_over_zero_cd": float(np.mean(np.maximum(actuator, 0.0))),
            "mean_absolute_minus_omega_cm_pitch_over_zero_cd": float(np.mean(np.abs(actuator))),
            "guard": "fluid-torque work proxy only; not electrical motor energy; positive-only assumes no regenerative recovery",
        }
    result = {
        "status": "LOW_ACTION_PHASE94_CANONICAL_PHYSICAL_AUDIT_V3_COMPLETE",
        "analysis_window_provenance": {
            "window": list(WINDOW),
            "window_length_D_over_U": 60.0,
            "discarded_initial_transient_D_over_U": 20.0,
            "baseline": ZERO,
            "inherited_from": "artifacts/tandem_cylinders/two_phase_alternating_predeclared_20261003.json",
            "inherited_file_sha256": "34355adfe55a019c8615514c1634531c76bcc230940b9c0e80319746374e627b",
            "inherited_file_mtime_utc": "2026-10-03T04:36:54.157373525Z",
            "low_action_predeclaration_limitation": "low_action_phase94_validation_v1_predeclared.json declared only the run interval [94,174], not this analysis interval; [114,174] is inherited from the earlier same-phase t94 zero/control audit",
            "canonical_joint_gate": {
                "mean_total_cd_reduction_percent_minimum": 2.0,
                "rear_cl_fluctuation_rms_maximum_ratio_to_zero": 1.05,
                "abs_mean_rear_cl_maximum_ratio_to_zero_cl_fluctuation_rms": 0.1,
            },
            "no_posthoc_window_selection": True,
        },
        "supersedes": [
            {
                "file": "low_action_phase94_physical_audit.json",
                "reason": "v1 used a stricter but non-canonical two-term gate",
            },
            {
                "file": "low_action_phase94_physical_audit_v2_canonical.json",
                "reason": "v2 overstated the low-action file's own analysis-window predeclaration and omitted torque-work proxies",
            },
        ],
        "shared_provenance": {
            "source_restart_time": 94.0,
            "source_restart_u_sha256": configs[ZERO]["source_restart_u_sha256"],
            "source_restart_p_sha256": configs[ZERO]["source_restart_p_sha256"],
            "delta_t": configs[ZERO]["delta_t"],
            "geometry_and_mesh": "same copied tandem_backward_dt005 constant/mesh, 19290 cells",
        },
        "cases": rows,
        "comparisons_to_same_phase_zero": comparisons,
        "scientific_scope": "open-loop same-phase CFD diagnostic; not closed-loop control or model-selection evidence",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
