#!/usr/bin/env python3
"""Audit the predeclared two-phase long-dwell OpenFOAM panel."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CFD = REPO / "cfd/tandem_cylinders"
CASES = CFD / "cases"
sys.path.insert(0, str(CFD))
from analyze_baseline import log_health

PHASES = {
    "t90": {
        "zero": "longdwell075_t90_zero_20261003",
        "control": "longdwell075_t90_control_20261003",
        "window": (130.0, 210.0),
    },
    "t94": {
        "zero": "longdwell075_t94_zero_20261003",
        "control": "longdwell075_t94_control_20261003",
        "window": (134.0, 214.0),
    },
}
EXPECTED_ANALYSIS_SAMPLES = 16001
EXPECTED_SOLVER_STEPS = 24000


def read_coefficients(case: Path, force_name: str) -> np.ndarray:
    samples: dict[float, np.ndarray] = {}
    paths = sorted(case.glob(f"postProcessing/{force_name}/*/coefficient.dat"))
    if not paths:
        raise FileNotFoundError(f"missing {force_name}: {case}")
    for path in paths:
        header = None
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# Time"):
                header = line[1:].split()
                continue
            if not line or line.startswith("#"):
                continue
            if header is None:
                raise ValueError(f"missing coefficient header: {path}")
            fields = line.split()
            required = ("Time", "Cd", "Cl", "CmPitch")
            if any(name not in header for name in required):
                raise ValueError(f"missing required coefficient in {path}")
            row = np.asarray([float(fields[header.index(name)]) for name in required])
            if not np.isfinite(row).all():
                raise ValueError(f"non-finite coefficient row: {path}")
            key = round(float(row[0]), 8)
            if key in samples and not np.allclose(samples[key], row, rtol=0, atol=1e-10):
                raise ValueError(f"conflicting restart row at t={key}")
            samples[key] = row
    return np.asarray([samples[key] for key in sorted(samples)])


def validate_analysis_time(time: np.ndarray, begin: float, end: float) -> None:
    expected = np.linspace(begin, end, EXPECTED_ANALYSIS_SAMPLES)
    if len(time) != EXPECTED_ANALYSIS_SAMPLES:
        raise ValueError(
            f"expected {EXPECTED_ANALYSIS_SAMPLES} analysis samples, got {len(time)}"
        )
    if not np.allclose(time, expected, rtol=0, atol=1e-8):
        raise ValueError("analysis timestamps do not match the complete 0.005 grid")


def selected_series(case: Path, begin: float, end: float) -> tuple[np.ndarray, np.ndarray]:
    front = read_coefficients(case, "forceFront")
    rear = read_coefficients(case, "forceRear")
    front = front[(front[:, 0] >= begin - 1e-9) & (front[:, 0] <= end + 1e-9)]
    rear = rear[(rear[:, 0] >= begin - 1e-9) & (rear[:, 0] <= end + 1e-9)]
    validate_analysis_time(front[:, 0], begin, end)
    validate_analysis_time(rear[:, 0], begin, end)
    if not np.allclose(front[:, 0], rear[:, 0], rtol=0, atol=1e-9):
        raise ValueError(f"front/rear timestamps differ: {case}")
    return front, rear


def fluctuation_rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values - np.mean(values)))))


def action_table(case: Path) -> np.ndarray:
    config = json.loads((case / "case_config.json").read_text(encoding="utf-8"))
    return np.asarray(config["action_points"], dtype=float)


def metrics(
    case: Path,
    begin: float,
    end: float,
    baseline_drag: float | None,
) -> tuple[dict, np.ndarray]:
    front, rear = selected_series(case, begin, end)
    time = front[:, 0]
    total_cd = front[:, 1] + rear[:, 1]
    total_cl = front[:, 2] + rear[:, 2]
    action = action_table(case)
    omega = np.interp(time, action[:, 0], action[:, 1])
    moment = rear[:, 3]
    cycles = []
    for index in range(2):
        left = begin + 40.0 * index
        right = left + 40.0
        selected = (time >= left - 1e-9) & (
            time <= right + 1e-9 if index == 1 else time < right - 1e-9
        )
        cycles.append(
            {
                "window": [left, right],
                "total_cd_mean": float(np.mean(total_cd[selected])),
                "rear_cl_mean": float(np.mean(rear[selected, 2])),
                "rear_cl_fluctuation_rms": fluctuation_rms(rear[selected, 2]),
            }
        )
    result = {
        "samples": len(time),
        "window": [begin, end],
        "total_cd_mean": float(np.mean(total_cd)),
        "front_cd_mean": float(np.mean(front[:, 1])),
        "rear_cd_mean": float(np.mean(rear[:, 1])),
        "front_cl_mean": float(np.mean(front[:, 2])),
        "rear_cl_mean": float(np.mean(rear[:, 2])),
        "front_cl_fluctuation_rms": fluctuation_rms(front[:, 2]),
        "rear_cl_fluctuation_rms": fluctuation_rms(rear[:, 2]),
        "total_cl_fluctuation_rms": fluctuation_rms(total_cl),
        "omega_time_mean": float(np.mean(omega)),
        "omega_rms": float(np.sqrt(np.mean(np.square(omega)))),
        "cm_pitch_mean": float(np.mean(moment)),
        "cm_pitch_rms": float(np.sqrt(np.mean(np.square(moment)))),
        "mean_omega_times_cm_pitch": float(np.mean(omega * moment)),
        "action_cycle_metrics": cycles,
    }
    if baseline_drag is not None:
        actuator = -omega * moment / baseline_drag
        result["torque_cost"] = {
            "mean_signed_actuator_power_over_zero_drag_power": float(np.mean(actuator)),
            "mean_positive_only_actuator_power_over_zero_drag_power": float(
                np.mean(np.maximum(actuator, 0.0))
            ),
            "mean_abs_rotation_work_over_zero_drag_power": float(
                np.mean(np.abs(actuator))
            ),
        }
    return result, np.column_stack((time, omega, moment))


def comparison(control: dict, zero: dict) -> dict:
    reduction = 1.0 - control["total_cd_mean"] / zero["total_cd_mean"]
    lift_ratio = (
        control["rear_cl_fluctuation_rms"] / zero["rear_cl_fluctuation_rms"]
    )
    mean_ratio = abs(control["rear_cl_mean"]) / zero["rear_cl_fluctuation_rms"]
    checks = {
        "total_drag_reduction_at_least_2pct": reduction >= 0.02,
        "rear_cl_fluctuation_rms_increase_at_most_5pct": lift_ratio <= 1.05,
        "rear_mean_lift_at_most_10pct_zero_fluctuation_rms": mean_ratio <= 0.10,
    }
    return {
        "total_drag_reduction": reduction,
        "rear_cl_fluctuation_rms_ratio": lift_ratio,
        "abs_rear_cl_mean_over_zero_fluctuation_rms": mean_ratio,
        "canonical_checks": checks,
        "canonical_joint_check": all(checks.values()),
    }


def validate_case_config(name: str, predeclared: dict) -> None:
    case = CASES / name
    config = json.loads((case / "case_config.json").read_text(encoding="utf-8"))
    frozen = predeclared["schedules"][name]
    if config["action_points"] != frozen["action_points"]:
        raise ValueError(f"action table differs from predeclared audit: {name}")
    if config["analysis_window"] != frozen["analysis_window"]:
        raise ValueError(f"analysis window differs from predeclared audit: {name}")
    source = predeclared["leakage_audit"]["source_hashes"][
        f"t{config['source_restart_time']:g}"
    ]
    if {
        "U": config["source_restart_u_sha256"],
        "p": config["source_restart_p_sha256"],
    } != source:
        raise ValueError(f"source hashes differ from predeclared audit: {name}")


def write_raw_manifest(output_path: Path) -> str:
    paths = []
    for phase in PHASES.values():
        for role in ("zero", "control"):
            case = CASES / phase[role]
            paths.extend(
                [
                    case / "case_config.json",
                    case / "log.pimpleFoam.longdwell075",
                ]
            )
            paths.extend(case.glob("postProcessing/force*/*/coefficient.dat"))
    unique_paths = sorted(set(paths))
    if len(unique_paths) != 16 or any(not path.is_file() for path in unique_paths):
        raise ValueError("expected 16 key raw files for the two-phase panel")
    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(CASES)}"
        for path in unique_paths
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return hashlib.sha256(output_path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predeclared-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"refusing to overwrite {args.output_dir}")
    predeclared = json.loads(args.predeclared_audit.read_text(encoding="utf-8"))
    if predeclared["status"] != "PREDECLARED_BEFORE_CFD":
        raise ValueError("invalid predeclared audit")
    if predeclared["leakage_audit"]["exact_u_and_p_matches"] != 0:
        raise ValueError("initial-state leakage detected")
    expected_checks = {
        "total_drag_reduction_at_least_2pct": 0.02,
        "rear_cl_fluctuation_rms_ratio_at_most": 1.05,
        "abs_mean_rear_cl_over_zero_fluctuation_rms_at_most": 0.10,
    }
    if predeclared["canonical_checks"] != expected_checks:
        raise ValueError("canonical checks differ from the project gate")

    phases = {}
    torque_rows = []
    for phase_name, phase in PHASES.items():
        for name in (phase["zero"], phase["control"]):
            validate_case_config(name, predeclared)
        begin, end = phase["window"]
        zero_metrics, zero_rows = metrics(CASES / phase["zero"], begin, end, None)
        zero_metrics["torque_cost"] = {
            "mean_signed_actuator_power_over_zero_drag_power": 0.0,
            "mean_positive_only_actuator_power_over_zero_drag_power": 0.0,
            "mean_abs_rotation_work_over_zero_drag_power": 0.0,
        }
        control_metrics, control_rows = metrics(
            CASES / phase["control"],
            begin,
            end,
            zero_metrics["total_cd_mean"],
        )
        health = {}
        for label in ("zero", "control"):
            health[label] = log_health(
                CASES / phase[label] / "log.pimpleFoam.longdwell075"
            )
            if not health[label]["solver_ended_cleanly"]:
                raise ValueError(f"incomplete solver log: {phase[label]}")
            if health[label]["steps"] != EXPECTED_SOLVER_STEPS:
                raise ValueError(f"unexpected solver step count: {phase[label]}")
        phase_comparison = comparison(control_metrics, zero_metrics)
        phases[phase_name] = {
            "cases": {label: phase[label] for label in ("zero", "control")},
            "window": [begin, end],
            "metrics": {"zero": zero_metrics, "control": control_metrics},
            "comparison": phase_comparison,
            "solver_health": health,
        }
        for label, rows in (("zero", zero_rows), ("control", control_rows)):
            for time, omega, moment in rows:
                actuator = -omega * moment / zero_metrics["total_cd_mean"]
                torque_rows.append(
                    [phase_name, label, time, omega, moment, actuator]
                )

    first = phases["t90"]["comparison"]
    second = phases["t94"]["comparison"]
    robustness = {
        "drag_reduction_phase_difference_percentage_points": 100.0
        * abs(first["total_drag_reduction"] - second["total_drag_reduction"]),
        "rear_cl_fluctuation_ratio_phase_difference": abs(
            first["rear_cl_fluctuation_rms_ratio"]
            - second["rear_cl_fluctuation_rms_ratio"]
        ),
        "canonical_joint_pass_both_phases": bool(
            first["canonical_joint_check"] and second["canonical_joint_check"]
        ),
    }
    args.output_dir.mkdir(parents=True)
    raw_manifest = args.output_dir / "raw_key_files.sha256"
    raw_manifest_sha256 = write_raw_manifest(raw_manifest)
    torque_csv = args.output_dir / "rear_cylinder_torque_timeseries.csv"
    with torque_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "phase",
                "case_role",
                "time",
                "omega",
                "cm_pitch_fluid_on_cylinder",
                "signed_actuator_power_over_phase_zero_drag_power",
            ]
        )
        writer.writerows(torque_rows)
    report = {
        "status": "TWO_PHASE_LONG_DWELL075_OPENFOAM_AUDIT_COMPLETED",
        "scope": "predeclared real-CFD low-amplitude long-dwell physical screen; not tuning, learned closed loop, or final acceptance",
        "predeclared_audit": str(args.predeclared_audit),
        "predeclared_audit_sha256": hashlib.sha256(
            args.predeclared_audit.read_bytes()
        ).hexdigest(),
        "phases": phases,
        "robustness": robustness,
        "torque_definition": "OpenFOAM CmPitch=M_fluid/(0.5*rho*U_inf^2*Aref*lRef); signed ideal actuator proxy=-omega*CmPitch/Cd_total_zero",
        "torque_guard": "fluid-torque work proxy, not electrical motor power; positive-only assumes no regenerative recovery",
        "torque_timeseries": str(torque_csv),
        "torque_timeseries_sha256": hashlib.sha256(torque_csv.read_bytes()).hexdigest(),
        "raw_key_files_manifest": str(raw_manifest),
        "raw_key_files_manifest_sha256": raw_manifest_sha256,
    }
    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "comparisons": {
                    phase: data["comparison"] for phase, data in phases.items()
                },
                "robustness": robustness,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
