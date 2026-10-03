#!/usr/bin/env python3
"""Audit two-phase long-window signed rotation, including CmPitch cost."""

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
    "phase_a_t80": {
        "window": (92.0, 116.0),
        "cases": {
            "zero": "landscape_val_zero_20261003",
            "positive": "landscape_val_p100_20261003",
            "negative": "landscape_val_m100_20261003",
        },
        "logs": {key: "log.pimpleFoam" for key in ("zero", "positive", "negative")},
    },
    "phase_b_t90": {
        "window": (102.0, 126.0),
        "cases": {
            "zero": "phase90_long_zero_20261003",
            "positive": "phase90_long_p100_20261003",
            "negative": "phase90_long_m100_20261003",
        },
        "logs": {
            key: "log.pimpleFoam.phase90_long"
            for key in ("zero", "positive", "negative")
        },
    },
}


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
                raise ValueError(f"conflicting coefficient restart row at {key}")
            samples[key] = row
    return np.asarray([samples[key] for key in sorted(samples)])


def selected_series(case: Path, begin: float, end: float) -> tuple[np.ndarray, np.ndarray]:
    front = read_coefficients(case, "forceFront")
    rear = read_coefficients(case, "forceRear")
    front = front[(front[:, 0] >= begin - 1e-9) & (front[:, 0] <= end + 1e-9)]
    rear = rear[(rear[:, 0] >= begin - 1e-9) & (rear[:, 0] <= end + 1e-9)]
    if len(front) < 4000 or len(front) != len(rear):
        raise ValueError(f"insufficient or unpaired force samples: {case}")
    if not np.allclose(front[:, 0], rear[:, 0], rtol=0, atol=1e-9):
        raise ValueError(f"front/rear timestamps differ: {case}")
    return front, rear


def fluctuation_rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values - np.mean(values)))))


def action_table(case: Path) -> np.ndarray:
    config = json.loads((case / "case_config.json").read_text(encoding="utf-8"))
    return np.asarray(config["action_points"], dtype=float)


def case_metrics(
    case: Path,
    begin: float,
    end: float,
    baseline_total_cd: float | None,
) -> tuple[dict, np.ndarray]:
    front, rear = selected_series(case, begin, end)
    time = front[:, 0]
    total_cd = front[:, 1] + rear[:, 1]
    total_cl = front[:, 2] + rear[:, 2]
    omega = np.interp(time, action_table(case)[:, 0], action_table(case)[:, 1])
    cm_pitch = rear[:, 3]
    block_means = []
    for index in range(4):
        left = begin + 6.0 * index
        right = left + 6.0
        selected = (time >= left - 1e-9) & (
            time <= right + 1e-9 if index == 3 else time < right - 1e-9
        )
        block_means.append(float(np.mean(total_cd[selected])))
    result = {
        "samples": len(time),
        "window": [begin, end],
        "total_cd_mean": float(np.mean(total_cd)),
        "total_cd_block_means_6du": block_means,
        "front_cd_mean": float(np.mean(front[:, 1])),
        "rear_cd_mean": float(np.mean(rear[:, 1])),
        "front_cl_mean": float(np.mean(front[:, 2])),
        "rear_cl_mean": float(np.mean(rear[:, 2])),
        "front_cl_fluctuation_rms": fluctuation_rms(front[:, 2]),
        "rear_cl_fluctuation_rms": fluctuation_rms(rear[:, 2]),
        "total_cl_fluctuation_rms": fluctuation_rms(total_cl),
        "omega_rms": float(np.sqrt(np.mean(np.square(omega)))),
        "cm_pitch_mean": float(np.mean(cm_pitch)),
        "cm_pitch_rms": float(np.sqrt(np.mean(np.square(cm_pitch)))),
        "mean_omega_times_cm_pitch": float(np.mean(omega * cm_pitch)),
    }
    if baseline_total_cd is not None:
        actuator = -omega * cm_pitch / baseline_total_cd
        result["torque_cost"] = {
            "mean_signed_actuator_power_over_zero_drag_power": float(np.mean(actuator)),
            "mean_positive_only_actuator_power_over_zero_drag_power": float(
                np.mean(np.maximum(actuator, 0.0))
            ),
            "mean_abs_rotation_work_over_zero_drag_power": float(
                np.mean(np.abs(actuator))
            ),
        }
    rows = np.column_stack((time, omega, cm_pitch))
    return result, rows


def comparison(control: dict, zero: dict) -> dict:
    drag_reduction = 1.0 - control["total_cd_mean"] / zero["total_cd_mean"]
    lift_ratio = (
        control["rear_cl_fluctuation_rms"] / zero["rear_cl_fluctuation_rms"]
    )
    mean_lift_ratio = abs(control["rear_cl_mean"]) / zero["rear_cl_fluctuation_rms"]
    checks = {
        "total_drag_reduction_at_least_2pct": drag_reduction >= 0.02,
        "rear_cl_fluctuation_rms_increase_at_most_5pct": lift_ratio <= 1.05,
        "rear_mean_lift_at_most_10pct_zero_fluctuation_rms": mean_lift_ratio <= 0.10,
    }
    return {
        "total_drag_reduction": drag_reduction,
        "rear_cl_fluctuation_rms_ratio": lift_ratio,
        "abs_rear_cl_mean_over_zero_fluctuation_rms": mean_lift_ratio,
        "canonical_checks": checks,
        "canonical_joint_check": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predeclared-audit", type=Path, required=True)
    parser.add_argument("--reference-correction", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"refusing to overwrite {args.output_dir}")
    predeclared = json.loads(args.predeclared_audit.read_text(encoding="utf-8"))
    if predeclared["status"] != "PREDECLARED_BEFORE_CFD":
        raise ValueError("invalid predeclared audit")
    if predeclared["leakage_audit"]["exact_u_and_p_matches"] != 0:
        raise ValueError("t=90 initial state leakage detected")
    correction = json.loads(args.reference_correction.read_text(encoding="utf-8"))
    if correction["status"] != "PRE_RESULT_PROTOCOL_REFERENCE_CORRECTION":
        raise ValueError("invalid phase-A reference correction")

    report_phases = {}
    torque_rows = []
    for phase_name, phase in PHASES.items():
        begin, end = phase["window"]
        if phase_name == "phase_b_t90":
            if [begin, end] != predeclared["analysis_window"]:
                raise ValueError("Phase-B analysis window differs from predeclared audit")
            for name in phase["cases"].values():
                config = json.loads(
                    (CASES / name / "case_config.json").read_text(encoding="utf-8")
                )
                expected = predeclared["actions"][name]["points"]
                if config["action_points"] != expected:
                    raise ValueError(f"action table differs from predeclared audit: {name}")
                if {
                    "U": config["source_restart_u_sha256"],
                    "p": config["source_restart_p_sha256"],
                } != predeclared["source_hashes"]:
                    raise ValueError(f"source hashes differ from predeclared audit: {name}")
        elif list(phase["cases"].values()) != correction["correct_phase_a_cases"]:
            raise ValueError("Phase-A cases differ from pre-result correction")
        zero_case = CASES / phase["cases"]["zero"]
        zero_metrics, zero_torque = case_metrics(zero_case, begin, end, None)
        zero_metrics["torque_cost"] = {
            "mean_signed_actuator_power_over_zero_drag_power": 0.0,
            "mean_positive_only_actuator_power_over_zero_drag_power": 0.0,
            "mean_abs_rotation_work_over_zero_drag_power": 0.0,
        }
        metrics = {"zero": zero_metrics}
        raw_rows = {"zero": zero_torque}
        for label in ("positive", "negative"):
            metrics[label], raw_rows[label] = case_metrics(
                CASES / phase["cases"][label],
                begin,
                end,
                zero_metrics["total_cd_mean"],
            )
        health = {}
        for label, name in phase["cases"].items():
            health[label] = log_health(CASES / name / phase["logs"][label])
            if not health[label]["solver_ended_cleanly"]:
                raise ValueError(f"incomplete solver log: {name}")
        comparisons = {
            label: comparison(metrics[label], metrics["zero"])
            for label in ("positive", "negative")
        }
        report_phases[phase_name] = {
            "cases": phase["cases"],
            "window": [begin, end],
            "metrics": metrics,
            "comparisons": comparisons,
            "solver_health": health,
        }
        for label, rows in raw_rows.items():
            baseline = zero_metrics["total_cd_mean"]
            for time, omega, moment in rows:
                actuator = -omega * moment / baseline
                torque_rows.append(
                    [phase_name, label, time, omega, moment, actuator]
                )

    robustness = {}
    for label in ("positive", "negative"):
        phase_a = report_phases["phase_a_t80"]["comparisons"][label]
        phase_b = report_phases["phase_b_t90"]["comparisons"][label]
        robustness[label] = {
            "drag_reduction_phase_a": phase_a["total_drag_reduction"],
            "drag_reduction_phase_b": phase_b["total_drag_reduction"],
            "absolute_phase_difference_percentage_points": 100.0
            * abs(phase_a["total_drag_reduction"] - phase_b["total_drag_reduction"]),
            "same_drag_reduction_sign": bool(
                phase_a["total_drag_reduction"] * phase_b["total_drag_reduction"] > 0
            ),
            "canonical_joint_pass_both_phases": bool(
                phase_a["canonical_joint_check"] and phase_b["canonical_joint_check"]
            ),
        }

    args.output_dir.mkdir(parents=True)
    torque_csv = args.output_dir / "rear_cylinder_torque_timeseries.csv"
    with torque_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "phase",
                "action",
                "time",
                "omega",
                "cm_pitch_fluid_on_cylinder",
                "signed_actuator_power_over_phase_zero_drag_power",
            ]
        )
        writer.writerows(torque_rows)
    report = {
        "status": "TWO_PHASE_LONG_SIGNED_OPENFOAM_AUDIT_COMPLETED",
        "scope": "cross-phase long-window robustness and drag/lift/torque-cost audit; not tuning, closed-loop benefit, or final acceptance",
        "predeclared_audit": str(args.predeclared_audit),
        "protocol_reference_correction": str(args.reference_correction),
        "phases": report_phases,
        "robustness": robustness,
        "torque_definition": "OpenFOAM CmPitch=M_fluid/(0.5*rho*U_inf^2*Aref*lRef); signed ideal actuator proxy=-omega*CmPitch/Cd_total_zero",
        "torque_guard": "fluid-torque work proxy, not electrical motor power; positive-only assumes no regenerative recovery",
        "torque_timeseries": str(torque_csv),
        "torque_timeseries_sha256": hashlib.sha256(torque_csv.read_bytes()).hexdigest(),
    }
    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "robustness": robustness}, indent=2))


if __name__ == "__main__":
    main()
