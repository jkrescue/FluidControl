#!/usr/bin/env python3
"""Predeclare low-amplitude long-dwell alternating-rotation CFD cases."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from make_expanded_control_dataset import replace_once, replace_rear_patch
from make_two_phase_alternating_panel import leakage_audit

ROOT = Path(__file__).resolve().parent
CASES = ROOT / "cases"
SOURCE = CASES / "tandem_backward_dt005"
DURATION = 120.0
ANALYSIS_OFFSET = 40.0
ACTION_PERIOD = 40.0
AMPLITUDE = 0.75


@dataclass(frozen=True)
class PanelCase:
    name: str
    start: float
    controlled: bool


PANEL = (
    PanelCase("longdwell075_t90_zero_20261003", 90.0, False),
    PanelCase("longdwell075_t90_control_20261003", 90.0, True),
    PanelCase("longdwell075_t94_zero_20261003", 94.0, False),
    PanelCase("longdwell075_t94_control_20261003", 94.0, True),
)


def relative_control_points() -> list[tuple[float, float]]:
    points = [(0.0, 0.0)]
    for cycle in range(3):
        base = cycle * ACTION_PERIOD
        points.extend(
            [
                (base + 2.0, -AMPLITUDE),
                (base + 18.0, -AMPLITUDE),
                (base + 22.0, AMPLITUDE),
                (base + 38.0, AMPLITUDE),
                (base + 40.0, 0.0),
            ]
        )
    return points


def action_points(spec: PanelCase) -> list[tuple[float, float]]:
    if not spec.controlled:
        return [(spec.start, 0.0), (spec.start + DURATION, 0.0)]
    return [(spec.start + time, omega) for time, omega in relative_control_points()]


def action_metrics(points: list[tuple[float, float]]) -> dict:
    signed_integral = 0.0
    max_rate = 0.0
    for (t0, value0), (t1, value1) in pairwise(points):
        delta = t1 - t0
        signed_integral += 0.5 * (value0 + value1) * delta
        max_rate = max(max_rate, abs(value1 - value0) / delta)
    duration = points[-1][0] - points[0][0]
    return {
        "signed_time_mean_omega": signed_integral / duration,
        "max_abs_omega": max(abs(value) for _, value in points),
        "max_abs_domega_dt": max_rate,
    }


def predeclared_audit() -> dict:
    leakage = leakage_audit()
    schedules = {}
    for spec in PANEL:
        points = action_points(spec)
        schedules[spec.name] = {
            "start": spec.start,
            "controlled": spec.controlled,
            "run_window": [spec.start, spec.start + DURATION],
            "analysis_window": [
                spec.start + ANALYSIS_OFFSET,
                spec.start + DURATION,
            ],
            "action_points": points,
            "action_metrics": action_metrics(points),
        }
    return {
        "status": "PREDECLARED_BEFORE_CFD",
        "scope": "two-phase low-amplitude long-dwell alternating rotation versus fresh same-phase zero; validation-only physical experiment",
        "unique_design": {
            "amplitude": AMPLITUDE,
            "action_period": ACTION_PERIOD,
            "negative_plateau_each_period": [2.0, 18.0],
            "positive_plateau_each_period": [22.0, 38.0],
            "sign_transition_duration": 4.0,
            "motivation": "halve sign-change frequency relative to failed omega=+/-1 T=20 square wave and permit roughly 2.6 shedding periods at each low-amplitude plateau, testing nonlinear quasi-steady drag preservation rather than linear duty mixing",
        },
        "analysis_duration": DURATION - ANALYSIS_OFFSET,
        "analysis_action_periods": (DURATION - ANALYSIS_OFFSET) / ACTION_PERIOD,
        "approximate_analysis_shedding_periods": (DURATION - ANALYSIS_OFFSET) / 6.154,
        "schedules": schedules,
        "canonical_checks": {
            "total_drag_reduction_at_least_2pct": 0.02,
            "rear_cl_fluctuation_rms_ratio_at_most": 1.05,
            "abs_mean_rear_cl_over_zero_fluctuation_rms_at_most": 0.10,
        },
        "torque_cost": "report signed -omega*CmPitch/Cd_zero, positive-only, and absolute proxies; not electrical motor energy",
        "failure_stop_rule": "if either phase fails the canonical joint gate, retain the result and do not tune amplitude, period, duty, phase, or window",
        "data_guard": "never merge cases into Curator/FNO train, validation, or frozen-test profiles; physical audit only",
        "leakage_audit": leakage,
    }


def generate(spec: PanelCase, source_hashes: dict) -> Path:
    target = CASES / spec.name
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    source_name = f"{spec.start:g}"
    source_time = SOURCE / source_name
    points = action_points(spec)
    with tempfile.TemporaryDirectory(prefix=f".{spec.name}.", dir=CASES) as temporary:
        stage = Path(temporary) / spec.name
        stage.mkdir()
        shutil.copytree(SOURCE / "constant", stage / "constant")
        shutil.copytree(SOURCE / "system", stage / "system")
        shutil.copytree(source_time, stage / source_name)
        velocity = stage / source_name / "U"
        velocity.write_text(
            replace_rear_patch(velocity.read_text(encoding="utf-8"), points),
            encoding="utf-8",
        )
        control_path = stage / "system/controlDict"
        control = control_path.read_text(encoding="utf-8")
        control = replace_once(
            control, "startTime 0;", f"startTime {source_name};", control_path
        )
        control = replace_once(
            control,
            "endTime 160;",
            f"endTime {spec.start + DURATION:g};",
            control_path,
        )
        control_path.write_text(control, encoding="utf-8")
        metadata = {
            "case": spec.name,
            "panel": "two_phase_longdwell075_validation_20261003",
            "split": "validation_only_not_curator_or_frozen_test",
            "source_restart_case": SOURCE.name,
            "source_restart_time": spec.start,
            "source_restart_u_sha256": source_hashes["U"],
            "source_restart_p_sha256": source_hashes["p"],
            "start_time": spec.start,
            "end_time": spec.start + DURATION,
            "analysis_window": [spec.start + ANALYSIS_OFFSET, spec.start + DURATION],
            "controlled": spec.controlled,
            "action_points": points,
            "action_metrics": action_metrics(points),
            "field_write_interval": 2.0,
            "force_write_interval": 0.005,
            "delta_t": 0.005,
            "purpose": "validation-only nonlinear long-dwell physical control screen",
            "interpretation_guard": "not training data, frozen test, tuning, or learned closed-loop evidence",
        }
        (stage / "case_config.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        stage.rename(target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    if args.audit.exists():
        parser.error(f"refusing to overwrite {args.audit}")
    if any((CASES / spec.name).exists() for spec in PANEL):
        parser.error("one or more long-dwell cases already exist")
    audit = predeclared_audit()
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    for spec in PANEL:
        hashes = audit["leakage_audit"]["source_hashes"][f"t{spec.start:g}"]
        print(generate(spec, hashes))


if __name__ == "__main__":
    main()
