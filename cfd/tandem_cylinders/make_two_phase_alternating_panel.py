#!/usr/bin/env python3
"""Predeclare two-phase zero-mean alternating-rotation OpenFOAM cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import h5py
from make_expanded_control_dataset import replace_once, replace_rear_patch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
CASES = ROOT / "cases"
SOURCE_CASE = CASES / "tandem_backward_dt005"
DURATION = 80.0
ANALYSIS_OFFSET = 20.0
ACTION_PERIOD = 20.0
CURATED_ROOTS = (
    REPO / "data/curated/tandem_cylinders_control_gap_v4",
    REPO / "data/curated/tandem_cylinders_phase_v1",
)


@dataclass(frozen=True)
class PanelCase:
    name: str
    start: float
    controlled: bool


PANEL = (
    PanelCase("alternating_t90_zero_20261003", 90.0, False),
    PanelCase("alternating_t90_square20_20261003", 90.0, True),
    PanelCase("alternating_t94_zero_20261003", 94.0, False),
    PanelCase("alternating_t94_square20_20261003", 94.0, True),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_alternating_points() -> list[tuple[float, float]]:
    points = [(0.0, 0.0)]
    for cycle in range(4):
        base = cycle * ACTION_PERIOD
        points.extend(
            [
                (base + 1.0, 1.0),
                (base + 9.0, 1.0),
                (base + 11.0, -1.0),
                (base + 19.0, -1.0),
                (base + 20.0, 0.0),
            ]
        )
    return points


def action_points(spec: PanelCase) -> list[tuple[float, float]]:
    if not spec.controlled:
        return [(spec.start, 0.0), (spec.start + DURATION, 0.0)]
    return [(spec.start + time, omega) for time, omega in relative_alternating_points()]


def action_audit(points: list[tuple[float, float]]) -> dict:
    signed_integral = 0.0
    max_rate = 0.0
    for (time_0, value_0), (time_1, value_1) in pairwise(points):
        delta = time_1 - time_0
        if delta <= 0:
            raise ValueError("action table is not strictly increasing")
        signed_integral += 0.5 * (value_0 + value_1) * delta
        max_rate = max(max_rate, abs(value_1 - value_0) / delta)
    duration = points[-1][0] - points[0][0]
    return {
        "signed_time_mean_omega": signed_integral / duration,
        "max_abs_omega": max(abs(value) for _, value in points),
        "max_abs_domega_dt": max_rate,
    }


def leakage_audit() -> dict:
    records = []
    references = {}
    for start in (90.0, 94.0):
        source = SOURCE_CASE / f"{start:g}"
        reference = {field: sha256(source / field) for field in ("U", "p")}
        references[f"t{start:g}"] = reference
        for root in CURATED_ROOTS:
            for split in ("train", "validation", "test"):
                for h5_path in sorted((root / split).glob("*.h5")):
                    with h5py.File(h5_path) as handle:
                        config = json.loads(handle.attrs["config_json"])
                        if handle.attrs["split"] != split:
                            raise ValueError(f"split mismatch: {h5_path}")
                    raw_time = CASES / config["case"] / f"{start:g}"
                    hashes = {}
                    for field in ("U", "p"):
                        path = raw_time / field
                        if not path.is_file():
                            raise FileNotFoundError(path)
                        hashes[field] = sha256(path)
                    records.append(
                        {
                            "start": start,
                            "dataset": str(root.relative_to(REPO)),
                            "split": split,
                            "case": config["case"],
                            "u_and_p_exact_match": bool(hashes == reference),
                        }
                    )
    exact = [row for row in records if row["u_and_p_exact_match"]]
    if exact:
        raise ValueError("one or more alternating-panel starts occur in curated splits")
    return {
        "method": "raw OpenFOAM U/p SHA-256 at each start compared with every current-v4 and independent-phase train/validation/test case",
        "datasets": [str(root.relative_to(REPO)) for root in CURATED_ROOTS],
        "comparisons": len(records),
        "split_counts": {
            split: sum(row["split"] == split for row in records)
            for split in ("train", "validation", "test")
        },
        "exact_u_and_p_matches": len(exact),
        "source_hashes": references,
        "records": records,
    }


def predeclared_audit() -> dict:
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
            "action_metrics": action_audit(points),
        }
    return {
        "status": "PREDECLARED_BEFORE_CFD",
        "scope": "two-phase long-window zero-mean alternating rotation versus fresh zero; no tuning and no closed-loop claim",
        "physical_hypothesis": "equal long dwell at +1 and -1 may preserve the sign-symmetric constant-rotation drag reduction while cancelling mean rear lift",
        "action_period": ACTION_PERIOD,
        "analysis_duration": DURATION - ANALYSIS_OFFSET,
        "analysis_action_periods": (DURATION - ANALYSIS_OFFSET) / ACTION_PERIOD,
        "approximate_analysis_shedding_periods": (DURATION - ANALYSIS_OFFSET) / 6.154,
        "schedules": schedules,
        "canonical_checks": {
            "total_drag_reduction_at_least_2pct": 0.02,
            "rear_cl_fluctuation_rms_ratio_at_most": 1.05,
            "abs_mean_rear_cl_over_zero_fluctuation_rms_at_most": 0.10,
        },
        "torque_cost": "report signed -omega*CmPitch/Cd_zero, positive-only, and absolute proxies; never call these electrical motor energy",
        "stop_rule": "no parameter changes after CFD; retain negative result if either phase fails any canonical check",
        "leakage_audit": leakage_audit(),
    }


def generate(spec: PanelCase, source_hashes: dict) -> Path:
    target = CASES / spec.name
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    source_name = f"{spec.start:g}"
    source_time = SOURCE_CASE / source_name
    points = action_points(spec)
    with tempfile.TemporaryDirectory(prefix=f".{spec.name}.", dir=CASES) as temporary:
        stage = Path(temporary) / spec.name
        stage.mkdir()
        shutil.copytree(SOURCE_CASE / "constant", stage / "constant")
        shutil.copytree(SOURCE_CASE / "system", stage / "system")
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
            "panel": "two_phase_alternating_square20_validation_20261003",
            "split": "validation_only_not_training_or_frozen_test",
            "source_restart_case": SOURCE_CASE.name,
            "source_restart_time": spec.start,
            "source_restart_u_sha256": source_hashes["U"],
            "source_restart_p_sha256": source_hashes["p"],
            "start_time": spec.start,
            "end_time": spec.start + DURATION,
            "analysis_window": [spec.start + ANALYSIS_OFFSET, spec.start + DURATION],
            "controlled": spec.controlled,
            "action_points": points,
            "action_metrics": action_audit(points),
            "field_write_interval": 2.0,
            "force_write_interval": 0.005,
            "delta_t": 0.005,
            "purpose": "two-phase alternating-sign drag/lift/torque-cost audit",
            "interpretation_guard": "not training/frozen test, not tuning, and not closed-loop acceptance",
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
        parser.error("one or more alternating-panel cases already exist")
    audit = predeclared_audit()
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    for spec in PANEL:
        hashes = audit["leakage_audit"]["source_hashes"][f"t{spec.start:g}"]
        print(generate(spec, hashes))


if __name__ == "__main__":
    main()
