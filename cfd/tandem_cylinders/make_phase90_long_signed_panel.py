#!/usr/bin/env python3
"""Predeclare the t=90 long-window signed-rotation robustness panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import h5py
from make_expanded_control_dataset import replace_once, replace_rear_patch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
CASES = ROOT / "cases"
SOURCE_CASE = CASES / "tandem_backward_dt005"
SOURCE_TIME = SOURCE_CASE / "90"
START_TIME = 90.0
END_TIME = 126.0
ANALYSIS_START = 102.0
ACTION_INTERVAL = 0.1
RAMP_DURATION = 2.0
CURATED_ROOTS = (
    REPO / "data/curated/tandem_cylinders_gate_b_aug_v3",
    REPO / "data/curated/tandem_cylinders_phase_v1",
)


@dataclass(frozen=True)
class PanelCase:
    name: str
    target: float


PANEL = (
    PanelCase("phase90_long_zero_20261003", 0.0),
    PanelCase("phase90_long_p100_20261003", 1.0),
    PanelCase("phase90_long_m100_20261003", -1.0),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def action_points(spec: PanelCase) -> list[tuple[float, float]]:
    count = round((END_TIME - START_TIME) / ACTION_INTERVAL)
    return [
        (
            round(START_TIME + index * ACTION_INTERVAL, 10),
            spec.target * min(index * ACTION_INTERVAL / RAMP_DURATION, 1.0),
        )
        for index in range(count + 1)
    ]


def leakage_audit() -> dict:
    reference = {field: sha256(SOURCE_TIME / field) for field in ("U", "p")}
    rows = []
    for root in CURATED_ROOTS:
        for split in ("train", "validation", "test"):
            for h5_path in sorted((root / split).glob("*.h5")):
                with h5py.File(h5_path) as handle:
                    config = json.loads(handle.attrs["config_json"])
                    if handle.attrs["split"] != split:
                        raise ValueError(f"split mismatch: {h5_path}")
                raw_time = CASES / config["case"] / "90"
                hashes = {}
                for field in ("U", "p"):
                    path = raw_time / field
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    hashes[field] = sha256(path)
                rows.append(
                    {
                        "dataset": str(root.relative_to(REPO)),
                        "split": split,
                        "case": config["case"],
                        "u_exact_match": hashes["U"] == reference["U"],
                        "p_exact_match": hashes["p"] == reference["p"],
                    }
                )
    exact = [row for row in rows if row["u_exact_match"] and row["p_exact_match"]]
    if exact:
        raise ValueError("t=90 restart exactly occurs in an existing curated split")
    return {
        "method": "compare raw OpenFOAM t=90 U/p SHA-256 against every case in current checkpoint and independent-phase train/validation/test splits",
        "datasets": [str(root.relative_to(REPO)) for root in CURATED_ROOTS],
        "cases_checked": len(rows),
        "split_counts": {
            split: sum(row["split"] == split for row in rows)
            for split in ("train", "validation", "test")
        },
        "exact_u_and_p_matches": len(exact),
        "reference_hashes": reference,
        "records": rows,
    }


def predeclared_audit() -> dict:
    leakage = leakage_audit()
    return {
        "status": "PREDECLARED_BEFORE_CFD",
        "scope": "second-phase long-window signed-action robustness and drag/lift/torque-cost audit; not tuning or closed-loop acceptance",
        "source_restart": "tandem_backward_dt005/t=90 uncontrolled",
        "source_hashes": leakage["reference_hashes"],
        "run_window": [START_TIME, END_TIME],
        "analysis_window": [ANALYSIS_START, END_TIME],
        "discarded_transient": [START_TIME, ANALYSIS_START],
        "reference_phase_a": {
            "source_restart": "tandem_backward_dt005/t=80 uncontrolled",
            "cases": [
                "landscape_val_zero_20261003",
                "landscape_val_p100_20261003",
                "landscape_val_m100_20261003",
            ],
            "analysis_window": [92.0, 116.0],
        },
        "action_rule": "omega ramps linearly from 0 to target during first 2 D/U, then holds",
        "actions": {
            spec.name: {"target_omega": spec.target, "points": action_points(spec)}
            for spec in PANEL
        },
        "canonical_checks": {
            "total_drag_reduction_at_least_2pct": 0.02,
            "rear_cl_fluctuation_rms_ratio_at_most": 1.05,
            "abs_mean_rear_cl_over_zero_fluctuation_rms_at_most": 0.10,
        },
        "torque_cost_definition": {
            "cm_pitch": "M_fluid/(0.5*rho*U_inf^2*A_ref*l_ref)",
            "signed_ideal_actuator_proxy": "-omega*CmPitch under force-on-body sign convention",
            "guards": "also report positive-only and absolute proxies; do not call any proxy electrical motor energy",
        },
        "leakage_audit": leakage,
    }


def generate(spec: PanelCase) -> Path:
    target = CASES / spec.name
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    points = action_points(spec)
    with tempfile.TemporaryDirectory(prefix=f".{spec.name}.", dir=CASES) as temporary:
        stage = Path(temporary) / spec.name
        stage.mkdir()
        shutil.copytree(SOURCE_CASE / "constant", stage / "constant")
        shutil.copytree(SOURCE_CASE / "system", stage / "system")
        shutil.copytree(SOURCE_TIME, stage / "90")
        velocity = stage / "90/U"
        velocity.write_text(
            replace_rear_patch(velocity.read_text(encoding="utf-8"), points),
            encoding="utf-8",
        )
        control_path = stage / "system/controlDict"
        control = control_path.read_text(encoding="utf-8")
        control = replace_once(control, "startTime 0;", "startTime 90;", control_path)
        control = replace_once(control, "endTime 160;", "endTime 126;", control_path)
        control_path.write_text(control, encoding="utf-8")
        metadata = {
            "case": spec.name,
            "panel": "phase90_long_signed_validation_20261003",
            "split": "validation_only_not_training_or_frozen_test",
            "source_restart_case": SOURCE_CASE.name,
            "source_restart_time": START_TIME,
            "source_restart_u_sha256": sha256(SOURCE_TIME / "U"),
            "source_restart_p_sha256": sha256(SOURCE_TIME / "p"),
            "start_time": START_TIME,
            "end_time": END_TIME,
            "analysis_window": [ANALYSIS_START, END_TIME],
            "target_omega": spec.target,
            "action_points": points,
            "action_interval": ACTION_INTERVAL,
            "field_write_interval": 2.0,
            "force_write_interval": 0.005,
            "delta_t": 0.005,
            "purpose": "cross-phase long-window robustness and torque-cost audit",
            "interpretation_guard": "not training/frozen test, not parameter tuning, not closed-loop acceptance",
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
        parser.error("one or more phase-B cases already exist")
    audit = predeclared_audit()
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    for spec in PANEL:
        print(generate(spec))


if __name__ == "__main__":
    main()
