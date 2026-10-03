#!/usr/bin/env python3
"""Predeclare a validation-only t=86 action-ranking panel.

The three real-CFD cases share the untouched uncontrolled t=86 restart.  This
panel is a short model decision diagnostic, not a closed-loop control result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from make_expanded_control_dataset import replace_once, replace_rear_patch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
CASES = ROOT / "cases"
SOURCE_CASE = CASES / "tandem_backward_dt005"
SOURCE_TIME = SOURCE_CASE / "86"
TRAIN_ROOT = REPO / "data/curated/tandem_cylinders_gate_b_aug_v3/train"
VALIDATION_H5 = (
    REPO
    / "data/curated/tandem_cylinders_phase_v1/validation"
    / "phase_validation_86_multisine.h5"
)
START_TIME = 86.0
END_TIME = 88.0
ACTION_INTERVAL = 0.1
RAMP_DURATION = 1.0


@dataclass(frozen=True)
class PanelCase:
    name: str
    target: float


PANEL = (
    PanelCase("crossphase86_rank_zero_20261003", 0.0),
    PanelCase("crossphase86_rank_p100_20261003", 1.0),
    PanelCase("crossphase86_rank_m100_20261003", -1.0),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def action_points(spec: PanelCase) -> list[tuple[float, float]]:
    points = []
    count = round((END_TIME - START_TIME) / ACTION_INTERVAL)
    for index in range(count + 1):
        tau = index * ACTION_INTERVAL
        omega = spec.target * min(tau / RAMP_DURATION, 1.0)
        points.append((round(START_TIME + tau, 10), omega))
    return points


def leakage_audit() -> dict:
    with h5py.File(VALIDATION_H5) as handle:
        config = json.loads(handle.attrs["config_json"])
        if handle.attrs["split"] != "validation":
            raise ValueError("t=86 reference is not marked validation")
        if config["source_restart_time"] != START_TIME:
            raise ValueError("unexpected validation restart time")
        reference = np.asarray(handle["state"][0])
        if not math.isclose(float(handle["time"][0, 0]), START_TIME):
            raise ValueError("validation tensor does not begin at t=86")

    candidates = []
    exact_matches = []
    files = sorted(TRAIN_ROOT.glob("*.h5"))
    for path in files:
        with h5py.File(path) as handle:
            if handle.attrs["split"] != "train":
                raise ValueError(f"non-train file under train root: {path}")
            times = np.asarray(handle["time"][:, 0])
            for raw_index in np.flatnonzero(np.isclose(times, START_TIME, atol=1e-10)):
                index = int(raw_index)
                state = np.asarray(handle["state"][index])
                same = bool(np.array_equal(state, reference))
                row = {
                    "file": path.name,
                    "frame": index,
                    "state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
                    "max_abs_difference_from_validation_start": float(
                        np.max(np.abs(state - reference))
                    ),
                    "exact_match": same,
                }
                candidates.append(row)
                if same:
                    exact_matches.append(row)
    if exact_matches:
        raise ValueError("validation t=86 initial state occurs in training split")
    return {
        "checkpoint_training_root": str(TRAIN_ROOT.relative_to(REPO)),
        "train_files_checked": len(files),
        "train_frames_at_physical_time_86_checked": len(candidates),
        "exact_initial_state_matches": len(exact_matches),
        "validation_initial_state_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
        "candidates": candidates,
    }


def predeclared_audit() -> dict:
    for field in (SOURCE_TIME / "U", SOURCE_TIME / "p"):
        if not field.is_file():
            raise FileNotFoundError(field)
    return {
        "status": "PREDECLARED_BEFORE_CFD",
        "scope": "validation-only cross-phase FNO-vs-real-CFD action ranking; not closed-loop benefit or Gate-B passage",
        "source_restart": "tandem_backward_dt005/t=86 uncontrolled",
        "source_u_sha256": sha256(SOURCE_TIME / "U"),
        "source_p_sha256": sha256(SOURCE_TIME / "p"),
        "window": [START_TIME, END_TIME],
        "sampled_ranking_window": [86.1, END_TIME],
        "horizon_steps": 20,
        "action_interval": ACTION_INTERVAL,
        "action_rule": "linear ramp from omega=0 to target over t=86..87, then hold",
        "objective": "mean system-total Cd on the 20 t=86.1..88.0 action frames; lower is better",
        "cases": {
            spec.name: {"target_omega": spec.target, "action_points": action_points(spec)}
            for spec in PANEL
        },
        "leakage_audit": leakage_audit(),
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
        shutil.copytree(SOURCE_TIME, stage / "86")
        velocity = stage / "86/U"
        velocity.write_text(
            replace_rear_patch(velocity.read_text(encoding="utf-8"), points),
            encoding="utf-8",
        )
        control_path = stage / "system/controlDict"
        control = control_path.read_text(encoding="utf-8")
        control = replace_once(control, "startTime 0;", "startTime 86;", control_path)
        control = replace_once(control, "endTime 160;", "endTime 88;", control_path)
        control = replace_once(control, "writeInterval 2;", "writeInterval 0.1;", control_path)
        control_path.write_text(control, encoding="utf-8")
        metadata = {
            "case": spec.name,
            "panel": "crossphase86_validation_ranking_20261003",
            "split": "validation_only_not_training_or_frozen_test",
            "source_restart_case": SOURCE_CASE.name,
            "source_restart_time": START_TIME,
            "source_restart_u_sha256": sha256(SOURCE_TIME / "U"),
            "source_restart_p_sha256": sha256(SOURCE_TIME / "p"),
            "start_time": START_TIME,
            "end_time": END_TIME,
            "action_points": points,
            "target_omega": spec.target,
            "action_interval": ACTION_INTERVAL,
            "delta_t": 0.005,
            "purpose": "short cross-phase real-CFD action-ranking diagnostic",
            "interpretation_guard": "not training data, not frozen test, not closed-loop performance",
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
        parser.error("one or more panel cases already exist")
    audit = predeclared_audit()
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    for spec in PANEL:
        print(generate(spec))


if __name__ == "__main__":
    main()
