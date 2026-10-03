#!/usr/bin/env python3
"""Predeclare and stage eight train-only dynamic-action OpenFOAM cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from itertools import pairwise
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CASES = HERE / "cases"
SOURCE = CASES / "tandem_backward_dt005"
sys.path.insert(0, str(HERE))
from make_expanded_control_dataset import replace_once, replace_rear_patch

FULL40 = REPO / "artifacts/tandem_cylinders/matched_start_full40_predeclared_20261003.json"
FULL40_SHA256 = "d7ff174ef10194a8739357376335ca13ff9b45c8079970846bb71f15a715d24b"
PREDECLARATION = REPO / "artifacts/tandem_cylinders/dynamic_train8_predeclared_20261003.json"
APPROVED_PREDECLARATION_SHA256 = (
    "4cf4e7c9b9da27b71e58db2e94b0750736b7f79f09a2aebc3ffa97729e882c5a"
)
GENERATION_TOKEN = "GENERATE_REVIEWED_DYNAMIC_TRAIN8"
IMAGE = (
    "opencfd/openfoam-default@sha256:"
    "33fb575aa9980d2bc42fd58c75ae698c489293ba30c991380fe3f899c622f319"
)
STATE_FIELDS = ("U", "U_0", "p", "phi", "phi_0")
PHASE_BINS = (0, 2, 4, 6)
DURATION = 20.0
DELTA_T = 0.005
CONTROL_INTERVAL = 0.1

# These target sequences and analytic phases are fixed without consulting the
# b01/b05 dynamic validation outcomes.  They are intentionally train-only.
PRBS_TARGETS = {
    0: (0.75, -0.75, 0.375, -0.375, 0.75, -0.75, 0.375, -0.375),
    2: (-0.375, 0.75, -0.75, 0.375, -0.375, 0.75, -0.75, 0.375),
    4: (0.375, -0.75, 0.75, -0.375, 0.375, -0.75, 0.75, -0.375),
    6: (-0.75, 0.375, -0.375, 0.75, -0.75, 0.375, -0.375, 0.75),
}
MULTISINE_PHASE = {0: 0.0, 2: math.pi / 2, 4: math.pi, 6: 3 * math.pi / 2}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prbs_profile(phase_bin: int) -> list[list[float]]:
    """Return a bounded slew-limited deterministic excitation on the 0.1 grid."""
    values = [0.0]
    targets = PRBS_TARGETS[phase_bin]
    for index in range(1, 201):
        t = index * CONTROL_INTERVAL
        if t >= 16.0:
            target = 0.0
        else:
            target = targets[min(int((t - 0.1) // 2.0), len(targets) - 1)]
        delta = max(-0.1, min(0.1, target - values[-1]))
        values.append(round(values[-1] + delta, 10))
    values[-1] = 0.0
    return [[round(i * CONTROL_INTERVAL, 10), value] for i, value in enumerate(values)]


def multisine_profile(phase_bin: int) -> list[list[float]]:
    """Return a fixed tapered multisine/chirp profile on the decision grid."""
    phase = MULTISINE_PHASE[phase_bin]
    rows = []
    for index in range(201):
        t = index * CONTROL_INTERVAL
        envelope = min(1.0, t / 1.0, (DURATION - t) / 1.0)
        chirp_phase = 2 * math.pi * (t / 7.5 + 0.5 * t * t / 300.0)
        value = envelope * (
            0.34 * math.sin(2 * math.pi * t / 6.15 + phase)
            + 0.20 * math.sin(2 * math.pi * t / 3.1 - 0.5 * phase)
            + 0.10 * math.sin(chirp_phase + 0.25 * phase)
        )
        rows.append([round(t, 10), round(value, 10)])
    rows[0][1] = rows[-1][1] = 0.0
    return rows


def validate_points(points: list[list[float]]) -> dict[str, float]:
    if len(points) != 201 or points[0] != [0.0, 0.0] or points[-1] != [20.0, 0.0]:
        raise ValueError("schedule must contain the complete 201-point decision grid")
    expected = [round(i * CONTROL_INTERVAL, 10) for i in range(201)]
    if [row[0] for row in points] != expected:
        raise ValueError("action schedule does not use the exact 0.1-D/U grid")
    deltas = [abs(b[1] - a[1]) for a, b in pairwise(points)]
    maximum = max(abs(row[1]) for row in points)
    if maximum > 0.75 + 1e-10 or max(deltas) > 0.1 + 1e-10:
        raise ValueError("action magnitude/rate exceeds full40 support")
    return {
        "max_abs_omega": maximum,
        "max_abs_delta_omega_per_0p1": max(deltas),
        "mean_omega": sum(row[1] for row in points) / len(points),
    }


def schedules() -> dict[int, dict[str, list[list[float]]]]:
    result = {}
    for phase_bin in PHASE_BINS:
        result[phase_bin] = {
            "prbs": prbs_profile(phase_bin),
            "multisine": multisine_profile(phase_bin),
        }
        for points in result[phase_bin].values():
            validate_points(points)
    return result


def load_contract() -> tuple[dict, dict[str, dict]]:
    if sha256(FULL40) != FULL40_SHA256:
        raise ValueError("full40 predeclaration SHA differs")
    full40 = json.loads(FULL40.read_text(encoding="utf-8"))
    profiles = schedules()
    planned = {}
    for phase_bin in PHASE_BINS:
        source_name = f"matched_start_acquisition_train_b{phase_bin:02d}_zero"
        source = full40["cases"].get(source_name)
        if source is None or source.get("split") != "train" or source.get("action_target") != 0.0:
            raise ValueError(f"train-only zero source absent: {source_name}")
        start = float(source["source_restart_time"])
        for field, digest in source["source_state_sha256"].items():
            if field not in STATE_FIELDS or sha256(SOURCE / f"{start:g}" / field) != digest:
                raise ValueError(f"source state differs: b{phase_bin:02d}/{field}")
        for profile, relative in profiles[phase_bin].items():
            name = f"dynamic_train8_b{phase_bin:02d}_{profile}"
            absolute = [[round(start + t, 10), omega] for t, omega in relative]
            planned[name] = {
                "split": "train",
                "phase_bin": phase_bin,
                "profile": profile,
                "source_restart_case": source["source_restart_case"],
                "source_restart_time": start,
                "source_state_sha256": source["source_state_sha256"],
                "action_points": absolute,
                "action_metrics": validate_points(relative),
                "run_window": [start, start + DURATION],
            }
    return full40, planned


def predeclaration() -> dict:
    full40, cases = load_contract()
    return {
        "status": "DYNAMIC_TRAIN8_PREDECLARED_NOT_EXECUTED",
        "scope": "train-only real OpenFOAM action-history acquisition",
        "selection_policy": "schedules frozen without reading b01/b05 validation or b03/b07 frozen outcomes",
        "full40_predeclaration_sha256": FULL40_SHA256,
        "phase_manifest_sha256": full40["phase_manifest_sha256"],
        "matrix": {"phase_bins": list(PHASE_BINS), "profiles": ["prbs", "multisine"], "case_count": 8},
        "cases": cases,
        "solver_contract": {
            "image": IMAGE,
            "delta_t": DELTA_T,
            "control_interval": CONTROL_INTERVAL,
            "field_write_interval": CONTROL_INTERVAL,
            "force_write_interval": DELTA_T,
            "expected_solver_steps": 4000,
            "expected_field_frames": 201,
            "maximum_total_raw_GiB": 30,
        },
        "execution_guards": {
            "spark_max_parallel_cases": 4,
            "minimum_MemAvailable_GiB": 40,
            "approval_token_required": GENERATION_TOKEN,
            "refuse_existing_case_or_output": True,
        },
        "curation_contract": {
            "official_api": "PhysicsNeMo Curator Source/Filter/Sink + run_pipeline",
            "split": "train_only",
            "expected_frames_per_case": 201,
            "force_channels": ["front_cd", "front_cl", "rear_cd", "rear_cl"],
            "normalization": "recompute from augmented train only after all eight HDF pass QC",
            "validation_or_frozen_access": "forbidden",
        },
    }


def write_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def stage(name: str, expected: dict) -> Path:
    target = CASES / name
    if target.exists():
        raise FileExistsError(f"refusing existing case: {target}")
    start, end = expected["run_window"]
    source = SOURCE / f"{start:g}"
    with tempfile.TemporaryDirectory(prefix=f".{name}.", dir=CASES) as directory:
        work = Path(directory) / name
        work.mkdir()
        shutil.copytree(SOURCE / "constant", work / "constant")
        shutil.copytree(SOURCE / "system", work / "system")
        shutil.copytree(source, work / f"{start:g}")
        shutil.copytree(source, work / "source_restart_provenance")
        velocity = work / f"{start:g}" / "U"
        velocity.write_text(replace_rear_patch(velocity.read_text(), expected["action_points"]))
        control_path = work / "system/controlDict"
        control = control_path.read_text()
        control = replace_once(control, "startTime 0;", f"startTime {start:g};", control_path)
        control = replace_once(control, "endTime 160;", f"endTime {end:g};", control_path)
        control = replace_once(control, "writeInterval 2;", "writeInterval 0.1;", control_path)
        control_path.write_text(control)
        config = dict(expected)
        config.update(
            case=name,
            panel="dynamic_train8_v1",
            predeclaration=str(PREDECLARATION.relative_to(REPO)),
            predeclaration_sha256=APPROVED_PREDECLARATION_SHA256,
            expected_solver_steps=4000,
            openfoam_image=IMAGE,
        )
        (work / "case_config.json").write_text(json.dumps(config, indent=2) + "\n")
        work.rename(target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-predeclaration", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--case")
    parser.add_argument("--approval-token")
    args = parser.parse_args()
    payload = predeclaration()
    if args.write_predeclaration:
        write_exclusive(PREDECLARATION, payload)
        print(PREDECLARATION)
        return
    if args.list:
        print("\n".join(payload["cases"]))
        return
    if args.approval_token != GENERATION_TOKEN or args.case not in payload["cases"]:
        parser.error("reviewed token and one declared train8 case are required")
    if sha256(PREDECLARATION) != APPROVED_PREDECLARATION_SHA256:
        raise ValueError("reviewed predeclaration SHA differs")
    print(stage(args.case, payload["cases"][args.case]))


if __name__ == "__main__":
    main()
