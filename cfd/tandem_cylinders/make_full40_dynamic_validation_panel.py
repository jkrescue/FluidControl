#!/usr/bin/env python3
"""Predeclare or stage the frozen-blind full40 dynamic validation panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
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
PREDECLARATION = REPO / "artifacts/tandem_cylinders/full40_dynamic_validation_predeclared_20261003.json"
APPROVED_PREDECLARATION_SHA256 = (
    "0478c8532bd2ded504ccd5f89303001eb8359f69b3036f296e31428a085d1272"
)
GENERATION_TOKEN = "GENERATE_REVIEWED_FULL40_DYNAMIC_VALIDATION"
IMAGE = (
    "opencfd/openfoam-default@sha256:"
    "33fb575aa9980d2bc42fd58c75ae698c489293ba30c991380fe3f899c622f319"
)
STATE_FIELDS = ("U", "U_0", "p", "phi", "phi_0")
DURATION = 20.0
DELTA_T = 0.005
WRITE_INTERVAL = 0.1

# Times are elapsed D/U.  Every knot is on the 0.1-D/U decision grid.  Linear
# ramps therefore also obey the per-decision action increment bound.
POSITIVE_PROFILE = (
    (0.0, 0.0),
    (0.8, 0.75),
    (2.0, 0.75),
    (3.5, -0.75),
    (4.7, -0.75),
    (5.5, 0.0),
    (6.5, 0.0),
    (6.9, 0.375),
    (8.0, 0.375),
    (8.8, -0.375),
    (9.9, -0.375),
    (10.3, 0.0),
    (11.5, 0.0),
    (12.3, 0.75),
    (13.5, 0.75),
    (15.0, -0.75),
    (16.2, -0.75),
    (17.0, 0.0),
    (20.0, 0.0),
)
PROFILES = {
    "plus": POSITIVE_PROFILE,
    "minus": tuple((time, -omega) for time, omega in POSITIVE_PROFILE),
    "zero": ((0.0, 0.0), (DURATION, 0.0)),
}


@dataclass(frozen=True)
class Case:
    name: str
    phase_bin: int
    source_time: float
    source_hashes: dict[str, str]
    role: str
    points: tuple[tuple[float, float], ...]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_points(points: list[list[float]]) -> dict[str, float]:
    if points[0][1] != 0.0 or points[-1][1] != 0.0:
        raise ValueError("schedule must begin and finish at zero")
    if any(abs(time * 10 - round(time * 10)) > 1e-9 for time, _ in points):
        raise ValueError("schedule knots must be on the 0.1-D/U decision grid")
    rates = []
    for (t0, a0), (t1, a1) in pairwise(points):
        if t1 <= t0:
            raise ValueError("action timestamps must increase")
        rates.append(abs(a1 - a0) / (t1 - t0))
    result = {
        "max_abs_omega": max(abs(row[1]) for row in points),
        "max_abs_delta_omega_per_0p1": max(rates, default=0.0) * 0.1,
        "max_abs_domega_dt": max(rates, default=0.0),
    }
    if result["max_abs_omega"] > 0.75 + 1e-12:
        raise ValueError("action magnitude exceeds full40 support")
    if result["max_abs_delta_omega_per_0p1"] > 0.1 + 1e-12:
        raise ValueError("action-rate bound exceeds full40 support")
    return result


def load_cases() -> tuple[dict, list[Case]]:
    if sha256(FULL40) != FULL40_SHA256:
        raise ValueError("full40 predeclaration SHA differs")
    full40 = json.loads(FULL40.read_text(encoding="utf-8"))
    cases = []
    for phase_bin in (1, 5):
        zero_name = f"matched_start_acquisition_validation_b{phase_bin:02d}_zero"
        source = full40.get("cases", {}).get(zero_name)
        if not source or source.get("split") != "validation":
            raise ValueError(f"validation source is absent: b{phase_bin:02d}")
        start = float(source["source_restart_time"])
        hashes = source["source_state_sha256"]
        for field in STATE_FIELDS:
            path = SOURCE / f"{start:g}" / field
            if sha256(path) != hashes[field]:
                raise ValueError(f"source state differs: b{phase_bin:02d}/{field}")
        for role, relative in PROFILES.items():
            points = tuple((round(start + t, 10), omega) for t, omega in relative)
            cases.append(
                Case(
                    name=f"full40_dynamic_validation_b{phase_bin:02d}_{role}",
                    phase_bin=phase_bin,
                    source_time=start,
                    source_hashes=hashes,
                    role=role,
                    points=points,
                )
            )
    return full40, cases


def predeclaration(full40: dict, cases: list[Case]) -> dict:
    planned = {}
    for case in cases:
        points = [[time, omega] for time, omega in case.points]
        planned[case.name] = {
            "split": "validation",
            "phase_bin": case.phase_bin,
            "profile": case.role,
            "paired_zero": f"full40_dynamic_validation_b{case.phase_bin:02d}_zero",
            "source_restart_case": SOURCE.name,
            "source_restart_time": case.source_time,
            "source_state_sha256": case.source_hashes,
            "action_points": points,
            "action_metrics": validate_points(points),
            "run_window": [case.source_time, case.source_time + DURATION],
            "fixed_force_windows_elapsed": [[0.0, 10.0], [10.0, 20.0], [0.0, 20.0]],
            "horizon_start_frames": [0, 20, 47, 65, 80, 100],
        }
    return {
        "status": "FULL40_DYNAMIC_VALIDATION_PREDECLARED_NOT_EXECUTED",
        "scope": "real OpenFOAM time-varying-action validation at train-excluded b01/b05 phases",
        "full40_predeclaration": str(FULL40.relative_to(REPO)),
        "full40_predeclaration_sha256": FULL40_SHA256,
        "phase_manifest_sha256": full40["phase_manifest_sha256"],
        "frozen_test_accessed": False,
        "matrix": {
            "phase_bins": [1, 5],
            "profiles": ["minus", "zero", "plus"],
            "case_count": 6,
            "duration_D_over_U": DURATION,
        },
        "cases": planned,
        "solver_contract": {
            "image": IMAGE,
            "delta_t": DELTA_T,
            "field_write_interval": WRITE_INTERVAL,
            "force_write_interval": DELTA_T,
            "expected_solver_steps": 4000,
            "expected_field_frames": 201,
            "expected_raw_force_samples": 4000,
        },
        "fno_validation_contract": {
            "horizons_frames": [1, 10, 50, 100],
            "strict_counterfactual_pairs": "only start_frame=0 within each phase",
            "later_starts": "matched elapsed time but action-diverged states; diagnostic only",
            "force_channels": ["front_cd", "front_cl", "rear_cd", "rear_cl"],
            "terminal_total_cd_pooled_nrmse_max": 0.10,
            "strict_start0_delta_total_cd_mae_max": 0.023,
            "fixed_window_total_cd_ranking": "report exact order for minus/zero/plus at both phases",
            "interpretation": "development validation only; not a formal frozen-test Gate or PPO benefit claim",
        },
        "execution_guards": {
            "one_case_per_invocation": True,
            "existing_case_or_output_refused": True,
            "minimum_MemAvailable_GiB": 40,
            "do_not_run_with_Curator_or_training": True,
            "approval_token_required": GENERATION_TOKEN,
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


def stage(case: Case, reviewed: dict) -> Path:
    target = CASES / case.name
    if target.exists():
        raise FileExistsError(f"refusing existing case: {target}")
    expected = reviewed["cases"].get(case.name)
    if expected != predeclaration(*load_cases())["cases"][case.name]:
        raise ValueError("reviewed case contract differs")
    source = SOURCE / f"{case.source_time:g}"
    with tempfile.TemporaryDirectory(prefix=f".{case.name}.", dir=CASES) as temporary:
        work = Path(temporary) / case.name
        work.mkdir()
        shutil.copytree(SOURCE / "constant", work / "constant")
        shutil.copytree(SOURCE / "system", work / "system")
        shutil.copytree(source, work / f"{case.source_time:g}")
        shutil.copytree(source, work / "source_restart_provenance")
        velocity = work / f"{case.source_time:g}" / "U"
        velocity.write_text(
            replace_rear_patch(velocity.read_text(encoding="utf-8"), list(case.points)),
            encoding="utf-8",
        )
        control_path = work / "system/controlDict"
        control = control_path.read_text(encoding="utf-8")
        control = replace_once(control, "startTime 0;", f"startTime {case.source_time:g};", control_path)
        control = replace_once(control, "endTime 160;", f"endTime {case.source_time + DURATION:g};", control_path)
        control = replace_once(control, "writeInterval 2;", "writeInterval 0.1;", control_path)
        control_path.write_text(control, encoding="utf-8")
        config = dict(expected)
        config.update(
            case=case.name,
            panel="full40_dynamic_validation_v1",
            predeclaration=str(PREDECLARATION.relative_to(REPO)),
            predeclaration_sha256=APPROVED_PREDECLARATION_SHA256,
            start_time=case.source_time,
            end_time=case.source_time + DURATION,
            expected_solver_steps=4000,
            openfoam_image=IMAGE,
        )
        (work / "case_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        work.rename(target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-predeclaration", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--case")
    parser.add_argument("--approval-token")
    args = parser.parse_args()
    full40, cases = load_cases()
    payload = predeclaration(full40, cases)
    if args.write_predeclaration:
        write_exclusive(PREDECLARATION, payload)
        print(PREDECLARATION)
        return
    if args.list:
        for case in cases:
            print(case.name)
        return
    if args.approval_token != GENERATION_TOKEN or not args.case:
        parser.error("reviewed approval token and exactly one case are required")
    if len(APPROVED_PREDECLARATION_SHA256) != 64 or sha256(PREDECLARATION) != APPROVED_PREDECLARATION_SHA256:
        raise ValueError("dynamic validation predeclaration is not reviewed/hard-bound")
    reviewed = json.loads(PREDECLARATION.read_text(encoding="utf-8"))
    if reviewed != payload:
        raise ValueError("reviewed dynamic validation contract differs")
    by_name = {case.name: case for case in cases}
    if args.case not in by_name:
        parser.error("case is outside the reviewed dynamic validation panel")
    print(stage(by_name[args.case], reviewed))


if __name__ == "__main__":
    main()
