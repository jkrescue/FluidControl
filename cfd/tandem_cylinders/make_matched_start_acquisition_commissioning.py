#!/usr/bin/env python3
"""Atomically stage the frozen nine-case matched-start commissioning panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from make_expanded_control_dataset import replace_once, replace_rear_patch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
CASES = ROOT / "cases"
SOURCE = CASES / "tandem_backward_dt005"
EXPECTED_BINS = (0, 2, 4)
EXPECTED_ACTIONS = (-0.75, 0.0, 0.75)
STATE_FIELDS = ("U", "U_0", "p", "phi", "phi_0")
DURATION = 80.0
DELTA_T = 0.005
WRITE_INTERVAL = 0.1
RAMP_RATE = 1.0
IMAGE = (
    "opencfd/openfoam-default@sha256:"
    "33fb575aa9980d2bc42fd58c75ae698c489293ba30c991380fe3f899c622f319"
)
APPROVED_MANIFEST = (
    REPO
    / "artifacts/tandem_cylinders/"
    "matched_start_phase_restart_predeclared_v3_20261003.json"
)
APPROVED_MANIFEST_SHA256 = (
    "6279492bd3a79eff868a4333e1f642e3be4a4d58a78e46dc47dde040e4e39603"
)


@dataclass(frozen=True)
class CommissioningCase:
    name: str
    phase_bin: int
    source_time: float
    action: float
    source_u_sha256: str
    source_p_sha256: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def action_label(value: float) -> str:
    labels = {-0.75: "m075", 0.0: "zero", 0.75: "p075"}
    try:
        return labels[value]
    except KeyError as error:
        raise ValueError(f"unsupported commissioning action: {value}") from error


def case_name(phase_bin: int, action: float) -> str:
    return f"matched_start_acquisition_train_b{phase_bin:02d}_{action_label(action)}"


def action_points(start: float, action: float) -> list[tuple[float, float]]:
    end = start + DURATION
    if action == 0.0:
        return [(start, 0.0), (end, 0.0)]
    ramp_end = start + abs(action) / RAMP_RATE
    return [(start, 0.0), (ramp_end, action), (end, action)]


def load_panel(manifest_path: Path) -> tuple[dict, list[CommissioningCase]]:
    if manifest_path.resolve() != APPROVED_MANIFEST.resolve():
        raise ValueError("only the reviewed v3 phase manifest is accepted")
    actual_sha256 = sha256(manifest_path)
    if actual_sha256 != APPROVED_MANIFEST_SHA256:
        raise ValueError("phase manifest SHA-256 is not the reviewed value")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "MATCHED_START_PHASE_RESTART_AUDIT_COMPLETE":
        raise ValueError("phase manifest audit is not complete")
    if manifest.get("decision") != "GO_9_CASE_COMMISSIONING":
        raise ValueError("phase manifest does not authorize commissioning")
    commissioning = manifest.get("commissioning", {})
    if commissioning.get("phase_bins") != list(EXPECTED_BINS):
        raise ValueError("commissioning bins differ from the frozen contract")
    if commissioning.get("actions") != list(EXPECTED_ACTIONS):
        raise ValueError("commissioning actions differ from the frozen contract")
    if commissioning.get("case_count") != 9:
        raise ValueError("commissioning case count differs from nine")
    commissioning_rows = {
        row.get("phase_bin")
        for row in manifest.get("selections", [])
        if row.get("commissioning") is True
    }
    if commissioning_rows != set(EXPECTED_BINS):
        raise ValueError("manifest marks unexpected phase bins for commissioning")
    force_info = manifest.get("signal_qc", {})
    force_path = REPO / force_info.get("force_path", "")
    if not force_path.is_file() or force_info.get("force_sha256") != sha256(force_path):
        raise ValueError("raw phase-source force SHA-256 is missing or mismatched")

    selected = {}
    for row in manifest.get("selections", []):
        phase_bin = row.get("phase_bin")
        if phase_bin in EXPECTED_BINS:
            if row.get("split") != "train" or row.get("commissioning") is not True:
                raise ValueError(f"phase bin {phase_bin} is not a commissioning train bin")
            if row.get("coverage_pass") is not True:
                raise ValueError(f"phase bin {phase_bin} failed phase coverage")
            if phase_bin in selected:
                raise ValueError(f"duplicate selected phase bin {phase_bin}")
            selected[phase_bin] = row["selected"]
    if tuple(sorted(selected)) != EXPECTED_BINS:
        raise ValueError("manifest does not contain exactly the three commissioning bins")

    specs = []
    for phase_bin in EXPECTED_BINS:
        state = selected[phase_bin]
        start = float(state["restart_time"])
        source = SOURCE / f"{start:g}"
        if sha256(source / "U") != state["u_sha256"]:
            raise ValueError(f"selected U hash mismatch for phase bin {phase_bin}")
        if sha256(source / "p") != state["p_sha256"]:
            raise ValueError(f"selected p hash mismatch for phase bin {phase_bin}")
        names = {path.name for path in source.iterdir() if path.is_file()}
        if names != set(STATE_FIELDS):
            raise ValueError(f"incomplete real restart state at t={start:g}: {names}")
        for action in EXPECTED_ACTIONS:
            specs.append(
                CommissioningCase(
                    name=case_name(phase_bin, action),
                    phase_bin=phase_bin,
                    source_time=start,
                    action=action,
                    source_u_sha256=state["u_sha256"],
                    source_p_sha256=state["p_sha256"],
                )
            )
    return manifest, specs


def validate_action(points: list[tuple[float, float]]) -> dict[str, float]:
    if points[0][1] != 0.0:
        raise ValueError("action must start at zero")
    rates = []
    for (t0, a0), (t1, a1) in pairwise(points):
        if t1 <= t0 or not all(math.isfinite(value) for value in (t0, a0, t1, a1)):
            raise ValueError("action table is non-finite or non-monotone")
        rates.append(abs(a1 - a0) / (t1 - t0))
    metrics = {
        "max_abs_omega": max(abs(value) for _, value in points),
        "max_abs_domega_dt": max(rates),
    }
    if metrics["max_abs_omega"] > 0.75 + 1e-12 or metrics["max_abs_domega_dt"] > 1.0 + 1e-12:
        raise ValueError("action violates commissioning bounds")
    return metrics


def predeclaration(manifest: dict, specs: list[CommissioningCase]) -> dict:
    source_force_sha256 = {
        force: sha256(SOURCE / f"postProcessing/{force}/0/coefficient.dat")
        for force in ("forceFront", "forceRear")
    }
    cases = {}
    for spec in specs:
        source = SOURCE / f"{spec.source_time:g}"
        points = action_points(spec.source_time, spec.action)
        cases[spec.name] = {
            "phase_bin": spec.phase_bin,
            "split": "train",
            "source_restart_time": spec.source_time,
            "source_state_sha256": {
                field: sha256(source / field) for field in STATE_FIELDS
            },
            "action_target": spec.action,
            "action_points": points,
            "action_metrics": validate_action(points),
            "run_window": [spec.source_time, spec.source_time + DURATION],
            "analysis_window": [spec.source_time + 20.0, spec.source_time + DURATION],
        }
    return {
        "status": "MATCHED_START_9_CASE_COMMISSIONING_PREDECLARED_NO_CASES_GENERATED",
        "scope": "three train phase bins by three actions; pipeline commissioning only",
        "phase_manifest": str(APPROVED_MANIFEST.relative_to(REPO)),
        "phase_manifest_sha256": APPROVED_MANIFEST_SHA256,
        "phase_force_source_sha256": manifest["signal_qc"]["force_sha256"],
        "baseline_force_source_sha256": source_force_sha256,
        "cases": cases,
        "solver_contract": {
            "image": IMAGE,
            "network": "none",
            "worker": "CPU-only temporary solve; all raw output returned to Spark",
            "delta_t": DELTA_T,
            "duration": DURATION,
            "field_write_interval": WRITE_INTERVAL,
            "force_write_interval": DELTA_T,
            "expected_solver_steps_per_case": 16000,
            "expected_field_frames_per_case": 801,
            "expected_raw_solver_force_samples_per_case": 16000,
            "expected_aligned_force_samples_with_source_t0_per_case": 16001,
            "source_restart_provenance_fields_per_case": list(STATE_FIELDS),
        },
        "fail_closed_qc": {
            "solver_end_required": True,
            "max_courant_strictly_below": 0.3,
            "max_abs_global_continuity_per_step_strictly_below": 1e-9,
            "worker_to_spark_all_raw_file_sha256_required": True,
            "same_phase_three_branch_five_field_source_hashes_required": True,
            "retain_failed_cases": True,
        },
        "resource_budget": {
            "maximum_parallel_cases": 2,
            "container_limits_per_case": {"cpus": 4, "memory_gib": 8},
            "minimum_worker_mem_available_gib": 40,
            "estimated_worker_raw_gib": 18,
            "estimated_spark_raw_plus_vtk_gib": 38,
            "estimated_solver_wall_minutes": [33, 40],
        },
        "stop_rule": "commission only these nine cases; do not generate the 40-case matrix or train without a separate reviewed authorization",
    }


def generate(spec: CommissioningCase, manifest_path: Path) -> Path:
    target = CASES / spec.name
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    source_time = SOURCE / f"{spec.source_time:g}"
    source_hashes = {field: sha256(source_time / field) for field in STATE_FIELDS}
    if source_hashes["U"] != spec.source_u_sha256 or source_hashes["p"] != spec.source_p_sha256:
        raise ValueError("source state changed after manifest validation")
    points = action_points(spec.source_time, spec.action)
    metrics = validate_action(points)
    with tempfile.TemporaryDirectory(prefix=f".{spec.name}.", dir=CASES) as temporary:
        stage = Path(temporary) / spec.name
        stage.mkdir()
        shutil.copytree(SOURCE / "constant", stage / "constant")
        shutil.copytree(SOURCE / "system", stage / "system")
        shutil.copytree(source_time, stage / f"{spec.source_time:g}")
        shutil.copytree(source_time, stage / "source_restart_provenance")
        velocity_path = stage / f"{spec.source_time:g}" / "U"
        velocity_path.write_text(
            replace_rear_patch(velocity_path.read_text(encoding="utf-8"), points),
            encoding="utf-8",
        )
        control_path = stage / "system/controlDict"
        control = control_path.read_text(encoding="utf-8")
        control = replace_once(
            control, "startTime 0;", f"startTime {spec.source_time:g};", control_path
        )
        control = replace_once(
            control,
            "endTime 160;",
            f"endTime {spec.source_time + DURATION:g};",
            control_path,
        )
        control = replace_once(
            control, "writeInterval 2;", "writeInterval 0.1;", control_path
        )
        control_path.write_text(control, encoding="utf-8")
        config = {
            "case": spec.name,
            "panel": "matched_start_acquisition_commissioning_v1",
            "split": "train",
            "phase_bin": spec.phase_bin,
            "source_restart_case": SOURCE.name,
            "source_restart_time": spec.source_time,
            "source_state_sha256": source_hashes,
            "source_state_provenance_dir": "source_restart_provenance",
            "source_force_sha256": {
                force: sha256(SOURCE / f"postProcessing/{force}/0/coefficient.dat")
                for force in ("forceFront", "forceRear")
            },
            "phase_manifest": str(manifest_path),
            "phase_manifest_sha256": APPROVED_MANIFEST_SHA256,
            "action_target": spec.action,
            "action_points": points,
            "action_metrics": metrics,
            "start_time": spec.source_time,
            "end_time": spec.source_time + DURATION,
            "analysis_window": [spec.source_time + 20.0, spec.source_time + DURATION],
            "delta_t": DELTA_T,
            "expected_solver_steps": 16000,
            "field_write_interval": WRITE_INTERVAL,
            "expected_field_frames": 801,
            "force_write_interval": DELTA_T,
            "expected_raw_solver_force_samples": 16000,
            "expected_aligned_force_samples_with_source_t0": 16001,
            "openfoam_image": IMAGE,
            "compute_guard": "worker CPU only; network none; raw output copied to Spark before curation",
            "interpretation_guard": "commissioning pipeline evidence only; not training or a physical control claim",
        }
        (stage / "case_config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
        stage.rename(target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-manifest", required=True, type=Path)
    parser.add_argument("--confirm-generate", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--predeclaration-output", type=Path)
    args = parser.parse_args()
    manifest, specs = load_panel(args.phase_manifest)
    if args.predeclaration_output:
        if args.predeclaration_output.exists():
            parser.error(f"refusing to overwrite {args.predeclaration_output}")
        args.predeclaration_output.parent.mkdir(parents=True, exist_ok=True)
        args.predeclaration_output.write_text(
            json.dumps(predeclaration(manifest, specs), indent=2) + "\n",
            encoding="utf-8",
        )
    if args.list:
        for spec in specs:
            print(spec.name)
        return
    if not args.confirm_generate:
        if args.predeclaration_output:
            print(args.predeclaration_output)
            return
        parser.error("--confirm-generate is required; no cases were generated")
    if any((CASES / spec.name).exists() for spec in specs):
        parser.error("one or more commissioning cases already exist")
    for spec in specs:
        print(generate(spec, args.phase_manifest))


if __name__ == "__main__":
    main()
