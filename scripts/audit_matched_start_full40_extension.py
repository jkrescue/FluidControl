#!/usr/bin/env python3
"""Fail-closed raw-solver QC for one authorized full40 remainder case."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CFD = REPO / "cfd/tandem_cylinders"
CASES = CFD / "cases"
sys.path.insert(0, str(CFD))
from analyze_baseline import log_health

PREDECLARATION = (
    REPO / "artifacts/tandem_cylinders/matched_start_full40_predeclared_20261003.json"
)
PREDECLARATION_SHA256 = "d7ff174ef10194a8739357376335ca13ff9b45c8079970846bb71f15a715d24b"
AUTHORIZATION_SHA256 = "da8bccaf18a86666ac78804e775c1608d8fc390bfca1484cbd1eaabe93c17151"
AUTHORIZATION = (
    REPO / "artifacts/tandem_cylinders/"
    "matched_start_full40_extension_authorized_20261003.json"
)
STATE_FIELDS = ("U", "U_0", "p", "phi", "phi_0")


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def load_authorization(authorization_sha256: str) -> dict:
    if len(authorization_sha256) != 64 or sha256(AUTHORIZATION) != authorization_sha256:
        raise ValueError("extension authorization SHA-256 differs")
    data = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    baseline = data.get("baseline_force_source_sha256")
    if data.get("status") != "MATCHED_START_FULL40_EXTENSION_AUTHORIZED" or not (
        isinstance(baseline, dict) and set(baseline) == {"forceFront", "forceRear"}
    ):
        raise ValueError("extension authorization content differs")
    return data


def parse_rear_omega_table(path: Path) -> list[list[float]]:
    text = path.read_text(encoding="utf-8")
    patch = re.search(r"rearCylinder\s*\{(?P<body>.*?)\n\s*\}", text, re.DOTALL)
    if not patch:
        raise ValueError(f"rearCylinder patch missing: {path}")
    table = re.search(
        r"omega\s+table\s*\((?P<rows>.*?)\)\s*;", patch.group("body"), re.DOTALL
    )
    if not table:
        raise ValueError(f"rearCylinder omega table missing: {path}")
    return [
        [float(left), float(right)]
        for left, right in re.findall(
            r"\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)", table.group("rows")
        )
    ]


def planned_cases() -> dict[str, dict]:
    if sha256(PREDECLARATION) != PREDECLARATION_SHA256:
        raise ValueError("full40 predeclaration SHA-256 differs")
    data = json.loads(PREDECLARATION.read_text(encoding="utf-8"))
    selected = {
        name: row
        for name, row in data["cases"].items()
        if row["disposition"] == "planned_new_remainder_case"
    }
    if len(selected) != 31:
        raise ValueError("full40 remainder is not exactly 31 cases")
    return selected


def validate_case_config(name: str, config: dict, authorization_sha256: str) -> dict:
    expected = planned_cases().get(name)
    if expected is None:
        raise ValueError("case is not in the 31-case remainder")
    checks = {
        "case": name,
        "panel": "matched_start_acquisition_full40_v1",
        "split": expected["split"],
        "phase_bin": expected["phase_bin"],
        "source_restart_time": expected["source_restart_time"],
        "source_state_sha256": expected["source_state_sha256"],
        "action_target": expected["action_target"],
        "action_points": expected["action_points"],
        "start_time": expected["run_window"][0],
        "end_time": expected["run_window"][1],
        "analysis_window": expected["analysis_window"],
        "full40_predeclaration_sha256": PREDECLARATION_SHA256,
        "full40_extension_authorization_sha256": authorization_sha256,
    }
    for key, value in checks.items():
        if config.get(key) != value:
            raise ValueError(f"case config field differs: {name}/{key}")
    if config.get("expected_solver_steps") != 16000:
        raise ValueError("solver step contract differs")
    return expected


def exact_grid(time: np.ndarray, begin: float, end: float, step: float) -> None:
    expected = np.linspace(begin, end, round((end - begin) / step) + 1)
    if len(time) != len(expected) or not np.allclose(time, expected, rtol=0, atol=1e-8):
        raise ValueError(f"incomplete fixed grid [{begin}, {end}] at dt={step}")


def read_force_rows(case: Path, force_name: str) -> np.ndarray:
    samples = {}
    for path in sorted(case.glob(f"postProcessing/{force_name}/*/coefficient.dat")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            row = np.asarray([float(value) for value in line.split()])
            if len(row) < 8 or not np.isfinite(row).all():
                raise ValueError(f"invalid force row: {path}")
            key = round(float(row[0]), 8)
            if key in samples and not np.allclose(samples[key], row, rtol=0, atol=1e-10):
                raise ValueError(f"conflicting force row at t={key}")
            samples[key] = row
    if not samples:
        raise FileNotFoundError(f"missing force output: {case}/{force_name}")
    return np.asarray([samples[key] for key in sorted(samples)])


def numeric_times(case: Path) -> np.ndarray:
    values = []
    for path in case.iterdir():
        if path.is_dir():
            try:
                values.append(float(path.name))
            except ValueError:
                pass
    return np.asarray(sorted(values))


def parse_manifest(path: Path) -> dict[str, str]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if separator != "  " or len(digest) != 64 or not relative or relative in rows:
            raise ValueError("invalid or duplicate worker SHA-256 manifest row")
        rows[relative] = digest
    if not rows:
        raise ValueError("empty worker SHA-256 manifest")
    return rows


def local_hashes(case: Path) -> dict[str, str]:
    return {
        str(path.relative_to(CASES)): sha256(path)
        for path in sorted(case.rglob("*"))
        if path.is_file()
        and not any(part.startswith(".matched_start_") for part in path.parts)
    }


def audit_case(name: str, worker_manifest: Path, authorization_sha256: str) -> dict:
    authorization = load_authorization(authorization_sha256)
    case = CASES / name
    config = json.loads((case / "case_config.json").read_text(encoding="utf-8"))
    validate_case_config(name, config, authorization_sha256)
    actual_action = parse_rear_omega_table(
        case / f"{config['source_restart_time']:g}" / "U"
    )
    if actual_action != config["action_points"]:
        raise ValueError(f"OpenFOAM action table differs: {name}")
    provenance = case / config["source_state_provenance_dir"]
    actual_source = {field: sha256(provenance / field) for field in STATE_FIELDS}
    if actual_source != config["source_state_sha256"]:
        raise ValueError(f"source-state provenance differs: {name}")
    staged = case / f"{config['source_restart_time']:g}"
    for field in STATE_FIELDS:
        if field != "U" and sha256(staged / field) != actual_source[field]:
            raise ValueError(f"staged source state differs: {name}/{field}")
    expected_force = authorization["baseline_force_source_sha256"]
    if config.get("source_force_sha256") != expected_force:
        raise ValueError(f"case baseline force contract differs: {name}")
    for force_name, expected_sha in expected_force.items():
        source = CASES / config["source_restart_case"]
        path = source / f"postProcessing/{force_name}/0/coefficient.dat"
        if sha256(path) != expected_sha:
            raise ValueError(f"baseline force provenance differs: {force_name}")
        source_rows = read_force_rows(source, force_name)
        matches = np.flatnonzero(
            np.isclose(source_rows[:, 0], float(config["start_time"]), rtol=0, atol=1e-8)
        )
        if len(matches) != 1:
            raise ValueError(f"baseline force has no unique source t0: {force_name}")
    begin, end = float(config["start_time"]), float(config["end_time"])
    exact_grid(numeric_times(case), begin, end, 0.1)
    for force in ("forceFront", "forceRear"):
        rows = read_force_rows(case, force)
        exact_grid(rows[:, 0], begin + 0.005, end, 0.005)
    health = log_health(case / "log.pimpleFoam.matched_start_full40")
    if (
        not health["solver_ended_cleanly"]
        or health["steps"] != 16000
        or health["max_courant"] >= 0.3
        or health["max_abs_global_continuity_per_step"] >= 1e-9
    ):
        raise ValueError(f"solver health failed: {name}")
    marker = json.loads((case / "solver_complete.full40.json").read_text(encoding="utf-8"))
    if (
        marker.get("status") != "FULL40_SOLVER_COMPLETED_PENDING_TRANSFER_QC"
        or marker.get("case") != name
        or marker.get("full40_extension_authorization_sha256") != authorization_sha256
        or marker.get("solver_steps") != 16000
    ):
        raise ValueError(f"solver completion marker differs: {name}")
    if marker.get("solver_log_sha256") != sha256(
        case / "log.pimpleFoam.matched_start_full40"
    ) or marker.get("solver_log_qc_sha256") != sha256(
        case / "solver_log_qc.full40.json"
    ):
        raise ValueError(f"solver completion marker hashes differ: {name}")
    worker = parse_manifest(worker_manifest)
    local = local_hashes(case)
    if worker != local:
        raise ValueError(f"worker/Spark raw file SHA-256 differs: {name}")
    return {
        "status": "FULL40_RAW_TRANSFER_VERIFIED",
        "case": name,
        "split": config["split"],
        "phase_bin": config["phase_bin"],
        "action_target": config["action_target"],
        "full40_predeclaration_sha256": PREDECLARATION_SHA256,
        "full40_extension_authorization_sha256": authorization_sha256,
        "worker_raw_manifest": str(worker_manifest.relative_to(REPO)),
        "worker_raw_manifest_sha256": sha256(worker_manifest),
        "raw_file_count": len(local),
        "coverage": "raw OpenFOAM solver files only; VTK is not included",
        "next_required_stage": "SPARK_PINNED_FOAMTOVTK_801_FRAMES",
        "solver_health": health,
        "source_state_sha256": actual_source,
        "baseline_force_source_sha256": expected_force,
    }


def aggregate(receipt_dir: Path) -> dict:
    planned = planned_cases()
    rows = []
    for name, expected in planned.items():
        path = receipt_dir / f"{name}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        required = {
            "status": "FULL40_RAW_TRANSFER_VERIFIED",
            "case": name,
            "split": expected["split"],
            "phase_bin": expected["phase_bin"],
            "action_target": expected["action_target"],
            "full40_predeclaration_sha256": PREDECLARATION_SHA256,
            "full40_extension_authorization_sha256": AUTHORIZATION_SHA256,
        }
        if any(data.get(key) != value for key, value in required.items()):
            raise ValueError(f"strict receipt differs: {name}")
        rows.append(data)
    counts = {
        split: sum(row["split"] == split for row in rows)
        for split in ("train", "validation", "frozen_test")
    }
    if counts != {"train": 11, "validation": 10, "frozen_test": 10}:
        raise ValueError("remainder split counts differ")
    for phase_bin in range(8):
        group = [row for row in rows if row["phase_bin"] == phase_bin]
        if group and len({json.dumps(row["source_state_sha256"], sort_keys=True) for row in group}) != 1:
            raise ValueError(f"phase b{phase_bin:02d} source states differ")
    return {
        "status": "MATCHED_START_FULL40_EXTENSION_31_CASE_RAW_QC_PASS",
        "case_count": 31,
        "split_counts": counts,
        "full40_predeclaration_sha256": PREDECLARATION_SHA256,
        "full40_extension_authorization_sha256": AUTHORIZATION_SHA256,
        "frozen_guard": "integrity QC only; no reward, drag summary, or model selection",
    }


def write_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(payload, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise ValueError(f"refusing to overwrite {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--worker-raw-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if len(AUTHORIZATION_SHA256) != 64:
        parser.error("extension authorization is not committed and SHA-bound")
    result = audit_case(args.case, args.worker_raw_manifest, AUTHORIZATION_SHA256)
    write_exclusive(args.output, result)
    print(json.dumps({"status": result["status"], "case": args.case}, indent=2))


if __name__ == "__main__":
    main()
