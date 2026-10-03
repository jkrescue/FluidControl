#!/usr/bin/env python3
"""Fail-closed QC for the nine-case matched-start commissioning panel."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CFD = REPO / "cfd/tandem_cylinders"
CASES = CFD / "cases"
sys.path.insert(0, str(CFD))
from analyze_baseline import log_health
from make_matched_start_acquisition_commissioning import (
    APPROVED_MANIFEST,
    APPROVED_MANIFEST_SHA256,
    EXPECTED_ACTIONS,
    EXPECTED_BINS,
    SOURCE,
    STATE_FIELDS,
    case_name,
    sha256,
)


def exact_time_grid(time: np.ndarray, begin: float, end: float, step: float) -> None:
    expected_count = round((end - begin) / step) + 1
    expected = np.linspace(begin, end, expected_count)
    if len(time) != expected_count:
        raise ValueError(f"expected {expected_count} timestamps, got {len(time)}")
    if not np.allclose(time, expected, rtol=0, atol=1e-8):
        raise ValueError("timestamps do not match the complete fixed grid")


def read_force_rows(case: Path, force_name: str) -> np.ndarray:
    samples = {}
    paths = sorted(case.glob(f"postProcessing/{force_name}/*/coefficient.dat"))
    if not paths:
        raise FileNotFoundError(f"missing {force_name} output: {case}")
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            values = np.asarray([float(value) for value in fields])
            if len(values) < 5 or not np.isfinite(values).all():
                raise ValueError(f"invalid force record: {path}")
            key = round(float(values[0]), 8)
            if key in samples and not np.allclose(samples[key], values, rtol=0, atol=1e-10):
                raise ValueError(f"conflicting force restart record at t={key}")
            samples[key] = values
    return np.asarray([samples[key] for key in sorted(samples)], dtype=float)


def aligned_force_rows(
    raw_rows: np.ndarray,
    source_rows: np.ndarray,
    begin: float,
    end: float,
) -> np.ndarray:
    exact_time_grid(raw_rows[:, 0], begin + 0.005, end, 0.005)
    selected = source_rows[np.isclose(source_rows[:, 0], begin, rtol=0, atol=1e-9)]
    if len(selected) != 1:
        raise ValueError(f"expected one source force row at t={begin}, got {len(selected)}")
    if selected.shape[1] != raw_rows.shape[1]:
        raise ValueError("source and raw force columns differ")
    aligned = np.vstack((selected, raw_rows))
    exact_time_grid(aligned[:, 0], begin, end, 0.005)
    return aligned


def numeric_output_times(case: Path) -> np.ndarray:
    times = []
    for path in case.iterdir():
        if not path.is_dir():
            continue
        try:
            times.append(float(path.name))
        except ValueError:
            continue
    return np.asarray(sorted(times), dtype=float)


def parse_sha256_manifest(path: Path) -> dict[str, str]:
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if separator != "  " or len(digest) != 64 or not relative:
            raise ValueError(f"invalid SHA-256 manifest row: {line}")
        if relative in records:
            raise ValueError(f"duplicate SHA-256 manifest path: {relative}")
        records[relative] = digest
    if not records:
        raise ValueError("empty worker raw manifest")
    return records


def local_raw_hashes(case_names: list[str]) -> dict[str, str]:
    records = {}
    for name in case_names:
        case = CASES / name
        for path in sorted(
            item
            for item in case.rglob("*")
            if item.is_file() and ".matched_start_solver_lock" not in item.parts
        ):
            relative = str(path.relative_to(CASES))
            records[relative] = sha256(path)
    return records


def write_json_exclusive_atomic(path: Path, payload: dict) -> None:
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


def audit_case(name: str, phase_bin: int, action: float) -> dict:
    case = CASES / name
    config = json.loads((case / "case_config.json").read_text(encoding="utf-8"))
    if config["case"] != name or config["phase_bin"] != phase_bin:
        raise ValueError(f"case identity mismatch: {name}")
    if config["action_target"] != action or config["split"] != "train":
        raise ValueError(f"case action/split mismatch: {name}")
    if config["phase_manifest_sha256"] != APPROVED_MANIFEST_SHA256:
        raise ValueError(f"case was not generated from approved v3 manifest: {name}")
    velocity = (case / f"{config['source_restart_time']:g}" / "U").read_text(
        encoding="utf-8"
    )
    for time, omega in config["action_points"]:
        if f"({time:.10g} {omega:.10g})" not in velocity:
            raise ValueError(f"OpenFOAM action table differs from case config: {name}")
    source = CASES / config["source_restart_case"] / f"{config['source_restart_time']:g}"
    actual_state = {field: sha256(source / field) for field in STATE_FIELDS}
    if config["source_state_sha256"] != actual_state:
        raise ValueError(f"source restart hash mismatch: {name}")
    provenance = case / config["source_state_provenance_dir"]
    provenance_state = {field: sha256(provenance / field) for field in STATE_FIELDS}
    if provenance_state != actual_state:
        raise ValueError(f"staged source restart provenance differs: {name}")
    begin, end = config["start_time"], config["end_time"]
    exact_time_grid(numeric_output_times(case), begin, end, 0.1)
    source_force = {}
    aligned_force = {}
    for force_name in ("forceFront", "forceRear"):
        source_path = SOURCE / f"postProcessing/{force_name}/0/coefficient.dat"
        if config["source_force_sha256"][force_name] != sha256(source_path):
            raise ValueError(f"source force provenance differs: {name}/{force_name}")
        source_rows = read_force_rows(SOURCE, force_name)
        raw_rows = read_force_rows(case, force_name)
        aligned_force[force_name] = aligned_force_rows(raw_rows, source_rows, begin, end)
        source_force[force_name] = {
            "path": str(source_path.relative_to(REPO)),
            "sha256": sha256(source_path),
            "source_t0_row": aligned_force[force_name][0].tolist(),
        }
    if not np.array_equal(
        aligned_force["forceFront"][:, 0], aligned_force["forceRear"][:, 0]
    ):
        raise ValueError(f"front/rear force times differ: {name}")
    health = log_health(case / "log.pimpleFoam.matched_start_acquisition")
    if not health["solver_ended_cleanly"] or health["steps"] != 16000:
        raise ValueError(f"solver completion failed: {name}")
    if health["max_courant"] >= 0.3:
        raise ValueError(f"Courant guard failed: {name}")
    if health["max_abs_global_continuity_per_step"] >= 1e-9:
        raise ValueError(f"continuity guard failed: {name}")
    marker = json.loads((case / "solver_complete.json").read_text(encoding="utf-8"))
    if (
        marker.get("status") != "SOLVER_COMPLETED_PENDING_TRANSFER_QC"
        or marker.get("case") != name
        or marker.get("phase_manifest_sha256") != APPROVED_MANIFEST_SHA256
        or marker.get("solver_steps") != 16000
    ):
        raise ValueError(f"invalid solver completion marker: {name}")
    if marker.get("solver_log_sha256") != sha256(
        case / "log.pimpleFoam.matched_start_acquisition"
    ) or marker.get("solver_log_qc_sha256") != sha256(case / "solver_log_qc.json"):
        raise ValueError(f"solver completion marker hashes differ: {name}")
    return {
        "case": name,
        "phase_bin": phase_bin,
        "action_target": action,
        "source_restart_time": config["source_restart_time"],
        "source_state_sha256": actual_state,
        "field_frames": len(numeric_output_times(case)),
        "raw_solver_force_samples": 16000,
        "aligned_force_samples": len(aligned_force["forceFront"]),
        "source_force_provenance": source_force,
        "solver_health": health,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-raw-manifest-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt-dir", required=True, type=Path)
    parser.add_argument("--case", action="append", dest="selected_cases")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    if sha256(APPROVED_MANIFEST) != APPROVED_MANIFEST_SHA256:
        raise ValueError("approved phase manifest changed")

    expected = {
        case_name(phase_bin, action): (phase_bin, action)
        for phase_bin in EXPECTED_BINS
        for action in EXPECTED_ACTIONS
    }
    selected_names = args.selected_cases or list(expected)
    if len(selected_names) != len(set(selected_names)) or not set(selected_names) <= set(expected):
        raise ValueError("selected case list is duplicated or outside the commissioning panel")

    results = []
    phase_source_hashes = {}
    transfer = {}
    for name in selected_names:
        phase_bin, action = expected[name]
        result = audit_case(name, phase_bin, action)
        results.append(result)
        state = result["source_state_sha256"]
        if phase_bin in phase_source_hashes and phase_source_hashes[phase_bin] != state:
            raise ValueError(f"phase bin {phase_bin} branches are not matched-start")
        phase_source_hashes[phase_bin] = state
        worker_path = args.worker_raw_manifest_dir / f"{name}.sha256"
        worker_hashes = parse_sha256_manifest(worker_path)
        spark_hashes = local_raw_hashes([name])
        if worker_hashes != spark_hashes:
            missing = sorted(set(worker_hashes) ^ set(spark_hashes))
            differing = sorted(
                path
                for path in set(worker_hashes) & set(spark_hashes)
                if worker_hashes[path] != spark_hashes[path]
            )
            raise ValueError(
                f"worker/Spark raw mismatch for {name}; "
                f"path mismatch={missing[:5]}, hash mismatch={differing[:5]}"
            )
        transfer[name] = {
            "worker_raw_manifest": str(worker_path),
            "worker_raw_manifest_sha256": sha256(worker_path),
            "raw_file_count": len(spark_hashes),
        }

    complete_panel = set(selected_names) == set(expected)
    report = {
        "status": (
            "MATCHED_START_9_CASE_COMMISSIONING_QC_PASS"
            if complete_panel
            else "MATCHED_START_CASE_TRANSFER_QC_PASS"
        ),
        "scope": "solver/data-pipeline commissioning only; not training or physical control evidence",
        "phase_manifest": str(APPROVED_MANIFEST.relative_to(REPO)),
        "phase_manifest_sha256": APPROVED_MANIFEST_SHA256,
        "cases": results,
        "transfer": transfer,
        "complete_nine_case_panel": complete_panel,
        "matched_source_hashes_for_every_phase_group_present": True,
    }
    args.receipt_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        name = result["case"]
        receipt = {
            "status": "RAW_TRANSFER_VERIFIED",
            "case": name,
            "phase_manifest_sha256": APPROVED_MANIFEST_SHA256,
            **transfer[name],
            "coverage": "raw OpenFOAM solver files only; VTK is not included",
            "next_required_stage": "SPARK_PINNED_FOAMTOVTK_801_FRAMES",
            "curator_guard": "do not curate until a separate VTK_READY receipt exists",
        }
        receipt_path = args.receipt_dir / f"{name}.json"
        if complete_panel:
            if not receipt_path.is_file():
                raise ValueError(f"missing prior per-case transfer receipt: {name}")
            prior = json.loads(receipt_path.read_text(encoding="utf-8"))
            if prior != receipt:
                raise ValueError(f"prior per-case transfer receipt differs: {name}")
        else:
            write_json_exclusive_atomic(receipt_path, receipt)
    write_json_exclusive_atomic(args.output, report)
    print(json.dumps({"status": report["status"], "cases": len(results)}, indent=2))


if __name__ == "__main__":
    main()
