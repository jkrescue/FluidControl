#!/usr/bin/env python3
"""Fail-closed raw OpenFOAM QC for the complete dynamic train8 panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CASES = REPO / "cfd/tandem_cylinders/cases"
PREDECL_SHA = "4cf4e7c9b9da27b71e58db2e94b0750736b7f79f09a2aebc3ffa97729e882c5a"
FORCE_SHA = {
    "forceFront": "bce88443ce3d19a6411c31b9266af16a3dd4e992f7adb1ddabc659238a1e88d1",
    "forceRear": "654c5bbf64b758505bdfe4d379e153f460862902309e96d2b01974e45536680b",
}
EXPECTED = {
    f"dynamic_train8_b{phase:02d}_{profile}"
    for phase in (0, 2, 4, 6)
    for profile in ("prbs", "multisine")
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def omega_table(path: Path) -> list[list[float]]:
    text = path.read_text(encoding="utf-8")
    patch = re.search(r"rearCylinder\s*\{(?P<body>.*?)\n\s*\}", text, re.S)
    table = patch and re.search(r"omega\s+table\s*\((?P<rows>.*?)\)\s*;", patch.group("body"), re.S)
    if table is None:
        raise ValueError(f"omega table absent: {path}")
    return [[float(a), float(b)] for a, b in re.findall(
        r"\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)", table.group("rows"))]


def force_grid(case: Path, object_name: str, start: float, end: float) -> np.ndarray:
    samples: dict[float, tuple[float, ...]] = {}
    paths = sorted(case.glob(f"postProcessing/{object_name}/*/coefficient.dat"))
    if not paths:
        raise FileNotFoundError(f"missing {object_name}: {case.name}")
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            row = tuple(float(value) for value in line.split())
            if not start + 0.005 - 1e-9 <= row[0] <= end + 1e-9:
                continue
            key = round(row[0], 8)
            if key in samples and not np.allclose(samples[key], row, rtol=0, atol=1e-10):
                raise ValueError(f"conflicting force restart row: {case.name}/{object_name}/{key}")
            samples[key] = row
    values = np.asarray([samples[key] for key in sorted(samples)], dtype=np.float64)
    expected = start + np.arange(1, 4001, dtype=np.float64) * 0.005
    if (
        values.shape[0] != 4000
        or values.shape[1] < 5
        or not np.isfinite(values).all()
        or not np.allclose(values[:, 0], expected, rtol=0, atol=1e-8)
    ):
        raise ValueError(f"{case.name}/{object_name}: force grid differs")
    return values


def audit(predeclaration: Path) -> dict:
    if sha256(predeclaration) != PREDECL_SHA:
        raise ValueError("reviewed predeclaration SHA differs")
    predecl = load(predeclaration)
    if (
        predecl.get("status") != "DYNAMIC_TRAIN8_PREDECLARED_NOT_EXECUTED"
        or set(predecl.get("cases", {})) != EXPECTED
        or predecl.get("matrix", {}).get("phase_bins") != [0, 2, 4, 6]
    ):
        raise ValueError("train8 matrix differs")
    baseline = CASES / "tandem_backward_dt005/postProcessing"
    for force, digest in FORCE_SHA.items():
        if sha256(baseline / force / "0/coefficient.dat") != digest:
            raise ValueError(f"baseline source force differs: {force}")
    results = {}
    for name in sorted(EXPECTED):
        expected = predecl["cases"][name]
        case = CASES / name
        config_path = case / "case_config.json"
        marker_path = case / "solver_complete.dynamic_train8.json"
        qc_path = case / "solver_log_qc.dynamic_train8.json"
        log_path = case / "log.pimpleFoam.dynamic_train8"
        config, marker, qc = load(config_path), load(marker_path), load(qc_path)
        if any(config.get(key) != value for key, value in expected.items()):
            raise ValueError(f"case contract differs: {name}")
        start, end = map(float, expected["run_window"])
        if (
            config.get("panel") != "dynamic_train8_v1"
            or config.get("split") != "train"
            or marker.get("status") != "DYNAMIC_TRAIN8_SOLVER_COMPLETE_PENDING_AGGREGATE_QC"
            or marker.get("case") != name
            or marker.get("predeclaration_sha256") != PREDECL_SHA
            or marker.get("solver_log_sha256") != sha256(log_path)
            or marker.get("solver_qc_sha256") != sha256(qc_path)
            or qc.get("steps") != 4000
            or not math.isclose(float(qc.get("terminal_time", math.nan)), end, abs_tol=2e-6)
        ):
            raise ValueError(f"solver provenance differs: {name}")
        for field, digest in expected["source_state_sha256"].items():
            if sha256(case / "source_restart_provenance" / field) != digest:
                raise ValueError(f"source state differs: {name}/{field}")
        if omega_table(case / f"{start:g}" / "U") != expected["action_points"]:
            raise ValueError(f"action table differs: {name}")
        front = force_grid(case, "forceFront", start, end)
        rear = force_grid(case, "forceRear", start, end)
        if not np.allclose(front[:, 0], rear[:, 0], rtol=0, atol=1e-10):
            raise ValueError(f"front/rear force times differ: {name}")
        results[name] = {
            "split": "train",
            "phase_bin": expected["phase_bin"],
            "profile": expected["profile"],
            "source_state_sha256": expected["source_state_sha256"],
            "action_metrics": expected["action_metrics"],
            "solver_log_sha256": sha256(log_path),
            "solver_qc_sha256": sha256(qc_path),
            "raw_force_samples": 4000,
            "front_force_file_sha256": {str(path.relative_to(REPO)): sha256(path) for path in sorted(case.glob("postProcessing/forceFront/*/coefficient.dat"))},
            "rear_force_file_sha256": {str(path.relative_to(REPO)): sha256(path) for path in sorted(case.glob("postProcessing/forceRear/*/coefficient.dat"))},
        }
    return {
        "status": "DYNAMIC_TRAIN8_REAL_OPENFOAM_QC_COMPLETE",
        "scope": "eight train-only prescribed dynamic-action trajectories; not validation or control benefit",
        "predeclaration": str(predeclaration.relative_to(REPO)),
        "predeclaration_sha256": PREDECL_SHA,
        "baseline_force_source_sha256": FORCE_SHA,
        "case_count": 8,
        "validation_or_frozen_accessed": False,
        "cases": results,
    }


def write_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError(f"refusing overwrite: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predeclaration", type=Path, default=REPO / "artifacts/tandem_cylinders/dynamic_train8_predeclared_20261003.json")
    parser.add_argument("--output", type=Path, default=REPO / "artifacts/tandem_cylinders/dynamic_train8_real_cfd_qc_20261003.json")
    args = parser.parse_args()
    write_exclusive(args.output.resolve(), audit(args.predeclaration.resolve()))
    print(args.output)


if __name__ == "__main__":
    main()
