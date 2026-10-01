#!/usr/bin/env python3
"""Audit segmented zero-action OpenFOAM observations against the monolithic CFD run."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from fluid_control.openfoam_observation import observation_at


def compare(cases_root: Path, name: str, result: Path) -> dict:
    case = cases_root / name
    config = json.loads((case / "case_config.json").read_text(encoding="utf-8"))
    smoke = json.loads(result.read_text(encoding="utf-8"))
    if config.get("status") != "segmented_zero_observation_smoke_completed":
        raise ValueError("segmented CFD zero run is incomplete")
    if smoke.get("status") != "REAL_OPENFOAM_SEGMENTED_ZERO_OBSERVATION_SMOKE_OK":
        raise ValueError("smoke result is not accepted")
    if smoke.get("case") != name or len(smoke.get("rows", [])) != config["steps"]:
        raise ValueError("case/result mismatch")
    reference_name = config["source_restart_case"]
    if reference_name != "tandem_backward_dt005":
        raise ValueError("unexpected reference source case")
    reference = cases_root / reference_name
    comparisons = []
    for index, row in enumerate(smoke["rows"], 1):
        target = round(80 + 0.1 * index, 6)
        if row["step"] != index or not math.isclose(row["end_time"], target, abs_tol=1e-6):
            raise ValueError("nonconsecutive segmented CFD time")
        if row["requested_omega"] != 0 or row["applied_omega"] != 0:
            raise ValueError("not a matched zero-action run")
        segmented, _ = observation_at(case, target, 0.0)
        monolithic, _ = observation_at(reference, target, 0.0)
        error = np.abs(segmented.astype(np.float64) - monolithic.astype(np.float64))
        comparisons.append({
            "step": index,
            "time": target,
            "probe_max_abs_error": float(error[:64].max()),
            "rear_force_max_abs_error": float(error[64:66].max()),
            "omega_abs_error": float(error[66]),
            "segmented_rear_cd": float(segmented[64]),
            "reference_rear_cd": float(monolithic[64]),
            "segmented_rear_cl": float(segmented[65]),
            "reference_rear_cl": float(monolithic[65]),
        })
    probe_limit = 1e-5
    force_limit = 1e-5
    passed = all(
        row["probe_max_abs_error"] <= probe_limit
        and row["rear_force_max_abs_error"] <= force_limit
        and row["omega_abs_error"] == 0
        for row in comparisons
    )
    return {
        "status": (
            "SEGMENTED_ZERO_MATCHES_MONOLITHIC_CFD"
            if passed else "SEGMENTED_ZERO_DIFFERS_FROM_MONOLITHIC_CFD"
        ),
        "scientific_scope": f"{len(comparisons)}_short_zero_action_intervals_only_not_rl_control",
        "segmented_case": name,
        "monolithic_reference": reference_name,
        "probe_error_limit": probe_limit,
        "rear_force_error_limit": force_limit,
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-root", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--run-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    report = compare(args.cases_root, args.case, args.run_result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "SEGMENTED_ZERO_MATCHES_MONOLITHIC_CFD":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
