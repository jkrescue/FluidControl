#!/usr/bin/env python3
"""Compare true-state H1 and recursive H100 at identical CFD target times.

Project diagnostic, not a model-admission gate. Input is existing evaluator
segments JSON; no CFD/HDF/model execution occurs here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def compare(payload: dict) -> dict:
    if payload.get("split") != "validation" or payload.get("action_mode") != "observed":
        raise ValueError("Requires observed-action validation evidence")
    lookup = {}
    for row in payload["segments"]:
        key = (row["case"], row["horizon"], row["start"])
        if key in lookup:
            raise ValueError(f"Duplicate segment {key}")
        lookup[key] = row
    groups = {}
    for (case, horizon, start), long in lookup.items():
        if horizon != 100:
            continue
        short = lookup.get((case, 1, start + 99))
        if short is None:
            raise ValueError(f"Missing same-target H1: {case}/{start}")
        values = [short["target_total_drag"], long["target_total_drag"],
                  short["rear_cl_mae"], long["rear_cl_mae"]]
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise ValueError("Non-finite force evidence")
        if values[0] != values[1]:
            raise ValueError("Different recorded CFD target total drag")
        if min(values[2:]) < 0:
            raise ValueError("Negative absolute error")
        groups.setdefault(case, []).append(values[2:])
    expected = {f"full40_dynamic_validation_{phase}_{action}"
                for phase in ("b01", "b05") for action in ("minus", "zero", "plus")}
    if set(groups) != expected or any(len(rows) != 101 for rows in groups.values()):
        raise ValueError("Requires full dynamic6 panel: six cases, 101 matched endpoints each")
    return {
        "status": "DIAGNOSTIC_COMPLETE_NOT_ADMISSION",
        "matched_endpoints": 606,
        "metric": "instantaneous_rear_cl_absolute_error_arithmetic_mean",
        "units": "lift_coefficient",
        "target_total_drag_max_difference": 0.0,
        "cases": {case: {
            "matched_endpoints": len(rows),
            "true_state_h1_mae": math.fsum(row[0] for row in rows) / len(rows),
            "recursive_h100_mae": math.fsum(row[1] for row in rows) / len(rows),
        } for case, rows in sorted(groups.items())},
        "limitations": "Not a Cl-prime RMS window error, causal attribution, or frozen test. "
                        "Target agreement is checked using recorded total drag; original evaluator "
                        "case/time identity remains required.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    content = args.segments.read_bytes()
    result = compare(json.loads(content))
    result["source"] = str(args.segments)
    result["source_sha256"] = hashlib.sha256(content).hexdigest()
    result["implementation_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(result["status"], result["matched_endpoints"])


if __name__ == "__main__":
    main()
