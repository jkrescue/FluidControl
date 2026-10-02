#!/usr/bin/env python3
"""Build HydroGym/CEM phase baselines from the audited real-CFD objective report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fluid_control.stage_c_objective import validate_stage_c_baseline


def build_phase_baselines(audit: dict, source: str) -> dict:
    """Extract one phase-matched zero-action baseline per audited CFD case."""
    cases = audit.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("objective audit contains no cases")
    result: dict[str, dict] = {}
    for row in cases:
        identity = (str(row["dataset"]), str(row["split"]), str(row["case"]))
        key = "/".join(identity)
        if key in result:
            raise ValueError(f"duplicate CFD case baseline: {key}")
        raw = row["phase_matched_zero_action_baseline"]
        baseline = {
            "total_drag": float(raw["total_cd_mean"]),
            "front_lift_rms": float(raw["front_cl"]["root_mean_square"]),
            "rear_lift_rms": float(raw["rear_cl"]["root_mean_square"]),
            "source": f"{source}::{key}::phase_matched_zero_action",
        }
        checked = validate_stage_c_baseline(baseline)
        assert checked is not None
        result[key] = {
            **checked,
            "window": [float(value) for value in raw["window"]],
            "samples": int(raw["samples"]),
        }
    return {
        "schema_version": 1,
        "objective": "system_total_drag_with_front_and_rear_lift_guards",
        "source_audit": source,
        "case_count": len(result),
        "cases": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    result = build_phase_baselines(audit, str(args.audit))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "case_count": result["case_count"]}))


if __name__ == "__main__":
    main()
