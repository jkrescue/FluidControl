#!/usr/bin/env python3
"""Group existing FC-P001 segment errors by phase, action and horizon."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(case: str) -> tuple[str, str]:
    phase = re.search(r"_(b\d{2})_", case)
    if not phase:
        raise ValueError(f"phase absent from case: {case}")
    action = case.rsplit("_", 1)[-1]
    return phase.group(1), action


def aggregate(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    rows = payload.get("segments")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"segments absent: {path}")
    groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        phase, action = identity(row["case"])
        groups[(phase, action, int(row["horizon"]))].append(row)
    result = []
    for (phase, action, horizon), values in sorted(groups.items()):
        predicted = [float(row["predicted_total_drag"]) for row in values]
        target = [float(row["target_total_drag"]) for row in values]
        result.append({
            "phase": phase,
            "action": action,
            "horizon": horizon,
            "segments": len(values),
            "aggregation": {
                "state_mae_macro_over_segments": sum(float(row["state_mae_physical_units"]) for row in values) / len(values),
                "total_cd_mae_macro_over_segments": sum(float(row["total_drag_absolute_error"]) for row in values) / len(values),
                "rear_cl_mae_macro_over_segments": sum(float(row["rear_cl_mae"]) for row in values) / len(values),
                "total_cd_nrmse_pooled_over_endpoints": math.sqrt(sum((p - t) ** 2 for p, t in zip(predicted, target, strict=True))) / math.sqrt(sum(t * t for t in target)),
                "mean_abs_omega_macro_over_segments": sum(float(row["mean_abs_omega"]) for row in values) / len(values),
                "mean_abs_domega_dt_macro_over_segments": sum(float(row["mean_abs_domega_dt"]) for row in values) / len(values),
            },
        })
    return result


def build(validation: Path, dynamic: Path, gate: Path) -> dict:
    development = json.loads(gate.read_text())
    branches = development["window_gate"]["branches"]
    return {
        "status": "FC_P002_PHASE_ACTION_HORIZON_FAILURE_MAP_COMPLETE",
        "scope": {
            "existing_inference_only": True,
            "frozen_test_accessed": False,
            "claim_limit": "Grouped development evidence; repeated validation is not independent testing.",
        },
        "sources": {
            "validation10_segments_sha256": sha256(validation),
            "dynamic6_segments_sha256": sha256(dynamic),
            "development_gate_sha256": sha256(gate),
        },
        "aggregation_definitions": {
            "macro": "arithmetic mean of per-segment absolute errors within exact phase/action/horizon",
            "pooled_total_cd_nrmse": "sqrt(sum endpoint squared error)/sqrt(sum endpoint target squared value) within the group",
        },
        "validation10": aggregate(validation),
        "dynamic6": aggregate(dynamic),
        "force_window_branches": branches,
    }


def write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-segments", type=Path, required=True)
    parser.add_argument("--dynamic-segments", type=Path, required=True)
    parser.add_argument("--development-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.validation_segments, args.dynamic_segments, args.development_gate)
    write_atomic(args.output, result)
    print(json.dumps({"status": result["status"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
