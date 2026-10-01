#!/usr/bin/env python3
"""Reject duplicate CFD trajectories across curated train/validation/test splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def signature(config: dict) -> str:
    """Identify runs with the same initial field, physics, and boundary action."""
    fields = (
        "source_restart_case",
        "source_restart_time",
        "geometry",
        "reynolds_number",
        "kinematic_viscosity",
        "delta_t",
        "start_time",
        "end_time",
        "action_points",
    )
    missing = [field for field in fields if field not in config]
    if missing:
        raise ValueError(f"case metadata misses split-audit fields: {missing}")
    payload = {field: config[field] for field in fields}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def audit(data: Path, cases_root: Path) -> dict:
    rows = []
    by_signature: dict[str, list[dict]] = defaultdict(list)
    for split in ("train", "validation", "test"):
        for path in sorted((data / split).glob("*.h5")):
            config_path = cases_root / path.stem / "case_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if config["split"] != split:
                raise ValueError(f"{path.stem}: HDF5 split {split} != CFD split {config['split']}")
            row = {"case": path.stem, "split": split, "signature": signature(config)}
            rows.append(row)
            by_signature[row["signature"]].append(row)
    duplicates = [group for group in by_signature.values() if len(group) > 1]
    return {
        "status": "SPLIT_INTEGRITY_OK" if not duplicates else "DUPLICATE_TRAJECTORIES",
        "trajectory_count": len(rows),
        "split_counts": {
            split: sum(row["split"] == split for row in rows)
            for split in ("train", "validation", "test")
        },
        "duplicate_groups": duplicates,
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cases-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.data, args.cases_root)
    result = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding="utf-8")
    print(result, end="")
    if report["duplicate_groups"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
