#!/usr/bin/env python3
"""Reject duplicate CFD trajectories across curated train/validation/test splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


EXPECTED_COUNTS = {"train": 24, "validation": 4, "test": 4}


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


def audit(data: Path, cases_root: Path, expected_counts: dict[str, int] | None = None) -> dict:
    expected_counts = expected_counts or EXPECTED_COUNTS
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
    counts = {
        split: sum(row["split"] == split for row in rows)
        for split in expected_counts
    }
    status = (
        "DUPLICATE_TRAJECTORIES" if duplicates else
        "INCOMPLETE_SPLIT" if counts != expected_counts else "SPLIT_INTEGRITY_OK"
    )
    return {
        "status": status,
        "trajectory_count": len(rows),
        "split_counts": counts,
        "expected_counts": expected_counts,
        "duplicate_groups": duplicates,
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cases-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-train", type=int, default=24)
    parser.add_argument("--expected-validation", type=int, default=4)
    parser.add_argument("--expected-test", type=int, default=4)
    args = parser.parse_args()
    expected = {"train": args.expected_train, "validation": args.expected_validation, "test": args.expected_test}
    if any(value < 1 for value in expected.values()):
        parser.error("expected split counts must be positive")
    report = audit(args.data, args.cases_root, expected_counts=expected)
    result = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding="utf-8")
    print(result, end="")
    if report["status"] != "SPLIT_INTEGRITY_OK":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
