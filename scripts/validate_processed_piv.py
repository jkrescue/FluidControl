"""Validate processed public PIV arrays and leakage-safe actuation splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train_public_piv import select_cases, split_controls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/piv"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/processed_piv_validation.json"))
    args = parser.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    selected = select_cases(manifest)
    splits = split_controls(sorted(selected))
    if len(manifest["excluded_unactuated_reference_files"]) != 28:
        raise ValueError("Expected 28 excluded unactuated reference files")
    if len(manifest["cases"]) != 29 or len(selected) != 28:
        raise ValueError("Expected 29 controlled HDF5 files and 28 nominal actuation values")
    if set(splits["train"]) & set(splits["val"]) or set(splits["train"]) & set(splits["test"]) or set(splits["val"]) & set(splits["test"]):
        raise ValueError("Actuation leakage across data splits")
    sizes, ranges = [], []
    for case in manifest["cases"]:
        if "_zero" in case["source"].lower():
            raise ValueError("Unactuated reference included in controlled manifest")
        with np.load(args.data / case["processed"]) as data:
            velocity, valid, xy = data["velocity"], data["valid"], data["xy"]
            if velocity.shape[1:] != (2, 32, 64) or valid.shape != (len(velocity), 1, 32, 64) or xy.shape != (2, 32, 64):
                raise ValueError(f"Unexpected processed shapes for {case['source']}")
            if not np.isfinite(velocity).all() or not np.isfinite(xy).all() or not np.isfinite(data["cd"]):
                raise ValueError(f"Nonfinite flow/drag values for {case['source']}")
            if not np.isclose(float(data["p"]), case["p"], atol=1e-4):
                raise ValueError(f"Action mismatch for {case['source']}")
            if not np.isclose(float(data["cd"]), case["cd"], atol=1e-4):
                raise ValueError(f"Drag mismatch for {case['source']}")
            sizes.append(len(velocity))
            ranges.append((float(velocity.min()), float(velocity.max())))
    report = {
        "status": "passed",
        "source": manifest["source"],
        "controlled_files": len(manifest["cases"]),
        "distinct_controls": len(selected),
        "excluded_unactuated_references": len(manifest["excluded_unactuated_reference_files"]),
        "retained_snapshot_count_range": [min(sizes), max(sizes)],
        "velocity_u_infty_range": [min(item[0] for item in ranges), max(item[1] for item in ranges)],
        "split": splits,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
