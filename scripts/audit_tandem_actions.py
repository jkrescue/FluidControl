#!/usr/bin/env python3
"""Audit angular-velocity amplitude and rate coverage by dataset split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/curated/tandem_cylinders"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report: dict = {"cases": {}}
    rates_by_split: dict[str, list[np.ndarray]] = {}
    for split_dir in sorted(path for path in args.data.iterdir() if path.is_dir()):
        split = split_dir.name
        rates_by_split[split] = []
        report["cases"][split] = {}
        for path in sorted(split_dir.glob("*.h5")):
            with h5py.File(path, "r") as handle:
                omega = np.asarray(handle["omega"][:, 0], dtype=np.float64)
                time = np.asarray(handle["time"][:, 0], dtype=np.float64)
            rate = np.diff(omega) / np.diff(time)
            rates_by_split[split].append(rate)
            report["cases"][split][path.stem] = {
                "omega_min": float(omega.min()),
                "omega_max": float(omega.max()),
                "max_abs_domega_dt": float(np.abs(rate).max()),
                "abs_domega_dt_quantiles": {
                    str(q): float(np.quantile(np.abs(rate), q))
                    for q in (0.5, 0.9, 0.95, 0.99)
                },
            }

    train_rates = np.concatenate(rates_by_split["train"])
    train_limit = float(np.abs(train_rates).max())
    report["train_max_abs_domega_dt"] = train_limit
    report["split_rate_coverage"] = {}
    for split, arrays in rates_by_split.items():
        values = np.concatenate(arrays)
        report["split_rate_coverage"][split] = {
            "samples": int(values.size),
            "max_abs_domega_dt": float(np.abs(values).max()),
            "fraction_above_train_max": float(np.mean(np.abs(values) > train_limit + 1.0e-9)),
        }

    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
