#!/usr/bin/env python3
"""Render PhysicsNeMo training history as a four-panel curve figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--title", default="Expanded action-domain PhysicsNeMo FNO training")
    args = parser.parse_args()

    rows = json.loads(args.history.read_text(encoding="utf-8"))
    if not rows:
        raise ValueError(f"empty training history: {args.history}")
    epochs = [row["epoch"] for row in rows]
    panels = (
        ("train_loss", "Training loss", True),
        ("state_mae_physical_units", "Validation field MAE", False),
        ("state_rmse_physical_units", "Validation field RMSE", False),
        ("force_mae_normalized", "Validation force MAE (normalized)", False),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
    for axis, (key, label, log_scale) in zip(axes.flat, panels, strict=True):
        values = [row[key] for row in rows]
        axis.plot(epochs, values, color="#087bb7", linewidth=1.8)
        best_index = min(range(len(values)), key=values.__getitem__)
        axis.scatter(
            [epochs[best_index]], [values[best_index]], color="#168653", s=32,
            label=f"best: epoch {epochs[best_index]}, {values[best_index]:.4g}", zorder=3,
        )
        if log_scale:
            axis.set_yscale("log")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=9)
    figure.suptitle(args.title)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
