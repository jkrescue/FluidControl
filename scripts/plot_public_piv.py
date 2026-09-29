"""Plot actual time-averaged experimental PIV at baseline and boat tailing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train_public_piv import load_arrays, mean_field, select_cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/piv"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/piv_baseline_vs_boattailing.png"))
    args = parser.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    selected = select_cases(manifest)
    controls = (0.0, 1.8)
    observations = []
    for p in controls:
        case = load_arrays(args.data, selected[p])
        mean, _ = mean_field(case["velocity"], case["valid"])
        observations.append((p, mean, case["xy"], float(case["cd"])))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True, sharey=True)
    for axis, (p, mean, xy, cd) in zip(axes, observations, strict=True):
        speed = np.linalg.norm(mean, axis=0)
        contours = axis.pcolormesh(xy[0], xy[1], speed, shading="auto", vmin=0, vmax=1.6, cmap="viridis")
        axis.set_title(f"Measured PIV: p={p:g}, $C_D$={cd:.3f}")
        axis.set_xlabel("x/D")
        axis.set_aspect("equal")
    axes[0].set_ylabel("y/D")
    fig.colorbar(contours, ax=axes, label=r"$|\overline{\mathbf{u}}|/U_\infty$", shrink=0.85)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    plt.close(fig)
    print(args.out, flush=True)


if __name__ == "__main__":
    main()
