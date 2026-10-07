#!/usr/bin/env python3
"""Plot already-saved retained-B field arrays; performs no model inference."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/workspace/fluid_control")
SOURCE = ROOT / "artifacts/p064_arm_b_development_h1_h5_20261006/fields_b01_0000.npz"
OUT = ROOT / "docs/report_20261007/assets"
EXPECTED = "bf07dcc3796adf29cda99a31036ebaf6ccb5ef0dfa0893ec465745fac2ecd406"

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

assert sha(SOURCE) == EXPECTED
z = np.load(SOURCE, allow_pickle=False)
assert z["truth_states"].shape == (5, 3, 128, 256)
assert z["predicted_states"].shape == (5, 3, 128, 256)
mask = z["mask"][0].astype(bool)
x, y = z["x"], z["y"]
extent = [float(x[0]), float(x[-1]), float(y[0]), float(y[-1])]
labels = [r"u / U∞", r"v / U∞", r"gauge p / (ρ U∞²)"]
cmaps = ["RdBu_r", "RdBu_r", "RdBu_r"]

assets = {}
for horizon, idx in [(1, 0), (5, 4)]:
    truth = z["truth_states"][idx].astype(np.float64)
    pred = z["predicted_states"][idx].astype(np.float64)
    err = pred - truth
    fig, axes = plt.subplots(3, 3, figsize=(16, 10.2), constrained_layout=True)
    for row in range(3):
        t = np.ma.masked_where(~mask, truth[row])
        p = np.ma.masked_where(~mask, pred[row])
        e = np.ma.masked_where(~mask, err[row])
        vals = np.concatenate([t.compressed(), p.compressed()])
        if row == 0:
            lo, hi = float(vals.min()), float(vals.max())
        else:
            lim = float(np.max(np.abs(vals)))
            lo, hi = -lim, lim
        elim = float(np.max(np.abs(e.compressed())))
        ims = [
            axes[row, 0].imshow(t, origin="lower", extent=extent, cmap=cmaps[row], vmin=lo, vmax=hi, aspect="auto"),
            axes[row, 1].imshow(p, origin="lower", extent=extent, cmap=cmaps[row], vmin=lo, vmax=hi, aspect="auto"),
            axes[row, 2].imshow(e, origin="lower", extent=extent, cmap="coolwarm", vmin=-elim, vmax=elim, aspect="auto"),
        ]
        for col, title in enumerate(["Saved CFD truth", "Retained-B prediction", "Signed error (prediction − truth)"]):
            axes[row, col].set_title(f"{labels[row]}: {title}")
            axes[row, col].set_xlabel("x / D")
            axes[row, col].set_ylabel("y / D")
            axes[row, col].set_aspect("equal", adjustable="box")
            fig.colorbar(ims[col], ax=axes[row, col], shrink=.86)
    t0 = float(z["time"][0, 0])
    th = float(z["time"][horizon, 0])
    fig.suptitle(
        f"Retained B (frozen K1 flow branch): saved b01 development replay, H{horizon}\n"
        f"t*={t0:.1f} → {th:.1f}; realized-action retrospective; opened development origin",
        fontsize=15,
    )
    path = OUT / f"retained_b_fields_h{horizon}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    assets[path.name] = {
        "sha256": sha(path), "bytes": path.stat().st_size,
        "source_sha256": EXPECTED, "source_time": [t0, th],
        "description": "Saved physical u/v/gauge-p truth, retained-B prediction, and signed error; no new inference."
    }
print(json.dumps(assets, indent=2))
