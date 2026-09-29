"""Compare published experimental and companion URANS force tables.

The URANS p grid is coarser; experimental values are linearly interpolated.
This is a source-consistency check, not validation of our own CFD solver.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/piv/manifest.json"))
    parser.add_argument("--urans-forces", type=Path, default=Path("data/raw/zenodo_20794709/URANSDataset/AerodynamicForces.mat"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/published_force_comparison.json"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = sorted((float(p), float(item["cd"])) for p, item in manifest["force_table"].items())
    experimental_p = np.asarray([row[0] for row in rows])
    experimental_cd = np.asarray([row[1] for row in rows])
    urans = loadmat(args.urans_forces)
    simulation_p = np.asarray(urans["p"], dtype=np.float64).ravel()
    simulation_cd = np.asarray(urans["Cd"], dtype=np.float64).ravel()
    comparisons = []
    for p, cd in zip(simulation_p, simulation_cd, strict=True):
        if experimental_p.min() <= p <= experimental_p.max():
            measured = float(np.interp(p, experimental_p, experimental_cd))
            comparisons.append({"p": float(p), "cd_experimental_interpolated": measured, "cd_urans": float(cd), "urans_minus_experimental": float(cd - measured)})
    report = {
        "source": manifest["source"],
        "note": "Published experimental vs published URANS forces; not new CFD and no numerical casewise experimental uncertainties available",
        "rows": comparisons,
        "mae_cd": float(np.mean([abs(item["urans_minus_experimental"]) for item in comparisons])),
        "experimental_discrete_optimum": {"p": float(experimental_p[np.argmin(experimental_cd)]), "cd": float(np.min(experimental_cd))},
        "urans_discrete_optimum": {"p": float(simulation_p[np.argmin(simulation_cd)]), "cd": float(np.min(simulation_cd))},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
