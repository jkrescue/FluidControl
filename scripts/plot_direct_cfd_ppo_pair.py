#!/usr/bin/env python3
"""Plot raw paired OpenFOAM PPO/zero forces and the applied rear-cylinder action."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_force(case: Path, function: str) -> tuple[np.ndarray, list[Path]]:
    paths = sorted(case.glob(f"postProcessing/{function}/*/coefficient.dat"))
    if not paths:
        raise FileNotFoundError(f"no {function} coefficient.dat in {case}")
    chunks = [np.loadtxt(path, comments="#", ndmin=2) for path in paths]
    values = np.concatenate(chunks)
    values = values[np.argsort(values[:, 0])]
    values = values[np.concatenate(([True], np.diff(values[:, 0]) > 1.0e-12))]
    if values.shape[1] < 5 or not np.isfinite(values).all():
        raise ValueError(f"invalid/non-finite {function} coefficients")
    return values, paths


def paired_force(case: Path, start: float, end: float) -> tuple[np.ndarray, list[Path]]:
    front, front_paths = load_force(case, "forceFront")
    rear, rear_paths = load_force(case, "forceRear")
    front = front[(front[:, 0] >= start - 1.0e-9) & (front[:, 0] <= end + 1.0e-9)]
    rear = rear[(rear[:, 0] >= start - 1.0e-9) & (rear[:, 0] <= end + 1.0e-9)]
    if len(front) != len(rear) or not np.allclose(
        front[:, 0], rear[:, 0], rtol=0.0, atol=1.0e-9
    ):
        raise ValueError("front/rear force time grids differ")
    # time, total Cd, rear Cl; no filtering or smoothing.
    return np.column_stack((front[:, 0], front[:, 1] + rear[:, 1], rear[:, 4])), [
        *front_paths,
        *rear_paths,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cases-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    rollout_path = args.run / "rollout/rollout_result.json"
    physical_path = args.run / "physical_result.json"
    rollout = json.loads(rollout_path.read_text(encoding="utf-8"))
    physical = json.loads(physical_path.read_text(encoding="utf-8"))
    if (
        rollout.get("status") != "DIRECT_CFD_FROZEN_PPO_PAIR_ROLLOUT_COMPLETE"
        or physical.get("status") != "DIRECT_CFD_B00_FROZEN_PPO_PAIR_EVALUATED"
        or rollout.get("steps_per_branch") != 800
    ):
        raise ValueError("completed 800-step paired CFD result is required")
    rows = rollout["rows"]
    ppo_rows = [row for row in rows if row.get("role") == "ppo"]
    zero_rows = [row for row in rows if row.get("role") == "zero"]
    if len(ppo_rows) != 800 or len(zero_rows) != 800:
        raise ValueError("paired rollout must contain exactly 800 rows per branch")
    start = min(float(row["cfd_time"]) for row in rows) - 0.1
    end = max(float(row["cfd_time"]) for row in rows)
    cases = {
        role: args.cases_root / rollout["branches"][role] for role in ("ppo", "zero")
    }
    forces = {}
    sources = {}
    for role, case in cases.items():
        forces[role], paths = paired_force(case, start, end)
        sources[role] = [
            {
                "path": str(path),
                "sha256": sha256(path),
            }
            for path in paths
        ]

    figure, axes = plt.subplots(
        3, 1, figsize=(12, 9), sharex=True, constrained_layout=True
    )
    labels = {"ppo": "PPO control", "zero": "zero rotation"}
    colors = {"ppo": "tab:blue", "zero": "tab:orange"}
    for role in ("ppo", "zero"):
        time = forces[role][:, 0] - start
        axes[0].plot(time, forces[role][:, 1], linewidth=0.8, label=labels[role], color=colors[role])
        axes[1].plot(time, forces[role][:, 2], linewidth=0.8, label=labels[role], color=colors[role])
    action_time = np.asarray([float(row["cfd_time"]) - start for row in ppo_rows])
    omega = np.asarray([float(row["applied_omega"]) for row in ppo_rows])
    axes[2].step(action_time, omega, where="post", linewidth=0.9, color="tab:green", label="PPO applied ω")
    axes[2].axhline(0.0, linewidth=0.8, color="tab:orange", linestyle="--", label="zero rotation")
    for axis in axes:
        axis.axvspan(0.0, 20.0, color="0.75", alpha=0.18, label="transition 0–20 D/U")
        axis.axvspan(20.0, 80.0, color="tab:blue", alpha=0.05, label="statistics 20–80 D/U")
        axis.axvline(20.0, color="black", linewidth=0.8, linestyle=":")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Total Cd")
    axes[1].set_ylabel("Rear Cl")
    axes[2].set_ylabel("Rear-cylinder ω")
    axes[2].set_xlabel("Elapsed nondimensional time D/U")
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].legend(ncol=2, fontsize=8)
    axes[2].legend(ncol=2, fontsize=8)
    figure.suptitle("Direct OpenFOAM PPO vs zero rotation · raw force histories")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=170)
    plt.close(figure)
    evidence = {
        "status": "DIRECT_CFD_PPO_PAIR_RAW_TIMESERIES_PLOT_COMPLETE",
        "plot": str(args.output),
        "plot_sha256": sha256(args.output),
        "rollout_result_sha256": sha256(rollout_path),
        "physical_result_sha256": sha256(physical_path),
        "elapsed_windows_du": {"transition": [0.0, 20.0], "statistics": [20.0, 80.0]},
        "signals": ["total_cd", "rear_cl", "applied_rear_cylinder_omega"],
        "raw_unsmoothed": True,
        "force_sources": sources,
        "scientific_scope": physical["scientific_scope"],
    }
    args.evidence.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
