#!/usr/bin/env python3
"""Compare an MPC OpenFOAM replay with the matched zero-rotation baseline."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parent
CASES = ROOT / "cases"
BASELINE = CASES / "tandem_backward_dt005"


def load_force(case: Path) -> np.ndarray:
    files = sorted(case.glob("postProcessing/forceRear/*/coefficient.dat"))
    if not files:
        raise FileNotFoundError(f"no rear force coefficients in {case}")
    chunks = [np.loadtxt(path, comments="#", ndmin=2) for path in files]
    data = np.concatenate(chunks)
    data = data[np.argsort(data[:, 0])]
    return data[np.concatenate(([True], np.diff(data[:, 0]) > 1.0e-12))]


def summarize(data: np.ndarray, start: float, end: float) -> dict[str, float | int]:
    selected = data[(data[:, 0] > start) & (data[:, 0] <= end + 1.0e-9)]
    if len(selected) < 100:
        raise ValueError(f"only {len(selected)} force samples in ({start}, {end}]")
    cd = selected[:, 1]
    cl = selected[:, 4]
    return {
        "samples": len(selected),
        "rear_cd_mean": float(cd.mean()),
        "rear_cl_mean": float(cl.mean()),
        "rear_cl_rms": float(np.sqrt(np.mean(np.square(cl)))),
        "finite": bool(np.isfinite(selected).all()),
    }


def write_plot(path: Path, controlled: np.ndarray, baseline: np.ndarray, config: dict) -> None:
    start = float(config["source_restart_time"])
    end = float(config["end_time"])
    controlled = controlled[(controlled[:, 0] > start) & (controlled[:, 0] <= end + 1.0e-9)]
    baseline = baseline[(baseline[:, 0] > start) & (baseline[:, 0] <= end + 1.0e-9)]
    points = np.asarray(config["action_points"], dtype=np.float64)
    omega = np.interp(controlled[:, 0], points[:, 0], points[:, 1])

    figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    axes[0].plot(controlled[:, 0], controlled[:, 1], label="MPC action replay", linewidth=1)
    axes[0].plot(baseline[:, 0], baseline[:, 1], label="zero rotation", linewidth=1, alpha=0.8)
    axes[0].set_ylabel("Rear Cd")
    axes[0].legend()
    axes[1].plot(controlled[:, 0], controlled[:, 4], label="MPC action replay", linewidth=1)
    axes[1].plot(baseline[:, 0], baseline[:, 4], label="zero rotation", linewidth=1, alpha=0.8)
    axes[1].set_ylabel("Rear Cl")
    axes[1].legend()
    axes[2].plot(controlled[:, 0], omega, color="tab:green", linewidth=1.2)
    axes[2].set_ylabel("omega")
    axes[2].set_xlabel("Nondimensional time")
    figure.suptitle("OpenFOAM replay of frozen PhysicsNeMo MPC actions")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_name")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()

    case = CASES / args.case_name
    config = json.loads((case / "case_config.json").read_text(encoding="utf-8"))
    log = (case / "log.pimpleFoam").read_text(encoding="utf-8", errors="replace")
    if not re.search(r"^End\s*$", log, re.MULTILINE):
        raise ValueError("controlled solver log did not reach End")
    if re.search(r"FOAM FATAL|segmentation fault|core dumped|\bnan\b|\binf\b", log, re.I):
        raise ValueError("controlled solver log contains a fatal or non-finite marker")

    controlled = load_force(case)
    baseline = load_force(BASELINE)
    start = float(config["source_restart_time"])
    end = float(config["end_time"])
    windows = []
    for window_start in (start, min(start + 2.0, end)):
        controlled_summary = summarize(controlled, window_start, end)
        baseline_summary = summarize(baseline, window_start, end)
        windows.append({
            "window": [window_start, end],
            "controlled": controlled_summary,
            "zero_rotation_baseline": baseline_summary,
            "relative_change_percent": {
                "rear_cd_mean": 100.0 * (
                    controlled_summary["rear_cd_mean"] / baseline_summary["rear_cd_mean"] - 1.0
                ),
                "rear_cl_rms": 100.0 * (
                    controlled_summary["rear_cl_rms"] / baseline_summary["rear_cl_rms"] - 1.0
                ),
            },
        })
    courant = [
        (float(mean), float(maximum))
        for mean, maximum in re.findall(
            r"Courant Number mean: ([0-9.eE+-]+) max: ([0-9.eE+-]+)", log
        )
    ]
    report = {
        "case": args.case_name,
        "status": "independent_openfoam_action_replay",
        "control_is_closed_loop": False,
        "action_count": config["action_count"],
        "action_range": config["rear_angular_velocity_range"],
        "max_abs_delta_omega": config["max_abs_delta_omega"],
        "solver_steps": len(re.findall(r"^Time = ", log, re.MULTILINE)),
        "courant_mean_max": max(value[0] for value in courant),
        "courant_absolute_max": max(value[1] for value in courant),
        "windows": windows,
        "note": "The action sequence is frozen before CFD; this validates replay transfer, not state-feedback closure.",
    }
    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    if args.plot:
        write_plot(args.plot, controlled, baseline, config)
    print(output, end="")


if __name__ == "__main__":
    main()
