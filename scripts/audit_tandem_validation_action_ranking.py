#!/usr/bin/env python3
"""Compare FNO and matched OpenFOAM action ranking on a validation-only panel.

This is a control-sufficiency diagnostic, never a Gate-B replacement or a
claim of learned control. All candidate actions were declared before CFD.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import h5py
import numpy as np
import torch
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import load_checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cfd" / "tandem_cylinders"))
from analyze_baseline import load_coefficients  # noqa: E402
from make_control_landscape_panel import PANEL, CASES  # noqa: E402
from screen_tandem_cem_mpc import load_composed_config, rollout_population  # noqa: E402
from train_tandem_fno import build_model  # noqa: E402


def cfd_force_at_frames(case: Path, horizon: int) -> np.ndarray:
    front = load_coefficients(case / "postProcessing/forceFront/80/coefficient.dat")
    rear = load_coefficients(case / "postProcessing/forceRear/80/coefficient.dat")
    if len(front) != len(rear):
        raise ValueError(f"unpaired CFD coefficients: {case}")
    by_time: dict[int, list[float]] = {}
    for first, last in zip(front, rear):
        if not math.isclose(first[0], last[0], abs_tol=1e-8):
            raise ValueError(f"CFD force time mismatch: {case}")
        index = round((first[0] - 80.0) / 0.1)
        expected_time = 80.0 + index * 0.1
        if 1 <= index <= horizon and abs(first[0] - expected_time) < 1e-8:
            if index in by_time:
                raise ValueError(f"duplicate CFD force frame {index}: {case}")
            by_time[index] = [first[1], first[2], last[1], last[2]]
    if set(by_time) != set(range(1, horizon + 1)):
        raise ValueError(f"missing CFD force frame: {case}")
    return np.asarray([by_time[index] for index in range(1, horizon + 1)])


def case_actions(case: Path, horizon: int) -> np.ndarray:
    cfg = json.loads((case / "case_config.json").read_text(encoding="utf-8"))
    if cfg["split"] != "validation" or cfg["panel"] != "tandem_control_landscape_validation_20261003":
        raise ValueError(f"unexpected panel provenance: {case}")
    points = cfg["action_points"]
    if len(points) < horizon + 1 or points[0] != [80.0, 0.0]:
        raise ValueError(f"invalid initial action table: {case}")
    values = []
    for index, (time, omega) in enumerate(points[1 : horizon + 1], start=1):
        if not math.isclose(time, 80.0 + index * 0.1, abs_tol=1e-8):
            raise ValueError(f"action grid mismatch: {case}")
        values.append(float(omega))
    return np.asarray(values)


def pairwise_accuracy(prediction: np.ndarray, cfd: np.ndarray) -> float:
    pairs = [(i, j) for i in range(len(cfd)) for j in range(i + 1, len(cfd))]
    correct = sum((prediction[i] - prediction[j]) * (cfd[i] - cfd[j]) > 0 for i, j in pairs)
    return correct / len(pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    if args.horizon != 100:
        parser.error("this predeclared ranking audit is fixed at 100 steps")

    data = Path("data/curated/tandem_cylinders_gate_b_aug_v3")
    source = data / "train/expanded_train_24.h5"
    with h5py.File(source, "r") as handle:
        attrs = json.loads(handle.attrs["config_json"])
        if attrs["source_restart_case"] != "tandem_backward_dt005" or attrs["source_restart_time"] != 80.0:
            raise ValueError("initial state does not match CFD source restart")
        if float(handle["time"][0, 0]) != 80.0 or float(handle["omega"][0, 0]) != 0.0:
            raise ValueError("initial state/action not at uncontrolled t=80")
        physical = torch.from_numpy(handle["state"][0]).float()
        mask = torch.from_numpy(handle["mask"][0:1]).float()
    stats = json.loads((data / "normalization.json").read_text(encoding="utf-8"))
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    names = [spec.name for spec in PANEL]
    schedules = np.stack([case_actions(CASES / name, args.horizon) for name in names])
    cfd = np.stack([cfd_force_at_frames(CASES / name, args.horizon) for name in names])
    if not np.isfinite(cfd).all():
        raise ValueError("nonfinite CFD force")

    DistributedManager.initialize()
    dist = DistributedManager()
    cfg = load_composed_config(Path("conf/tandem_fno_total_drag.yaml"))
    if dist.cuda:
        torch.cuda.set_per_process_memory_fraction(float(cfg.training.gpu_memory_fraction), device=dist.device)
    network = build_model(cfg).to(dist.device)
    metadata: dict = {}
    epoch = load_checkpoint(args.checkpoint_dir, models=network, metadata_dict=metadata, device=dist.device)
    if epoch < 1 or metadata.get("force_channels") != ["front_cd", "front_cl", "rear_cd", "rear_cl"]:
        raise ValueError("checkpoint missing or not the four-force tandem FNO")
    network.eval().requires_grad_(False)
    mask = mask.to(dist.device)
    mean = torch.as_tensor(stats["state_mean"], device=dist.device)[:, None, None]
    std = torch.as_tensor(stats["state_std"], device=dist.device)[:, None, None]
    initial = (((physical.to(dist.device) - mean) / std) * mask[0]).unsqueeze(0)
    predicted, bounds = rollout_population(
        network, initial, mask, schedules,
        current_omega=0.0,
        action_scale=float(manifest["max_abs_omega"]),
        force_mean=torch.as_tensor(stats["all_force_mean"], device=dist.device),
        force_std=torch.as_tensor(stats["all_force_std"], device=dist.device),
        batch_size=4,
    )
    cfd_drag = cfd[:, :, 0] + cfd[:, :, 2]
    predicted_drag = predicted[:, :, 0] + predicted[:, :, 2]
    true_cost = cfd_drag.mean(axis=1)
    predicted_cost = predicted_drag.mean(axis=1)
    best_pred = int(np.argmin(predicted_cost))
    best_true = int(np.argmin(true_cost))
    state_guard = 1.25 * float(stats["state_abs_normalized_max_train"])
    records = [
        {
            "case": name,
            "cfd_mean_total_cd_t80p1_to_t90": float(true_cost[index]),
            "fno_mean_total_cd_t80p1_to_t90": float(predicted_cost[index]),
            "fno_max_abs_normalized_state": float(bounds[index]),
            "fno_state_guard_exceeded": bool(bounds[index] > state_guard),
        }
        for index, name in enumerate(names)
    ]
    report = {
        "status": "VALIDATION_ACTION_RANKING_DIAGNOSTIC",
        "scientific_scope": "validation-only open-loop FNO-vs-CFD action ranking; not a closed-loop result or Gate-B pass",
        "checkpoint_epoch": int(epoch),
        "checkpoint_dir": str(args.checkpoint_dir),
        "initial_state": str(source) + ":frame0",
        "horizon_steps": args.horizon,
        "dt_action": 0.1,
        "objective": "mean system-total Cd over t=80.1..90, lower is better",
        "state_guard": state_guard,
        "cases": records,
        "pairwise_ranking_accuracy": pairwise_accuracy(predicted_cost, true_cost),
        "fno_selected_case": names[best_pred],
        "cfd_best_case": names[best_true],
        "cfd_regret_of_fno_selection": float(true_cost[best_pred] - true_cost[best_true]),
        "cfd_change_of_fno_selection_vs_zero_percent": float(100 * (true_cost[best_pred] / true_cost[0] - 1)),
        "interpretation_guard": "10-unit startup window, one phase, coarse mesh; do not extrapolate to late-window or long-horizon control",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "pairwise_ranking_accuracy", "fno_selected_case", "cfd_best_case",
        "cfd_regret_of_fno_selection", "cfd_change_of_fno_selection_vs_zero_percent"
    )}))


if __name__ == "__main__":
    main()
