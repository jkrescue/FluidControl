#!/usr/bin/env python3
"""Audit FNO action ranking against the predeclared t=86 real-CFD panel."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import load_checkpoint

REPO = Path(__file__).resolve().parents[1]
CFD = REPO / "cfd/tandem_cylinders"
CASES = CFD / "cases"
sys.path.insert(0, str(CFD))
from analyze_baseline import load_coefficients, log_health
from make_crossphase_ranking_panel import PANEL, action_points

sys.path.insert(0, str(REPO / "scripts"))
from screen_tandem_cem_mpc import load_composed_config, rollout_population
from train_tandem_fno import build_model

START_TIME = 86.0
HORIZON = 20
DATA = REPO / "data/curated/tandem_cylinders_gate_b_aug_v3"
INITIAL_H5 = (
    REPO
    / "data/curated/tandem_cylinders_phase_v1/validation"
    / "phase_validation_86_multisine.h5"
)


def sampled_forces(case: Path) -> np.ndarray:
    front = np.asarray(
        load_coefficients(case / "postProcessing/forceFront/86/coefficient.dat")
    )
    rear = np.asarray(
        load_coefficients(case / "postProcessing/forceRear/86/coefficient.dat")
    )
    if len(front) != len(rear) or not np.allclose(front[:, 0], rear[:, 0], atol=1e-9):
        raise ValueError(f"unpaired force series: {case}")
    rows = []
    for index in range(1, HORIZON + 1):
        expected = START_TIME + 0.1 * index
        matches = np.flatnonzero(np.isclose(front[:, 0], expected, atol=1e-9))
        if len(matches) != 1:
            raise ValueError(f"missing or duplicate force frame {expected}: {case}")
        row = int(matches[0])
        rows.append([front[row, 1], front[row, 2], rear[row, 1], rear[row, 2]])
    return np.asarray(rows)


def pairwise_accuracy(prediction: np.ndarray, truth: np.ndarray) -> float:
    pairs = [(i, j) for i in range(len(truth)) for j in range(i + 1, len(truth))]
    correct = sum(
        (prediction[i] - prediction[j]) * (truth[i] - truth[j]) > 0
        for i, j in pairs
    )
    return correct / len(pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--predeclared-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    predeclared = json.loads(args.predeclared_audit.read_text(encoding="utf-8"))
    if predeclared["status"] != "PREDECLARED_BEFORE_CFD":
        raise ValueError("invalid predeclared audit")
    if predeclared["leakage_audit"]["exact_initial_state_matches"] != 0:
        raise ValueError("validation initial state leaked into training")

    names = [spec.name for spec in PANEL]
    schedules = np.asarray(
        [[omega for _, omega in action_points(spec)[1:]] for spec in PANEL]
    )
    cfd = []
    health = {}
    for name in names:
        case = CASES / name
        config = json.loads((case / "case_config.json").read_text(encoding="utf-8"))
        if config["split"] != "validation_only_not_training_or_frozen_test":
            raise ValueError(f"invalid case split: {name}")
        cfd.append(sampled_forces(case))
        health[name] = log_health(case / "log.pimpleFoam.crossphase86")
        if not health[name]["solver_ended_cleanly"]:
            raise ValueError(f"incomplete solver log: {name}")
    cfd_array = np.asarray(cfd)

    with h5py.File(INITIAL_H5) as handle:
        config = json.loads(handle.attrs["config_json"])
        if handle.attrs["split"] != "validation" or config["source_restart_time"] != 86.0:
            raise ValueError("initial tensor is not the t=86 validation start")
        if not math.isclose(float(handle["time"][0, 0]), START_TIME):
            raise ValueError("initial tensor time mismatch")
        physical = torch.from_numpy(handle["state"][0]).float()
        mask = torch.from_numpy(handle["mask"][0:1]).float()

    stats = json.loads((DATA / "normalization.json").read_text(encoding="utf-8"))
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    DistributedManager.initialize()
    dist = DistributedManager()
    cfg = load_composed_config(REPO / "conf/tandem_fno_total_drag.yaml")
    network = build_model(cfg).to(dist.device)
    metadata: dict = {}
    epoch = load_checkpoint(
        args.checkpoint_dir,
        models=network,
        metadata_dict=metadata,
        device=dist.device,
    )
    if metadata.get("force_channels") != [
        "front_cd",
        "front_cl",
        "rear_cd",
        "rear_cl",
    ]:
        raise ValueError("checkpoint is not the four-force tandem FNO")
    network.eval().requires_grad_(False)
    mask = mask.to(dist.device)
    mean = torch.as_tensor(stats["state_mean"], device=dist.device)[:, None, None]
    std = torch.as_tensor(stats["state_std"], device=dist.device)[:, None, None]
    initial = (((physical.to(dist.device) - mean) / std) * mask[0]).unsqueeze(0)
    predicted, bounds = rollout_population(
        network,
        initial,
        mask,
        schedules,
        current_omega=0.0,
        action_scale=float(manifest["max_abs_omega"]),
        force_mean=torch.as_tensor(stats["all_force_mean"], device=dist.device),
        force_std=torch.as_tensor(stats["all_force_std"], device=dist.device),
        batch_size=3,
    )
    cfd_cost = (cfd_array[:, :, 0] + cfd_array[:, :, 2]).mean(axis=1)
    prediction_cost = (predicted[:, :, 0] + predicted[:, :, 2]).mean(axis=1)
    best_cfd = int(np.argmin(cfd_cost))
    best_prediction = int(np.argmin(prediction_cost))
    zero_cost = float(cfd_cost[0])
    records = []
    for index, name in enumerate(names):
        rear_lift = cfd_array[index, :, 3]
        records.append(
            {
                "case": name,
                "target_omega": PANEL[index].target,
                "cfd_mean_total_cd": float(cfd_cost[index]),
                "fno_mean_total_cd": float(prediction_cost[index]),
                "cfd_front_cd_mean": float(cfd_array[index, :, 0].mean()),
                "cfd_rear_cd_mean": float(cfd_array[index, :, 2].mean()),
                "cfd_rear_cl_mean": float(rear_lift.mean()),
                "cfd_rear_cl_fluctuation_rms": float(np.std(rear_lift)),
                "fno_max_abs_normalized_state": float(bounds[index]),
            }
        )
    report = {
        "status": "VALIDATION_ONLY_CROSSPHASE_ACTION_RANKING_DIAGNOSTIC",
        "scope": "20-step open-loop FNO-vs-real-CFD action ranking; not closed-loop benefit, not a Gate-B replacement, and not final physical acceptance",
        "predeclared_audit": str(args.predeclared_audit),
        "initial_state": str(INITIAL_H5.relative_to(REPO)) + ":frame0",
        "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoint_epoch": int(epoch),
        "inference_device": str(dist.device),
        "window": [86.1, 88.0],
        "horizon_steps": HORIZON,
        "objective": "mean system-total Cd over 20 action frames; lower is better",
        "solver_health": health,
        "cases": records,
        "pairwise_ranking_accuracy": pairwise_accuracy(prediction_cost, cfd_cost),
        "fno_selected_case": names[best_prediction],
        "cfd_best_case": names[best_cfd],
        "cfd_regret_of_fno_selection": float(cfd_cost[best_prediction] - cfd_cost[best_cfd]),
        "cfd_change_of_fno_selection_vs_zero_percent": float(
            100.0 * (cfd_cost[best_prediction] / zero_cost - 1.0)
        ),
        "interpretation_guard": "single validation phase and only 2 D/U; lift mean is phase-biased and no drag-reduction or closed-loop claim is permitted",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "pairwise_ranking_accuracy",
                    "fno_selected_case",
                    "cfd_best_case",
                    "cfd_regret_of_fno_selection",
                    "cfd_change_of_fno_selection_vs_zero_percent",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
