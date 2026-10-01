#!/usr/bin/env python3
"""Fail-closed parity check: official HydroGym FlowEnv vs existing FNO evaluator math."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from hydrogym import FlowEnv
from physicsnemo.utils import load_checkpoint

from evaluate_tandem_fno import load_composed_config
from fluid_control.tandem_hydrogym import TandemFNOStepper, TandemRewardAudit, TandemSurrogateFlow
from train_tandem_fno import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case", default="expanded_validation_00")
    parser.add_argument("--frame", type=int, default=100)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if not (args.data / "validation" / f"{args.case}.h5").is_file():
        parser.error("parity case must be a real validation HDF5")
    cfg = load_composed_config(args.config)
    network = build_model(cfg).cpu().eval()
    epoch = load_checkpoint(
        args.checkpoint_dir, models=network, device=torch.device("cpu")
    )
    if epoch < 1:
        raise FileNotFoundError("missing PhysicsNeMo checkpoint")
    with h5py.File(args.data / "validation" / f"{args.case}.h5", "r") as handle:
        if not 0 <= args.frame < len(handle["omega"]) - 1:
            parser.error("frame must have a next CFD action")
        next_omega = float(handle["omega"][args.frame + 1, 0])
    env = TandemRewardAudit(FlowEnv({
        "flow": TandemSurrogateFlow,
        "flow_config": {
            "data_root": args.data,
            "checkpoint_epoch": epoch,
            "split": "validation",
            "case": args.case,
            "frame": args.frame,
            "network": network,
            "device": "cpu",
            "max_delta_omega": 10.0,  # Disable rate clipping only for exact parity.
        },
        "solver": TandemFNOStepper,
        "solver_config": {"dt": 0.1},
        "max_steps": 2,
    }))
    initial_obs, _ = env.reset()
    flow = env.env.flow
    initial_field = flow.q.clone()
    mask = flow.mask
    height, width = mask.shape[-2:]
    input_tensor = torch.cat((
        initial_field[None],
        mask[None],
        torch.full((1, 1, height, width), flow.omega / flow.MAX_CONTROL),
        torch.full((1, 1, height, width), next_omega / flow.MAX_CONTROL),
    ), dim=1)
    with torch.inference_mode():
        raw = network(input_tensor)
        expected_field = (initial_field + raw[0, :3]) * mask
        expected_force_normalized = (
            (raw[0, 3:5] * mask).sum(dim=(-2, -1))
            / mask.sum(dim=(-2, -1)).clamp_min(1)
        )
        expected_force = (
            expected_force_normalized.numpy() * flow.force_std + flow.force_mean
        )
    observation, reward, terminated, truncated, info = env.step([next_omega])
    state_error = float((flow.q - expected_field).abs().max())
    force_error = float(np.max(np.abs(flow.force - expected_force)))
    reward_error = abs(reward + 0.1 * sum(flow.objective_terms().values()))
    if state_error > 1e-6 or force_error > 1e-6 or reward_error > 1e-8:
        raise AssertionError(
            f"adapter disagrees with evaluator: {state_error}, {force_error}, {reward_error}"
        )
    if observation.shape != (67,) or not np.isfinite(observation).all():
        raise AssertionError("invalid HydroGym observation")
    if terminated or truncated:
        raise AssertionError("unexpected first-step HydroGym termination")
    reward_components = [value for key, value in info.items() if key.startswith("reward_")]
    if abs(sum(reward_components) - reward) > 1e-8:
        raise AssertionError("reward components do not sum to HydroGym reward")
    if info["checkpoint_epoch"] != epoch or info["source_case"] != args.case:
        raise AssertionError("adapter omitted model/data provenance")
    if info["backend"] != "physicsnemo_fno_surrogate_not_cfd":
        raise AssertionError("adapter backend is not labeled surrogate-only")
    reset_obs, _ = env.reset()
    flow.max_delta_omega = 0.5
    opposite_limit = flow.MAX_CONTROL if flow.omega < 0 else -flow.MAX_CONTROL
    _, _, _, _, limited_info = env.step([opposite_limit])
    if not limited_info["rate_limited"]:
        raise AssertionError("rate limit did not activate")
    if abs(limited_info["applied_delta_omega"]) > 0.5 + 1e-6:
        raise AssertionError("applied action exceeded rate limit")
    env.reset()
    try:
        env.step([flow.MAX_CONTROL + 1.0])
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-support action was not rejected")
    reset_error = float(np.max(np.abs(initial_obs - reset_obs)))
    if reset_error > 1e-7 or flow.omega != float(initial_obs[-1]):
        raise AssertionError("HydroGym reset is not deterministic")
    report = {
        "status": "TANDEM_HYDROGYM_PARITY_OK",
        "case": args.case,
        "frame": args.frame,
        "checkpoint_epoch": epoch,
        "max_abs_state_difference": state_error,
        "reward_components_sum_difference": abs(sum(reward_components) - reward),
        "rate_limit_activated": bool(limited_info["rate_limited"]),
        "max_abs_force_difference": force_error,
        "out_of_support_action_rejected": True,
        "reward_sum_difference": reward_error,
        "reset_observation_difference": reset_error,
        "observation_width": len(observation),
        "source": "real Curator HDF5 frame; FNO prediction is surrogate-only",
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
