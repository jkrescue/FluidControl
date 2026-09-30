#!/usr/bin/env python3
"""Surrogate-only receding-horizon control with the frozen PhysicsNeMo FNO."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import load_checkpoint

from train_tandem_fno import build_model

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def load_composed_config(path: Path):
    """Resolve the Hydra defaults tree used by the training entry point."""
    with initialize_config_dir(config_dir=str(path.parent.resolve()), version_base="1.3"):
        return compose(config_name=path.stem)


def model_step(
    network, state, mask, omega_now, omega_next, action_scale, force_mean, force_std
):
    """Advance one normalized state and return physical rear Cd/Cl."""
    height, width = mask.shape[-2:]
    now = (omega_now / action_scale).reshape(1, 1, 1, 1).expand(1, 1, height, width)
    following = (omega_next / action_scale).reshape(1, 1, 1, 1).expand(1, 1, height, width)
    raw = network(torch.cat((state, mask, now, following), dim=1))
    next_state = (state + raw[:, :3]) * mask
    normalized_force = (
        (raw[:, 3:5] * mask).sum(dim=(-2, -1))
        / mask.sum(dim=(-2, -1)).clamp_min(1)
    )
    return next_state, normalized_force[0] * force_std + force_mean


def bounded_actions(parameters, current, cfg):
    increments = float(cfg.delta_omega_max) * torch.tanh(parameters)
    actions = torch.clamp(
        current + torch.cumsum(increments, dim=0),
        min=float(cfg.omega_min),
        max=float(cfg.omega_max),
    )
    return actions, torch.cat((actions[:1] - current, actions[1:] - actions[:-1]))


def optimize_actions(
    network, state, mask, current_omega, action_scale, force_mean, force_std, cfg, objective
):
    parameters = torch.zeros(int(cfg.horizon), device=state.device, requires_grad=True)
    optimizer = torch.optim.Adam([parameters], lr=float(cfg.optimization_learning_rate))
    last_components = None
    for _ in range(int(cfg.optimization_iterations)):
        optimizer.zero_grad(set_to_none=True)
        actions, rates = bounded_actions(parameters, current_omega, cfg)
        predicted = state
        previous = current_omega
        forces = []
        for action in actions:
            predicted, force = model_step(
                network, predicted, mask, previous, action, action_scale, force_mean, force_std
            )
            forces.append(force)
            previous = action
        forces = torch.stack(forces)
        cl_cost = (forces[:, 1] / float(objective.cl_scale)).square().mean()
        cd_cost = (forces[:, 0] / float(objective.cd_scale)).mean()
        action_cost = actions.square().mean()
        rate_cost = (rates / float(cfg.delta_omega_max)).square().mean()
        total = (
            float(objective.cl_weight) * cl_cost
            + float(objective.cd_weight) * cd_cost
            + float(objective.action_weight) * action_cost
            + float(objective.rate_weight) * rate_cost
        )
        total.backward()
        optimizer.step()
        last_components = (total, cl_cost, cd_cost, action_cost, rate_cost)
    actions, _ = bounded_actions(parameters, current_omega, cfg)
    assert last_components is not None
    return actions.detach(), [float(value.detach()) for value in last_components]


def simulate_zero_control(network, initial, mask, steps, action_scale, force_mean, force_std):
    state = initial.clone()
    zero = torch.zeros((), device=state.device)
    forces = []
    with torch.no_grad():
        for _ in range(steps):
            state, force = model_step(
                network, state, mask, zero, zero, action_scale, force_mean, force_std
            )
            forces.append(force.cpu().numpy())
    return np.asarray(forces), state


def summarize(force, omega, initial_omega=0.0):
    return {
        "rear_cd_mean": float(np.mean(force[:, 0])),
        "rear_cl_mean": float(np.mean(force[:, 1])),
        "rear_cl_rms": float(np.sqrt(np.mean(np.square(force[:, 1])))),
        "omega_rms": float(np.sqrt(np.mean(np.square(omega)))),
        "omega_abs_max": float(np.max(np.abs(omega))),
        "delta_omega_abs_max": float(np.max(np.abs(np.diff(np.r_[initial_omega, omega])))),
    }


def write_plots(output, rows, baseline_force, controlled_state, baseline_state,
                state_mean, state_std, mask, x, y):
    step = np.arange(1, len(rows) + 1)
    controlled_force = np.asarray([[row["rear_cd"], row["rear_cl"]] for row in rows])
    omega = np.asarray([row["omega"] for row in rows])
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    axes[0].plot(step, controlled_force[:, 0], label="MPC surrogate")
    axes[0].plot(step, baseline_force[:, 0], label="omega=0 surrogate", linestyle="--")
    axes[0].set_ylabel("Rear Cd")
    axes[0].legend()
    axes[1].plot(step, controlled_force[:, 1], label="MPC surrogate")
    axes[1].plot(step, baseline_force[:, 1], label="omega=0 surrogate", linestyle="--")
    axes[1].set_ylabel("Rear Cl")
    axes[1].legend()
    axes[2].step(step, omega, where="post")
    axes[2].set_ylabel("omega")
    axes[2].set_xlabel("Control step (Delta t=0.1)")
    fig.suptitle("Surrogate-only MPC smoke test")
    fig.savefig(output / "timeseries.png", dpi=160)
    plt.close(fig)

    controlled = ((controlled_state * state_std + state_mean) * mask)[0].cpu().numpy()
    baseline = ((baseline_state * state_std + state_mean) * mask)[0].cpu().numpy()
    valid = mask[0, 0].bool().cpu().numpy()
    extent = (float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))
    fig, axes = plt.subplots(3, 3, figsize=(16, 11), constrained_layout=True)
    for channel, label in enumerate(("u", "v", "p")):
        first = np.where(valid, controlled[channel], np.nan)
        second = np.where(valid, baseline[channel], np.nan)
        difference = np.abs(first - second)
        combined = np.concatenate((first[valid], second[valid]))
        low, high = np.nanpercentile(combined, (1, 99))
        for column, (values, title, vmin, vmax, cmap) in enumerate((
            (first, "MPC surrogate", low, high, "coolwarm"),
            (second, "omega=0 surrogate", low, high, "coolwarm"),
            (difference, "absolute difference", 0, max(np.nanpercentile(difference[valid], 99), 1e-12), "magma"),
        )):
            image = axes[channel, column].imshow(
                values, origin="lower", extent=extent, aspect="auto",
                vmin=vmin, vmax=vmax, cmap=cmap,
            )
            axes[channel, column].set_title(f"{label}: {title}")
            figure_colorbar = fig.colorbar(image, ax=axes[channel, column], shrink=0.82)
            figure_colorbar.ax.tick_params(labelsize=8)
    fig.suptitle("Final surrogate states; this is not CFD ground truth")
    fig.savefig(output / "final_fields.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("conf/tandem_mpc.yaml"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--override", action="append", default=[], metavar="KEY=VALUE",
        help="Override one config value; may be repeated.",
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    if args.smoke:
        cfg.control.steps = 3
        cfg.control.optimization_iterations = 2
        cfg.output_dir = "artifacts/tandem_mpc_smoke"
    if args.override:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.override))

    DistributedManager.initialize()
    dist = DistributedManager()
    model_cfg = load_composed_config(Path(cfg.model_config))
    if dist.cuda:
        torch.cuda.set_per_process_memory_fraction(
            float(model_cfg.training.gpu_memory_fraction), device=dist.device
        )
    network = build_model(model_cfg).to(dist.device)
    metadata = {}
    epoch = load_checkpoint(
        cfg.checkpoint_dir, models=network, metadata_dict=metadata, device=dist.device
    )
    if epoch == 0:
        raise FileNotFoundError(f"no checkpoint in {cfg.checkpoint_dir}")
    network.eval().requires_grad_(False)

    data_root = Path(cfg.data_root)
    candidates = list(data_root.glob(f"*/{cfg.case}.h5"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one HDF5 for {cfg.case}, found {candidates}")
    stats = json.loads((data_root / "normalization.json").read_text())
    data_manifest = json.loads((data_root / "manifest.json").read_text())
    action_scale = float(data_manifest.get("max_abs_omega", 1.0))
    if not np.isfinite(action_scale) or action_scale <= 0:
        raise ValueError(f"invalid max_abs_omega in data manifest: {action_scale}")
    state_mean = torch.tensor(stats["state_mean"], device=dist.device)[:, None, None]
    state_std = torch.tensor(stats["state_std"], device=dist.device)[:, None, None]
    force_mean = torch.tensor(stats["force_mean"], device=dist.device)
    force_std = torch.tensor(stats["force_std"], device=dist.device)
    with h5py.File(candidates[0], "r") as handle:
        index = int(cfg.initial_step)
        physical = torch.from_numpy(handle["state"][index]).float().to(dist.device)
        mask = torch.from_numpy(handle["mask"][index:index + 1]).float().to(dist.device)
        initial_omega = torch.tensor(float(handle["omega"][index, 0]), device=dist.device)
        x = np.asarray(handle["x"][:])
        y = np.asarray(handle["y"][:])
    initial = ((physical - state_mean) / state_std * mask[0]).unsqueeze(0)

    steps = int(cfg.control.steps)
    baseline_force, baseline_state = simulate_zero_control(
        network, initial, mask, steps, action_scale, force_mean, force_std
    )
    state = initial.clone()
    current_omega = initial_omega
    rows = []
    started = time.perf_counter()
    for control_step in range(steps):
        tick = time.perf_counter()
        actions, components = optimize_actions(
            network, state, mask, current_omega, action_scale, force_mean, force_std,
            cfg.control, cfg.objective,
        )
        next_omega = actions[0]
        with torch.no_grad():
            state, force = model_step(
                network, state, mask, current_omega, next_omega, action_scale,
                force_mean, force_std,
            )
        row = {
            "step": control_step + 1,
            "omega": float(next_omega),
            "delta_omega": float(next_omega - current_omega),
            "rear_cd": float(force[0]),
            "rear_cl": float(force[1]),
            "objective": components[0],
            "cl_cost": components[1],
            "cd_cost": components[2],
            "action_cost": components[3],
            "rate_cost": components[4],
            "optimization_seconds": time.perf_counter() - tick,
        }
        rows.append(row)
        print(json.dumps({"event": "control_step", **row}), flush=True)
        current_omega = next_omega
        if not torch.isfinite(state).all() or not torch.isfinite(force).all():
            raise FloatingPointError(f"non-finite surrogate state at control step {control_step + 1}")

    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "timeseries.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    controlled_force = np.asarray([[row["rear_cd"], row["rear_cl"]] for row in rows])
    omega = np.asarray([row["omega"] for row in rows])
    controlled_summary = summarize(controlled_force, omega, float(initial_omega))
    baseline_summary = summarize(baseline_force, np.zeros(steps))
    result = {
        "status": "surrogate_only",
        "checkpoint_epoch": epoch,
        "case": cfg.case,
        "initial_step": int(cfg.initial_step),
        "control_steps": steps,
        "horizon": int(cfg.control.horizon),
        "action_scale": action_scale,
        "controlled": controlled_summary,
        "zero_control_baseline": baseline_summary,
        "relative_change_percent": {
            "rear_cd_mean": 100.0 * (
                controlled_summary["rear_cd_mean"] / baseline_summary["rear_cd_mean"] - 1.0
            ),
            "rear_cl_rms": 100.0 * (
                controlled_summary["rear_cl_rms"] / baseline_summary["rear_cl_rms"] - 1.0
            ),
        },
        "constraint_checks": {
            "omega_within_bounds": bool(np.max(np.abs(omega)) <= float(cfg.control.omega_max) + 1e-6),
            "rate_within_bounds": bool(np.max(np.abs(np.diff(np.r_[float(initial_omega), omega]))) <= float(cfg.control.delta_omega_max) + 1e-6),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "mean_optimization_seconds": float(np.mean([row["optimization_seconds"] for row in rows])),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "note": "Surrogate-only feasibility result; not OpenFOAM or experimental validation.",
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    write_plots(
        output, rows, baseline_force, state, baseline_state,
        state_mean[None], state_std[None], mask, x, y,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
