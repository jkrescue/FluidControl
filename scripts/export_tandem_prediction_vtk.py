#!/usr/bin/env python3
"""Export PhysicsNeMo FNO rollouts as ParaView VTK time series."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pyvista as pv
import torch
from omegaconf import OmegaConf
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import load_checkpoint

from train_tandem_fno import build_model


def write_pvd(path: Path, records: list[dict]) -> None:
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
    ]
    lines.extend(
        f'    <DataSet timestep="{record["time"]:.8g}" group="" part="0" '
        f'file="{record["file"]}"/>'
        for record in records
    )
    lines.extend(("  </Collection>", "</VTKFile>"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_frame(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
    time_value: float,
    omega_value: float,
    truth_force: np.ndarray,
    predicted_force: np.ndarray,
) -> None:
    grid = pv.RectilinearGrid(x, y, np.asarray([0.0], dtype=np.float32))
    valid = mask.astype(bool)
    error = np.abs(prediction - truth)
    zeros = np.zeros_like(truth[0])
    grid.point_data["valid_mask"] = mask.ravel(order="C").astype(np.uint8)
    for prefix, values in (("ground_truth", truth), ("prediction", prediction), ("absolute_error", error)):
        vector = np.stack((values[0], values[1], zeros), axis=-1)
        grid.point_data[f"{prefix}_U"] = vector.reshape(-1, 3, order="C")
        grid.point_data[f"{prefix}_p"] = values[2].ravel(order="C")
    grid.field_data["time"] = np.asarray([time_value], dtype=np.float64)
    grid.field_data["omega"] = np.asarray([omega_value], dtype=np.float32)
    grid.field_data["ground_truth_rear_Cd_Cl"] = truth_force.astype(np.float32)
    grid.field_data["prediction_rear_Cd_Cl"] = predicted_force.astype(np.float32)
    grid.field_data["valid_point_fraction"] = np.asarray([valid.mean()], dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path, binary=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/curated/tandem_cylinders"))
    parser.add_argument("--config", type=Path, default=Path("conf/tandem_fno.yaml"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/tandem_fno_rollout/best"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--start", type=int, default=100)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--write-every", type=int, default=10)
    parser.add_argument("--include-steps", type=int, nargs="*", default=(1,))
    args = parser.parse_args()
    if args.steps < 1 or args.write_every < 1 or args.start < 0:
        raise ValueError("start must be nonnegative; steps and write-every must be positive")

    DistributedManager.initialize()
    dist = DistributedManager()
    if dist.distributed or dist.rank != 0:
        raise RuntimeError("VTK export must run as one process")
    cfg = OmegaConf.load(args.config)
    network = build_model(cfg).to(dist.device)
    metadata: dict = {}
    epoch = load_checkpoint(
        args.checkpoint_dir, models=network, metadata_dict=metadata, device=dist.device
    )
    if epoch == 0:
        raise FileNotFoundError(f"no PhysicsNeMo checkpoint found in {args.checkpoint_dir}")
    network.eval()

    stats = json.loads((args.data / "normalization.json").read_text(encoding="utf-8"))
    state_mean = torch.tensor(stats["state_mean"], device=dist.device)[:, None, None]
    state_std = torch.tensor(stats["state_std"], device=dist.device)[:, None, None]
    force_mean = torch.tensor(stats["force_mean"], device=dist.device)
    force_std = torch.tensor(stats["force_std"], device=dist.device)
    selected_steps = set(range(args.write_every, args.steps + 1, args.write_every))
    selected_steps.update(step for step in args.include_steps if 1 <= step <= args.steps)

    paths = sorted((args.data / args.split).glob("*.h5"))
    if args.cases:
        requested = set(args.cases)
        paths = [path for path in paths if path.stem in requested]
        missing = requested - {path.stem for path in paths}
        if missing:
            raise FileNotFoundError(f"cases not found in split {args.split}: {sorted(missing)}")
    if not paths:
        raise FileNotFoundError(f"no trajectories found in {args.data / args.split}")

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "checkpoint": str(args.checkpoint_dir),
        "checkpoint_epoch": epoch,
        "checkpoint_metadata": metadata,
        "split": args.split,
        "start": args.start,
        "steps": args.steps,
        "written_rollout_steps": sorted(selected_steps),
        "cases": [],
    }
    for path in paths:
        with h5py.File(path, "r") as handle:
            state = torch.from_numpy(handle["state"][:]).float().to(dist.device)
            mask = torch.from_numpy(handle["mask"][:]).float().to(dist.device)
            omega = torch.from_numpy(handle["omega"][:]).float().to(dist.device)
            rear_force = torch.from_numpy(handle["force"][:, 2:4]).float().to(dist.device)
            time_values = np.asarray(handle["time"][:]).reshape(-1)
            x = np.asarray(handle["x"][:])
            y = np.asarray(handle["y"][:])
        if args.start + args.steps >= len(state):
            raise IndexError(f"{path.stem}: start + steps exceeds {len(state) - 1}")

        active_mask = mask[args.start : args.start + 1]
        predicted = ((state[args.start : args.start + 1] - state_mean) / state_std) * active_mask
        predicted_force = None
        records = []
        case_dir = args.output / path.stem
        height, width = active_mask.shape[-2:]
        for rollout_step in range(1, args.steps + 1):
            index = args.start + rollout_step - 1
            omega_now = omega[index].reshape(1, 1, 1, 1).expand(1, 1, height, width)
            omega_next = omega[index + 1].reshape(1, 1, 1, 1).expand(1, 1, height, width)
            inputs = torch.cat((predicted, active_mask, omega_now, omega_next), dim=1)
            with torch.no_grad():
                raw = network(inputs)
                predicted = (predicted + raw[:, :3]) * active_mask
                predicted_force_normalized = (
                    (raw[:, 3:5] * active_mask).sum(dim=(-2, -1))
                    / active_mask.sum(dim=(-2, -1)).clamp_min(1)
                )
                predicted_force = predicted_force_normalized * force_std + force_mean
            if rollout_step not in selected_steps:
                continue
            target = args.start + rollout_step
            prediction_physical = ((predicted * state_std + state_mean) * active_mask)[0]
            destination = case_dir / f"rollout_{rollout_step:04d}.vtr"
            save_frame(
                destination,
                x,
                y,
                state[target].cpu().numpy(),
                prediction_physical.cpu().numpy(),
                active_mask[0, 0].cpu().numpy(),
                float(time_values[target]),
                float(omega[target]),
                rear_force[target].cpu().numpy(),
                predicted_force[0].cpu().numpy(),
            )
            records.append({
                "rollout_step": rollout_step,
                "frame": target,
                "time": float(time_values[target]),
                "omega": float(omega[target]),
                "file": destination.name,
            })
        write_pvd(case_dir / f"{path.stem}_prediction.pvd", records)
        manifest["cases"].append({"case": path.stem, "records": records})
        print(f"{path.stem}: wrote {len(records)} prediction VTK frames")

    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"PHYSICSNEMO_PREDICTION_VTK_OK epoch={epoch}")


if __name__ == "__main__":
    main()
