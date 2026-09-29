#!/usr/bin/env python3
"""Evaluate PhysicsNeMo FNO one-step and autoregressive predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch
from omegaconf import OmegaConf
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import load_checkpoint

from train_tandem_fno import build_model

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def save_rollout_figure(
    path: Path,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    valid_mask: np.ndarray,
    extent: tuple[float, float, float, float],
    title: str,
) -> None:
    """Write a physical-unit ground-truth/prediction/error comparison."""
    labels = ("u", "v", "p")
    figure, axes = plt.subplots(3, 3, figsize=(16, 11), constrained_layout=True)
    for channel, label in enumerate(labels):
        truth = np.where(valid_mask, ground_truth[channel], np.nan)
        predicted = np.where(valid_mask, prediction[channel], np.nan)
        error = np.abs(predicted - truth)
        combined = np.concatenate((truth[valid_mask], predicted[valid_mask]))
        vmin, vmax = np.nanpercentile(combined, (1.0, 99.0))
        if np.isclose(vmin, vmax):
            vmin, vmax = float(np.nanmin(combined)), float(np.nanmax(combined) + 1.0e-12)
        error_max = max(float(np.nanpercentile(error[valid_mask], 99.0)), 1.0e-12)
        for column, (values, heading, low, high, cmap) in enumerate((
            (truth, "Ground truth", vmin, vmax, "coolwarm"),
            (predicted, "Prediction", vmin, vmax, "coolwarm"),
            (error, "Absolute error", 0.0, error_max, "magma"),
        )):
            image = axes[channel, column].imshow(
                values,
                origin="lower",
                extent=extent,
                aspect="auto",
                vmin=low,
                vmax=high,
                cmap=cmap,
            )
            axes[channel, column].set_title(f"{label}: {heading}")
            axes[channel, column].set_xlabel("x/D")
            axes[channel, column].set_ylabel("y/D")
            figure.colorbar(image, ax=axes[channel, column], shrink=0.82)
    figure.suptitle(title, fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/curated/tandem_cylinders"))
    parser.add_argument("--config", type=Path, default=Path("conf/tandem_fno.yaml"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/tandem_fno/best"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/tandem_fno/evaluation.json"))
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--horizons", type=int, nargs="+", default=(1, 10, 50))
    parser.add_argument(
        "--visualization-dir",
        type=Path,
        default=Path("artifacts/tandem_fno/rollout_visualizations"),
    )
    parser.add_argument("--visualizations-per-horizon", type=int, default=3)
    args = parser.parse_args()

    DistributedManager.initialize()
    dist = DistributedManager()
    if dist.distributed and dist.rank != 0:
        raise RuntimeError("evaluation is a single-rank post-training step; run with python, not torchrun")
    cfg = OmegaConf.load(args.config)
    if dist.cuda:
        torch.cuda.set_per_process_memory_fraction(
            float(cfg.training.gpu_memory_fraction), device=dist.device
        )
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

    report = {
        "split": args.split,
        "checkpoint_epoch": epoch,
        "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoint_metadata": metadata,
        "cases": [],
    }
    for path in sorted((args.data / args.split).glob("*.h5")):
        with h5py.File(path, "r") as handle:
            state = torch.from_numpy(handle["state"][:]).to(dist.device)
            mask = torch.from_numpy(handle["mask"][:]).float().to(dist.device)
            omega = torch.from_numpy(handle["omega"][:]).float().to(dist.device)
            rear_force = torch.from_numpy(handle["force"][:, 2:4]).float().to(dist.device)
            x = np.asarray(handle["x"][:])
            y = np.asarray(handle["y"][:])
        normalized = (state - state_mean) / state_std
        normalized *= mask
        case_report = {"case": path.stem, "horizons": {}, "visualizations": []}
        for horizon in args.horizons:
            model_errors = []
            persistence_errors = []
            force_errors = []
            persistence_force_errors = []
            channel_max_abs = torch.zeros(3, device=dist.device)
            failed_segments = 0
            segment_starts = list(range(0, len(state) - horizon, horizon))
            visualization_count = min(args.visualizations_per_horizon, len(segment_starts))
            visualization_starts = {
                segment_starts[index]
                for index in np.linspace(
                    0, len(segment_starts) - 1, visualization_count, dtype=int
                )
            }
            for start in segment_starts:
                predicted = normalized[start : start + 1]
                active_mask = mask[start : start + 1]
                height, width = active_mask.shape[-2:]
                predicted_force_normalized = None
                segment_failed = False
                for step in range(start, start + horizon):
                    omega_now = omega[step].reshape(1, 1, 1, 1).expand(1, 1, height, width)
                    omega_next = omega[step + 1].reshape(1, 1, 1, 1).expand(1, 1, height, width)
                    inputs = torch.cat([predicted, active_mask, omega_now, omega_next], dim=1)
                    with torch.no_grad():
                        raw = network(inputs)
                        predicted = (predicted + raw[:, :3]) * active_mask
                        predicted_force_normalized = (
                            (raw[:, 3:5] * active_mask).sum(dim=(-2, -1))
                            / active_mask.sum(dim=(-2, -1)).clamp_min(1)
                        )
                    if not torch.isfinite(predicted).all() or not torch.isfinite(
                        predicted_force_normalized
                    ).all():
                        failed_segments += 1
                        segment_failed = True
                        break
                if segment_failed or predicted_force_normalized is None:
                    continue
                target = normalized[start + horizon : start + horizon + 1]
                physical_error = (predicted - target) * state_std
                persistence_error = (normalized[start : start + 1] - target) * state_std
                denom = active_mask.sum().clamp_min(1) * 3
                model_errors.append(float((physical_error.abs() * active_mask).sum() / denom))
                persistence_errors.append(float((persistence_error.abs() * active_mask).sum() / denom))
                predicted_force = predicted_force_normalized[0] * force_std + force_mean
                force_errors.append((predicted_force - rear_force[start + horizon]).abs().cpu().numpy())
                persistence_force_errors.append(
                    (rear_force[start] - rear_force[start + horizon]).abs().cpu().numpy()
                )
                predicted_physical = (predicted * state_std + state_mean) * active_mask
                channel_max_abs = torch.maximum(
                    channel_max_abs,
                    predicted_physical.abs().amax(dim=(0, 2, 3)),
                )
                if start in visualization_starts:
                    figure_path = (
                        args.visualization_dir
                        / path.stem
                        / f"horizon_{horizon:03d}_start_{start:04d}.png"
                    )
                    title = (
                        f"{path.stem} | start={start}, horizon={horizon} | "
                        f"omega={float(omega[start]):.3f}->{float(omega[start + horizon]):.3f} | "
                        f"GT Cd/Cl=({float(rear_force[start + horizon, 0]):.4f}, "
                        f"{float(rear_force[start + horizon, 1]):.4f}) | "
                        f"Pred Cd/Cl=({float(predicted_force[0]):.4f}, "
                        f"{float(predicted_force[1]):.4f})"
                    )
                    save_rollout_figure(
                        figure_path,
                        state[start + horizon].detach().cpu().numpy(),
                        predicted_physical[0].detach().cpu().numpy(),
                        active_mask[0, 0].bool().detach().cpu().numpy(),
                        (float(x[0]), float(x[-1]), float(y[0]), float(y[-1])),
                        title,
                    )
                    case_report["visualizations"].append({
                        "horizon": horizon,
                        "start": start,
                        "target": start + horizon,
                        "path": str(figure_path),
                    })
            force_errors_array = np.asarray(force_errors)
            persistence_force_array = np.asarray(persistence_force_errors)
            case_report["horizons"][str(horizon)] = {
                "state_mae_physical_units": float(np.mean(model_errors)) if model_errors else None,
                "persistence_state_mae_physical_units": (
                    float(np.mean(persistence_errors)) if persistence_errors else None
                ),
                "rear_force_mae": (
                    float(force_errors_array.mean()) if len(force_errors_array) else None
                ),
                "rear_cd_mae": (
                    float(force_errors_array[:, 0].mean()) if len(force_errors_array) else None
                ),
                "rear_cl_mae": (
                    float(force_errors_array[:, 1].mean()) if len(force_errors_array) else None
                ),
                "persistence_rear_force_mae": (
                    float(persistence_force_array.mean()) if len(persistence_force_array) else None
                ),
                "stable": failed_segments == 0,
                "failed_segments": failed_segments,
                "max_abs_predicted_u_v_p": [float(value) for value in channel_max_abs.cpu()],
                "segments": len(model_errors),
            }
        report["cases"].append(case_report)

    report["summary"] = {}
    for horizon in args.horizons:
        key = str(horizon)
        rows = [case["horizons"][key] for case in report["cases"]]
        aggregate_keys = (
            "state_mae_physical_units",
            "persistence_state_mae_physical_units",
            "rear_force_mae",
            "rear_cd_mae",
            "rear_cl_mae",
            "persistence_rear_force_mae",
        )
        report["summary"][key] = {
            metric: float(np.mean([row[metric] for row in rows if row[metric] is not None]))
            for metric in aggregate_keys
        }
        report["summary"][key].update({
            "stable": all(row["stable"] for row in rows),
            "failed_segments": sum(row["failed_segments"] for row in rows),
            "segments": sum(row["segments"] for row in rows),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
