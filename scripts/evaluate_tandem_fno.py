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
from hydra import compose, initialize_config_dir
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import load_checkpoint

from train_tandem_fno import build_model, configured_force_indices

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def load_composed_config(path: Path):
    """Resolve the Hydra defaults tree used by the training entry point."""
    with initialize_config_dir(config_dir=str(path.parent.resolve()), version_base="1.3"):
        return compose(config_name=path.stem)


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
    parser.add_argument(
        "--segment-stride", type=int, default=None,
        help="rollout start stride; defaults to the horizon for backward compatibility",
    )
    parser.add_argument("--evaluation-batch-size", type=int, default=8)
    parser.add_argument(
        "--action-mode",
        choices=("observed", "zero", "sign_flip", "shuffle"),
        default="observed",
        help="action input used during rollout; non-observed modes diagnose action conditioning",
    )
    parser.add_argument(
        "--action-seed", type=int, default=20260930,
        help="deterministic seed used by --action-mode shuffle",
    )
    parser.add_argument(
        "--segment-metrics-output", type=Path, default=None,
        help="optional JSON file with per-segment errors and action statistics",
    )
    args = parser.parse_args()

    DistributedManager.initialize()
    dist = DistributedManager()
    if dist.distributed and dist.rank != 0:
        raise RuntimeError("evaluation is a single-rank post-training step; run with python, not torchrun")
    cfg = load_composed_config(args.config)
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
    data_manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    action_scale = float(data_manifest.get("max_abs_omega", 1.0))
    state_mean = torch.tensor(stats["state_mean"], device=dist.device)[:, None, None]
    state_std = torch.tensor(stats["state_std"], device=dist.device)[:, None, None]
    force_indices = configured_force_indices(cfg)
    if force_indices == (2, 3):
        force_channels = tuple(stats["force_channels"])
        force_mean = torch.tensor(stats["force_mean"], device=dist.device)
        force_std = torch.tensor(stats["force_std"], device=dist.device)
    elif force_indices == (0, 1, 2, 3):
        force_channels = tuple(stats["all_force_channels"])
        force_mean = torch.tensor(stats["all_force_mean"], device=dist.device)
        force_std = torch.tensor(stats["all_force_std"], device=dist.device)
    else:
        raise ValueError(f"unsupported force_indices: {force_indices}")

    report = {
        "split": args.split,
        "checkpoint_epoch": epoch,
        "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoint_metadata": metadata,
        "segment_stride": args.segment_stride,
        "evaluation_batch_size": args.evaluation_batch_size,
        "action_scale": action_scale,
        "action_mode": args.action_mode,
        "action_seed": args.action_seed,
        "force_indices": list(force_indices),
        "force_channels": list(force_channels),
        "cases": [],
    }
    segment_records = []
    for case_index, path in enumerate(sorted((args.data / args.split).glob("*.h5"))):
        with h5py.File(path, "r") as handle:
            state = torch.from_numpy(handle["state"][:]).to(dist.device)
            mask = torch.from_numpy(handle["mask"][:]).float().to(dist.device)
            omega = torch.from_numpy(handle["omega"][:]).float().to(dist.device)
            selected_force = torch.from_numpy(
                handle["force"][:, list(force_indices)]
            ).float().to(dist.device)
            time = torch.from_numpy(handle["time"][:]).float().to(dist.device)
            x = np.asarray(handle["x"][:])
            y = np.asarray(handle["y"][:])
        normalized = (state - state_mean) / state_std
        normalized *= mask
        observed_model_omega = omega / action_scale
        if args.action_mode == "observed":
            model_omega = observed_model_omega
        elif args.action_mode == "zero":
            model_omega = torch.zeros_like(observed_model_omega)
        elif args.action_mode == "sign_flip":
            model_omega = -observed_model_omega
        else:
            rng = np.random.default_rng(args.action_seed + case_index)
            permutation = torch.as_tensor(
                rng.permutation(len(observed_model_omega)), device=dist.device
            )
            model_omega = observed_model_omega[permutation]
        case_report = {"case": path.stem, "horizons": {}, "visualizations": []}
        for horizon in args.horizons:
            model_errors = []
            channel_errors = []
            persistence_errors = []
            force_errors = []
            persistence_force_errors = []
            total_drag_errors = []
            persistence_total_drag_errors = []
            channel_max_abs = torch.zeros(3, device=dist.device)
            failed_segments = 0
            segment_stride = args.segment_stride or horizon
            if segment_stride < 1:
                raise ValueError(f"segment stride must be positive, got {segment_stride}")
            segment_starts = list(range(0, len(state) - horizon, segment_stride))
            visualization_count = min(args.visualizations_per_horizon, len(segment_starts))
            visualization_starts = {
                segment_starts[index]
                for index in np.linspace(
                    0, len(segment_starts) - 1, visualization_count, dtype=int
                )
            }
            for batch_offset in range(0, len(segment_starts), args.evaluation_batch_size):
                starts = segment_starts[batch_offset : batch_offset + args.evaluation_batch_size]
                start_indices = torch.tensor(starts, dtype=torch.long, device=dist.device)
                predicted = normalized[start_indices]
                active_mask = mask[start_indices]
                height, width = active_mask.shape[-2:]
                predicted_force_normalized = None
                for offset in range(horizon):
                    step_indices = start_indices + offset
                    omega_now = model_omega[step_indices].reshape(-1, 1, 1, 1).expand(-1, 1, height, width)
                    omega_next = model_omega[step_indices + 1].reshape(-1, 1, 1, 1).expand(-1, 1, height, width)
                    inputs = torch.cat([predicted, active_mask, omega_now, omega_next], dim=1)
                    with torch.no_grad():
                        raw = network(inputs)
                        if raw.shape[1] != 3 + len(force_indices):
                            raise ValueError(
                                "checkpoint output channels do not match configured force targets"
                            )
                        predicted = (predicted + raw[:, :3]) * active_mask
                        predicted_force_normalized = (
                            (raw[:, 3:] * active_mask).sum(dim=(-2, -1))
                            / active_mask.sum(dim=(-2, -1)).clamp_min(1)
                        )
                if predicted_force_normalized is None:
                    continue
                finite = torch.isfinite(predicted).flatten(1).all(dim=1)
                finite &= torch.isfinite(predicted_force_normalized).all(dim=1)
                failed_segments += int((~finite).sum())
                if not finite.any():
                    continue
                target_indices = start_indices + horizon
                target = normalized[target_indices]
                physical_error = (predicted - target) * state_std
                persistence_error = (normalized[start_indices] - target) * state_std
                point_count = active_mask.sum(dim=(1, 2, 3)).clamp_min(1)
                model_mae = (physical_error.abs() * active_mask).sum(dim=(1, 2, 3)) / (point_count * 3)
                channel_mae = (physical_error.abs() * active_mask).sum(dim=(2, 3)) / point_count[:, None]
                persistence_mae = (
                    (persistence_error.abs() * active_mask).sum(dim=(1, 2, 3))
                    / (point_count * 3)
                )
                predicted_force = predicted_force_normalized * force_std + force_mean
                force_error = (predicted_force - selected_force[target_indices]).abs()
                persistence_force_error = (
                    selected_force[start_indices] - selected_force[target_indices]
                ).abs()
                if "front_cd" in force_channels and "rear_cd" in force_channels:
                    front_cd_index = force_channels.index("front_cd")
                    rear_cd_index = force_channels.index("rear_cd")
                    predicted_total_cd = (
                        predicted_force[:, front_cd_index]
                        + predicted_force[:, rear_cd_index]
                    )
                    target_total_cd = (
                        selected_force[target_indices, front_cd_index]
                        + selected_force[target_indices, rear_cd_index]
                    )
                    persistence_total_cd = (
                        selected_force[start_indices, front_cd_index]
                        + selected_force[start_indices, rear_cd_index]
                    )
                    total_drag_errors.extend(
                        (predicted_total_cd - target_total_cd).abs()[finite].cpu().tolist()
                    )
                    persistence_total_drag_errors.extend(
                        (persistence_total_cd - target_total_cd).abs()[finite].cpu().tolist()
                    )
                model_errors.extend(model_mae[finite].cpu().tolist())
                channel_errors.extend(channel_mae[finite].cpu().tolist())
                persistence_errors.extend(persistence_mae[finite].cpu().tolist())
                force_errors.extend(force_error[finite].cpu().tolist())
                persistence_force_errors.extend(persistence_force_error[finite].cpu().tolist())
                if args.segment_metrics_output is not None:
                    for local_index, start in enumerate(starts):
                        if not bool(finite[local_index]):
                            continue
                        action_window = omega[start : start + horizon + 1, 0]
                        time_window = time[start : start + horizon + 1, 0]
                        action_rate = torch.diff(action_window) / torch.diff(time_window)
                        force_error_by_channel = {
                            channel: float(force_error[local_index, channel_index])
                            for channel_index, channel in enumerate(force_channels)
                        }
                        segment_record = {
                            "case": path.stem,
                            "horizon": horizon,
                            "start": start,
                            "mean_abs_omega": float(action_window.abs().mean()),
                            "max_abs_omega": float(action_window.abs().max()),
                            "mean_abs_domega_dt": float(action_rate.abs().mean()),
                            "max_abs_domega_dt": float(action_rate.abs().max()),
                            "state_mae_physical_units": float(model_mae[local_index]),
                            "force_channel_absolute_error": force_error_by_channel,
                        }
                        if "rear_cd" in force_channels and "rear_cl" in force_channels:
                            rear_cd_index = force_channels.index("rear_cd")
                            rear_cl_index = force_channels.index("rear_cl")
                            segment_record.update({
                                "rear_force_mae": float(
                                    force_error[local_index, [rear_cd_index, rear_cl_index]].mean()
                                ),
                                "rear_cd_mae": float(force_error[local_index, rear_cd_index]),
                                "rear_cl_mae": float(force_error[local_index, rear_cl_index]),
                            })
                        if "front_cd" in force_channels and "rear_cd" in force_channels:
                            front_cd_index = force_channels.index("front_cd")
                            rear_cd_index = force_channels.index("rear_cd")
                            predicted_total_cd = (
                                predicted_force[local_index, front_cd_index]
                                + predicted_force[local_index, rear_cd_index]
                            )
                            target_total_cd = (
                                selected_force[target_indices[local_index], front_cd_index]
                                + selected_force[target_indices[local_index], rear_cd_index]
                            )
                            segment_record["total_drag_absolute_error"] = float(
                                (predicted_total_cd - target_total_cd).abs()
                            )
                        segment_records.append(segment_record)
                predicted_physical = (predicted * state_std + state_mean) * active_mask
                channel_max_abs = torch.maximum(
                    channel_max_abs,
                    predicted_physical[finite].abs().amax(dim=(0, 2, 3)),
                )
                for local_index, start in enumerate(starts):
                    if start not in visualization_starts or not bool(finite[local_index]):
                        continue
                    figure_path = (
                        args.visualization_dir
                        / path.stem
                        / f"horizon_{horizon:03d}_start_{start:04d}.png"
                    )
                    force_summary = ", ".join(
                        f"{channel}={float(predicted_force[local_index, channel_index]):.3f}"
                        for channel_index, channel in enumerate(force_channels)
                    )
                    title = (
                        f"{path.stem} | start={start}, horizon={horizon} | "
                        f"omega={float(omega[start]):.3f}->{float(omega[start + horizon]):.3f} | "
                        f"Predicted forces: {force_summary}"
                    )
                    save_rollout_figure(
                        figure_path,
                        state[start + horizon].detach().cpu().numpy(),
                        predicted_physical[local_index].detach().cpu().numpy(),
                        active_mask[local_index, 0].bool().detach().cpu().numpy(),
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
            channel_errors_array = np.asarray(channel_errors)
            persistence_force_array = np.asarray(persistence_force_errors)
            force_channel_mae = (
                {
                    channel: float(force_errors_array[:, channel_index].mean())
                    for channel_index, channel in enumerate(force_channels)
                }
                if len(force_errors_array) else None
            )
            horizon_report = {
                "state_mae_physical_units": float(np.mean(model_errors)) if model_errors else None,
                "state_channel_mae_u_v_p": (
                    channel_errors_array.mean(axis=0).tolist() if len(channel_errors_array) else None
                ),
                "persistence_state_mae_physical_units": (
                    float(np.mean(persistence_errors)) if persistence_errors else None
                ),
                "force_mae": float(force_errors_array.mean()) if len(force_errors_array) else None,
                "force_channel_mae": force_channel_mae,
                "persistence_force_mae": (
                    float(persistence_force_array.mean()) if len(persistence_force_array) else None
                ),
                "stable": failed_segments == 0,
                "failed_segments": failed_segments,
                "max_abs_predicted_u_v_p": [float(value) for value in channel_max_abs.cpu()],
                "segments": len(model_errors),
                "segment_stride": segment_stride,
            }
            if len(force_errors_array) and "rear_cd" in force_channels:
                rear_cd_index = force_channels.index("rear_cd")
                rear_cl_index = force_channels.index("rear_cl")
                horizon_report.update({
                    "rear_force_mae": float(
                        force_errors_array[:, [rear_cd_index, rear_cl_index]].mean()
                    ),
                    "rear_cd_mae": float(force_errors_array[:, rear_cd_index].mean()),
                    "rear_cl_mae": float(force_errors_array[:, rear_cl_index].mean()),
                    "persistence_rear_force_mae": float(
                        persistence_force_array[:, [rear_cd_index, rear_cl_index]].mean()
                    ),
                })
            if len(force_errors_array) and "front_cd" in force_channels:
                front_cd_index = force_channels.index("front_cd")
                front_cl_index = force_channels.index("front_cl")
                horizon_report.update({
                    "front_force_mae": float(
                        force_errors_array[:, [front_cd_index, front_cl_index]].mean()
                    ),
                    "front_cd_mae": float(force_errors_array[:, front_cd_index].mean()),
                    "front_cl_mae": float(force_errors_array[:, front_cl_index].mean()),
                    "total_drag_mae": float(np.mean(total_drag_errors)),
                    "persistence_total_drag_mae": float(
                        np.mean(persistence_total_drag_errors)
                    ),
                })
            case_report["horizons"][str(horizon)] = horizon_report
        report["cases"].append(case_report)

    report["summary"] = {}
    for horizon in args.horizons:
        key = str(horizon)
        rows = [case["horizons"][key] for case in report["cases"]]
        aggregate_keys = (
            "state_mae_physical_units",
            "persistence_state_mae_physical_units",
            "force_mae",
            "persistence_force_mae",
            "rear_force_mae",
            "rear_cd_mae",
            "rear_cl_mae",
            "persistence_rear_force_mae",
            "front_force_mae",
            "front_cd_mae",
            "front_cl_mae",
            "total_drag_mae",
            "persistence_total_drag_mae",
        )
        report["summary"][key] = {
            metric: float(np.mean([row[metric] for row in rows if row.get(metric) is not None]))
            for metric in aggregate_keys
            if any(row.get(metric) is not None for row in rows)
        }
        report["summary"][key]["force_channel_mae"] = {
            channel: float(np.mean([
                row["force_channel_mae"][channel]
                for row in rows if row["force_channel_mae"] is not None
            ]))
            for channel in force_channels
        }
        report["summary"][key]["state_channel_mae_u_v_p"] = np.mean(
            [row["state_channel_mae_u_v_p"] for row in rows], axis=0
        ).tolist()
        report["summary"][key].update({
            "stable": all(row["stable"] for row in rows),
            "failed_segments": sum(row["failed_segments"] for row in rows),
            "segments": sum(row["segments"] for row in rows),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.segment_metrics_output is not None:
        args.segment_metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.segment_metrics_output.write_text(
            json.dumps({
                "action_mode": args.action_mode,
                "split": args.split,
                "segments": segment_records,
            }, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
