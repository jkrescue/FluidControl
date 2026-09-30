#!/usr/bin/env python3
"""Calibrate checkpoint-committee disagreement against tandem rollout error."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import load_checkpoint

from evaluate_tandem_fno import load_composed_config
from train_tandem_fno import build_model


def rank_correlation(left: list[float], right: list[float]) -> float:
    """Return a dependency-free Spearman-style rank correlation."""
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if len(x) < 2 or np.isclose(x.std(), 0.0) or np.isclose(y.std(), 0.0):
        return float("nan")
    x_rank = np.empty(len(x), dtype=np.float64)
    y_rank = np.empty(len(y), dtype=np.float64)
    x_rank[np.argsort(x, kind="stable")] = np.arange(len(x), dtype=np.float64)
    y_rank[np.argsort(y, kind="stable")] = np.arange(len(y), dtype=np.float64)
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def summarize(rows: list[dict], thresholds: dict[str, float] | None = None) -> dict:
    result = {"segments": len(rows)}
    pairs = (
        ("state_disagreement", "state_error"),
        ("force_disagreement", "force_error"),
    )
    for uncertainty_key, error_key in pairs:
        uncertainty = np.asarray([row[uncertainty_key] for row in rows])
        error = np.asarray([row[error_key] for row in rows])
        result[uncertainty_key] = {
            "mean": float(uncertainty.mean()),
            "p90": float(np.quantile(uncertainty, 0.9)),
            "error_mean": float(error.mean()),
            "rank_correlation": rank_correlation(uncertainty.tolist(), error.tolist()),
        }
        if thresholds is not None:
            threshold = thresholds[uncertainty_key]
            trusted = uncertainty <= threshold
            result[uncertainty_key]["validation_p90_threshold"] = threshold
            result[uncertainty_key]["trusted_fraction"] = float(trusted.mean())
            result[uncertainty_key]["trusted_error_mean"] = (
                float(error[trusted].mean()) if trusted.any() else None
            )
            result[uncertainty_key]["flagged_error_mean"] = (
                float(error[~trusted].mean()) if (~trusted).any() else None
            )
            result[uncertainty_key]["flagged_to_trusted_error_ratio"] = (
                float(error[~trusted].mean() / error[trusted].mean())
                if trusted.any() and (~trusted).any() and error[trusted].mean() > 0 else None
            )
    return result


def evaluate_split(
    data: Path,
    split: str,
    horizons: list[int],
    stride: int,
    batch_size: int,
    models: list[torch.nn.Module],
    device: torch.device,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    force_mean: torch.Tensor,
    force_std: torch.Tensor,
    action_scale: float,
) -> dict[int, list[dict]]:
    rows = {horizon: [] for horizon in horizons}
    for path in sorted((data / split).glob("*.h5")):
        with h5py.File(path, "r") as handle:
            state = torch.from_numpy(handle["state"][:]).to(device)
            mask = torch.from_numpy(handle["mask"][:]).float().to(device)
            omega = torch.from_numpy(handle["omega"][:]).float().to(device)
            force = torch.from_numpy(handle["force"][:, 2:4]).float().to(device)
            time = torch.from_numpy(handle["time"][:]).float().to(device)
        normalized = ((state - state_mean) / state_std) * mask
        model_omega = omega / action_scale
        for horizon in horizons:
            starts = list(range(0, len(state) - horizon, stride))
            for offset in range(0, len(starts), batch_size):
                batch_starts = starts[offset : offset + batch_size]
                indices = torch.tensor(batch_starts, device=device, dtype=torch.long)
                active_mask = mask[indices]
                height, width = active_mask.shape[-2:]
                predictions = [normalized[indices].clone() for _ in models]
                predicted_forces = [None for _ in models]
                for rollout_offset in range(horizon):
                    step = indices + rollout_offset
                    omega_now = model_omega[step].reshape(-1, 1, 1, 1).expand(-1, 1, height, width)
                    omega_next = model_omega[step + 1].reshape(-1, 1, 1, 1).expand(-1, 1, height, width)
                    for model_index, model in enumerate(models):
                        inputs = torch.cat(
                            (predictions[model_index], active_mask, omega_now, omega_next), dim=1
                        )
                        with torch.no_grad():
                            raw = model(inputs)
                        predictions[model_index] = (
                            predictions[model_index] + raw[:, :3]
                        ) * active_mask
                        predicted_forces[model_index] = (
                            (raw[:, 3:5] * active_mask).sum(dim=(-2, -1))
                            / active_mask.sum(dim=(-2, -1)).clamp_min(1)
                        ) * force_std + force_mean

                target_indices = indices + horizon
                target = normalized[target_indices]
                point_count = active_mask.sum(dim=(1, 2, 3)).clamp_min(1)
                production_error = (
                    ((predictions[0] - target) * state_std).abs() * active_mask
                ).sum(dim=(1, 2, 3)) / (point_count * 3)
                production_force_error = (
                    predicted_forces[0] - force[target_indices]
                ).abs().mean(dim=1)

                physical_predictions = torch.stack([
                    prediction * state_std + state_mean for prediction in predictions
                ])
                state_disagreement = (
                    physical_predictions.std(dim=0, unbiased=False) * active_mask
                ).sum(dim=(1, 2, 3)) / (point_count * 3)
                force_disagreement = torch.stack(predicted_forces).std(
                    dim=0, unbiased=False
                ).mean(dim=1)

                for local_index, start in enumerate(batch_starts):
                    action = omega[start : start + horizon + 1, 0]
                    action_time = time[start : start + horizon + 1, 0]
                    rate = torch.diff(action) / torch.diff(action_time)
                    rows[horizon].append({
                        "case": path.stem,
                        "start": start,
                        "state_error": float(production_error[local_index]),
                        "force_error": float(production_force_error[local_index]),
                        "state_disagreement": float(state_disagreement[local_index]),
                        "force_disagreement": float(force_disagreement[local_index]),
                        "max_abs_omega": float(action.abs().max()),
                        "max_abs_domega_dt": float(rate.abs().max()),
                    })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/curated/tandem_cylinders_expanded_v1"))
    parser.add_argument("--config", type=Path, default=Path("conf/tandem_fno_expanded.yaml"))
    parser.add_argument("--production-checkpoint", type=Path, required=True)
    parser.add_argument("--committee-checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=(10, 50))
    parser.add_argument("--segment-stride", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    DistributedManager.initialize()
    dist = DistributedManager()
    cfg = load_composed_config(args.config)
    checkpoint_dirs = [args.production_checkpoint, *args.committee_checkpoints]
    models = []
    epochs = []
    for checkpoint_dir in checkpoint_dirs:
        model = build_model(cfg).to(dist.device)
        epoch = load_checkpoint(checkpoint_dir, models=model, device=dist.device)
        if epoch == 0:
            raise FileNotFoundError(f"no checkpoint found in {checkpoint_dir}")
        model.eval()
        models.append(model)
        epochs.append(epoch)

    stats = json.loads((args.data / "normalization.json").read_text(encoding="utf-8"))
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    state_mean = torch.tensor(stats["state_mean"], device=dist.device)[:, None, None]
    state_std = torch.tensor(stats["state_std"], device=dist.device)[:, None, None]
    force_mean = torch.tensor(stats["force_mean"], device=dist.device)
    force_std = torch.tensor(stats["force_std"], device=dist.device)

    split_rows = {}
    for split in ("validation", "test"):
        split_rows[split] = evaluate_split(
            args.data, split, args.horizons, args.segment_stride, args.batch_size,
            models, dist.device, state_mean, state_std, force_mean, force_std,
            float(manifest["max_abs_omega"]),
        )

    report = {
        "method": "heterogeneous_checkpoint_committee",
        "limitation": "diagnostic proxy; committee members use different training objectives",
        "production_checkpoint": str(args.production_checkpoint),
        "committee": [
            {"checkpoint": str(path), "epoch": epoch}
            for path, epoch in zip(checkpoint_dirs, epochs)
        ],
        "segment_stride": args.segment_stride,
        "horizons": {},
    }
    for horizon in args.horizons:
        validation_rows = split_rows["validation"][horizon]
        thresholds = {
            key: float(np.quantile([row[key] for row in validation_rows], 0.9))
            for key in ("state_disagreement", "force_disagreement")
        }
        report["horizons"][str(horizon)] = {
            "validation": summarize(validation_rows),
            "test": summarize(split_rows["test"][horizon], thresholds),
            "thresholds_from_validation": thresholds,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
