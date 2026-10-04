#!/usr/bin/env python3
"""Read-only train-pair scale audit for a fixed official FNO checkpoint."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import hydra
import torch
from fluid_control.paired_force_statistics import paired_statistic_loss
from fluid_control.paired_stat_datapipe import MatchedPairStatDataset
from fluid_control.paired_training import combine_paired_rollout_batch
from omegaconf import DictConfig
from physicsnemo.datapipes import DataLoader
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import load_checkpoint

from train_tandem_fno import build_model, configured_force_indices, predict


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@hydra.main(version_base="1.3", config_path="../conf", config_name="tandem_fno_paired_stats_h100")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()
    if not dist.cuda or dist.world_size != 1:
        raise RuntimeError("paired parent probe requires exactly one CUDA GPU")
    torch.cuda.set_per_process_memory_fraction(
        float(cfg.training.gpu_memory_fraction), device=dist.device
    )
    dataset = MatchedPairStatDataset(
        cfg.data.paired_root, cfg.data.paired_manifest, num_workers=1
    )
    loader = DataLoader(
        dataset, batch_size=int(cfg.training.paired_batch_size), shuffle=False,
        prefetch_factor=0, use_streams=False,
    )
    network = build_model(cfg).to(dist.device)
    metadata = {}
    epoch = load_checkpoint(
        Path(cfg.training.initial_checkpoint), models=network,
        metadata_dict=metadata, device=dist.device,
    )
    if epoch == 0:
        raise FileNotFoundError("fixed parent checkpoint did not load")
    channels = tuple(dataset.manifest["force_channels"])
    if tuple(configured_force_indices(cfg)) != (0, 1, 2, 3):
        raise ValueError("paired probe requires all four force channels")
    force_mean = dataset.force_mean.to(dist.device)
    force_std = dataset.force_std.to(dist.device)
    squared_error = torch.zeros(3, 3, dtype=torch.float64, device=dist.device)
    count = 0
    records = []
    with torch.no_grad():
        for pair in loader:
            pair = {key: value.to(dist.device) for key, value in pair.items()}
            combined = combine_paired_rollout_batch(pair)
            prediction = []
            state = combined["state"]
            mask = combined["mask"]
            omega = combined["omega"]
            height, width = mask.shape[-2:]
            for step in range(100):
                omega_now = omega[:, step].reshape(-1, 1, 1, 1).expand(-1, 1, height, width)
                omega_next = omega[:, step + 1].reshape(-1, 1, 1, 1).expand(-1, 1, height, width)
                delta, force = predict(
                    network, torch.cat((state, mask, omega_now, omega_next), dim=1), mask
                )
                state = (state + delta) * mask
                prediction.append(force)
            prediction = torch.stack(prediction, dim=1)
            batch = pair["action_state"].shape[0]
            loss, predicted_delta, target_delta = paired_statistic_loss(
                prediction[:batch], prediction[batch:], combined["action_force"],
                combined["zero_force"], force_mean, force_std, channels,
            )
            if not torch.allclose(target_delta, combined["paired_targets"], rtol=2e-5, atol=2e-6):
                raise RuntimeError("runtime pair targets differ")
            front, rear, lift = channels.index("front_cd"), channels.index("rear_cd"), channels.index("rear_cl")
            scales = torch.stack((
                torch.sqrt(force_std[front].square() + force_std[rear].square()),
                force_std[lift], force_std[lift],
            ))
            normalized_error = (predicted_delta - target_delta) / scales
            squared_error += normalized_error.double().square().sum(dim=0)
            count += batch
            records.append({"batch_size": batch, "loss": float(loss)})
    component_rmse = torch.sqrt(squared_error / count).cpu()
    result = {
        "status": "TRAIN20_PAIRED_STAT_PARENT_SCALE_AUDIT_COMPLETE",
        "training_executed": False,
        "validation_or_frozen_accessed": False,
        "parent_checkpoint_sha256": sorted(
            (path.name, sha(path)) for path in Path(cfg.training.initial_checkpoint).glob("*") if path.is_file()
        ),
        "pair_manifest_sha256": sha(Path(cfg.data.paired_manifest)),
        "pair_count": count,
        "horizons": [20, 50, 100],
        "statistics": ["mean_total_cd", "mean_rear_cl", "rear_cl_fluctuation_rms"],
        "normalized_component_rmse": component_rmse.tolist(),
        "normalized_mse": float(squared_error.sum().cpu() / (count * 9)),
        "records": records,
    }
    output = Path(cfg.probe_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.link(temporary, output)
    temporary.unlink()
    print(json.dumps({"status": result["status"], "normalized_mse": result["normalized_mse"]}))
    dataset.close()


if __name__ == "__main__":
    main()
