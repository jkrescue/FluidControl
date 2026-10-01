#!/usr/bin/env python3
"""Train the action-conditioned FNO with the standard PhysicsNeMo runtime."""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

import hydra
import numpy as np
import torch
from fluid_control.tandem_datapipe import TandemWindowDataset
from omegaconf import DictConfig, OmegaConf
from physicsnemo.datapipes import DataLoader
from physicsnemo.distributed import DistributedManager
from physicsnemo.models.fno import FNO
from physicsnemo.utils import (
    StaticCaptureEvaluateNoGrad,
    StaticCaptureTraining,
    load_checkpoint,
    save_checkpoint,
)
from physicsnemo.utils.logging import LaunchLogger, PythonLogger
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler


def build_model(cfg: DictConfig) -> FNO:
    """Build the PhysicsNeMo FNO declared in the Hydra configuration."""
    return FNO(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        dimension=2,
        latent_channels=cfg.model.latent_channels,
        num_fno_layers=cfg.model.num_fno_layers,
        num_fno_modes=list(cfg.model.num_fno_modes),
        decoder_layers=cfg.model.decoder_layers,
        decoder_layer_size=cfg.model.decoder_layer_size,
        padding=cfg.model.padding,
        coord_features=cfg.model.coord_features,
    )


def predict(network: torch.nn.Module, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    raw = network(x)
    force = (raw[:, 3:5] * mask).sum(dim=(-2, -1)) / mask.sum(dim=(-2, -1)).clamp_min(1)
    return raw[:, :3], force


def reduce_totals(values: torch.Tensor, dist: DistributedManager) -> torch.Tensor:
    if dist.distributed:
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
    return values


def prune_checkpoint_dir(directory: Path, keep_last: int = 2) -> None:
    """Retain only the newest complete PhysicsNeMo checkpoint generations."""
    if keep_last < 1 or not directory.exists():
        return
    pattern = re.compile(r"^(?:checkpoint|.+)\.0\.(\d+)\.(?:pt|mdlus)$")
    by_epoch: dict[int, list[Path]] = {}
    for path in directory.iterdir():
        match = pattern.match(path.name)
        if match:
            by_epoch.setdefault(int(match.group(1)), []).append(path)
    retained = set(sorted(by_epoch)[-keep_last:])
    for epoch, paths in by_epoch.items():
        if epoch not in retained:
            for path in paths:
                path.unlink()


def validation_metrics(forward_eval, loader: DataLoader, device: torch.device,
                       state_std: torch.Tensor, dist: DistributedManager) -> dict[str, float]:
    totals = torch.zeros(5, dtype=torch.float64, device=device)
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        delta, force = forward_eval(batch["x"], batch["mask"])
        error = (delta - batch["delta"]) * state_std
        totals[0] += (error.abs() * batch["mask"]).sum().double()
        totals[1] += (error.square() * batch["mask"]).sum().double()
        totals[2] += (batch["mask"].sum() * 3).double()
        totals[3] += (force - batch["force"]).abs().sum().double()
        totals[4] += force.numel()
    totals = reduce_totals(totals, dist).cpu()
    return {
        "state_mae_physical_units": float(totals[0] / totals[2]),
        "state_rmse_physical_units": float(torch.sqrt(totals[1] / totals[2])),
        "force_mae_normalized": float(totals[3] / totals[4]),
    }


@hydra.main(version_base="1.3", config_path="../conf", config_name="tandem_fno")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()
    if not dist.cuda:
        raise RuntimeError("CUDA is required for the planned PhysicsNeMo training run")

    memory_fraction = float(cfg.training.gpu_memory_fraction)
    if not 0.0 < memory_fraction <= 1.0:
        raise ValueError(f"gpu_memory_fraction must be in (0, 1], got {memory_fraction}")
    torch.cuda.set_per_process_memory_fraction(memory_fraction, device=dist.device)
    device_properties = torch.cuda.get_device_properties(dist.device)
    memory_budget_bytes = int(device_properties.total_memory * memory_fraction)
    memory_reserve_bytes = device_properties.total_memory - memory_budget_bytes

    output = Path(cfg.output_dir).resolve()
    checkpoint_dir = output / "checkpoints"
    best_dir = output / "best"
    if dist.rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
        runtime = {
            "world_size": dist.world_size,
            "device_name": device_properties.name,
            "device_total_bytes": device_properties.total_memory,
            "gpu_memory_fraction": memory_fraction,
            "per_process_budget_bytes": memory_budget_bytes,
            "reserved_bytes": memory_reserve_bytes,
            "nccl_environment": {
                key: os.environ.get(key)
                for key in (
                    "NCCL_P2P_DISABLE",
                    "NCCL_SHM_DISABLE",
                    "NCCL_IB_DISABLE",
                    "NCCL_CUMEM_ENABLE",
                    "NCCL_CUMEM_HOST_ENABLE",
                    "NCCL_SOCKET_IFNAME",
                )
            },
        }
        (output / "runtime_metadata.json").write_text(
            json.dumps(runtime, indent=2) + "\n", encoding="utf-8"
        )
    if dist.distributed:
        torch.distributed.barrier()

    pylog = PythonLogger(name="tandem_fno")
    if dist.rank == 0:
        pylog.file_logging(str(output / "physicsnemo.log"))
    LaunchLogger.initialize()

    seed = int(cfg.training.seed) + dist.rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    smoke = bool(cfg.training.smoke)
    train = TandemWindowDataset(
        cfg.data.root, "train", stride=int(cfg.training.get("smoke_train_stride", 8)) if smoke else 1,
        num_workers=cfg.training.workers,
    )
    validation = TandemWindowDataset(
        cfg.data.root, "validation", stride=int(cfg.training.get("smoke_validation_stride", 16)) if smoke else 2,
        num_workers=cfg.training.workers,
    )
    train_sampler = DistributedSampler(train, shuffle=True, seed=cfg.training.seed) if dist.distributed else None
    validation_sampler = DistributedSampler(validation, shuffle=False) if dist.distributed else None
    loader_args = {
        "batch_size": int(cfg.training.batch_size),
        "prefetch_factor": int(cfg.data.prefetch_factor),
        "num_streams": int(cfg.data.num_streams),
        "use_streams": True,
        "seed": int(cfg.training.seed),
    }
    train_loader = DataLoader(train, shuffle=train_sampler is None, sampler=train_sampler, **loader_args)
    validation_loader = DataLoader(validation, shuffle=False, sampler=validation_sampler, **loader_args)

    network: torch.nn.Module = build_model(cfg).to(dist.device)
    if dist.distributed:
        network = DistributedDataParallel(
            network,
            device_ids=[dist.local_rank],
            output_device=dist.local_rank,
            broadcast_buffers=dist.broadcast_buffers,
            find_unused_parameters=dist.find_unused_parameters,
        )
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay
    )
    epochs = 1 if smoke else int(cfg.training.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    @StaticCaptureTraining(
        model=network, optim=optimizer, logger=pylog, use_graphs=False,
        use_amp=False,
        gradient_clip_norm=cfg.training.gradient_clip_norm,
        label="tandem_fno_train",
    )
    def forward_train(x, target_delta, target_force, mask):
        delta, force = predict(network, x, mask)
        field_loss = ((delta - target_delta).square() * mask).sum() / (mask.sum().clamp_min(1) * 3)
        force_loss = (force - target_force).square().mean()
        return field_loss + cfg.training.force_loss_weight * force_loss

    @StaticCaptureEvaluateNoGrad(
        model=network, logger=pylog, use_graphs=False,
        use_amp=False, label="tandem_fno_eval",
    )
    def forward_eval(x, mask):
        return predict(network, x, mask)

    metadata: dict = {}
    loaded_epoch = load_checkpoint(
        checkpoint_dir, models=network, optimizer=optimizer, scheduler=scheduler,
        metadata_dict=metadata, device=dist.device,
    )
    best = float(metadata.get("best_validation_state_mae", "inf"))
    history_path = output / "training_history.json"
    history = json.loads(history_path.read_text()) if dist.rank == 0 and history_path.exists() else []
    state_std = train.state_std.to(dist.device)[None]

    for epoch in range(loaded_epoch + 1, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_total = torch.zeros(2, dtype=torch.float64, device=dist.device)
        with LaunchLogger("train", epoch=epoch, num_mini_batch=len(train_loader)) as logger:
            for batch in train_loader:
                batch = {key: value.to(dist.device, non_blocking=True) for key, value in batch.items()}
                loss = forward_train(batch["x"], batch["delta"], batch["force"], batch["mask"])
                logger.log_minibatch({"loss": loss.detach()})
                train_total[0] += loss.detach().double()
                train_total[1] += 1
            logger.log_epoch({"learning_rate": optimizer.param_groups[0]["lr"]})
        train_total = reduce_totals(train_total, dist).cpu()
        train_loss = float(train_total[0] / train_total[1])

        with LaunchLogger("validation", epoch=epoch) as logger:
            metrics = validation_metrics(forward_eval, validation_loader, dist.device, state_std, dist)
            logger.log_epoch(metrics)

        scheduler.step()
        score = metrics["state_mae_physical_units"]
        checkpoint_interval = max(int(cfg.training.get("checkpoint_interval", 5)), 1)
        save_now = epoch % checkpoint_interval == 0 or epoch == epochs
        improved = save_now and score < best
        if save_now:
            best = min(best, score)
        checkpoint_metadata = {
            "best_validation_state_mae": best,
            "validation": metrics,
            "data_root": str(Path(cfg.data.root).resolve()),
            "action_scale": train.action_scale,
            "model_config": OmegaConf.to_container(cfg.model, resolve=True),
        }
        if dist.rank == 0 and save_now:
            save_checkpoint(
                checkpoint_dir, models=network, optimizer=optimizer, scheduler=scheduler,
                epoch=epoch, metadata=checkpoint_metadata,
            )
            if improved:
                save_checkpoint(
                    best_dir, models=network, optimizer=optimizer, scheduler=scheduler,
                    epoch=epoch, metadata=checkpoint_metadata,
                )
            keep_last = int(cfg.training.get("checkpoint_keep_last", 2))
            prune_checkpoint_dir(checkpoint_dir, keep_last)
            if improved:
                prune_checkpoint_dir(
                    best_dir, int(cfg.training.get("best_checkpoint_keep_last", 1))
                )
        if dist.rank == 0:
            row = {"epoch": epoch, "train_loss": train_loss, **metrics}
            history.append(row)
            history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(row), flush=True)

    if dist.rank == 0:
        print(
            json.dumps(
                {
                    "gpu_memory_fraction": memory_fraction,
                    "gpu_budget_gib": memory_budget_bytes / 1024**3,
                    "gpu_reserve_gib": memory_reserve_bytes / 1024**3,
                    "world_size": dist.world_size,
                    "action_scale": train.action_scale,
                }
            ),
            flush=True,
        )
        pylog.success(f"Training complete; best validation state MAE: {best:.6g}")
    train.close()
    validation.close()
    if dist.distributed:
        DistributedManager.cleanup()


if __name__ == "__main__":
    main()
