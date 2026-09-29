#!/usr/bin/env python3
"""Fine-tune the PhysicsNeMo FNO with autoregressive rollout loss."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from fluid_control.tandem_datapipe import TandemRolloutDataset
from omegaconf import DictConfig, OmegaConf
from physicsnemo.datapipes import DataLoader
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import (
    StaticCaptureEvaluateNoGrad,
    StaticCaptureTraining,
    load_checkpoint,
    save_checkpoint,
)
from physicsnemo.utils.logging import LaunchLogger, PythonLogger
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler

from train_tandem_fno import build_model, predict, reduce_totals


def rollout(
    network: torch.nn.Module,
    state: torch.Tensor,
    mask: torch.Tensor,
    omega: torch.Tensor,
    target_state: torch.Tensor | None = None,
    teacher_forcing_ratio: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Autoregress over the supplied action sequence."""
    predictions = []
    forces = []
    predicted = state
    height, width = mask.shape[-2:]
    for step in range(omega.shape[1] - 1):
        if step and target_state is not None and teacher_forcing_ratio > 0.0:
            predicted = (
                teacher_forcing_ratio * target_state[:, step - 1]
                + (1.0 - teacher_forcing_ratio) * predicted
            )
        omega_now = omega[:, step].reshape(-1, 1, 1, 1).expand(-1, 1, height, width)
        omega_next = omega[:, step + 1].reshape(-1, 1, 1, 1).expand(-1, 1, height, width)
        inputs = torch.cat([predicted, mask, omega_now, omega_next], dim=1)
        delta, force = predict(network, inputs, mask)
        predicted = (predicted + delta) * mask
        predictions.append(predicted)
        forces.append(force)
    return torch.stack(predictions, dim=1), torch.stack(forces, dim=1)


def teacher_forcing_ratio(cfg: DictConfig, epoch: int) -> float:
    start = float(cfg.training.teacher_forcing_start)
    end = float(cfg.training.teacher_forcing_end)
    duration = max(int(cfg.training.teacher_forcing_epochs), 1)
    fraction = min(max((epoch - 1) / duration, 0.0), 1.0)
    return start + fraction * (end - start)


@hydra.main(version_base="1.3", config_path="../conf", config_name="tandem_fno_rollout")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()
    if not dist.cuda:
        raise RuntimeError("CUDA is required for PhysicsNeMo rollout training")

    memory_fraction = float(cfg.training.gpu_memory_fraction)
    torch.cuda.set_per_process_memory_fraction(memory_fraction, device=dist.device)
    device_properties = torch.cuda.get_device_properties(dist.device)
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
            "rollout_steps": int(cfg.training.rollout_steps),
            "initial_checkpoint": str(cfg.training.initial_checkpoint),
            "nccl_environment": {
                key: os.environ.get(key)
                for key in (
                    "NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE", "NCCL_IB_DISABLE",
                    "NCCL_CUMEM_ENABLE", "NCCL_CUMEM_HOST_ENABLE", "NCCL_SOCKET_IFNAME",
                )
            },
        }
        (output / "runtime_metadata.json").write_text(
            json.dumps(runtime, indent=2) + "\n", encoding="utf-8"
        )
    if dist.distributed:
        torch.distributed.barrier()

    pylog = PythonLogger(name="tandem_fno_rollout")
    if dist.rank == 0:
        pylog.file_logging(str(output / "physicsnemo.log"))
    LaunchLogger.initialize()

    seed = int(cfg.training.seed) + dist.rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    smoke = bool(cfg.training.smoke)
    rollout_steps = 2 if smoke else int(cfg.training.rollout_steps)
    train = TandemRolloutDataset(
        cfg.data.root, "train", rollout_steps,
        stride=32 if smoke else 1, num_workers=cfg.training.workers,
    )
    validation = TandemRolloutDataset(
        cfg.data.root, "validation", rollout_steps,
        stride=64 if smoke else int(cfg.training.validation_stride),
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

    current_teacher_forcing = 0.0

    @StaticCaptureTraining(
        model=network, optim=optimizer, logger=pylog, use_graphs=False,
        use_amp=False, gradient_clip_norm=cfg.training.gradient_clip_norm,
        label="tandem_fno_rollout_train",
    )
    def forward_train(state, target_state, omega, target_force, mask):
        predicted_state, predicted_force = rollout(
            network, state, mask, omega, target_state, current_teacher_forcing
        )
        steps = predicted_state.shape[1]
        weights = torch.pow(
            torch.as_tensor(float(cfg.training.rollout_discount), device=state.device),
            torch.arange(steps, device=state.device, dtype=state.dtype),
        )
        weights = weights / weights.sum()
        field_by_step = (
            ((predicted_state - target_state).square() * mask[:, None]).sum(dim=(2, 3, 4))
            / (mask.sum(dim=(1, 2, 3)).clamp_min(1)[:, None] * 3)
        )
        force_by_step = (predicted_force - target_force).square().mean(dim=2)
        field_loss = (field_by_step * weights[None]).sum(dim=1).mean()
        force_loss = (force_by_step * weights[None]).sum(dim=1).mean()
        return field_loss + float(cfg.training.force_loss_weight) * force_loss

    @StaticCaptureEvaluateNoGrad(
        model=network, logger=pylog, use_graphs=False, use_amp=False,
        label="tandem_fno_rollout_eval",
    )
    def forward_eval(state, mask, omega):
        return rollout(network, state, mask, omega)

    metadata: dict = {}
    loaded_epoch = load_checkpoint(
        checkpoint_dir, models=network, optimizer=optimizer, scheduler=scheduler,
        metadata_dict=metadata, device=dist.device,
    )
    initialized_from = None
    if loaded_epoch == 0 and str(cfg.training.initial_checkpoint):
        initial_metadata: dict = {}
        initial_epoch = load_checkpoint(
            Path(cfg.training.initial_checkpoint), models=network,
            metadata_dict=initial_metadata, device=dist.device,
        )
        if initial_epoch == 0:
            raise FileNotFoundError(f"initial checkpoint not found: {cfg.training.initial_checkpoint}")
        initialized_from = {"path": str(cfg.training.initial_checkpoint), "epoch": initial_epoch}

    best = float(metadata.get("best_rollout_score", "inf"))
    history_path = output / "training_history.json"
    history = json.loads(history_path.read_text()) if dist.rank == 0 and history_path.exists() else []
    state_std = train.state_std.to(dist.device)[None, None]
    force_std = train.force_std.to(dist.device)[None, None]

    for epoch in range(loaded_epoch + 1, epochs + 1):
        current_teacher_forcing = teacher_forcing_ratio(cfg, epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        totals = torch.zeros(2, dtype=torch.float64, device=dist.device)
        max_train_batches = cfg.training.get("max_train_batches")
        train_batches = min(len(train_loader), int(max_train_batches)) if max_train_batches else len(train_loader)
        with LaunchLogger("train", epoch=epoch, num_mini_batch=train_batches) as logger:
            for batch_index, batch in enumerate(train_loader):
                if max_train_batches and batch_index >= int(max_train_batches):
                    break
                batch = {key: value.to(dist.device, non_blocking=True) for key, value in batch.items()}
                loss = forward_train(
                    batch["state"], batch["target_state"], batch["omega"],
                    batch["target_force"], batch["mask"],
                )
                logger.log_minibatch({"loss": loss.detach()})
                totals[0] += loss.detach().double()
                totals[1] += 1
            logger.log_epoch({
                "learning_rate": optimizer.param_groups[0]["lr"],
                "teacher_forcing_ratio": current_teacher_forcing,
            })
        totals = reduce_totals(totals, dist).cpu()
        train_loss = float(totals[0] / totals[1])

        validation_totals = torch.zeros(8, dtype=torch.float64, device=dist.device)
        max_validation_batches = cfg.training.get("max_validation_batches")
        with LaunchLogger("validation", epoch=epoch) as logger:
            for batch_index, batch in enumerate(validation_loader):
                if max_validation_batches and batch_index >= int(max_validation_batches):
                    break
                batch = {key: value.to(dist.device, non_blocking=True) for key, value in batch.items()}
                predicted_state, predicted_force = forward_eval(
                    batch["state"], batch["mask"], batch["omega"]
                )
                state_error = (predicted_state - batch["target_state"]) * state_std
                force_error = (predicted_force - batch["target_force"]) * force_std
                denom = batch["mask"].sum() * 3
                validation_totals[0] += (state_error.abs() * batch["mask"][:, None]).sum().double()
                validation_totals[1] += (denom * predicted_state.shape[1]).double()
                validation_totals[2] += (state_error[:, -1].abs() * batch["mask"]).sum().double()
                validation_totals[3] += denom.double()
                validation_totals[4] += force_error.abs().sum().double()
                validation_totals[5] += force_error.numel()
                validation_totals[6] += force_error[:, -1].abs().sum().double()
                validation_totals[7] += force_error[:, -1].numel()
            validation_totals = reduce_totals(validation_totals, dist).cpu()
            metrics = {
                "rollout_state_mae": float(validation_totals[0] / validation_totals[1]),
                "terminal_state_mae": float(validation_totals[2] / validation_totals[3]),
                "rollout_force_mae": float(validation_totals[4] / validation_totals[5]),
                "terminal_force_mae": float(validation_totals[6] / validation_totals[7]),
            }
            logger.log_epoch(metrics)

        scheduler.step()
        score = metrics["terminal_state_mae"] + float(cfg.training.selection_force_weight) * metrics["terminal_force_mae"]
        improved = score < best
        best = min(best, score)
        checkpoint_metadata = {
            "best_rollout_score": best,
            "validation": metrics,
            "rollout_steps": rollout_steps,
            "teacher_forcing_ratio": current_teacher_forcing,
            "initialized_from": initialized_from,
            "action_scale": train.action_scale,
            "model_config": OmegaConf.to_container(cfg.model, resolve=True),
        }
        if dist.rank == 0:
            save_checkpoint(
                checkpoint_dir, models=network, optimizer=optimizer, scheduler=scheduler,
                epoch=epoch, metadata=checkpoint_metadata,
            )
            if improved:
                save_checkpoint(
                    best_dir, models=network, optimizer=optimizer, scheduler=scheduler,
                    epoch=epoch, metadata=checkpoint_metadata,
                )
            row = {
                "epoch": epoch, "train_loss": train_loss,
                "teacher_forcing_ratio": current_teacher_forcing,
                "selection_score": score, **metrics,
            }
            history.append(row)
            history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(row), flush=True)

    train.close()
    validation.close()
    if dist.rank == 0:
        pylog.success(f"Rollout training complete; best selection score: {best:.6g}")
    if dist.distributed:
        DistributedManager.cleanup()


if __name__ == "__main__":
    main()
