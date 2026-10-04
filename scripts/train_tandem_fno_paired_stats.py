#!/usr/bin/env python3
"""Fine-tune official PhysicsNeMo FNO with a train-only paired-stat objective.

The FNO architecture and autoregressive field/force losses remain unchanged.
The optional paired statistic term is project code and is evaluated on exact
matched-start train trajectories only.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from fluid_control.tandem_datapipe import TandemRolloutDataset
from fluid_control.paired_force_statistics import paired_statistic_loss
from fluid_control.paired_stat_datapipe import MatchedPairStatDataset
from fluid_control.paired_training import (
    combine_paired_rollout_batch,
    paired_batch_indices,
    validate_paired_identity_passes,
)
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

from train_tandem_fno import (
    build_model,
    configured_force_indices,
    predict,
    prune_checkpoint_dir,
    reduce_totals,
)


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
        omega_next = (
            omega[:, step + 1].reshape(-1, 1, 1, 1).expand(-1, 1, height, width)
        )
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


def force_channel_weights(
    cfg: DictConfig, force_indices: tuple[int, ...] | list[int], device: torch.device
) -> torch.Tensor:
    """Validate optional objective-focused weights without changing the FNO model."""
    configured = cfg.training.get("force_channel_weights")
    values = (
        np.ones(len(force_indices), dtype=np.float64)
        if configured is None
        else np.asarray(list(configured), dtype=np.float64)
    )
    if (
        values.shape != (len(force_indices),)
        or not np.isfinite(values).all()
        or np.any(values <= 0)
    ):
        raise ValueError("force_channel_weights must be one positive finite value per force channel")
    return torch.as_tensor(values / values.sum(), dtype=torch.float32, device=device)


def total_drag_error_sums(predicted, target, mean, std, channels):
    """Pooled physical total-Cd errors, without averaging case-wise ratios.

    Return terminal squared error/reference sums and all-step sums. These
    diagnostics do not change the existing checkpoint-selection objective.
    """
    if "front_cd" not in channels or "rear_cd" not in channels:
        return None
    indices = [channels.index("front_cd"), channels.index("rear_cd")]
    physical_target = target.double() * std.double() + mean.double()
    physical_error = (predicted.double() - target.double()) * std.double()
    total_target = physical_target[..., indices].sum(dim=-1)
    total_error = physical_error[..., indices].sum(dim=-1)
    return torch.stack([
        total_error[:, -1].square().sum(),
        total_target[:, -1].square().sum(),
        total_error.square().sum(),
        total_target.square().sum(),
    ])


@hydra.main(
    version_base="1.3",
    config_path="../conf",
    config_name="tandem_fno_paired_stats_h100",
)
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
        (output / "resolved_config.yaml").write_text(
            OmegaConf.to_yaml(cfg), encoding="utf-8"
        )
        runtime = {
            "world_size": dist.world_size,
            "device_name": device_properties.name,
            "device_total_bytes": device_properties.total_memory,
            "gpu_memory_fraction": memory_fraction,
            "rollout_steps": int(cfg.training.rollout_steps),
            "initial_checkpoint": str(cfg.training.initial_checkpoint),
            "paired_stat_loss_weight": float(cfg.training.paired_stat_loss_weight),
            "paired_stat_horizons": list(cfg.training.paired_stat_horizons),
            "paired_root": str(cfg.data.paired_root),
            "paired_manifest": str(cfg.data.paired_manifest),
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
    validation_rollout_steps = (
        rollout_steps if smoke else int(cfg.training.get("validation_rollout_steps", rollout_steps))
    )
    if validation_rollout_steps < 1:
        raise ValueError("validation_rollout_steps must be positive")
    force_indices = configured_force_indices(cfg)
    if int(cfg.model.out_channels) != 3 + len(force_indices):
        raise ValueError("model.out_channels must equal 3 + number of force targets")
    channel_weights = force_channel_weights(cfg, force_indices, dist.device)
    train = TandemRolloutDataset(
        cfg.data.root,
        "train",
        rollout_steps,
        stride=32 if smoke else int(cfg.training.get("train_stride", 1)),
        num_workers=cfg.training.workers,
        force_indices=force_indices,
    )
    validation = TandemRolloutDataset(
        cfg.data.root,
        "validation",
        validation_rollout_steps,
        stride=64 if smoke else int(cfg.training.validation_stride),
        num_workers=cfg.training.workers,
        force_indices=force_indices,
    )
    base_train = train
    additional_sources = []
    additional_roots = list(cfg.data.get("additional_train_roots", []))
    if additional_roots:
        from fluid_control.augmented_datapipe import compose_training_data

        train, additional_sources = compose_training_data(
            base_train, additional_roots, rollout_steps=rollout_steps,
            stride=int(cfg.training.get("additional_train_stride", 2)),
            workers=cfg.training.workers, force_indices=force_indices,
        )
        if dist.rank == 0:
            (output / "training_data_sources.json").write_text(
                json.dumps({"base_root": str(base_train.root),
                            "base_windows": len(base_train),
                            "total_windows": len(train),
                            "additional_sources": additional_sources}, indent=2) + "\n",
                encoding="utf-8",
            )
    train_sampler = (
        DistributedSampler(train, shuffle=True, seed=cfg.training.seed)
        if dist.distributed
        else None
    )
    validation_sampler = (
        DistributedSampler(validation, shuffle=False) if dist.distributed else None
    )
    loader_args = {
        "batch_size": int(cfg.training.batch_size),
        "prefetch_factor": int(cfg.data.prefetch_factor),
        "num_streams": int(cfg.data.num_streams),
        "use_streams": True,
        "seed": int(cfg.training.seed),
    }
    train_loader = DataLoader(
        train, shuffle=train_sampler is None, sampler=train_sampler, **loader_args
    )
    validation_loader = DataLoader(
        validation, shuffle=False, sampler=validation_sampler, **loader_args
    )
    paired_weight = float(cfg.training.paired_stat_loss_weight)
    if dist.world_size != 1:
        raise ValueError("paired-stat controlled experiment currently requires one GPU")
    if not np.isfinite(paired_weight) or paired_weight < 0.0:
        raise ValueError("paired_stat_loss_weight must be finite and non-negative")
    paired_horizons = tuple(int(value) for value in cfg.training.paired_stat_horizons)
    if paired_horizons != (20, 50, 100) or rollout_steps != 100:
        raise ValueError("paired-stat experiment requires exact H20/H50/H100 within H100")
    paired_dataset_kind = str(cfg.data.get("paired_dataset_kind", "static16"))
    if paired_dataset_kind == "static16":
        pair_dataset = MatchedPairStatDataset(
            cfg.data.paired_root,
            cfg.data.paired_manifest,
            num_workers=cfg.training.workers,
        )
        expected_pair_ids = {
            f"{row['phase']}:{row['action']}" for row in pair_dataset.manifest["pairs"]
        }
    elif paired_dataset_kind == "dynamic8":
        from fluid_control.dynamic_pair_stat_datapipe import (
            DynamicMatchedPairStatDataset,
        )

        pair_dataset = DynamicMatchedPairStatDataset(
            cfg.data.paired_action_root,
            cfg.data.paired_zero_root,
            cfg.data.paired_manifest,
            num_workers=cfg.training.workers,
        )
        expected_pair_ids = {
            f"{row['phase']}:{row['profile']}" for row in pair_dataset.manifest["pairs"]
        }
    else:
        raise ValueError(f"unsupported paired_dataset_kind: {paired_dataset_kind}")
    pair_loader = DataLoader(
        pair_dataset,
        batch_size=int(cfg.training.paired_batch_size),
        shuffle=True,
        collate_metadata=True,
        prefetch_factor=int(cfg.data.prefetch_factor),
        num_streams=int(cfg.data.num_streams),
        use_streams=True,
        seed=int(cfg.training.seed),
    )
    paired_batch_size = int(cfg.training.paired_batch_size)
    if paired_batch_size < 1 or len(pair_dataset) % paired_batch_size:
        raise ValueError("paired_batch_size must exactly divide the paired dataset")
    expected_pair_batches = len(pair_dataset) // paired_batch_size
    if len(pair_loader) != expected_pair_batches:
        raise ValueError("paired-stat loader does not cover each pair once")
    paired_dataset_repetitions = int(
        cfg.training.get("paired_dataset_repetitions", 1)
    )
    if paired_dataset_repetitions < 1:
        raise ValueError("paired_dataset_repetitions must be positive")
    paired_batches_per_epoch = int(
        cfg.training.get("paired_batches_per_epoch", expected_pair_batches)
    )
    if paired_batches_per_epoch != expected_pair_batches * paired_dataset_repetitions:
        raise ValueError(
            "paired_batches_per_epoch must equal complete paired dataset passes"
        )

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
        network.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    epochs = 1 if smoke else int(cfg.training.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    current_teacher_forcing = 0.0

    def base_objective(state, target_state, omega, target_force, mask):
        predicted_state, predicted_force = rollout(
            network, state, mask, omega, target_state, current_teacher_forcing
        )
        steps = predicted_state.shape[1]
        weights = torch.pow(
            torch.as_tensor(float(cfg.training.rollout_discount), device=state.device),
            torch.arange(steps, device=state.device, dtype=state.dtype),
        )
        weights = weights / weights.sum()
        field_by_step = ((predicted_state - target_state).square() * mask[:, None]).sum(
            dim=(2, 3, 4)
        ) / (mask.sum(dim=(1, 2, 3)).clamp_min(1)[:, None] * 3)
        force_by_step = (
            (predicted_force - target_force).square()
            * channel_weights[None, None]
        ).sum(dim=2)
        field_loss = (field_by_step * weights[None]).sum(dim=1).mean()
        force_loss = (force_by_step * weights[None]).sum(dim=1).mean()
        return field_loss + float(cfg.training.force_loss_weight) * force_loss

    @StaticCaptureTraining(
        model=network,
        optim=optimizer,
        logger=pylog,
        use_graphs=False,
        use_amp=False,
        gradient_clip_norm=cfg.training.gradient_clip_norm,
        label="tandem_fno_rollout_train",
    )
    def forward_train(state, target_state, omega, target_force, mask):
        loss = base_objective(state, target_state, omega, target_force, mask)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite autoregressive training loss; refusing optimizer update")
        return loss

    @StaticCaptureTraining(
        model=network,
        optim=optimizer,
        logger=pylog,
        use_graphs=False,
        use_amp=False,
        gradient_clip_norm=cfg.training.gradient_clip_norm,
        label="tandem_fno_paired_stats_train",
    )
    def forward_train_paired(
        state, target_state, omega, target_force, mask,
        paired_state, paired_target_state, paired_omega, paired_mask,
        action_force, zero_force,
    ):
        base_loss = base_objective(state, target_state, omega, target_force, mask)
        pair_batch = action_force.shape[0]
        _, combined_prediction = rollout(
            network, paired_state, paired_mask, paired_omega,
            paired_target_state, current_teacher_forcing,
        )
        pair_loss, _, _ = paired_statistic_loss(
            combined_prediction[:pair_batch],
            combined_prediction[pair_batch:],
            action_force,
            zero_force,
            force_mean.reshape(-1),
            force_std.reshape(-1),
            tuple(base_train.force_channels),
            paired_horizons,
        )
        loss = base_loss + paired_weight * pair_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite paired autoregressive loss; refusing update")
        return loss

    @StaticCaptureEvaluateNoGrad(
        model=network,
        logger=pylog,
        use_graphs=False,
        use_amp=False,
        label="tandem_fno_rollout_eval",
    )
    def forward_eval(state, mask, omega):
        return rollout(network, state, mask, omega)

    metadata: dict = {}
    loaded_epoch = load_checkpoint(
        checkpoint_dir,
        models=network,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata_dict=metadata,
        device=dist.device,
    )
    initialized_from = None
    if loaded_epoch == 0 and str(cfg.training.initial_checkpoint):
        initial_metadata: dict = {}
        initial_epoch = load_checkpoint(
            Path(cfg.training.initial_checkpoint),
            models=network,
            metadata_dict=initial_metadata,
            device=dist.device,
        )
        if initial_epoch == 0:
            raise FileNotFoundError(
                f"initial checkpoint not found: {cfg.training.initial_checkpoint}"
            )
        initialized_from = {
            "path": str(cfg.training.initial_checkpoint),
            "epoch": initial_epoch,
        }

    best = float(metadata.get("best_rollout_score", "inf"))
    history_path = output / "training_history.json"
    history = (
        json.loads(history_path.read_text())
        if dist.rank == 0 and history_path.exists()
        else []
    )
    state_std = base_train.state_std.to(dist.device)[None, None]
    force_std = base_train.force_std.to(dist.device)[None, None]
    force_mean = base_train.force_mean.to(dist.device)[None, None]

    for epoch in range(loaded_epoch + 1, epochs + 1):
        current_teacher_forcing = teacher_forcing_ratio(cfg, epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        totals = torch.zeros(2, dtype=torch.float64, device=dist.device)
        max_train_batches = cfg.training.get("max_train_batches")
        train_batches = (
            min(len(train_loader), int(max_train_batches))
            if max_train_batches
            else len(train_loader)
        )
        expected_regular_batches = cfg.training.get("expected_regular_batches")
        if expected_regular_batches and train_batches != int(expected_regular_batches):
            raise RuntimeError(
                f"regular batch count differs: {train_batches} != "
                f"{expected_regular_batches}"
            )
        paired_schedule = str(
            cfg.training.get("paired_batch_schedule", "frontloaded")
        )
        paired_indices = paired_batch_indices(
            train_batches, paired_batches_per_epoch, paired_schedule
        )
        paired_index_set = set(paired_indices)
        pair_iterator = iter(pair_loader)
        paired_batches = 0
        paired_identities = []
        paired_identity_passes = [[] for _ in range(paired_dataset_repetitions)]
        with LaunchLogger("train", epoch=epoch, num_mini_batch=train_batches) as logger:
            for batch_index, batch in enumerate(train_loader):
                if max_train_batches and batch_index >= int(max_train_batches):
                    break
                batch = {
                    key: value.to(dist.device, non_blocking=True)
                    for key, value in batch.items()
                }
                if batch_index not in paired_index_set:
                    pair = None
                else:
                    if paired_batches and paired_batches % expected_pair_batches == 0:
                        pair_iterator = iter(pair_loader)
                    try:
                        pair, pair_metadata = next(pair_iterator)
                    except StopIteration:
                        raise RuntimeError("paired loader ended before a complete pass")
                if pair is None:
                    loss = forward_train(
                        batch["state"], batch["target_state"], batch["omega"],
                        batch["target_force"], batch["mask"],
                    )
                else:
                    pair = {
                        key: value.to(dist.device, non_blocking=True)
                        for key, value in pair.items()
                    }
                    combined = combine_paired_rollout_batch(pair)
                    loss = forward_train_paired(
                        batch["state"], batch["target_state"], batch["omega"],
                        batch["target_force"], batch["mask"],
                        combined["state"], combined["target_state"],
                        combined["omega"], combined["mask"],
                        combined["action_force"], combined["zero_force"],
                    )
                    paired_batches += 1
                    identities = [
                        item.get("pair_id", f"{item['phase']}:{item.get('action')}")
                        for item in pair_metadata
                    ]
                    paired_identities.extend(identities)
                    pass_index = (paired_batches - 1) // expected_pair_batches
                    paired_identity_passes[pass_index].extend(identities)
                logger.log_minibatch({"loss": loss.detach()})
                totals[0] += loss.detach().double()
                totals[1] += 1
            logger.log_epoch(
                {
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "teacher_forcing_ratio": current_teacher_forcing,
                    "paired_stat_loss_weight": paired_weight,
                    "paired_batches": paired_batches,
                }
            )
        if paired_batches != paired_batches_per_epoch:
            raise RuntimeError(
                f"consumed {paired_batches} paired batches; expected "
                f"{paired_batches_per_epoch}"
            )
        validate_paired_identity_passes(
            paired_identity_passes,
            expected_pair_ids,
            paired_dataset_repetitions,
        )
        totals = reduce_totals(totals, dist).cpu()
        train_loss = float(totals[0] / totals[1])

        validation_totals = torch.zeros(12, dtype=torch.float64, device=dist.device)
        max_validation_batches = cfg.training.get("max_validation_batches")
        with LaunchLogger("validation", epoch=epoch) as logger:
            for batch_index, batch in enumerate(validation_loader):
                if max_validation_batches and batch_index >= int(
                    max_validation_batches
                ):
                    break
                batch = {
                    key: value.to(dist.device, non_blocking=True)
                    for key, value in batch.items()
                }
                predicted_state, predicted_force = forward_eval(
                    batch["state"], batch["mask"], batch["omega"]
                )
                state_error = (predicted_state - batch["target_state"]) * state_std
                force_error = (predicted_force - batch["target_force"]) * force_std
                denom = batch["mask"].sum() * 3
                validation_totals[0] += (
                    (state_error.abs() * batch["mask"][:, None]).sum().double()
                )
                validation_totals[1] += (denom * predicted_state.shape[1]).double()
                validation_totals[2] += (
                    (state_error[:, -1].abs() * batch["mask"]).sum().double()
                )
                validation_totals[3] += denom.double()
                validation_totals[4] += force_error.abs().sum().double()
                validation_totals[5] += force_error.numel()
                validation_totals[6] += force_error[:, -1].abs().sum().double()
                validation_totals[7] += force_error[:, -1].numel()
                drag_sums = total_drag_error_sums(
                    predicted_force, batch["target_force"], force_mean,
                    force_std, list(base_train.force_channels),
                )
                if drag_sums is not None:
                    validation_totals[8:] += drag_sums
            validation_totals = reduce_totals(validation_totals, dist).cpu()
            metrics = {
                "rollout_state_mae": float(validation_totals[0] / validation_totals[1]),
                "terminal_state_mae": float(
                    validation_totals[2] / validation_totals[3]
                ),
                "rollout_force_mae": float(validation_totals[4] / validation_totals[5]),
                "terminal_force_mae": float(
                    validation_totals[6] / validation_totals[7]
                ),
            }
            if validation_totals[9] > 0 and validation_totals[11] > 0:
                metrics.update(
                    terminal_total_drag_pooled_nrmse=float(
                        torch.sqrt(validation_totals[8] / validation_totals[9])
                    ),
                    rollout_total_drag_pooled_nrmse=float(
                        torch.sqrt(validation_totals[10] / validation_totals[11])
                    ),
                )
            logger.log_epoch(metrics)

        paired_eval_total = torch.zeros(2, dtype=torch.float64, device=dist.device)
        max_paired_eval_batches = int(
            cfg.training.get("max_paired_eval_batches", expected_pair_batches)
        )
        if not 1 <= max_paired_eval_batches <= len(pair_loader):
            raise ValueError(
                f"max_paired_eval_batches must be in [1, {expected_pair_batches}]"
            )
        for pair_index, (pair, _) in enumerate(pair_loader):
            if pair_index >= max_paired_eval_batches:
                break
            pair = {
                key: value.to(dist.device, non_blocking=True)
                for key, value in pair.items()
            }
            combined = combine_paired_rollout_batch(pair)
            pair_batch = pair["action_state"].shape[0]
            _, predicted_force = forward_eval(
                combined["state"], combined["mask"], combined["omega"]
            )
            paired_loss, _, target_delta = paired_statistic_loss(
                predicted_force[:pair_batch], predicted_force[pair_batch:],
                combined["action_force"], combined["zero_force"],
                force_mean.reshape(-1), force_std.reshape(-1),
                tuple(base_train.force_channels), paired_horizons,
            )
            if not torch.allclose(
                target_delta, combined["paired_targets"], rtol=2e-5, atol=2e-6
            ):
                raise RuntimeError("paired target tensor differs from force sequences")
            paired_eval_total[0] += paired_loss.double() * pair_batch
            paired_eval_total[1] += pair_batch
        metrics["train_only_paired_stat_loss"] = float(
            paired_eval_total[0] / paired_eval_total[1]
        )

        scheduler.step()
        score = (
            metrics["terminal_state_mae"]
            + float(cfg.training.selection_force_weight) * metrics["terminal_force_mae"]
        )
        checkpoint_interval = max(int(cfg.training.get("checkpoint_interval", 5)), 1)
        save_now = epoch % checkpoint_interval == 0 or epoch == epochs
        improved = save_now and score < best
        if save_now:
            best = min(best, score)
        checkpoint_metadata = {
            "best_rollout_score": best,
            "validation": metrics,
            "rollout_steps": rollout_steps,
            "validation_rollout_steps": validation_rollout_steps,
            "teacher_forcing_ratio": current_teacher_forcing,
            "initialized_from": initialized_from,
            "action_scale": base_train.action_scale,
            "force_channels": list(base_train.force_channels),
            "force_indices": list(base_train.force_indices),
            "additional_training_sources": additional_sources,
            "force_channel_weights": channel_weights.detach().cpu().tolist(),
            "paired_stat_loss_weight": paired_weight,
            "paired_stat_horizons": list(paired_horizons),
            "paired_pair_count": len(pair_dataset),
            "paired_dataset_kind": paired_dataset_kind,
            "paired_dataset_repetitions": paired_dataset_repetitions,
            "paired_batches_per_epoch": paired_batches_per_epoch,
            "paired_batch_schedule": paired_schedule,
            "paired_batch_indices": list(paired_indices),
            "paired_identities": paired_identities,
            "paired_identity_passes": paired_identity_passes,
            "paired_eval_batches": max_paired_eval_batches,
            "paired_manifest": str(cfg.data.paired_manifest),
            "model_config": OmegaConf.to_container(cfg.model, resolve=True),
        }
        if dist.rank == 0 and save_now:
            save_checkpoint(
                checkpoint_dir,
                models=network,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metadata=checkpoint_metadata,
            )
            if improved:
                save_checkpoint(
                    best_dir,
                    models=network,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metadata=checkpoint_metadata,
                )
            keep_last = int(cfg.training.get("checkpoint_keep_last", 2))
            prune_checkpoint_dir(checkpoint_dir, keep_last)
            if improved:
                prune_checkpoint_dir(
                    best_dir, int(cfg.training.get("best_checkpoint_keep_last", 1))
                )
        if dist.rank == 0:
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "teacher_forcing_ratio": current_teacher_forcing,
                "train_rollout_steps": rollout_steps,
                "validation_rollout_steps": validation_rollout_steps,
                "selection_score": score,
                "paired_stat_loss_weight": paired_weight,
                "paired_batches": paired_batches,
                "paired_batch_schedule": paired_schedule,
                "paired_batch_indices": list(paired_indices),
                "paired_identities": paired_identities,
                "paired_identity_passes": paired_identity_passes,
                **metrics,
            }
            history.append(row)
            history_path.write_text(
                json.dumps(history, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(row), flush=True)

    train.close()
    validation.close()
    pair_dataset.close()
    if dist.rank == 0:
        pylog.success(f"Rollout training complete; best selection score: {best:.6g}")
    if dist.distributed:
        DistributedManager.cleanup()


if __name__ == "__main__":
    main()
