#!/usr/bin/env python3
"""Train-only FC-P003C gradient-balance diagnostic (no optimizer or model save)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

POSITIONS = tuple(index * 1367 // 15 for index in range(16))
CHANNELS = ("front_cd", "front_cl", "rear_cd", "rear_cl")


class FixedSampler:
    """Replay already-audited official sampler indices without reading other HDF rows."""

    def __init__(self, indices: Sequence[int]) -> None:
        self.indices = tuple(int(value) for value in indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def model_state_sha256(model: torch.nn.Module) -> str:
    """Hash parameter and buffer values without serializing a checkpoint."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        contiguous = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(contiguous.dtype).encode())
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def gradients(
    loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    *,
    retain_graph: bool = False,
) -> list[torch.Tensor]:
    """Return a dense CPU gradient vector split by parameter, without touching ``.grad``."""
    values = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    return [
        (torch.zeros_like(parameter) if value is None else value).detach().cpu()
        for parameter, value in zip(parameters, values, strict=True)
    ]


def add(
    left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]
) -> list[torch.Tensor]:
    if len(left) != len(right):
        raise ValueError("gradient parameter counts differ")
    return [a + b for a, b in zip(left, right, strict=True)]


def scale(values: Sequence[torch.Tensor], factor: float) -> list[torch.Tensor]:
    return [value * factor for value in values]


def zeros(parameters: Sequence[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [torch.zeros_like(parameter, device="cpu") for parameter in parameters]


def vector_stats(values: Sequence[torch.Tensor]) -> dict[str, float | int]:
    squares = sum(float(value.double().square().sum()) for value in values)
    maximum = max(
        (float(value.abs().max()) for value in values if value.numel()), default=0.0
    )
    finite = sum(int(torch.isfinite(value).sum()) for value in values)
    nonzero = sum(int(torch.count_nonzero(value)) for value in values)
    count = sum(value.numel() for value in values)
    if finite != count:
        raise FloatingPointError("nonfinite gradient component")
    return {
        "l2": math.sqrt(squares),
        "linf": maximum,
        "finite_count": finite,
        "nonzero_count": nonzero,
        "element_count": count,
    }


def compare(
    left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]
) -> dict[str, float | bool | None]:
    left_stats, right_stats = vector_stats(left), vector_stats(right)
    dot = sum(
        float(a.double().mul(b.double()).sum())
        for a, b in zip(left, right, strict=True)
    )
    denominator = float(left_stats["l2"]) * float(right_stats["l2"])
    conflict = total = 0
    for a, b in zip(left, right, strict=True):
        active = (a != 0) & (b != 0)
        conflict += int(((a * b < 0) & active).sum())
        total += int(active.sum())
    return {
        "cosine_defined": denominator > 0,
        "cosine": dot / denominator if denominator else None,
        "sign_conflict_defined": total > 0,
        "sign_conflict_fraction": conflict / total if total else None,
        "ratio_defined": float(right_stats["l2"]) > 0,
        "left_over_right_l2": (
            float(left_stats["l2"]) / float(right_stats["l2"])
            if right_stats["l2"]
            else None
        ),
    }


def assert_gradient_close(
    actual: Sequence[torch.Tensor],
    expected: Sequence[torch.Tensor],
    *,
    label: str,
    rtol: float = 2e-5,
) -> dict[str, float | int]:
    """Require a decomposed gradient to reproduce its independent definition."""
    residual = vector_stats(add(actual, scale(expected, -1.0)))
    reference = float(vector_stats(expected)["l2"])
    actual_stats = vector_stats(actual)
    relative = float(residual["l2"]) / max(reference, 1e-12)
    if relative > rtol:
        print(
            json.dumps(
                {
                    "status": "GRADIENT_DECOMPOSITION_MISMATCH",
                    "scope": "train_only_no_optimizer_debug_evidence",
                    "label": label,
                    "residual_l2": residual["l2"],
                    "residual_linf": residual["linf"],
                    "reference_l2": reference,
                    "reference_linf": vector_stats(expected)["linf"],
                    "actual_l2": actual_stats["l2"],
                    "actual_linf": actual_stats["linf"],
                    "relative_residual_l2": relative,
                    "relative_tolerance": rtol,
                },
                sort_keys=True,
                allow_nan=False,
            ),
            flush=True,
        )
        raise RuntimeError(f"{label} gradient decomposition differs")
    return residual


def summarize_rows(rows: Sequence[dict[str, Any]], key: str) -> dict[str, float]:
    values = sorted(float(row[key]) for row in rows)
    if not values:
        raise ValueError("cannot summarize empty rows")
    middle = len(values) // 2
    median = (
        (values[middle - 1] + values[middle]) / 2
        if len(values) % 2 == 0
        else values[middle]
    )
    return {"min": values[0], "median": median, "max": values[-1]}


def _accumulate(target: list[torch.Tensor], update: Sequence[torch.Tensor]) -> None:
    if len(target) != len(update):
        raise ValueError("gradient accumulator differs")
    for destination, value in zip(target, update, strict=True):
        destination.add_(value)


def _write_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--train8", type=Path, required=True)
    parser.add_argument("--train16", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--sampling-receipt", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.35)
    parser.add_argument("--max-positions", type=int, choices=(1, 16), default=16)
    return parser.parse_args()


def main() -> None:
    started = time.monotonic()
    args = parse_args()
    if not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("diagnostic requires exactly one CUDA process")
    if not 0 < args.gpu_memory_fraction <= 0.35:
        raise ValueError("allocator fraction must be in (0,0.35]")
    if args.output.exists():
        raise FileExistsError(args.output)
    progress_path = args.output.with_suffix(args.output.suffix + ".progress.jsonl")
    if progress_path.exists():
        raise FileExistsError(progress_path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    from fluid_control.augmented_datapipe import compose_training_data
    from fluid_control.dynamic_pair_stat_datapipe import DynamicMatchedPairStatDataset
    from fluid_control.paired_force_statistics import paired_statistic_loss
    from fluid_control.paired_step_training import predict_true_state_force_chunk
    from fluid_control.paired_step_force import paired_step_force_delta_loss
    from fluid_control.paired_training import combine_paired_rollout_batch
    from fluid_control.tandem_datapipe import TandemRolloutDataset
    from omegaconf import OmegaConf
    from physicsnemo.datapipes import DataLoader
    from physicsnemo.utils import load_checkpoint
    from train_tandem_fno import build_model, configured_force_indices, predict
    from train_tandem_fno_paired_stats import (
        force_channel_weights,
        regular_rollout_objective,
        rollout,
    )

    cfg = OmegaConf.load(args.config)
    expected = {
        "rollout_steps": 100,
        "seed": 20261003,
        "paired_batches_per_epoch": 16,
        "paired_dataset_repetitions": 2,
        "paired_objective_kind": "true_state_step_force",
    }
    if any(cfg.training.get(key) != value for key, value in expected.items()):
        raise ValueError("resolved FC-P003C diagnostic contract differs")
    sampling = json.loads(args.sampling_receipt.read_text())
    if (
        sampling.get("status") != "FC_P003B_REAL_OFFICIAL_DATALOADER_SAMPLING_PASS"
        or sampling.get("paired_batch_indices") != list(POSITIONS)
        or sampling.get("validation_or_frozen_accessed") is not False
    ):
        raise ValueError("sampling receipt differs")

    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device)
    force_indices = configured_force_indices(cfg)
    if tuple(force_indices) != (0, 1, 2, 3):
        raise ValueError("all four canonical force channels are required")
    base = TandemRolloutDataset(
        args.base,
        "train",
        100,
        stride=20,
        num_workers=int(cfg.training.workers),
        force_indices=force_indices,
    )
    regular = pair_dataset = None
    progress_stream = progress_path.open("x", encoding="utf-8")
    try:
        regular, _ = compose_training_data(
            base,
            [args.train8, args.train16],
            rollout_steps=100,
            stride=2,
            workers=int(cfg.training.workers),
            force_indices=force_indices,
        )
        pair_dataset = DynamicMatchedPairStatDataset(
            args.train8,
            args.base,
            args.pair_manifest,
            num_workers=int(cfg.training.workers),
        )
        regular_order_loader = DataLoader(
            regular,
            batch_size=1,
            shuffle=True,
            prefetch_factor=0,
            use_streams=False,
            seed=20261003,
        )
        regular_indices = list(iter(regular_order_loader.sampler))
        positions = POSITIONS[: args.max_positions]
        selected_indices = [regular_indices[position] for position in positions]
        pair_order_loader = DataLoader(
            pair_dataset,
            batch_size=1,
            shuffle=True,
            collate_metadata=True,
            prefetch_factor=0,
            use_streams=False,
            seed=20261003,
        )
        pair_indices = [
            *iter(pair_order_loader.sampler),
            *iter(pair_order_loader.sampler),
        ]
        if hashlib.sha256(
            json.dumps(regular_indices, separators=(",", ":")).encode()
        ).hexdigest() != sampling["regular_sequence_sha256"] or pair_indices != [
            *sampling["pair_pass_indices"][0],
            *sampling["pair_pass_indices"][1],
        ]:
            raise ValueError("live sampler order differs from receipt")
        expected_pair_ids = [
            pair_id
            for identity_pass in sampling["pair_identity_passes"]
            for pair_id in identity_pass
        ]
        loader_args = {
            "batch_size": 1,
            "shuffle": False,
            "collate_metadata": True,
            "prefetch_factor": 0,
            "use_streams": False,
            "seed": 20261003,
        }
        selected_regular = DataLoader(
            regular, sampler=FixedSampler(selected_indices), **loader_args
        )
        selected_pairs = DataLoader(
            pair_dataset,
            sampler=FixedSampler(pair_indices[: args.max_positions]),
            **loader_args,
        )

        model = build_model(cfg).to(device)
        metadata: dict[str, Any] = {}
        if (
            load_checkpoint(
                args.parent, models=model, metadata_dict=metadata, device=device
            )
            != 2
        ):
            raise ValueError("Main-e2 parent epoch differs")
        model.train()
        torch.cuda.reset_peak_memory_stats(device)
        model_state_before = model_state_sha256(model)
        parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        weights = force_channel_weights(cfg, force_indices, device)
        force_mean = torch.as_tensor(base.force_mean, device=device).reshape(-1)
        force_std = torch.as_tensor(base.force_std, device=device).reshape(-1)
        mean_gradients = {
            name: zeros(parameters)
            for name in (
                "field",
                "regular_force_0p2",
                "regular",
                "old_pair_lambda10",
                "true_state_pair_lambda10",
            )
        }
        rows = []

        def predict_force(model_arg, inputs, mask):
            return predict(model_arg, inputs, mask)[1]

        for position_index, (position, (regular_item, pair_item)) in enumerate(
            zip(
                positions,
                zip(selected_regular, selected_pairs, strict=True),
                strict=True,
            )
        ):
            batch, regular_metadata = regular_item
            pair, pair_metadata = pair_item
            regular_identity = regular_metadata[0]
            pair_identity = pair_metadata[0]
            if (
                regular_identity.get("split") != "train"
                or regular_identity.get("rollout_steps") != 100
                or not isinstance(regular_identity.get("case"), str)
                or isinstance(regular_identity.get("step"), bool)
                or not isinstance(regular_identity.get("step"), int)
                or regular_identity["step"] < 0
            ):
                raise RuntimeError("selected regular metadata contract differs")
            if (
                pair_identity.get("pair_id") != expected_pair_ids[position_index]
                or pair_identity.get("split") != "train"
                or pair_identity.get("horizon") != 100
                or pair_identity.get("start") != 0
            ):
                raise RuntimeError(
                    "selected paired identity differs from sampling receipt"
                )
            batch = {key: value.to(device) for key, value in batch.items()}
            pair = {key: value.to(device) for key, value in pair.items()}
            total, field, force = regular_rollout_objective(
                model,
                batch["state"],
                batch["target_state"],
                batch["omega"],
                batch["target_force"],
                batch["mask"],
                weights,
                force_loss_weight=0.2,
                rollout_discount=1.0,
                teacher_forcing_ratio=0.0,
            )
            g_total = gradients(total, parameters, retain_graph=True)
            g_field = gradients(field, parameters, retain_graph=True)
            g_force = gradients(0.2 * force, parameters)
            g_regular = add(g_field, g_force)
            regular_gradient_residual = assert_gradient_close(
                g_regular, g_total, label="regular total"
            )
            regular_gradient_residual_relative = float(
                regular_gradient_residual["l2"]
            ) / max(float(vector_stats(g_total)["l2"]), 1e-12)

            # Diagnostic-only decomposition of the unchanged regular force loss.
            # Its four gradients must sum to the authoritative gradient above.
            _, predicted_regular_force = rollout(
                model,
                batch["state"],
                batch["mask"],
                batch["omega"],
                batch["target_state"],
                0.0,
            )
            regular_channel_mse = (
                (predicted_regular_force - batch["target_force"])
                .square()
                .mean(dim=(0, 1))
            )
            regular_force_channels = [zeros(parameters) for _ in CHANNELS]
            for channel in range(4):
                contribution = 0.2 * weights[channel] * regular_channel_mse[channel]
                _accumulate(
                    regular_force_channels[channel],
                    gradients(contribution, parameters, retain_graph=channel < 3),
                )
            g_force_decomposed = zeros(parameters)
            for value in regular_force_channels:
                _accumulate(g_force_decomposed, value)
            force_gradient_residual = assert_gradient_close(
                g_force_decomposed,
                g_force,
                label="regular force channel",
            )
            force_gradient_residual_relative = float(
                force_gradient_residual["l2"]
            ) / max(float(vector_stats(g_force)["l2"]), 1e-12)
            force_loss_decomposed = 0.2 * (regular_channel_mse * weights).sum()
            if not torch.allclose(
                force_loss_decomposed.detach(),
                (0.2 * force).detach(),
                rtol=2e-6,
                atol=2e-7,
            ):
                raise RuntimeError(
                    "regular force channel losses do not reproduce objective"
                )

            combined = combine_paired_rollout_batch(pair)
            _, predicted = rollout(
                model,
                combined["state"],
                combined["mask"],
                combined["omega"],
                combined["target_state"],
                0.0,
            )
            pair_batch = pair["action_force"].shape[0]
            old_loss, predicted_delta, target_delta = paired_statistic_loss(
                predicted[:pair_batch],
                predicted[pair_batch:],
                combined["action_force"],
                combined["zero_force"],
                force_mean,
                force_std,
                CHANNELS,
                (20, 50, 100),
            )
            if not torch.allclose(
                target_delta,
                combined["paired_targets"],
                rtol=2e-5,
                atol=2e-6,
            ):
                raise RuntimeError(
                    "old paired targets differ from aligned force sequence"
                )
            g_old_total = gradients(10.0 * old_loss, parameters, retain_graph=True)
            old_error = predicted_delta - target_delta
            cd_scale = torch.sqrt(force_std[0].square() + force_std[2].square())
            old_scales = torch.stack((cd_scale, force_std[3], force_std[3]))
            old_groups = {
                name: zeros(parameters)
                for name in ("mean_total_cd", "mean_rear_cl", "rear_cl_rms")
            }
            for statistic, name in enumerate(old_groups):
                component = (
                    (old_error[..., statistic] / old_scales[statistic]).square().mean()
                ) / 3
                _accumulate(
                    old_groups[name],
                    gradients(10.0 * component, parameters, retain_graph=statistic < 2),
                )
            g_old = zeros(parameters)
            for value in old_groups.values():
                _accumulate(g_old, value)
            old_gradient_residual = assert_gradient_close(
                g_old, g_old_total, label="old paired statistic"
            )
            old_gradient_residual_relative = float(old_gradient_residual["l2"]) / max(
                float(vector_stats(g_old_total)["l2"]), 1e-12
            )

            new_channels = [zeros(parameters) for _ in CHANNELS]
            new_channel_loss = torch.zeros(4, dtype=torch.float64)
            new_gradient_residual_max = 0.0
            new_gradient_residual_relative_max = 0.0
            for start in range(0, 100, 10):
                stop = start + 10
                action, zero = predict_true_state_force_chunk(
                    model, pair, start, stop, predict_force
                )
                error = (action - zero) - (
                    pair["action_force"][:, 1 + start : 1 + stop]
                    - pair["zero_force"][:, 1 + start : 1 + stop]
                )
                channel_loss = error.square().mean(dim=(0, 1))
                new_channel_loss += channel_loss.detach().cpu().double() * 0.1
                chunk_loss = paired_step_force_delta_loss(
                    action,
                    zero,
                    pair["action_force"][:, 1 + start : 1 + stop],
                    pair["zero_force"][:, 1 + start : 1 + stop],
                    weights,
                )
                if not torch.allclose(
                    chunk_loss.detach(),
                    (channel_loss * weights).sum().detach(),
                    rtol=2e-6,
                    atol=2e-7,
                ):
                    raise RuntimeError("true-state channel losses differ from helper")
                g_chunk_total = gradients(
                    10.0 * chunk_loss * 0.1, parameters, retain_graph=True
                )
                chunk_channels = []
                for channel in range(4):
                    contribution = 10.0 * weights[channel] * channel_loss[channel] * 0.1
                    channel_gradient = gradients(
                        contribution, parameters, retain_graph=channel < 3
                    )
                    chunk_channels.append(channel_gradient)
                    _accumulate(new_channels[channel], channel_gradient)
                g_chunk_decomposed = zeros(parameters)
                for value in chunk_channels:
                    _accumulate(g_chunk_decomposed, value)
                residual = assert_gradient_close(
                    g_chunk_decomposed,
                    g_chunk_total,
                    label=f"true-state chunk {start}:{stop}",
                )
                new_gradient_residual_max = max(
                    new_gradient_residual_max, float(residual["l2"])
                )
                new_gradient_residual_relative_max = max(
                    new_gradient_residual_relative_max,
                    float(residual["l2"])
                    / max(float(vector_stats(g_chunk_total)["l2"]), 1e-12),
                )
            g_new = zeros(parameters)
            for value in new_channels:
                _accumulate(g_new, value)
            components = {
                "field": g_field,
                "regular_force_0p2": g_force,
                "regular": g_regular,
                "old_pair_lambda10": g_old,
                "true_state_pair_lambda10": g_new,
            }
            for name, value in components.items():
                _accumulate(mean_gradients[name], value)
            mixed_old, mixed_new = add(g_regular, g_old), add(g_regular, g_new)
            regular_norm = float(vector_stats(g_regular)["l2"])
            if regular_norm == 0:
                raise RuntimeError("regular gradient is identically zero")
            row = {
                "position": position,
                "regular_dataset_index": selected_indices[position_index],
                "regular_metadata": regular_identity,
                "pair_dataset_index": pair_indices[position_index],
                "pair_metadata": pair_identity,
                "losses": {
                    "regular_total": float(total.detach()),
                    "regular_field": float(field.detach()),
                    "regular_force_raw": float(force.detach()),
                    "regular_force_0p2": float((0.2 * force).detach()),
                    "old_pair_raw": float(old_loss.detach()),
                    "old_pair_lambda10": float((10 * old_loss).detach()),
                    "true_state_pair_raw": float(
                        (new_channel_loss * weights.detach().cpu().double()).sum()
                    ),
                },
                "gradient": {
                    name: vector_stats(value) for name, value in components.items()
                },
                "regular_total_gradient_residual": regular_gradient_residual,
                "regular_total_gradient_residual_relative": regular_gradient_residual_relative,
                "regular_force_channel_loss_0p2": dict(
                    zip(
                        CHANNELS,
                        (
                            0.2
                            * weights.detach().cpu()
                            * regular_channel_mse.detach().cpu()
                        ).tolist(),
                        strict=True,
                    )
                ),
                "regular_force_channel_gradient": {
                    name: vector_stats(value)
                    for name, value in zip(
                        CHANNELS, regular_force_channels, strict=True
                    )
                },
                "regular_force_decomposition_gradient_residual": force_gradient_residual,
                "regular_force_decomposition_gradient_residual_relative": force_gradient_residual_relative,
                "old_statistic_gradient": {
                    name: vector_stats(value) for name, value in old_groups.items()
                },
                "old_statistic_gradient_residual": old_gradient_residual,
                "old_statistic_gradient_residual_relative": old_gradient_residual_relative,
                "true_state_channel_loss": dict(
                    zip(CHANNELS, new_channel_loss.tolist(), strict=True)
                ),
                "true_state_channel_weighted_lambda10_loss": dict(
                    zip(
                        CHANNELS,
                        (
                            10.0 * weights.detach().cpu().double() * new_channel_loss
                        ).tolist(),
                        strict=True,
                    )
                ),
                "true_state_channel_gradient": {
                    name: vector_stats(value)
                    for name, value in zip(CHANNELS, new_channels, strict=True)
                },
                "true_state_chunk_gradient_residual_l2_max": new_gradient_residual_max,
                "true_state_chunk_gradient_residual_relative_max": new_gradient_residual_relative_max,
                "old_vs_regular": compare(g_old, g_regular),
                "new_vs_regular": compare(g_new, g_regular),
                "new_vs_regular_force": compare(g_new, g_force),
                "old_vs_new": compare(g_old, g_new),
                "mixed_old": {
                    **vector_stats(mixed_old),
                    "clip_scale_at_1": min(
                        1.0, 1.0 / max(float(vector_stats(mixed_old)["l2"]), 1e-30)
                    ),
                },
                "mixed_new": {
                    **vector_stats(mixed_new),
                    "clip_scale_at_1": min(
                        1.0, 1.0 / max(float(vector_stats(mixed_new)["l2"]), 1e-30)
                    ),
                },
                "new_over_regular_l2": float(vector_stats(g_new)["l2"]) / regular_norm,
            }
            rows.append(row)
            progress_stream.write(
                json.dumps(
                    {
                        "status": "INCOMPLETE_PROGRESS",
                        "completed_positions": position_index + 1,
                        "position": position,
                        "elapsed_seconds": time.monotonic() - started,
                        "finite": True,
                        "regular_gradient_l2": row["gradient"]["regular"]["l2"],
                        "old_pair_gradient_l2": row["gradient"]["old_pair_lambda10"][
                            "l2"
                        ],
                        "true_state_pair_gradient_l2": row["gradient"][
                            "true_state_pair_lambda10"
                        ]["l2"],
                    },
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
            progress_stream.flush()
            os.fsync(progress_stream.fileno())
            model.zero_grad(set_to_none=True)

        mean_vectors = {
            name: scale(value, 1 / args.max_positions)
            for name, value in mean_gradients.items()
        }
        aggregate = {name: vector_stats(value) for name, value in mean_vectors.items()}
        model_state_after = model_state_sha256(model)
        if model_state_after != model_state_before:
            raise RuntimeError("model state changed during no-optimizer diagnostic")
        result = {
            "status": "FC_P003C_TRAIN_ONLY_GRADIENT_DIAGNOSTIC_COMPLETE",
            "scope": (
                "fixed Main-e2 model at the audited epoch-1 insertion sequence; "
                "not the 32 evolving optimizer states from two training epochs"
            ),
            "scientific_admission": False,
            "optimizer_created_or_stepped": False,
            "candidate_saved": False,
            "validation_or_frozen_accessed": False,
            "parent_model_sha256": sha256(args.parent / "FNO.0.2.mdlus"),
            "parent_state_sha256": sha256(args.parent / "checkpoint.0.2.pt"),
            "model_state_sha256_before": model_state_before,
            "model_state_sha256_after": model_state_after,
            "image_id": args.image_id,
            "config_sha256": sha256(args.config),
            "sampling_receipt_sha256": sha256(args.sampling_receipt),
            "input_sha256": {
                "base_manifest": sha256(args.base / "manifest.json"),
                "base_train_split_manifest": sha256(
                    args.base / "splits" / "train.json"
                ),
                "base_normalization": sha256(args.base / "normalization.json"),
                "train8_manifest": sha256(args.train8 / "manifest.json"),
                "train8_normalization": sha256(args.train8 / "normalization.json"),
                "train16_manifest": sha256(args.train16 / "manifest.json"),
                "train16_normalization": sha256(args.train16 / "normalization.json"),
                "pair_manifest": sha256(args.pair_manifest),
                "source_manifest": sha256(args.source_manifest),
            },
            "positions": list(positions),
            "max_positions": args.max_positions,
            "rows": rows,
            "mean_gradient": aggregate,
            "mean_gradient_directions": {
                "old_vs_regular": compare(
                    mean_vectors["old_pair_lambda10"], mean_vectors["regular"]
                ),
                "new_vs_regular": compare(
                    mean_vectors["true_state_pair_lambda10"],
                    mean_vectors["regular"],
                ),
                "old_vs_new": compare(
                    mean_vectors["old_pair_lambda10"],
                    mean_vectors["true_state_pair_lambda10"],
                ),
            },
            "summaries": {
                "new_over_regular_l2": summarize_rows(rows, "new_over_regular_l2")
            },
            "elapsed_seconds": time.monotonic() - started,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
        _write_exclusive(args.output, result)
    finally:
        progress_stream.close()
        if pair_dataset is not None:
            pair_dataset.close()
        if regular is not None:
            regular.close()
        else:
            base.close()


if __name__ == "__main__":
    main()
