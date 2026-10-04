#!/usr/bin/env python3
"""Bounded train-only FC-P003C calibration; never reads validation/frozen data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

TOTAL_STEPS = 128
PAIRED_STEPS = 64
PAIR_SIZE = 8
PAIR_PASSES = 8
PARENT_MODEL_SHA = "f78c2f3341663ed2f6e7f4c64a0bf6539a065d2e7f11ada8f19f493320697eb4"
PARENT_STATE_SHA = "a63186685c998115507c423c5db8d467e845469dc0a5b087b47f341b361a8b4a"
CONFIG_SHA = "07e55fd11df8030313338cef0344490c3453e515aae9c6b6122e997bad5085d9"
BASE_MANIFEST_SHA = "5213c7bb07c824c6e601636c3cfa974b80ec7c051633b0c8368a045cea41ddd2"
TRAIN8_MANIFEST_SHA = "a0bd0e3b79d4d43ace407292c31a595d552dbd6e1a5de1fd149b1ca446f53f35"
TRAIN16_MANIFEST_SHA = (
    "7c62dab94e317442cecf7ea3be547cf6e03c594a2c7ec2a1a695caba2445cf5b"
)
PAIR_MANIFEST_SHA = "b756c6d777b68fe9dc6c5a81e40733e24d2c248b315723ddb40cc3273902d28c"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def calibration_positions() -> tuple[int, ...]:
    from fluid_control.paired_training import paired_batch_indices

    values = tuple(paired_batch_indices(TOTAL_STEPS, PAIRED_STEPS, "interleaved"))
    if (
        len(values) != PAIRED_STEPS
        or len(set(values)) != PAIRED_STEPS
        or values[0] != 0
        or values[-1] != TOTAL_STEPS - 1
    ):
        raise RuntimeError("bounded paired schedule differs")
    return values


def validate_pair_passes(passes: list[list[str]], expected: set[str]) -> None:
    if len(passes) != PAIR_PASSES or any(len(row) != PAIR_SIZE for row in passes):
        raise RuntimeError("requires eight complete dynamic8 passes")
    if any(set(row) != expected or len(set(row)) != PAIR_SIZE for row in passes):
        raise RuntimeError("a paired pass is missing or duplicating an identity")


def regular_optimizer_step(model, optimizer, loss, clip: float) -> dict:
    import torch

    optimizer.zero_grad(set_to_none=True)
    if loss.ndim or not torch.isfinite(loss):
        raise FloatingPointError("regular loss must be a finite scalar")
    loss.backward()
    parameters = [p for p in model.parameters() if p.requires_grad]
    gradients = [p.grad for p in parameters if p.grad is not None]
    if not gradients or any(not torch.isfinite(g).all() for g in gradients):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("regular gradients are missing or non-finite")
    norm = torch.nn.utils.clip_grad_norm_(parameters, clip)
    if not torch.isfinite(norm):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("regular preclip norm is non-finite")
    optimizer.step()
    return {"loss": float(loss.detach()), "preclip_gradient_norm": float(norm)}


def tensor_state_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _jsonable(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _identity(metadata) -> str:
    payload = _jsonable(metadata)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _pair_ids(metadata) -> list[str]:
    rows = metadata if isinstance(metadata, (list, tuple)) else [metadata]
    result = []
    for row in rows:
        if "pair_id" in row:
            result.append(str(row["pair_id"]))
        else:
            result.append(f"{row['phase']}:{row.get('profile', row.get('action'))}")
    return result


def _branch_h1(model, pair, branch: str, predict):
    import torch
    from fluid_control.paired_step_force import true_state_step_input

    state = pair[f"{branch}_state"]
    omega = pair[f"{branch}_omega"]
    predicted_state, predicted_force = [], []
    with torch.no_grad():
        for step in range(state.shape[1] - 1):
            inputs = true_state_step_input(
                state[:, 0], state[:, 1:], pair["mask"], omega, step
            )
            delta, force = predict(model, inputs, pair["mask"])
            predicted_state.append((state[:, step] + delta) * pair["mask"])
            predicted_force.append(force)
    return torch.stack(predicted_state, 1), torch.stack(predicted_force, 1)


def _summary(values) -> dict:
    import numpy as np

    if not values or any(not math.isfinite(float(x)) for x in values):
        raise RuntimeError("non-finite/empty readout")
    array = np.asarray(values, dtype=np.float64)
    return {
        "mae": float(np.abs(array).mean()),
        "rmse": float(np.sqrt(np.square(array).mean())),
        "bias": float(array.mean()),
        "count": int(array.size),
    }


def train_h1_readout(model, items, state_mean, state_std, force_std, predict) -> dict:
    import torch

    if tuple(force_std.shape) != (4,):
        raise ValueError("force_std must have shape [4]")
    from evaluate_tandem_fno import field_error_sums, relative_field_metrics

    model.eval()
    panels = {}
    zero_seen = set()
    for item in items:
        panel, identity, pair = item["panel"], item["identity"], item["pair"]
        pair = {k: v.to(state_std.device) for k, v in pair.items()}
        if (
            pair["action_state"].ndim != 5
            or pair["action_state"].shape[0] != 1
            or pair["action_state"].shape[2] != 3
            or pair["action_force"].shape != (1, pair["action_state"].shape[1], 4)
            or pair["action_omega"].shape != (1, pair["action_state"].shape[1], 1)
        ):
            raise ValueError("readout pair shapes differ")
        current = panels.setdefault(
            panel,
            {
                "field_sums": torch.zeros(
                    (2, 3), dtype=torch.float64, device=state_std.device
                ),
                "absolute": {
                    name: [] for name in ("front_cd", "front_cl", "rear_cd", "rear_cl")
                },
                "delta": {
                    name: [] for name in ("front_cd", "front_cl", "rear_cd", "rear_cl")
                },
                "per_pair": {},
            },
        )
        ap_state, ap_force = _branch_h1(model, pair, "action", predict)
        zp_state, zp_force = _branch_h1(model, pair, "zero", predict)
        action_target_state, zero_target_state = (
            pair["action_state"][:, 1:],
            pair["zero_state"][:, 1:],
        )
        action_target_force, zero_target_force = (
            pair["action_force"][:, 1:],
            pair["zero_force"][:, 1:],
        )
        phase = identity.split(":", 1)[0].split("/", 1)[0]
        branches = [(ap_state, action_target_state, ap_force, action_target_force)]
        zero_key = (panel, phase)
        if zero_key not in zero_seen:
            branches.append((zp_state, zero_target_state, zp_force, zero_target_force))
            zero_seen.add(zero_key)
        for prediction, target, predicted_force, target_force in branches:
            error = (prediction - target) * state_std
            physical_target = target * state_std + state_mean
            batch, steps, _, height, width = error.shape
            masks = pair["mask"][:, None].expand(-1, steps, -1, -1, -1)
            current["field_sums"] += field_error_sums(
                error.reshape(batch * steps, 3, height, width),
                physical_target.reshape(batch * steps, 3, height, width),
                masks.reshape(batch * steps, 1, height, width),
            )
            physical_force_error = (predicted_force - target_force) * force_std
            for channel, values in zip(
                current["absolute"], physical_force_error.unbind(-1), strict=True
            ):
                current["absolute"][channel].extend(values.cpu().reshape(-1).tolist())
        action_error = (ap_force - action_target_force) * force_std
        zero_error = (zp_force - zero_target_force) * force_std
        delta = action_error - zero_error
        for channel, values in zip(current["delta"], delta.unbind(-1), strict=True):
            current["delta"][channel].extend(values.cpu().reshape(-1).tolist())
        action_field_error = (ap_state - action_target_state) * state_std
        action_field_target = action_target_state * state_std + state_mean
        _, steps, _, height, width = action_field_error.shape
        action_masks = pair["mask"][:, None].expand(-1, steps, -1, -1, -1)
        current["per_pair"][identity] = {
            "field_action": relative_field_metrics(
                field_error_sums(
                    action_field_error.reshape(steps, 3, height, width),
                    action_field_target.reshape(steps, 3, height, width),
                    action_masks.reshape(steps, 1, height, width),
                )
                .cpu()
                .numpy()
            ),
            "absolute": {
                channel: _summary(values.cpu().reshape(-1).tolist())
                for channel, values in zip(
                    current["absolute"], action_error.unbind(-1), strict=True
                )
            },
            "action_minus_zero": {
                channel: _summary(values.cpu().reshape(-1).tolist())
                for channel, values in zip(
                    current["delta"], delta.unbind(-1), strict=True
                )
            },
        }

    result = {}
    for panel, values in panels.items():
        result[panel] = {
            "field": relative_field_metrics(values.pop("field_sums").cpu().numpy()),
            "force_absolute": {
                k: _summary(v) for k, v in values.pop("absolute").items()
            },
            "force_action_minus_zero": {
                k: _summary(v) for k, v in values.pop("delta").items()
            },
            "per_pair": values["per_pair"],
        }
    model.train()
    return result


def build_train_readout_items(cfg, device) -> list[dict]:
    from diagnose_fcp003c_train_vs_validation_true_state_h1 import (
        _load_hdf,
        _normalized_pair,
    )

    action_root = Path(cfg.data.paired_action_root)
    zero_root = Path(cfg.data.paired_zero_root)
    manifest = json.loads(Path(cfg.data.paired_manifest).read_text())
    stats = json.loads((action_root / "normalization.json").read_text())
    force_mean = stats["all_force_mean"]
    force_std = stats["all_force_std"]
    items = []
    for row in manifest["pairs"]:
        action = _load_hdf(
            action_root / row["action_file"], row["action_hdf_sha256"], 201
        )
        zero = _load_hdf(zero_root / row["zero_file"], row["zero_hdf_sha256"], 801)
        identity = f"{row['phase']}:{row['profile']}"
        for panel, endpoints in (
            ("train_paired_window", list(range(1, 101))),
            ("train_late_window", list(range(100, 201))),
        ):
            pair = _normalized_pair(action, zero, endpoints, stats, device)
            first, last = endpoints[0] - 1, endpoints[-1]
            import torch

            mean = torch.tensor(force_mean, dtype=torch.float32, device=device)
            std = torch.tensor(force_std, dtype=torch.float32, device=device)
            pair["action_force"] = (
                torch.as_tensor(action["force"][first : last + 1], device=device)[None]
                - mean
            ) / std
            pair["zero_force"] = (
                torch.as_tensor(zero["force"][first : last + 1], device=device)[None]
                - mean
            ) / std
            items.append({"panel": panel, "identity": identity, "pair": pair})
    if len(items) != 16:
        raise RuntimeError("requires eight pairs across two train panels")
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    import numpy as np
    import torch
    from fluid_control.augmented_datapipe import compose_training_data
    from fluid_control.dynamic_pair_stat_datapipe import DynamicMatchedPairStatDataset
    from fluid_control.paired_step_training import (
        prepare_true_state_training_model,
        true_state_paired_optimizer_step,
    )
    from fluid_control.tandem_datapipe import TandemRolloutDataset
    from omegaconf import OmegaConf
    from physicsnemo.datapipes import DataLoader
    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils import load_checkpoint, save_checkpoint
    from train_tandem_fno import build_model, configured_force_indices, predict
    from train_tandem_fno_paired_stats import (
        force_channel_weights,
        regular_rollout_objective,
    )

    if sha256(args.config) != CONFIG_SHA:
        raise ValueError("C resolved config SHA differs")
    cfg = OmegaConf.load(args.config)
    required = {
        "rollout_steps": 100,
        "batch_size": 1,
        "paired_batch_size": 1,
        "paired_objective_kind": "true_state_step_force",
        "paired_step_chunk_size": 10,
        "paired_stat_loss_weight": 10.0,
        "teacher_forcing_start": 0.0,
        "teacher_forcing_end": 0.0,
    }
    if any(cfg.training.get(k) != value for k, value in required.items()):
        raise ValueError("C training contract differs")
    if cfg.data.get("paired_dataset_kind") != "dynamic8":
        raise ValueError("requires dynamic8")
    additional_roots = [Path(value) for value in cfg.data.additional_train_roots]
    if (
        sha256(Path(cfg.data.root) / "manifest.json") != BASE_MANIFEST_SHA
        or len(additional_roots) != 2
        or sha256(additional_roots[0] / "manifest.json") != TRAIN8_MANIFEST_SHA
        or sha256(additional_roots[1] / "manifest.json") != TRAIN16_MANIFEST_SHA
        or sha256(Path(cfg.data.paired_manifest)) != PAIR_MANIFEST_SHA
    ):
        raise ValueError("fixed train-only data manifests differ")
    DistributedManager.initialize()
    dist = DistributedManager()
    if dist.distributed or not dist.cuda:
        raise RuntimeError("exactly one CUDA device required")
    torch.cuda.set_per_process_memory_fraction(
        float(cfg.training.gpu_memory_fraction), device=dist.device
    )
    seed = int(cfg.training.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    force_indices = configured_force_indices(cfg)
    base = TandemRolloutDataset(
        cfg.data.root,
        "train",
        100,
        stride=int(cfg.training.train_stride),
        num_workers=cfg.training.workers,
        force_indices=force_indices,
    )
    train, sources = compose_training_data(
        base,
        additional_roots,
        rollout_steps=100,
        stride=int(cfg.training.additional_train_stride),
        workers=cfg.training.workers,
        force_indices=force_indices,
    )
    regular_loader = DataLoader(
        train,
        batch_size=1,
        shuffle=True,
        collate_metadata=True,
        prefetch_factor=int(cfg.data.prefetch_factor),
        num_streams=int(cfg.data.num_streams),
        use_streams=True,
        seed=seed,
    )
    pairs = DynamicMatchedPairStatDataset(
        cfg.data.paired_action_root,
        cfg.data.paired_zero_root,
        cfg.data.paired_manifest,
        num_workers=cfg.training.workers,
    )
    pair_loader = DataLoader(
        pairs,
        batch_size=1,
        shuffle=True,
        collate_metadata=True,
        prefetch_factor=int(cfg.data.prefetch_factor),
        num_streams=int(cfg.data.num_streams),
        use_streams=True,
        seed=seed,
    )
    if len(pairs) != PAIR_SIZE or len(regular_loader) < TOTAL_STEPS:
        raise RuntimeError("dataset sizes differ")
    model = build_model(cfg).to(dist.device)
    parent = Path(cfg.training.initial_checkpoint)
    models, states = (
        list(parent.glob("FNO.0.*.mdlus")),
        list(parent.glob("checkpoint.0.*.pt")),
    )
    if (
        len(models) != 1
        or len(states) != 1
        or sha256(models[0]) != PARENT_MODEL_SHA
        or sha256(states[0]) != PARENT_STATE_SHA
    ):
        raise ValueError("C parent differs")
    if load_checkpoint(parent, models=model, device=dist.device) != 2:
        raise ValueError("requires C epoch2")
    before_sha = tensor_state_sha256(model)
    channel_weights = force_channel_weights(cfg, force_indices, dist.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
    )
    prepare_true_state_training_model(model)
    state_mean = base.state_mean.to(dist.device)[None, None]
    state_std = base.state_std.to(dist.device)[None, None]
    force_std = base.force_std.to(dist.device)
    readout_items = build_train_readout_items(cfg, dist.device)
    before = train_h1_readout(
        model, readout_items, state_mean, state_std, force_std, predict
    )
    positions = set(calibration_positions())
    pair_iterator = iter(pair_loader)
    records, identities, passes = [], [], [[] for _ in range(PAIR_PASSES)]
    regular_iterator = iter(regular_loader)
    paired_count = 0
    for step in range(TOTAL_STEPS):
        batch, metadata = next(regular_iterator)
        batch = {k: v.to(dist.device) for k, v in batch.items()}
        base_loss, field_loss, force_loss = regular_rollout_objective(
            model,
            batch["state"],
            batch["target_state"],
            batch["omega"],
            batch["target_force"],
            batch["mask"],
            channel_weights,
            force_loss_weight=float(cfg.training.force_loss_weight),
            rollout_discount=float(cfg.training.rollout_discount),
            teacher_forcing_ratio=0.0,
        )
        if step in positions:
            if paired_count and paired_count % PAIR_SIZE == 0:
                pair_iterator = iter(pair_loader)
            pair, pair_metadata = next(pair_iterator)
            pair = {k: v.to(dist.device) for k, v in pair.items()}
            result = true_state_paired_optimizer_step(
                model=model,
                optimizer=optimizer,
                base_loss=base_loss,
                pair=pair,
                channel_weights=channel_weights,
                paired_weight=10.0,
                predict_force=lambda n, x, m: predict(n, x, m)[1],
                total_steps=100,
                chunk_size=10,
                gradient_clip_norm=float(cfg.training.gradient_clip_norm),
            )
            ids = _pair_ids(pair_metadata)
            passes[paired_count // PAIR_SIZE].extend(ids)
            paired_count += 1
            kind = "regular_plus_paired"
        else:
            result = regular_optimizer_step(
                model, optimizer, base_loss, float(cfg.training.gradient_clip_norm)
            )
            ids = []
            kind = "regular_only"
        identities.append(_identity(metadata))
        record = {
            "step": step,
            "kind": kind,
            "regular_identity": identities[-1],
            "pair_ids": ids,
            "loss": result["loss"],
            "field_loss": float(field_loss.detach()),
            "force_loss": float(force_loss.detach()),
            "preclip_gradient_norm": result["preclip_gradient_norm"],
        }
        for key in (
            "base_loss",
            "paired_step_force_loss",
            "paired_step_force_weighted_loss",
            "paired_step_force_per_channel_mse",
            "paired_step_force_per_channel_weighted_contribution",
            "optimizer_steps",
            "chunk_size",
            "chunk_count",
        ):
            if key in result:
                record[key] = result[key]
        records.append(record)
        if (step + 1) % 8 == 0:
            print(
                json.dumps(
                    {
                        "event": "fcp003c_train_fit_progress",
                        "completed_steps": step + 1,
                        "total_steps": TOTAL_STEPS,
                        "paired_steps_completed": paired_count,
                        "latest_kind": kind,
                        "latest_loss": result["loss"],
                    },
                    allow_nan=False,
                    sort_keys=True,
                ),
                flush=True,
            )
    expected_ids = {
        f"b{i:02d}:{profile}" for i in (0, 2, 4, 6) for profile in ("multisine", "prbs")
    }
    validate_pair_passes(passes, expected_ids)
    if paired_count != PAIRED_STEPS or len(identities) != TOTAL_STEPS:
        raise RuntimeError("bounded step counts differ")
    after = train_h1_readout(
        model, readout_items, state_mean, state_std, force_std, predict
    )
    after_sha = tensor_state_sha256(model)
    if after_sha == before_sha:
        raise RuntimeError("calibration did not change scratch model")
    args.output.mkdir(parents=True)
    (args.output / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg))
    receipt = {
        "status": "FCP003C_TRAIN_FIT_CALIBRATION_COMPLETE_NOT_ADMISSION",
        "optimizer_steps": TOTAL_STEPS,
        "paired_steps": paired_count,
        "paired_passes": passes,
        "regular_identities": identities,
        "records": records,
        "before_readout": before,
        "after_readout": after,
        "parent_model_sha256": PARENT_MODEL_SHA,
        "parent_state_sha256": PARENT_STATE_SHA,
        "model_state_before_sha256": before_sha,
        "model_state_after_sha256": after_sha,
        "training_sources": sources,
        "validation_accessed": False,
        "frozen_test_accessed": False,
        "ppo_executed": False,
        "scheduler_steps": 0,
        "execution_contract": {
            "config_path": str(args.config.resolve()),
            "config_sha256": CONFIG_SHA,
            "source_path": str(Path(__file__).resolve()),
            "source_sha256": sha256(Path(__file__).resolve()),
            "effective_optimizer_steps": TOTAL_STEPS,
            "effective_paired_steps": PAIRED_STEPS,
            "effective_paired_passes": PAIR_PASSES,
            "fixed_learning_rate": float(cfg.training.learning_rate),
            "scheduler_used": False,
            "selection_performed": False,
            "note": "Effective bounded calibration contract; resolved config retains the parent two-epoch training declaration.",
        },
    }
    final_dir = args.output / "final"
    save_checkpoint(
        final_dir,
        models=model,
        optimizer=optimizer,
        epoch=1,
        metadata={
            "status": receipt["status"],
            "optimizer_steps": TOTAL_STEPS,
            "paired_steps": paired_count,
            "selection_performed": False,
            "validation_accessed": False,
            "frozen_test_accessed": False,
        },
    )
    final_models = list(final_dir.glob("FNO.0.*.mdlus"))
    final_states = list(final_dir.glob("checkpoint.0.*.pt"))
    if len(final_models) != 1 or len(final_states) != 1:
        raise RuntimeError("requires one final model/state pair")
    receipt["final_model_sha256"] = sha256(final_models[0])
    receipt["final_state_sha256"] = sha256(final_states[0])
    receipt["selection_performed"] = False
    (args.output / "result.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    )
    train.close()
    pairs.close()


if __name__ == "__main__":
    main()
