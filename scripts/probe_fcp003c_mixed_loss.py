#!/usr/bin/env python3
"""One-update FC-P003C mixed-loss engineering probe.

This script consumes train-only data and a scratch model loaded from the
immutable Main-e2 parent.  It delegates both the regular objective and the
mixed optimizer step to the reviewed training implementation.  It never
saves a model or checkpoint and is not a scientific accuracy experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


EXPECTED_REGULAR_METADATA = {
    "case": "matched_start_acquisition_train_b04_m075",
    "step": 180,
    "rollout_steps": 100,
    "split": "train",
    "dataset_index": 0,
}
EXPECTED_REGULAR_SHAPES = {
    "state": [1, 3, 128, 256],
    "target_state": [1, 100, 3, 128, 256],
    "omega": [1, 101, 1],
    "target_force": [1, 100, 4],
    "mask": [1, 1, 128, 256],
}
EXPECTED_PAIR_ID = "b00:multisine"
FORCE_CHANNELS = ["front_cd", "front_cl", "rear_cd", "rear_cl"]


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def validate_regular_identity(metadata: list[dict[str, Any]], batch: Any) -> None:
    if metadata != [EXPECTED_REGULAR_METADATA]:
        raise ValueError(f"regular batch identity differs: {metadata}")
    observed = {key: list(batch[key].shape) for key in EXPECTED_REGULAR_SHAPES}
    if observed != EXPECTED_REGULAR_SHAPES:
        raise ValueError(f"regular batch shapes differ: {observed}")


def validate_pair_identity(metadata: list[dict[str, Any]], pair: Any) -> None:
    if len(metadata) != 1 or metadata[0].get("pair_id") != EXPECTED_PAIR_ID:
        raise ValueError(f"paired batch identity differs: {metadata}")
    expected = {
        "action_state": [1, 101, 3, 128, 256],
        "zero_state": [1, 101, 3, 128, 256],
        "action_omega": [1, 101, 1],
        "zero_omega": [1, 101, 1],
        "action_force": [1, 101, 4],
        "zero_force": [1, 101, 4],
        "mask": [1, 1, 128, 256],
    }
    observed = {key: list(pair[key].shape) for key in expected}
    if observed != expected:
        raise ValueError(f"paired batch shapes differ: {observed}")


class CountingAdamW(torch.optim.AdamW):
    """AdamW that makes the one-step execution budget auditable."""

    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.step_count = 0

    def step(self, closure=None):
        self.step_count += 1
        if self.step_count > 1:
            raise RuntimeError("mixed-loss technical probe permits exactly one step")
        return super().step(closure)


class MemorySampler:
    def __init__(self, interval: float = 0.2) -> None:
        self.interval = interval
        self.minimum_gib = math.inf
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def available_gib() -> float:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / 1024**2
        raise RuntimeError("MemAvailable is absent from /proc/meminfo")

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.minimum_gib = min(self.minimum_gib, self.available_gib())

    def __enter__(self):
        self.minimum_gib = self.available_gib()
        self._thread.start()
        return self

    def __exit__(self, *_):
        self.minimum_gib = min(self.minimum_gib, self.available_gib())
        self._stop.set()
        self._thread.join()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--train8-root", required=True)
    parser.add_argument("--train16-root", required=True)
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--launch-receipt", required=True)
    parser.add_argument("--cpu-equivalence-receipt", required=True)
    parser.add_argument("--expected-cpu-equivalence-sha256", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.25)
    parser.add_argument("--min-mem-available-gib", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.gpu_memory_fraction <= 0.25:
        raise ValueError("gpu memory fraction must be in (0,0.25]")
    if args.min_mem_available_gib < 20:
        raise ValueError("physical MemAvailable floor cannot be below 20 GiB")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"exclusive probe output exists: {output}")
    if sha256(args.cpu_equivalence_receipt) != args.expected_cpu_equivalence_sha256:
        raise ValueError("CPU equivalence receipt SHA differs")
    cpu_receipt = json.loads(Path(args.cpu_equivalence_receipt).read_text())
    if cpu_receipt.get("status") != "FC_P003C_CPU_EQUIVALENCE_PASS":
        raise ValueError("CPU equivalence receipt is not PASS")

    from fluid_control.augmented_datapipe import compose_training_data
    from fluid_control.dynamic_pair_stat_datapipe import DynamicMatchedPairStatDataset
    from fluid_control.paired_step_training import (
        prepare_true_state_training_model,
        true_state_paired_optimizer_step,
    )
    from fluid_control.tandem_datapipe import TandemRolloutDataset
    from omegaconf import OmegaConf
    from physicsnemo.datapipes import DataLoader
    from physicsnemo.utils import load_checkpoint
    from train_tandem_fno import build_model, configured_force_indices, predict
    from train_tandem_fno_paired_stats import (
        force_channel_weights,
        regular_rollout_objective,
    )

    cfg = OmegaConf.load(args.config)
    required = {
        "rollout_steps": 100,
        "batch_size": 1,
        "seed": 20261003,
        "teacher_forcing_start": 0.0,
        "teacher_forcing_end": 0.0,
        "paired_stat_loss_weight": 10.0,
        "paired_step_chunk_size": 10,
        "paired_objective_kind": "true_state_step_force",
    }
    for key, expected in required.items():
        if cfg.training.get(key) != expected:
            raise ValueError(f"resolved config training.{key} differs")
    if list(cfg.training.force_channel_weights) != [1.0, 1.0, 4.0, 1.0]:
        raise ValueError("force channel weights differ")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("technical probe requires exactly one process/GPU")

    seed = int(cfg.training.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device=0)
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    parent_model_path = Path(args.parent) / "FNO.0.2.mdlus"
    parent_state_path = Path(args.parent) / "checkpoint.0.2.pt"
    parent_file_sha_before = {
        "model": sha256(parent_model_path),
        "state": sha256(parent_state_path),
    }

    force_indices = configured_force_indices(cfg)
    if tuple(force_indices) != (0, 1, 2, 3):
        raise ValueError("probe requires all four force channels")
    base = TandemRolloutDataset(
        args.base_root, "train", 100, stride=int(cfg.training.train_stride),
        num_workers=int(cfg.training.workers), force_indices=force_indices,
    )
    train = pair_dataset = None
    try:
        train, _ = compose_training_data(
            base, [args.train8_root, args.train16_root], rollout_steps=100,
            stride=int(cfg.training.additional_train_stride),
            workers=int(cfg.training.workers), force_indices=force_indices,
        )
        loader = DataLoader(
            train, batch_size=1, shuffle=True, collate_metadata=True,
            prefetch_factor=int(cfg.data.prefetch_factor),
            num_streams=int(cfg.data.num_streams), use_streams=True, seed=seed,
        )
        if len(loader) != 1368:
            raise ValueError(f"regular loader length differs: {len(loader)}")
        batch, regular_metadata = next(iter(loader))
        validate_regular_identity(regular_metadata, batch)
        regular_hdf_sha = sha256(
            Path(args.base_root)
            / "train"
            / f"{EXPECTED_REGULAR_METADATA['case']}.h5"
        )

        pair_dataset = DynamicMatchedPairStatDataset(
            args.train8_root, args.base_root, args.pair_manifest,
            num_workers=int(cfg.training.workers),
        )
        pair_loader = DataLoader(
            pair_dataset, batch_size=1, shuffle=False, collate_metadata=True,
            prefetch_factor=int(cfg.data.prefetch_factor),
            num_streams=int(cfg.data.num_streams), use_streams=True, seed=seed,
        )
        pair, pair_metadata = next(iter(pair_loader))
        validate_pair_identity(pair_metadata, pair)

        model = build_model(cfg).to(device)
        metadata: dict[str, Any] = {}
        epoch = load_checkpoint(
            Path(args.parent), models=model, metadata_dict=metadata, device=device
        )
        if epoch != 2:
            raise ValueError(f"Main-e2 parent loaded epoch differs: {epoch}")
        prepare_true_state_training_model(model)
        before_state_sha = model_state_sha256(model)
        optimizer = CountingAdamW(
            model.parameters(), lr=float(cfg.training.learning_rate),
            weight_decay=float(cfg.training.weight_decay),
        )
        weights = force_channel_weights(cfg, force_indices, device)
        batch = {key: value.to(device) for key, value in batch.items()}
        pair = {key: value.to(device) for key, value in pair.items()}

        def predict_force(model_arg, inputs, mask):
            return predict(model_arg, inputs, mask)[1]

        started = time.monotonic()
        with MemorySampler() as memory:
            base_loss, field_loss, force_loss = regular_rollout_objective(
                model, batch["state"], batch["target_state"], batch["omega"],
                batch["target_force"], batch["mask"], weights,
                force_loss_weight=float(cfg.training.force_loss_weight),
                rollout_discount=float(cfg.training.rollout_discount),
                teacher_forcing_ratio=0.0,
            )
            metrics = true_state_paired_optimizer_step(
                model=model, optimizer=optimizer, base_loss=base_loss, pair=pair,
                channel_weights=weights, paired_weight=10.0,
                predict_force=predict_force, total_steps=100, chunk_size=10,
                gradient_clip_norm=float(cfg.training.gradient_clip_norm),
            )
            torch.cuda.synchronize(device)
        elapsed = time.monotonic() - started
        after_state_sha = model_state_sha256(model)
        if optimizer.step_count != 1 or metrics.get("optimizer_steps") != 1:
            raise RuntimeError("technical probe did not execute exactly one step")
        if before_state_sha == after_state_sha:
            raise RuntimeError("scratch model did not change after the unique step")
        if memory.minimum_gib < args.min_mem_available_gib:
            raise RuntimeError("physical MemAvailable floor was crossed")
        numeric = [
            metrics["loss"], metrics["base_loss"],
            metrics["paired_step_force_loss"],
            metrics["paired_step_force_weighted_loss"],
            metrics["preclip_gradient_norm"], float(field_loss.detach()),
            float(force_loss.detach()), *metrics["paired_step_force_per_channel_mse"],
            *metrics["paired_step_force_per_channel_weighted_contribution"],
        ]
        if not all(math.isfinite(value) for value in numeric):
            raise FloatingPointError("mixed probe metrics are non-finite")

        parent_file_sha_after = {
            "model": sha256(parent_model_path),
            "state": sha256(parent_state_path),
        }
        if parent_file_sha_after != parent_file_sha_before:
            raise RuntimeError("immutable parent files changed during probe")
        result = {
            "status": "FC_P003C_MIXED_LOSS_TECHNICAL_PROBE_PASS",
            "scientific_result": False,
            "candidate_saved": False,
            "validation_or_frozen_accessed": False,
            "optimizer_steps": 1,
            "regular_metadata": regular_metadata[0],
            "regular_hdf_sha256": regular_hdf_sha,
            "pair_metadata": pair_metadata[0],
            "force_channels": FORCE_CHANNELS,
            "force_channel_weights_normalized": weights.detach().cpu().tolist(),
            "regular_field_loss": float(field_loss.detach()),
            "regular_force_loss": float(force_loss.detach()),
            "metrics": metrics,
            "scratch_model_state_sha256_before": before_state_sha,
            "scratch_model_state_sha256_after": after_state_sha,
            "elapsed_seconds": elapsed,
            "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
            "minimum_mem_available_gib": memory.minimum_gib,
            "image_id": args.image_id,
            "torch_version": torch.__version__,
            "physicsnemo_version": importlib.metadata.version("nvidia-physicsnemo"),
            "config_sha256": sha256(args.config),
            "source_manifest_sha256": sha256(args.source_manifest),
            "launch_receipt_sha256": sha256(args.launch_receipt),
            "cpu_equivalence_receipt_sha256": sha256(args.cpu_equivalence_receipt),
            "parent_file_sha256_before": parent_file_sha_before,
            "parent_file_sha256_after": parent_file_sha_after,
        }
        output.mkdir(parents=False)
        temporary = output.with_name(f"{output.name}.json.tmp.{os.getpid()}")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        os.link(temporary, output / "result.json")
        temporary.unlink()
    finally:
        if pair_dataset is not None:
            pair_dataset.close()
        if train is not None:
            train.close()
        elif base is not None:
            base.close()


if __name__ == "__main__":
    main()
