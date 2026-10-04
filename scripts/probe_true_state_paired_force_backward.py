#!/usr/bin/env python3
"""Backward-only H100 technical probe for one real train-only dynamic pair."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import threading
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from physicsnemo.datapipes import DataLoader
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import load_checkpoint
from train_tandem_fno import build_model, configured_force_indices, predict

from fluid_control.dynamic_pair_stat_datapipe import DynamicMatchedPairStatDataset
from fluid_control.paired_step_force import (
    paired_step_force_delta_loss,
    true_state_step_input,
)

HELPER_SHA = "d382c7c886509fb12b85e7f79080dcf1b85d9c2315c15717e97310459dbfee30"
PAIR_MANIFEST_SHA = "b756c6d777b68fe9dc6c5a81e40733e24d2c248b315723ddb40cc3273902d28c"
NORMALIZATION_SHA = "f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1"
ACTION_HDF_SHA = "400b3b5f8ea381da59c7806aef19f7cd2b50da303f717ad93cebf505c5949e9f"
ZERO_HDF_SHA = "243caa79ac320b421adc1bf0c2cc830a32482dc758c3c0f9ce71d26171a61a01"
MODEL_SHA = "8466bd47f2de188f5e741197832ec3bee1223f72f54d8956e584c586a2774240"
STATE_SHA = "1e5d4c055812d8f92bc55f58708e839f7cc776071849544f6a26d3d62053cbd0"
RESOLVED_CONFIG_SHA = "feaaeeeb8def2dae789f084ea676ceac95767998b7b0865275b49a063330b76e"
TRAINER_SHA = "9e5bbebd338af02b1d73530c9cb74fc2f840455f56b7bf34ed2bb517be19a22a"
CHANNELS = ("front_cd", "front_cl", "rear_cd", "rear_cl")
WEIGHTS = (1 / 7, 1 / 7, 4 / 7, 1 / 7)
LOSS_RTOL, LOSS_ATOL, GRAD_RTOL, GRAD_ATOL = 2e-5, 1e-7, 3e-4, 3e-6


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def mem_gib():
    line = next(
        x
        for x in Path("/proc/meminfo").read_text().splitlines()
        if x.startswith("MemAvailable:")
    )
    return float(line.split()[1]) / 1024**2


class MemorySampler:
    def __init__(self, floor):
        self.floor = floor
        self.start = self.minimum = mem_gib()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop.wait(0.1):
            self.minimum = min(self.minimum, mem_gib())

    def __enter__(self):
        if self.start < self.floor:
            raise RuntimeError("physical MemAvailable guard failed before probe")
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.minimum = min(self.minimum, mem_gib())
        if self.minimum < self.floor:
            raise RuntimeError("physical MemAvailable fell below guard")


def scaled_loss(pa, pz, ta, tz, weights, total):
    steps = pa.shape[1]
    if total < 1 or not 0 < steps <= total:
        raise ValueError("invalid chunk/total step count")
    return paired_step_force_delta_loss(pa, pz, ta, tz, weights) * (steps / total)


def predict_chunk(model, pair, start, stop):
    action, zero = [], []
    mask = torch.cat((pair["mask"], pair["mask"]), 0)
    for step in range(start, stop):
        xa = true_state_step_input(
            pair["action_state"][:, 0],
            pair["action_state"][:, 1:],
            pair["mask"],
            pair["action_omega"],
            step,
        )
        xz = true_state_step_input(
            pair["zero_state"][:, 0],
            pair["zero_state"][:, 1:],
            pair["mask"],
            pair["zero_omega"],
            step,
        )
        _, force = predict(model, torch.cat((xa, xz), 0), mask)
        batch = xa.shape[0]
        action.append(force[:batch])
        zero.append(force[batch:])
    return torch.stack(action, 1), torch.stack(zero, 1)


def backward_range(model, pair, weights, total, chunk):
    value, pieces = 0.0, []
    channel_sse = torch.zeros(4, dtype=torch.float64, device=weights.device)
    for start in range(0, total, chunk):
        stop = min(total, start + chunk)
        pa, pz = predict_chunk(model, pair, start, stop)
        ta = pair["action_force"][:, 1 + start : 1 + stop]
        tz = pair["zero_force"][:, 1 + start : 1 + stop]
        error = (pa - pz) - (ta - tz)
        channel_sse += error.detach().double().square().sum((0, 1))
        loss = scaled_loss(pa, pz, ta, tz, weights, total)
        loss.backward()
        current = float(loss.detach())
        if not math.isfinite(current):
            raise FloatingPointError("non-finite chunk loss")
        value += current
        pieces.append(current)
    return value, pieces, channel_sse / (pair["action_state"].shape[0] * total)


def gradients(model, clone=False):
    result, square, maximum, count, nonzero = {}, 0.0, 0.0, 0, 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            grad = parameter.grad
            if grad is None or not torch.isfinite(grad).all():
                raise FloatingPointError(f"missing/non-finite gradient: {name}")
            if clone:
                result[name] = grad.detach().cpu().clone()
            square += float(grad.detach().abs().double().square().sum())
            maximum = max(maximum, float(grad.detach().abs().max()))
            count += grad.numel()
            nonzero += int(torch.count_nonzero(grad))
    summary = {
        "finite": True,
        "element_count": count,
        "nonzero_count": nonzero,
        "l2_norm": math.sqrt(square),
        "max_abs": maximum,
    }
    return result, summary


def model_tensor_sha(model):
    """Hash parameters and buffers without retaining a second model copy."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def args():
    parser = argparse.ArgumentParser()
    for name in (
        "repo",
        "action-root",
        "zero-root",
        "pair-manifest",
        "checkpoint",
        "output",
        "source-manifest",
        "launch-receipt",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--pair-id", default="b00:multisine")
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--equivalence-steps", type=int, default=20)
    parser.add_argument("--total-steps", type=int, default=100)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.20)
    parser.add_argument("--min-mem-available-gib", type=float, default=20.0)
    return parser.parse_args()


def main():
    a = args()
    if (a.chunk_size, a.equivalence_steps, a.total_steps) != (10, 20, 100):
        raise ValueError("reviewed contract is chunk10/equivalence20/H100")
    helper = a.repo / "src/fluid_control/paired_step_force.py"
    if sha(helper) != HELPER_SHA:
        raise ValueError("helper SHA differs")
    trainer = a.repo / "scripts/train_tandem_fno.py"
    if sha(trainer) != TRAINER_SHA:
        raise ValueError("model/predict implementation SHA differs")
    manifest = json.loads(a.pair_manifest.read_text())
    selected = manifest["pairs"][0]
    if manifest.get("validation_or_frozen_accessed") is not False:
        raise ValueError("not train-only")
    if f"{selected['phase']}:{selected['profile']}" != a.pair_id:
        raise ValueError("pair differs")
    fixed_inputs = {
        "pair_manifest": (a.pair_manifest, PAIR_MANIFEST_SHA),
        "normalization": (a.action_root / "normalization.json", NORMALIZATION_SHA),
        "action_hdf": (a.action_root / selected["action_file"], ACTION_HDF_SHA),
        "zero_hdf": (a.zero_root / selected["zero_file"], ZERO_HDF_SHA),
        "model": (a.checkpoint / "FNO.0.2.mdlus", MODEL_SHA),
        "state": (a.checkpoint / "checkpoint.0.2.pt", STATE_SHA),
    }
    for label, (path, expected) in fixed_inputs.items():
        if sha(path) != expected:
            raise ValueError(f"fixed {label} SHA differs")
    DistributedManager.initialize()
    dist = DistributedManager()
    if not dist.cuda or dist.world_size != 1:
        raise RuntimeError("one CUDA GPU required")
    torch.cuda.set_per_process_memory_fraction(
        a.gpu_memory_fraction, device=dist.device
    )
    torch.manual_seed(20261003)
    with initialize_config_dir(
        config_dir=str((a.repo / "conf").resolve()), version_base="1.3"
    ):
        cfg = compose(config_name="tandem_fno_control_train16_h100")
    resolved_config = OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True).encode()
    if hashlib.sha256(resolved_config).hexdigest() != RESOLVED_CONFIG_SHA:
        raise ValueError("resolved inherited config SHA differs")
    if tuple(configured_force_indices(cfg)) != (0, 1, 2, 3):
        raise ValueError("force channels differ")
    dataset = DynamicMatchedPairStatDataset(
        a.action_root,
        a.zero_root,
        a.pair_manifest,
        num_workers=1,
        verify_hdf_sha256=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_metadata=True,
        prefetch_factor=0,
        use_streams=False,
        seed=20261003,
    )
    pair, pair_metadata = next(iter(loader))
    if len(pair_metadata) != 1 or pair_metadata[0].get("pair_id") != a.pair_id:
        raise ValueError("official DataLoader pair identity differs")
    pair = {key: val.to(dist.device) for key, val in pair.items()}
    model = build_model(cfg).to(dist.device)
    metadata = {}
    if (
        load_checkpoint(
            a.checkpoint, models=model, metadata_dict=metadata, device=dist.device
        )
        != 2
    ):
        raise RuntimeError("Main-e2 checkpoint did not load")
    model_before_sha = model_tensor_sha(model)
    model.train()
    weights = torch.tensor(WEIGHTS, device=dist.device)
    torch.cuda.reset_peak_memory_stats(dist.device)
    started = time.monotonic()
    with MemorySampler(a.min_mem_available_gib) as memory:
        model.zero_grad(set_to_none=True)
        whole, _, _ = backward_range(model, pair, weights, 20, 20)
        reference, _ = gradients(model, clone=True)
        model.zero_grad(set_to_none=True)
        chunked, _, _ = backward_range(model, pair, weights, 20, 10)
        max_difference = 0.0
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                expected = reference[name].to(parameter.grad.device)
                max_difference = max(
                    max_difference, float((parameter.grad - expected).abs().max())
                )
                if not torch.allclose(
                    parameter.grad, expected, rtol=GRAD_RTOL, atol=GRAD_ATOL
                ):
                    raise RuntimeError(f"gradient equivalence failed: {name}")
        if not math.isclose(whole, chunked, rel_tol=LOSS_RTOL, abs_tol=LOSS_ATOL):
            raise RuntimeError("loss equivalence failed")
        del reference
        model.zero_grad(set_to_none=True)
        h100, pieces, channel_mse = backward_range(model, pair, weights, 100, 10)
        _, grad_summary = gradients(model)
        if not grad_summary["nonzero_count"]:
            raise RuntimeError("all gradients zero")
        torch.cuda.synchronize(dist.device)
        model_after_sha = model_tensor_sha(model)
        if model_after_sha != model_before_sha:
            raise RuntimeError("model parameter/buffer digest changed")
    runtime = time.monotonic() - started
    result = {
        "status": "TRUE_STATE_PAIRED_FORCE_BACKWARD_TECHNICAL_PROBE_PASS",
        "scope": "paired-term-only; not full-training memory or scientific evidence",
        "training_executed": False,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "candidate_weights_saved": False,
        "model_parameters_and_buffers_unchanged": True,
        "validation_or_frozen_accessed": False,
        "model_mode": "train",
        "image_id": a.image_id,
        "runtime_versions": {
            "physicsnemo": importlib.metadata.version("nvidia-physicsnemo"),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "pair_id": a.pair_id,
        "horizon": 100,
        "chunk_size": 10,
        "force_channels": list(CHANNELS),
        "channel_weights": list(WEIGHTS),
        "loss_space": "train-normalized force coefficients",
        "equivalence": {
            "steps": 20,
            "unchunked_loss": whole,
            "chunked_loss": chunked,
            "loss_rtol": LOSS_RTOL,
            "loss_atol": LOSS_ATOL,
            "gradient_rtol": GRAD_RTOL,
            "gradient_atol": GRAD_ATOL,
            "max_gradient_abs_difference": max_difference,
        },
        "h100_normalized_loss": h100,
        "h100_chunk_losses": pieces,
        "h100_per_channel_normalized_mse": channel_mse.cpu().tolist(),
        "gradients": grad_summary,
        "runtime_seconds": runtime,
        "torch_peak_allocated_gib": torch.cuda.max_memory_allocated(dist.device)
        / 1024**3,
        "torch_peak_reserved_gib": torch.cuda.max_memory_reserved(dist.device)
        / 1024**3,
        "physical_mem_available_gib": {
            "start": memory.start,
            "minimum": memory.minimum,
            "end": mem_gib(),
            "required_minimum": a.min_mem_available_gib,
        },
        "input_sha256": {
            "execution_script": sha(Path(__file__)),
            "helper": sha(helper),
            "train_tandem_fno": sha(trainer),
            "pair_manifest": sha(a.pair_manifest),
            "action_hdf": sha(a.action_root / selected["action_file"]),
            "zero_hdf": sha(a.zero_root / selected["zero_file"]),
            "normalization": sha(a.action_root / "normalization.json"),
            "model": sha(a.checkpoint / "FNO.0.2.mdlus"),
            "state": sha(a.checkpoint / "checkpoint.0.2.pt"),
            "config": sha(a.repo / "conf/tandem_fno_control_train16_h100.yaml"),
            "resolved_config": hashlib.sha256(resolved_config).hexdigest(),
            "model_parameters_and_buffers_before": model_before_sha,
            "model_parameters_and_buffers_after": model_after_sha,
            "source_snapshot_manifest": sha(a.source_manifest),
            "launch_receipt": sha(a.launch_receipt),
        },
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    temp = a.output.with_suffix(a.output.suffix + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(result, indent=2) + "\n")
    os.link(temp, a.output)
    temp.unlink()
    dataset.close()
    print(
        json.dumps(
            {"status": result["status"], "loss": h100, "runtime_seconds": runtime}
        )
    )


if __name__ == "__main__":
    main()
