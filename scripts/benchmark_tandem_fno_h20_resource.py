#!/usr/bin/env python3
"""Forward/backward resource benchmark for PhysicsNeMo FNO H20 on real CFD."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path


MIN_AVAILABLE_GIB = 20.0


@dataclass(frozen=True)
class Spec:
    name: str
    latent_channels: int
    batch_size: int


def benchmark_plan() -> list[Spec]:
    return [
        Spec("width48_batch4", 48, 4),
        Spec("width48_batch8", 48, 8),
        Spec("width64_batch4", 64, 4),
    ]


def mem_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / 1024**2
    raise RuntimeError("MemAvailable missing from /proc/meminfo")


def reserve_ok(available_gib: float, minimum_gib: float = MIN_AVAILABLE_GIB) -> bool:
    return available_gib >= minimum_gib


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class MemorySampler:
    def __init__(self, interval: float = 0.1) -> None:
        self.interval = interval
        self.values: list[float] = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop.is_set():
            self.values.append(mem_available_gib())
            self.stop.wait(self.interval)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.values.append(mem_available_gib())


def write_reports(output_json: Path, output_csv: Path, report: dict) -> None:
    temporary = output_json.with_suffix(output_json.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_json)
    fields = [
        "name", "latent_channels", "batch_size", "status", "timed_iterations",
        "elapsed_seconds", "sequences_per_second", "transitions_per_second",
        "peak_allocated_gib", "peak_reserved_gib", "min_mem_available_gib",
        "error",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in report["results"]:
            writer.writerow({key: row.get(key) for key in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=2)
    args = parser.parse_args()
    if args.output_json.exists() or args.output_csv.exists():
        parser.error("refusing to overwrite an existing benchmark report")
    if args.warmup < 1 or args.iterations < 1:
        parser.error("warmup and iterations must be positive")

    import numpy as np
    import physicsnemo
    import torch
    from fluid_control.tandem_datapipe import TandemRolloutDataset
    from omegaconf import OmegaConf
    from train_tandem_fno import build_model
    from train_tandem_fno_rollout import rollout

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(0.65, device=device)
    torch.manual_seed(20261003)
    torch.cuda.manual_seed_all(20261003)

    dataset = TandemRolloutDataset(
        args.data, "train", rollout_steps=20, stride=1, num_workers=1,
        force_indices=(0, 1, 2, 3),
    )
    if len(dataset) < 8:
        raise ValueError("real train split has fewer than eight H20 windows")
    sample_indices = np.linspace(0, len(dataset) - 1, 8, dtype=int).tolist()
    samples = []
    sample_metadata = []
    for index in sample_indices:
        sample, metadata = dataset._load(index)
        samples.append({key: value for key, value in sample.items()})
        sample_metadata.append(metadata)

    def real_batch(batch_size: int) -> dict[str, torch.Tensor]:
        return {
            key: torch.stack([samples[index][key] for index in range(batch_size)]).to(
                device, non_blocking=False
            )
            for key in ("state", "target_state", "omega", "target_force", "mask")
        }

    train_files = sorted((args.data / "train").glob("*.h5"))
    report = {
        "status": "RUNNING",
        "scope": "forward+backward only; no optimizer, parameter update, checkpoint write, validation or frozen-test access",
        "framework": "official NVIDIA PhysicsNeMo FNO",
        "runtime": {
            "physicsnemo_version": getattr(physicsnemo, "__version__", "unknown"),
            "torch_version": torch.__version__,
            "device_name": torch.cuda.get_device_name(device),
            "device_total_gib": torch.cuda.get_device_properties(device).total_memory / 1024**3,
            "pinned_container_image_id": os.environ.get("BENCHMARK_IMAGE_ID", "unrecorded"),
        },
        "precision": "float32; AMP disabled to match current H20 training",
        "rollout_steps": 20,
        "model_fixed": {
            "in_channels": 6, "out_channels": 7, "num_fno_layers": 5,
            "num_fno_modes": [32, 32], "decoder_layers": 2,
            "decoder_layer_size": 128, "padding": 8, "coord_features": True,
        },
        "data": {
            "root": str(args.data), "split": "train",
            "trajectory_count": len(train_files),
            "trajectory_sha256": {str(path): sha256(path) for path in train_files},
            "sample_indices": sample_indices,
            "sample_metadata": sample_metadata,
            "real_data_guard": "TandemRolloutDataset reads normalized contiguous H20 windows from curated OpenFOAM HDF5",
        },
        "warmup_iterations": args.warmup,
        "timed_iterations": args.iterations,
        "minimum_mem_available_gib": MIN_AVAILABLE_GIB,
        "results": [],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    width48_safe = True
    for spec in benchmark_plan():
        before = mem_available_gib()
        result = {**asdict(spec), "mem_available_before_gib": before}
        if not reserve_ok(before):
            result.update(status="SKIPPED_MEMORY_GUARD", error="MemAvailable below 20 GiB")
            report["results"].append(result)
            width48_safe = False
            write_reports(args.output_json, args.output_csv, report)
            continue
        if spec.latent_channels == 64 and not width48_safe:
            result.update(status="SKIPPED_AFTER_UNSAFE_WIDTH48", error="width48 prerequisite failed")
            report["results"].append(result)
            write_reports(args.output_json, args.output_csv, report)
            continue
        batch = real_batch(spec.batch_size)
        cfg = OmegaConf.create({
            "model": {
                "in_channels": 6, "out_channels": 7,
                "latent_channels": spec.latent_channels, "num_fno_layers": 5,
                "num_fno_modes": [32, 32], "decoder_layers": 2,
                "decoder_layer_size": 128, "padding": 8, "coord_features": True,
            }
        })
        network = None
        try:
            network = build_model(cfg).to(device).train()
            result["parameter_count"] = sum(parameter.numel() for parameter in network.parameters())

            def step() -> torch.Tensor:
                network.zero_grad(set_to_none=True)
                predicted_state, predicted_force = rollout(
                    network, batch["state"], batch["mask"], batch["omega"]
                )
                field = ((predicted_state - batch["target_state"]).square() * batch["mask"][:, None]).mean()
                force = (predicted_force - batch["target_force"]).square().mean()
                loss = field + 0.2 * force
                loss.backward()
                return loss

            for _ in range(args.warmup):
                step()
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            losses = []
            with MemorySampler() as sampler:
                started = time.perf_counter()
                for _ in range(args.iterations):
                    losses.append(float(step().detach().cpu()))
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
            minimum = min(sampler.values)
            if not reserve_ok(minimum):
                raise RuntimeError(f"20 GiB MemAvailable guard violated: {minimum:.3f} GiB")
            result.update(
                status="OK", timed_iterations=args.iterations,
                elapsed_seconds=elapsed,
                sequences_per_second=spec.batch_size * args.iterations / elapsed,
                transitions_per_second=spec.batch_size * 20 * args.iterations / elapsed,
                peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024**3,
                peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 1024**3,
                min_mem_available_gib=minimum,
                final_loss=losses[-1], error=None,
            )
        except torch.OutOfMemoryError as error:
            result.update(status="OOM", error=str(error), min_mem_available_gib=mem_available_gib())
            if spec.latent_channels == 48:
                width48_safe = False
        except RuntimeError as error:
            result.update(status="FAILED_GUARD", error=str(error), min_mem_available_gib=mem_available_gib())
            if spec.latent_channels == 48:
                width48_safe = False
        finally:
            del network, batch
            torch.cuda.empty_cache()
        report["results"].append(result)
        write_reports(args.output_json, args.output_csv, report)

    report["status"] = "COMPLETE"
    report["mem_available_after_gib"] = mem_available_gib()
    report["checkpoint_write_count"] = 0
    report["optimizer_created"] = False
    report["frozen_test_status"] = "NOT_ACCESSED"
    write_reports(args.output_json, args.output_csv, report)
    dataset.close()


if __name__ == "__main__":
    main()
