#!/usr/bin/env python3
"""Check the official PhysicsNeMo FNO CUDA backward pass on real CFD windows.

This is a runtime test only: raw fields are not normalized, no optimizer step is
taken, and its loss is not a model-quality metric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from hydra import compose, initialize_config_dir

from train_tandem_fno import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", type=Path,
        default=Path("data/curated/tandem_cylinders_expanded_v1/train/expanded_train_00.h5"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or not torch.cuda.is_available():
        parser.error("positive batch size and CUDA device 0 are required")

    torch.cuda.set_per_process_memory_fraction(0.20, device=0)
    with h5py.File(args.case, "r") as handle:
        if len(handle["state"]) < args.batch_size + 1:
            raise ValueError("real CFD case has too few frames")
        state = np.asarray(handle["state"][:args.batch_size + 1], dtype=np.float32)
        mask = np.asarray(handle["mask"][:args.batch_size], dtype=np.float32)
        omega = np.asarray(handle["omega"][:args.batch_size + 1, 0], dtype=np.float32)
    height, width = mask.shape[-2:]
    action_now = np.broadcast_to(
        (omega[:-1] / 5.0)[:, None, None, None], (args.batch_size, 1, height, width)
    )
    action_next = np.broadcast_to(
        (omega[1:] / 5.0)[:, None, None, None], (args.batch_size, 1, height, width)
    )
    inputs = np.concatenate((state[:-1], mask, action_now, action_next), axis=1)
    target = state[1:] - state[:-1]
    x = torch.from_numpy(inputs.copy()).to("cuda:0")
    y_target = torch.from_numpy(target.copy()).to("cuda:0")
    valid_mask = torch.from_numpy(mask).to("cuda:0")

    with initialize_config_dir(config_dir=str(Path("conf").resolve()), version_base="1.3"):
        cfg = compose(config_name="tandem_fno_expanded")
    model = build_model(cfg).to("cuda:0")
    output = model(x)
    if tuple(output.shape) != (args.batch_size, 5, height, width):
        raise AssertionError(f"unexpected FNO output shape {tuple(output.shape)}")
    loss = ((output[:, :3] - y_target).square() * valid_mask).sum()
    loss = loss / (valid_mask.sum().clamp_min(1) * 3)
    loss.backward()
    if not torch.isfinite(loss) or not all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters() if parameter.grad is not None
    ):
        raise FloatingPointError("non-finite loss or FNO gradient")
    report = {
        "case": args.case.name,
        "batch_size": args.batch_size,
        "real_window_indices": list(range(args.batch_size)),
        "model": "physicsnemo.models.fno.FNO",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "output_shape": list(output.shape),
        "raw_field_smoke_loss_not_quality_metric": float(loss.detach()),
        "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(0) / 1024**3,
        "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved(0) / 1024**3,
    }
    print(json.dumps(report), flush=True)
    print("REAL_CFD_PHYSICSNEMO_FNO_BACKWARD_OK", flush=True)


if __name__ == "__main__":
    main()
