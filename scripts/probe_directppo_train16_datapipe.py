#!/usr/bin/env python3
"""Read real train-only CFD windows through official PhysicsNeMo datapipes.

This CPU integration check performs no model training and never opens a
validation or frozen-test HDF. Synthetic unit-test metadata is not accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from physicsnemo.datapipes import DataLoader
from torch.utils.data import SubsetRandomSampler

from fluid_control.augmented_datapipe import compose_training_data
from fluid_control.tandem_datapipe import TandemRolloutDataset


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--train8", type=Path, required=True)
    parser.add_argument("--train16", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if (
        sha(args.base / "manifest.json")
        != "5213c7bb07c824c6e601636c3cfa974b80ec7c051633b0c8368a045cea41ddd2"
    ):
        raise ValueError("immutable dev30 manifest differs")
    if (
        sha(args.base / "normalization.json")
        != "f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1"
    ):
        raise ValueError("immutable train20 normalization differs")
    if (args.base / "frozen_test").exists():
        raise ValueError("base dataset exposes frozen-test files")
    torch.set_num_threads(2)
    base = TandemRolloutDataset(
        args.base, "train", 100, stride=20, num_workers=1, force_indices=(0, 1, 2, 3)
    )
    combined = None
    try:
        combined, sources = compose_training_data(
            base,
            [args.train8, args.train16],
            rollout_steps=100,
            stride=2,
            workers=1,
            force_indices=(0, 1, 2, 3),
        )
        if [row["windows"] for row in sources] != [408, 240]:
            raise ValueError("H100 additional window counts differ")
        indices = [0, len(base), len(base) + sources[0]["windows"], len(combined) - 1]
        sampler = SubsetRandomSampler(
            indices, generator=torch.Generator().manual_seed(20261004)
        )
        loader = DataLoader(
            combined,
            batch_size=1,
            sampler=sampler,
            collate_metadata=True,
            prefetch_factor=0,
            use_streams=False,
        )
        rows = []
        for batch, metadata in loader:
            expected = {
                "state": (1, 3, 128, 256),
                "target_state": (1, 100, 3, 128, 256),
                "target_force": (1, 100, 4),
                "omega": (1, 101, 1),
                "mask": (1, 1, 128, 256),
            }
            for name, shape in expected.items():
                if (
                    tuple(batch[name].shape) != shape
                    or not torch.isfinite(batch[name]).all()
                ):
                    raise ValueError(
                        f"real DataLoader shape/finite check failed: {name}"
                    )
            omega = batch["omega"] * 0.75
            if len(metadata) != 1 or metadata[0]["split"] != "train":
                raise ValueError("real DataLoader metadata split differs")
            # Old VTK-time-interpolated actions retain their reviewed 2e-5
            # representation tolerance; train16 exact endpoints use 2e-6.
            slew_tolerance = 2.0e-6 if metadata[0]["dataset_index"] == 2 else 2.0e-5
            if (
                omega.abs().max() > 0.750002
                or omega.diff(dim=1).abs().max() > 0.1 + slew_tolerance
            ):
                raise ValueError("real DataLoader action contract differs")
            rows.append(
                {
                    "metadata": metadata[0],
                    "shapes": {key: list(value.shape) for key, value in batch.items()},
                    "max_abs_omega": float(omega.abs().max()),
                    "max_delta_omega": float(omega.diff(dim=1).abs().max()),
                    "action_slew_representation_tolerance": slew_tolerance,
                }
            )
        if len(rows) != 4 or {row["metadata"]["dataset_index"] for row in rows} != {
            0,
            1,
            2,
        }:
            raise ValueError("probe did not read all three real training sources")
        report = {
            "status": "DIRECTPPO_TRAIN16_OFFICIAL_DATAPIPE_PROBE_PASS",
            "training_executed": False,
            "validation_or_frozen_hdf_opened": False,
            "official_components": [
                "DataLoader",
                "MultiDataset",
                "DatasetBase",
                "HDF5Reader",
            ],
            "rollout_steps": 100,
            "base_windows": len(base),
            "total_windows": len(combined),
            "additional_sources": sources,
            "samples": rows,
            "script_sha256": sha(Path(__file__)),
            "implementation_sha256": {
                name: sha(Path(__file__).resolve().parents[1] / name)
                for name in (
                    "src/fluid_control/augmented_datapipe.py",
                    "src/fluid_control/tandem_datapipe.py",
                )
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
        print(json.dumps(report))
    finally:
        if combined is not None:
            combined.close()
        else:
            base.close()


if __name__ == "__main__":
    main()
