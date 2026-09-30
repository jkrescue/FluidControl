#!/usr/bin/env python3
"""Exercise the official PhysicsNeMo HDF5/DataLoader path before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from fluid_control.tandem_datapipe import TandemWindowDataset
from physicsnemo.datapipes import DataLoader, DatasetBase
from physicsnemo.datapipes.readers.hdf5 import HDF5Reader
from tensordict import TensorDict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/curated/tandem_cylinders"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/tandem_cylinders/datapipe_validation.json"),
    )
    args = parser.parse_args()
    root = args.data
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    pairs = int(manifest["pairs_per_trajectory"])
    expected_lengths = {
        split: int(count) * pairs
        for split, count in manifest["trajectory_counts"].items()
    }
    report = {"official_components": {}, "splits": {}}

    report["official_components"] = {
        "dataset_base": f"{DatasetBase.__module__}.{DatasetBase.__name__}",
        "hdf5_reader": f"{HDF5Reader.__module__}.{HDF5Reader.__name__}",
        "data_loader": f"{DataLoader.__module__}.{DataLoader.__name__}",
    }

    datasets = {}
    for split, expected_length in expected_lengths.items():
        dataset = TandemWindowDataset(root, split, num_workers=4)
        datasets[split] = dataset
        assert isinstance(dataset, DatasetBase)
        assert len(dataset) == expected_length, (split, len(dataset), expected_length)
        sample, metadata = dataset[0]
        assert isinstance(sample, TensorDict)
        shapes = {key: list(value.shape) for key, value in sample.items()}
        assert shapes == {
            "x": [6, 128, 256],
            "delta": [3, 128, 256],
            "force": [2],
            "mask": [1, 128, 256],
            "time": [1],
        }
        assert all(torch.isfinite(value).all() for value in sample.values())
        report["splits"][split] = {
            "length": len(dataset),
            "sample_shapes": shapes,
            "metadata": metadata,
        }

    loader = DataLoader(
        datasets["train"],
        batch_size=4,
        shuffle=True,
        prefetch_factor=2,
        num_streams=1,
        use_streams=False,
        seed=20260929,
    )
    iterator = iter(loader)
    batch_shapes = []
    for _ in range(3):
        batch = next(iterator)
        assert isinstance(batch, TensorDict)
        assert all(torch.isfinite(value).all() for value in batch.values())
        batch_shapes.append({key: list(value.shape) for key, value in batch.items()})
    del iterator
    report["official_dataloader_batches"] = batch_shapes

    for dataset in datasets.values():
        dataset.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("PHYSICSNEMO_DATAPIPE_OK")


if __name__ == "__main__":
    main()
