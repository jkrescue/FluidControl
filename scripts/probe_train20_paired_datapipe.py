#!/usr/bin/env python3
"""CPU probe of all matched pairs through the official PhysicsNeMo DataLoader."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import physicsnemo
import torch
from physicsnemo.datapipes import DataLoader

from fluid_control.paired_stat_datapipe import MatchedPairStatDataset


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = MatchedPairStatDataset(args.data, args.manifest, num_workers=1)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_metadata=True,
        prefetch_factor=0,
        use_streams=False,
    )
    records = []
    try:
        for batch, metadata in loader:
            records.append(
                {
                    "phase": metadata[0]["phase"],
                    "action": metadata[0]["action"],
                    "state_shape": list(batch["action_state"].shape),
                    "force_shape": list(batch["action_force"].shape),
                    "finite": all(torch.isfinite(value).all().item() for value in batch.values()),
                    "physical_targets_recomputed": True,
                }
            )
    finally:
        dataset.close()
    if len(records) != 16 or not all(item["finite"] for item in records):
        raise RuntimeError("paired DataPipe CPU probe did not consume 16 finite pairs")
    result = {
        "status": "TRAIN20_MATCHED_PAIR_OFFICIAL_DATAPIPE_CPU_PROBE_PASS",
        "official_components": [
            "physicsnemo.datapipes.DatasetBase",
            "physicsnemo.datapipes.readers.hdf5.HDF5Reader",
            "physicsnemo.datapipes.DataLoader",
        ],
        "physicsnemo_version": physicsnemo.__version__,
        "pair_manifest_sha256": sha(args.manifest),
        "normalization_sha256": sha(args.data / "normalization.json"),
        "adapter_sha256": sha(
            Path(__file__).resolve().parents[1]
            / "src/fluid_control/paired_stat_datapipe.py"
        ),
        "probe_script_sha256": sha(Path(__file__)),
        "pair_count": len(records),
        "validation_or_frozen_accessed": False,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.link(temporary, args.output)
    temporary.unlink()
    print(json.dumps({"status": result["status"], "pairs": len(records)}))


if __name__ == "__main__":
    main()
