#!/usr/bin/env python3
"""Consume all eight FC-P003 fallback pairs with official PhysicsNeMo DataLoader."""

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch
from physicsnemo.datapipes import DataLoader

from fluid_control.dynamic_pair_stat_datapipe import DynamicMatchedPairStatDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-root", type=Path, required=True)
    parser.add_argument("--zero-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    dataset = DynamicMatchedPairStatDataset(args.action_root, args.zero_root, args.manifest, num_workers=2)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_metadata=True, prefetch_factor=0, use_streams=False, seed=20261003)
    identities = []
    for sample, metadata in loader:
        if not all(torch.isfinite(value).all() for value in sample.values()):
            raise RuntimeError("non-finite official DataLoader batch")
        identities.append(metadata[0]["pair_id"])
    if len(identities) != 8 or len(set(identities)) != 8:
        raise RuntimeError("official DataLoader did not consume exact eight pairs")
    result = {"status": "FC_P003_DYNAMIC8_OFFICIAL_DATAPIPE_CPU_PROBE_PASS", "pair_count": 8, "identities": identities, "validation_or_frozen_accessed": False}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{args.output.name}.", dir=args.output.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2, sort_keys=True); stream.write("\n")
            os.replace(temporary, args.output)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
