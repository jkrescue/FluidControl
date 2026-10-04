#!/usr/bin/env python3
"""CPU-only counterfactual audit of FC-P003 PhysicsNeMo loader ordering."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch
from physicsnemo.datapipes import DataLoader, DatasetBase
from tensordict import TensorDict


class IdentityDataset(DatasetBase):
    """Synthetic IDs used only to audit the official loader's order mechanics."""

    def __init__(self, count: int) -> None:
        super().__init__(num_workers=1)
        self.count = count

    def __len__(self) -> int:
        return self.count

    def _load(self, index: int):
        return TensorDict({"identity": torch.tensor(index)}, batch_size=[]), {"id": index}


def paired_indices(train_batches: int, pair_batches: int, schedule: str) -> tuple[int, ...]:
    if schedule == "frontloaded":
        return tuple(range(pair_batches))
    if schedule != "interleaved":
        raise ValueError(schedule)
    return tuple(i * (train_batches - 1) // (pair_batches - 1) for i in range(pair_batches))


def simulate(schedule: str, seed: int = 20261003) -> dict:
    regular_loader = DataLoader(
        IdentityDataset(1368), batch_size=1, shuffle=True, collate_metadata=True,
        prefetch_factor=0, use_streams=False, seed=seed,
    )
    pair_loader = DataLoader(
        IdentityDataset(16), batch_size=1, shuffle=True, collate_metadata=True,
        prefetch_factor=0, use_streams=False, seed=seed,
    )
    positions = paired_indices(1368, 16, schedule)
    position_set = set(positions)
    pair_iterator = iter(pair_loader)
    regular_ids, pair_ids = [], []
    torch.manual_seed(99173)
    initial_rng = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    for index, (_, metadata) in enumerate(regular_loader):
        regular_ids.append(int(metadata[0]["id"]))
        if index in position_set:
            _, pair_metadata = next(pair_iterator)
            pair_ids.append(int(pair_metadata[0]["id"]))
    final_rng = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    return {
        "schedule": schedule,
        "positions": list(positions),
        "regular_ids": regular_ids,
        "pair_ids": pair_ids,
        "initial_global_torch_rng_sha256": initial_rng,
        "final_global_torch_rng_sha256": final_rng,
    }


def audit() -> dict:
    frontloaded = simulate("frontloaded")
    interleaved = simulate("interleaved")
    checks = {
        "regular_id_sequence_identical": frontloaded["regular_ids"] == interleaved["regular_ids"],
        "pair_id_sequence_identical": frontloaded["pair_ids"] == interleaved["pair_ids"],
        "global_torch_rng_initial_identical": frontloaded["initial_global_torch_rng_sha256"] == interleaved["initial_global_torch_rng_sha256"],
        "global_torch_rng_final_identical": frontloaded["final_global_torch_rng_sha256"] == interleaved["final_global_torch_rng_sha256"],
        "only_positions_differ": frontloaded["positions"] != interleaved["positions"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"FC-P003 DataLoader counterfactual failed: {checks}")
    encode = lambda values: hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()
    return {
        "status": "FC_P003_OFFICIAL_DATALOADER_ORDER_COUNTERFACTUAL_PASS",
        "scope": "synthetic identities test sampling mechanics only; not scientific CFD data",
        "physicsnemo_dataloader": "physicsnemo.datapipes.DataLoader",
        "seed": 20261003,
        "regular_count": 1368,
        "pair_count": 16,
        "checks": checks,
        "regular_sequence_sha256": encode(frontloaded["regular_ids"]),
        "pair_sequence_sha256": encode(frontloaded["pair_ids"]),
        "pair_sequence": frontloaded["pair_ids"],
        "frontloaded_positions": frontloaded["positions"],
        "interleaved_positions": interleaved["positions"],
        "global_torch_rng_sha256": frontloaded["final_global_torch_rng_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{args.output.name}.", dir=args.output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True); stream.write("\n")
        os.replace(temporary, args.output)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
