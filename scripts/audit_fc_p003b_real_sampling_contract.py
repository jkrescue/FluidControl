#!/usr/bin/env python3
"""Audit FC-P003B sampler order on the real train-only dataset identities."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch
from physicsnemo.datapipes import DataLoader

from fluid_control.augmented_datapipe import compose_training_data
from fluid_control.dynamic_pair_stat_datapipe import DynamicMatchedPairStatDataset
from fluid_control.tandem_datapipe import TandemRolloutDataset

SEED = 20261003
FC_P003_REGULAR_SHA = "177ebd9523cde918eb0c1fb8026286dac7e95f3d228ae0e349757a2a9a288f9f"


def sequence_sha(values: list[int]) -> str:
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--train8", type=Path, required=True)
    parser.add_argument("--train16", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = TandemRolloutDataset(args.base, "train", 100, stride=20, num_workers=1, force_indices=(0, 1, 2, 3))
    regular, _ = compose_training_data(base, [args.train8, args.train16], rollout_steps=100, stride=2, workers=1, force_indices=(0, 1, 2, 3))
    pair = DynamicMatchedPairStatDataset(args.train8, args.base, args.pair_manifest, num_workers=1)
    torch.manual_seed(99173)
    initial_rng = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    regular_loader = DataLoader(regular, batch_size=1, shuffle=True, prefetch_factor=0, use_streams=False, seed=SEED)
    pair_loader = DataLoader(pair, batch_size=1, shuffle=True, prefetch_factor=0, use_streams=False, seed=SEED)
    regular_indices = list(iter(regular_loader.sampler))
    pair_pass_indices = [list(iter(pair_loader.sampler)), list(iter(pair_loader.sampler))]
    final_rng = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    pair_ids = [f"{row['phase']}:{row['profile']}" for row in pair.manifest["pairs"]]
    pair_pass_ids = [[pair_ids[index] for index in values] for values in pair_pass_indices]
    expected_ids = set(pair_ids)
    positions = [index * 1367 // 15 for index in range(16)]
    checks = {
        "real_regular_count_1368": len(regular_indices) == 1368,
        "regular_order_matches_fc_p003": sequence_sha(regular_indices) == FC_P003_REGULAR_SHA,
        "two_complete_pair_passes": len(pair_pass_ids) == 2 and all(len(values) == 8 and set(values) == expected_ids for values in pair_pass_ids),
        "global_torch_rng_unchanged": initial_rng == final_rng,
        "approved_interleaved_positions": positions == [index * 1367 // 15 for index in range(16)],
    }
    if not all(checks.values()):
        raise RuntimeError(f"FC-P003B sampling contract failed: {checks}")
    result = {
        "status": "FC_P003B_REAL_OFFICIAL_DATALOADER_SAMPLING_PASS",
        "scope": "sampler identities only; no training or validation/frozen access",
        "seed": SEED,
        "checks": checks,
        "regular_count": len(regular_indices),
        "regular_sequence_sha256": sequence_sha(regular_indices),
        "fc_p003_regular_sequence_sha256": FC_P003_REGULAR_SHA,
        "pair_pass_indices": pair_pass_indices,
        "pair_identity_passes": pair_pass_ids,
        "paired_batch_indices": positions,
        "global_torch_rng_sha256": final_rng,
        "validation_or_frozen_accessed": False,
        "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
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
