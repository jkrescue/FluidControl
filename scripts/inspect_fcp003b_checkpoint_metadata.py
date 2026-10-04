#!/usr/bin/env python3
"""Extract and validate FC-P003B checkpoint metadata in the official torch image."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import torch


IDENTITIES = {
    f"b{phase:02d}:{profile}"
    for phase in (0, 2, 4, 6)
    for profile in ("multisine", "prbs")
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect(path: Path, mode: str) -> dict:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict) or not isinstance(value.get("metadata"), dict):
        raise ValueError("checkpoint metadata missing")
    metadata = value["metadata"]
    repetitions = 1 if mode == "--probe" else 2
    expected_indices = list(range(8)) if mode == "--probe" else [i * 1367 // 15 for i in range(16)]
    passes = metadata.get("paired_identity_passes")
    checks = {
        "paired_dataset_kind": metadata.get("paired_dataset_kind") == "dynamic8",
        "paired_dataset_repetitions": metadata.get("paired_dataset_repetitions") == repetitions,
        "paired_batches_per_epoch": metadata.get("paired_batches_per_epoch") == 8 * repetitions,
        "paired_batch_schedule": metadata.get("paired_batch_schedule") == "interleaved",
        "paired_batch_indices": metadata.get("paired_batch_indices") == expected_indices,
        "paired_identity_passes": isinstance(passes, list)
        and len(passes) == repetitions
        and all(len(row) == 8 and set(row) == IDENTITIES for row in passes),
        "paired_manifest": metadata.get("paired_manifest") == "/workspace/dynamic_pair_manifest.json",
        "paired_stat_loss_weight": metadata.get("paired_stat_loss_weight") == 10.0,
        "paired_stat_horizons": metadata.get("paired_stat_horizons") == [20, 50, 100],
        "force_channel_weights": isinstance(metadata.get("force_channel_weights"), list)
        and len(metadata["force_channel_weights"]) == 4
        and all(
            math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-7)
            for actual, expected in zip(
                metadata["force_channel_weights"], [1 / 7, 1 / 7, 4 / 7, 1 / 7], strict=True
            )
        ),
        "rollout_steps": metadata.get("rollout_steps") == 100,
        "validation_rollout_steps": metadata.get("validation_rollout_steps") == 100,
    }
    if not all(checks.values()):
        raise ValueError(f"checkpoint metadata contract differs: {checks}")
    return {
        "status": "FC_P003B_CHECKPOINT_METADATA_PASS",
        "mode": mode,
        "checkpoint_state_sha256": sha256(path),
        "paired_dataset_kind": metadata["paired_dataset_kind"],
        "paired_dataset_repetitions": repetitions,
        "paired_batch_indices": expected_indices,
        "paired_identity_passes": passes,
        "checks": checks,
    }


def write_exclusive(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=("--probe", "--execute"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.checkpoint, args.mode)
    write_exclusive(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
