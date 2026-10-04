#!/usr/bin/env python3
"""CPU-only real-loader preflight for the bounded FC-P003C calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    from fluid_control.augmented_datapipe import compose_training_data
    from fluid_control.dynamic_pair_stat_datapipe import DynamicMatchedPairStatDataset
    from fluid_control.tandem_datapipe import TandemRolloutDataset
    from omegaconf import OmegaConf
    from physicsnemo.datapipes import DataLoader
    from train_tandem_fno import configured_force_indices

    cfg = OmegaConf.load(args.config)
    force_indices = configured_force_indices(cfg)
    base = TandemRolloutDataset(
        cfg.data.root, "train", 100, stride=int(cfg.training.train_stride),
        num_workers=1, force_indices=force_indices,
    )
    train = pairs = None
    try:
        train, sources = compose_training_data(
            base, [Path(value) for value in cfg.data.additional_train_roots],
            rollout_steps=100, stride=int(cfg.training.additional_train_stride),
            workers=1, force_indices=force_indices,
        )
        regular = DataLoader(
            train, batch_size=1, shuffle=True, collate_metadata=True,
            prefetch_factor=0, use_streams=False, seed=int(cfg.training.seed),
        )
        regular_batch, regular_metadata = next(iter(regular))
        pairs = DynamicMatchedPairStatDataset(
            cfg.data.paired_action_root, cfg.data.paired_zero_root,
            cfg.data.paired_manifest, num_workers=1,
        )
        pair_loader = DataLoader(
            pairs, batch_size=1, shuffle=True, collate_metadata=True,
            prefetch_factor=0, use_streams=False, seed=int(cfg.training.seed),
        )
        pair_batch, pair_metadata = next(iter(pair_loader))
        regular_shapes = {key: list(value.shape) for key, value in regular_batch.items()}
        pair_shapes = {key: list(value.shape) for key, value in pair_batch.items()}
        expected_regular = {
            "state": [1, 3, 128, 256],
            "target_state": [1, 100, 3, 128, 256],
            "omega": [1, 101, 1], "target_force": [1, 100, 4],
            "mask": [1, 1, 128, 256],
        }
        expected_pair = {
            "action_state": [1, 101, 3, 128, 256],
            "zero_state": [1, 101, 3, 128, 256],
            "action_omega": [1, 101, 1], "zero_omega": [1, 101, 1],
            "action_force": [1, 101, 4], "zero_force": [1, 101, 4],
            "mask": [1, 1, 128, 256],
        }
        if len(regular) != 1368 or len(pairs) != 8:
            raise RuntimeError("real loader lengths differ")
        if any(regular_shapes.get(key) != value for key, value in expected_regular.items()):
            raise RuntimeError(f"real regular-batch shape differs: {regular_shapes}")
        if any(pair_shapes.get(key) != value for key, value in expected_pair.items()):
            raise RuntimeError("real paired-batch shape differs")
        pair_id = pair_metadata[0].get("pair_id")
        if not isinstance(pair_id, str) or pair_id.count(":") != 1:
            raise RuntimeError("real paired metadata identity differs")
        regular_row = regular_metadata[0]
        if regular_row.get("split") != "train":
            raise RuntimeError("real regular metadata is not train-only")
        print(json.dumps({
            "status": "FCP003C_TRAIN_FIT_CALIBRATION_CPU_PREFLIGHT_PASS",
            "regular_loader_length": len(regular), "pair_dataset_length": len(pairs),
            "regular_metadata": regular_row, "pair_id": pair_id,
            "regular_shapes": regular_shapes, "pair_shapes": pair_shapes,
            "training_sources": sources, "validation_accessed": False,
            "frozen_test_accessed": False, "gpu_used": False,
        }, allow_nan=False, sort_keys=True))
    finally:
        if train is not None:
            train.close()
        if pairs is not None:
            pairs.close()


if __name__ == "__main__":
    main()
