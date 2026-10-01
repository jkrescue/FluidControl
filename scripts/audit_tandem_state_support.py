#!/usr/bin/env python3
"""Measure normalized-state support directly from curated OpenFOAM trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-frames", type=int, default=16)
    parser.add_argument(
        "--write-normalization-support",
        action="store_true",
        help="atomically add exact train-support metadata to normalization.json",
    )
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if not output.is_relative_to(project / "artifacts"):
        parser.error("output must be under project artifacts")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    if args.chunk_frames < 1:
        parser.error("chunk-frames must be positive")

    data = args.data.resolve()
    stats = json.loads((data / "normalization.json").read_text(encoding="utf-8"))
    mean = np.asarray(stats["state_mean"], dtype=np.float64)[None, :, None, None]
    std = np.asarray(stats["state_std"], dtype=np.float64)[None, :, None, None]
    splits: dict[str, dict] = {}
    global_max = {"value": -np.inf}
    for split in ("train", "validation", "test"):
        cases = []
        split_frame_bounds: list[np.ndarray] = []
        for path in sorted((data / split).glob("*.h5")):
            frame_bounds = []
            channel_max = np.zeros(3, dtype=np.float64)
            with h5py.File(path, "r") as handle:
                frames = len(handle["state"])
                for start in range(0, frames, args.chunk_frames):
                    stop = min(start + args.chunk_frames, frames)
                    state = np.asarray(handle["state"][start:stop], dtype=np.float64)
                    mask = np.asarray(handle["mask"][start:stop], dtype=np.float64)
                    normalized = np.abs((state - mean) / std * mask)
                    bounds = normalized.max(axis=(1, 2, 3))
                    channel_max = np.maximum(channel_max, normalized.max(axis=(0, 2, 3)))
                    local = int(np.argmax(bounds))
                    if float(bounds[local]) > global_max["value"]:
                        channel = int(np.unravel_index(np.argmax(normalized[local]), normalized[local].shape)[0])
                        global_max = {
                            "value": float(bounds[local]),
                            "split": split,
                            "case": path.stem,
                            "frame": start + local,
                            "channel": stats["state_channels"][channel],
                        }
                    frame_bounds.append(bounds)
            joined = np.concatenate(frame_bounds)
            split_frame_bounds.append(joined)
            cases.append({
                "case": path.stem,
                "frames": int(len(joined)),
                "maximum": float(joined.max()),
                "p99": float(np.quantile(joined, 0.99)),
                "channel_maximum_u_v_p": channel_max.tolist(),
            })
        all_bounds = np.concatenate(split_frame_bounds)
        split_channel_max = np.max(
            np.asarray([case["channel_maximum_u_v_p"] for case in cases]), axis=0
        )
        splits[split] = {
            "cases": cases,
            "frames": int(len(all_bounds)),
            "maximum": float(all_bounds.max()),
            "p99": float(np.quantile(all_bounds, 0.99)),
            "p999": float(np.quantile(all_bounds, 0.999)),
            "frames_above_20": int(np.count_nonzero(all_bounds > 20.0)),
            "fraction_above_20": float(np.mean(all_bounds > 20.0)),
            "channel_maximum_u_v_p": split_channel_max.tolist(),
        }
    train_max = splits["train"]["maximum"]
    report = {
        "status": "REAL_CFD_NORMALIZED_STATE_SUPPORT_AUDITED",
        "data_root": str(data),
        "normalization_source": "train split normalization.json",
        "splits": splits,
        "global_maximum": global_max,
        "recommended_guard": {
            "basis": "1.25_times_exact_train_split_maximum",
            "value": 1.25 * train_max,
            "heldout_maximum_covered": max(
                splits["validation"]["maximum"], splits["test"]["maximum"]
            ) <= 1.25 * train_max,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.write_normalization_support:
        stats["state_abs_normalized_channel_max_train"] = splits["train"][
            "channel_maximum_u_v_p"
        ]
        stats["state_abs_normalized_max_train"] = train_max
        stats["state_support_computed_from"] = "exact train split valid cells"
        target = data / "normalization.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        temporary.replace(target)
    print(json.dumps({
        "output": str(output),
        "split_maxima": {key: value["maximum"] for key, value in splits.items()},
        "global_maximum": global_max,
        "recommended_guard": report["recommended_guard"],
    }, indent=2))


if __name__ == "__main__":
    main()
