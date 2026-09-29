"""Generate action-conditioned CFD trajectories and averaged flow fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fluid_control.cfd import CFDConfig, CylinderCFD


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("artifacts/cfd_dataset.npz"))
    parser.add_argument("--count", type=int, default=48)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3600)
    parser.add_argument("--average-start", type=int, default=2000)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    cfg = CFDConfig(nx=args.nx, ny=args.ny, dt=args.dt, steps=args.steps, average_start=args.average_start)
    solver = CylinderCFD(cfg)
    rng = np.random.default_rng(args.seed)
    anchors = np.array([[0, 0], [1, -1], [-1, 1], [1, 1], [-1, -1], [2, -2], [-2, 2], [2, 2], [-2, -2]], dtype=np.float32)
    if args.count < len(anchors):
        raise ValueError(f"count must be at least {len(anchors)}")
    actions = np.concatenate([anchors, rng.uniform(-2, 2, size=(args.count - len(anchors), 2)).astype(np.float32)])
    results = []
    for start in range(0, len(actions), args.batch):
        print(f"CFD cases {start + 1}-{min(start + args.batch, len(actions))}/{len(actions)}", flush=True)
        results.append(solver.run(actions[start : start + args.batch], progress=True))
    merged = {key: np.concatenate([r[key] for r in results]) for key in results[0]}
    fields = torch.from_numpy(merged.pop("fields"))
    merged["fields"] = F.interpolate(fields, size=(64, 128), mode="area").numpy().astype(np.float32)
    merged["features"] = solver.feature_maps(torch.from_numpy(actions).to(solver.device), out_size=(64, 128)).cpu().numpy().astype(np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **merged)
    args.out.with_suffix(".json").write_text(json.dumps({"solver": "in_house_fourier_brinkman_2d", "status": "developmental_unvalidated", "config": cfg.metadata(), "seed": args.seed, "count": args.count, "output_grid": [128, 64]}, indent=2), encoding="utf-8")
    print(f"saved {args.out}; baseline objective={merged['objective'][0]:.6f}; best sampled={merged['objective'].min():.6f}", flush=True)


if __name__ == "__main__":
    main()
