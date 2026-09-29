"""Train a PhysicsNeMo FNO to forecast measured PIV fields at fixed actuation.

This uses real time-resolved PIV, not synthetic CFD trajectories. Whole
actuation cases are held out. The first baseline is field persistence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from physicsnemo.models.fno import FNO

from train_public_piv import load_arrays, mean_field, select_cases, split_controls


def model_factory() -> FNO:
    return FNO(
        in_channels=6,
        out_channels=2,
        dimension=2,
        latent_channels=16,
        num_fno_layers=3,
        num_fno_modes=[10, 10],
        decoder_layers=1,
        decoder_layer_size=32,
        padding=4,
    )


def load_pairs(root: Path, selected: dict, split: dict, horizon: int, pair_step: int, device: str) -> dict:
    values = {name: {"x": [], "y": [], "valid": [], "p": []} for name in split}
    for p in sorted(selected):
        data = load_arrays(root, selected[p])
        velocity = data["velocity"]
        valid = data["valid"]
        xy = data["xy"]
        h, w = xy.shape[1:]
        static = np.stack([xy[0] / 8.0, xy[1] / 3.0, np.full((h, w), p / 3.0, dtype=np.float32)])
        membership = next(name for name, controls in split.items() if p in controls)
        group = values[membership]
        for t in range(0, len(velocity) - horizon, pair_step):
            x = np.concatenate([velocity[t], valid[t].astype(np.float32), static], axis=0)
            group["x"].append(x)
            group["y"].append(velocity[t + horizon])
            group["valid"].append(valid[t + horizon].astype(np.float32))
            group["p"].append(p)
    return {
        name: {
            key: torch.as_tensor(np.stack(items), dtype=torch.float32, device=device)
            for key, items in group.items()
        }
        for name, group in values.items()
    }


def prediction(model: FNO, x: torch.Tensor) -> torch.Tensor:
    return x[:, :2] + model(x)


def masked_mse(yhat: torch.Tensor, y: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return ((yhat - y).square() * valid).sum() / (2 * valid.sum().clamp_min(1))


@torch.no_grad()
def evaluate(model: FNO, group: dict, batch: int) -> dict:
    model.eval()
    errors, baseline_errors, total = 0.0, 0.0, 0.0
    by_p: dict[float, dict[str, float]] = {}
    for start in range(0, len(group["x"]), batch):
        x = group["x"][start : start + batch]
        y = group["y"][start : start + batch]
        valid = group["valid"][start : start + batch]
        p = group["p"][start : start + batch]
        pred = prediction(model, x)
        sample_error = ((pred - y).abs() * valid).sum(dim=(1, 2, 3)) / (2 * valid.sum(dim=(1, 2, 3)).clamp_min(1))
        sample_baseline = ((x[:, :2] - y).abs() * valid).sum(dim=(1, 2, 3)) / (2 * valid.sum(dim=(1, 2, 3)).clamp_min(1))
        errors += float(sample_error.sum())
        baseline_errors += float(sample_baseline.sum())
        total += len(x)
        for value in torch.unique(p):
            key = float(value)
            mask = p == value
            entry = by_p.setdefault(key, {"samples": 0, "error_sum": 0.0, "baseline_sum": 0.0})
            entry["samples"] += int(mask.sum())
            entry["error_sum"] += float(sample_error[mask].sum())
            entry["baseline_sum"] += float(sample_baseline[mask].sum())
    return {
        "mae_u_infty": errors / total,
        "persistence_mae_u_infty": baseline_errors / total,
        "by_p": {str(round(p, 4)): {"samples": item["samples"], "mae_u_infty": item["error_sum"] / item["samples"], "persistence_mae_u_infty": item["baseline_sum"] / item["samples"]} for p, item in sorted(by_p.items())},
    }


@torch.no_grad()
def conditional_mean_baseline(root: Path, selected: dict, split: dict, group: dict, device: str) -> dict:
    means = {}
    for p in split["train"]:
        case = load_arrays(root, selected[p])
        means[p], _ = mean_field(case["velocity"], case["valid"])
    train_p = np.asarray(split["train"])
    by_p = {}
    for p in sorted({float(value) for value in group["p"].cpu().numpy()}):
        loc = int(np.searchsorted(train_p, p))
        left_index, right_index = max(0, loc - 1), min(len(train_p) - 1, loc)
        left, right = float(train_p[left_index]), float(train_p[right_index])
        if left == right:
            field = means[left]
        else:
            ratio = (p - left) / (right - left)
            field = (1 - ratio) * means[left] + ratio * means[right]
        baseline = torch.as_tensor(field, dtype=torch.float32, device=device)
        ids = torch.isclose(group["p"], torch.tensor(p, device=device))
        target = group["y"][ids]
        valid = group["valid"][ids]
        mae = ((target - baseline).abs() * valid).sum() / (2 * valid.sum().clamp_min(1))
        by_p[str(round(p, 4))] = {"samples": int(ids.sum()), "conditional_mean_interpolation_mae_u_infty": float(mae)}
    total = sum(item["samples"] for item in by_p.values())
    aggregate = sum(item["samples"] * item["conditional_mean_interpolation_mae_u_infty"] for item in by_p.values()) / total
    return {"mae_u_infty": aggregate, "by_p": by_p}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/piv"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/public_piv_dynamics_fno.pt"))
    parser.add_argument("--horizon", type=int, default=2, help="Forecast retained frames ahead, each separated by 0.05 s")
    parser.add_argument("--pair-step", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    selected = select_cases(manifest)
    split = split_controls(sorted(selected))
    data = load_pairs(args.data, selected, split, args.horizon, args.pair_step, device)
    model = model_factory().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=5e-5)
    best, best_epoch = float("inf"), 0
    history = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        indices = torch.randperm(len(data["train"]["x"]), device=device)
        losses = []
        for batch_ids in indices.split(args.batch):
            group = data["train"]
            pred = prediction(model, group["x"][batch_ids])
            loss = masked_mse(pred, group["y"][batch_ids], group["valid"][batch_ids])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        validation = evaluate(model, data["val"], args.batch)
        history.append({"epoch": epoch, "training_mse": float(np.mean(losses)), "validation_mae": validation["mae_u_infty"]})
        if validation["mae_u_infty"] < best:
            best, best_epoch = validation["mae_u_infty"], epoch
            torch.save({"model": model.state_dict(), "split": split, "horizon": args.horizon, "pair_step": args.pair_step, "epoch": epoch, "source": manifest["source"], "model_type": "PhysicsNeMo FNO"}, args.out)
        if epoch == 1 or epoch % 5 == 0:
            print(f"epoch {epoch}: train_mse={np.mean(losses):.6f} val_mae={validation['mae_u_infty']:.5f} persistence={validation['persistence_mae_u_infty']:.5f}", flush=True)
    model.load_state_dict(torch.load(args.out, map_location=device, weights_only=False)["model"])
    metrics = {"source": manifest["source"], "split": split, "best_epoch": best_epoch, "horizon_s": args.horizon / manifest["retained_hz"], "history": history, "validation": evaluate(model, data["val"], args.batch), "test": evaluate(model, data["test"], args.batch), "validation_conditional_mean_baseline": conditional_mean_baseline(args.data, selected, split, data["val"], device), "test_conditional_mean_baseline": conditional_mean_baseline(args.data, selected, split, data["test"], device), "note": "One-step PIV forecast under fixed actuation; not a closed-loop controller"}
    args.out.with_suffix(".json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"validation": metrics["validation"], "test": metrics["test"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
