"""Train an action-conditioned PhysicsNeMo FNO on CFD mean and energy fields."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from physicsnemo.models.fno import FNO


def make_model() -> FNO:
    return FNO(
        in_channels=6,
        out_channels=3,
        dimension=2,
        latent_channels=16,
        num_fno_layers=4,
        num_fno_modes=[12, 12],
        decoder_layers=1,
        decoder_layer_size=32,
        padding=8,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/cfd_dataset.npz"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/fno_surrogate.pt"))
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--allow-development-data", action="store_true", help="Explicitly permit exploratory training on unvalidated CFD")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    dataset = np.load(args.dataset)
    x = torch.from_numpy(dataset["features"]).to("cuda")
    y = torch.from_numpy(dataset["fields"]).to("cuda")
    y[:, 0] -= 1.0
    y *= 4.0
    n = len(x)
    # Keep baseline and open-loop anchor actions in train; reserve random controls.
    random_ids = np.arange(9, n)
    np.random.default_rng(args.seed + 1).shuffle(random_ids)
    validation_count = max(4, round(0.2 * len(random_ids)))
    val_ids = random_ids[:validation_count]
    train_ids = np.concatenate([np.arange(9), random_ids[validation_count:]])
    if not len(train_ids) or not len(val_ids):
        raise ValueError("Not enough CFD cases for train/validation split")

    h, w = y.shape[-2:]
    metadata = json.loads(args.dataset.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata.get("status") != "validated" and not args.allow_development_data:
        raise SystemExit("CFD data are not validated; use public PIV or explicitly pass --allow-development-data for exploratory work")
    cfg = metadata["config"]
    xx = torch.arange(w, device="cuda") * (cfg["lx"] / w)
    yy = (torch.arange(h, device="cuda") + 0.5) * (cfg["ly"] / h) - cfg["ly"] / 2
    wake = ((xx[None] >= cfg["main_x"] + 1.5) & (xx[None] <= cfg["main_x"] + 5.0) & (yy[:, None].abs() <= 1.5))
    solid = x[0, :3].sum(dim=0).clamp(0, 1)
    weight = (1.0 - 0.8 * solid + 4.0 * wake.float())[None, None]

    model = make_model().to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-4)
    best_val = float("inf")
    history = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        shuffled = np.random.permutation(train_ids)
        train_losses = []
        for start in range(0, len(shuffled), args.batch):
            ids = shuffled[start : start + args.batch]
            prediction = model(x[ids])
            loss = ((prediction - y[ids]).square() * weight).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()
        model.eval()
        with torch.no_grad():
            prediction = model(x[val_ids])
            val_loss = (((prediction - y[val_ids]).square()) * weight).mean().item()
            predicted_score = (prediction[:, 2, wake] / 4).mean(dim=1)
            score_mae = (predicted_score - torch.from_numpy(dataset["wake_error"][val_ids]).to("cuda")).abs().mean().item()
        train_loss = float(np.mean(train_losses))
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "wake_score_mae": score_mae})
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "train_ids": train_ids, "val_ids": val_ids, "dataset": str(args.dataset), "epoch": epoch + 1}, args.out)
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"epoch {epoch + 1:03d}: train={train_loss:.6f} val={val_loss:.6f} wake_MAE={score_mae:.6f}", flush=True)

    args.out.with_suffix(".json").write_text(json.dumps({"best_validation_loss": best_val, "history": history, "train_ids": train_ids.tolist(), "val_ids": val_ids.tolist()}, indent=2), encoding="utf-8")
    print(f"saved {args.out}; best validation loss={best_val:.6f}", flush=True)


if __name__ == "__main__":
    main()
