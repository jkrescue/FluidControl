"""Estimate drag from observed experimental PIV flow, using PhysicsNeMo FNO.

This is an observation-to-force model, *not* a counterfactual controller: a PIV
field at a proposed actuation must already exist before the model can use it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from physicsnemo.models.fno import FNO

from train_public_piv import load_arrays, mean_field, select_cases, split_controls


def build_model() -> FNO:
    return FNO(
        in_channels=4,
        out_channels=1,
        dimension=2,
        latent_channels=12,
        num_fno_layers=2,
        num_fno_modes=[8, 8],
        decoder_layers=1,
        decoder_layer_size=24,
        padding=4,
    )


def input_field(mean: np.ndarray, xy: np.ndarray) -> np.ndarray:
    return np.concatenate([mean, xy[:1] / 8.0, xy[1:] / 3.0], axis=0)


def load_data(root: Path, selected: dict, split: dict, window: int, device: str) -> tuple[dict, dict]:
    groups = {name: {"x": [], "cd": []} for name in split}
    whole = {}
    for p in sorted(selected):
        case = load_arrays(root, selected[p])
        mean, _ = mean_field(case["velocity"], case["valid"])
        whole[p] = {"x": input_field(mean, case["xy"]), "cd": float(case["cd"])}
        membership = next(name for name, values in split.items() if p in values)
        for start in range(0, len(case["velocity"]) - window + 1, window):
            averaged, _ = mean_field(case["velocity"][start : start + window], case["valid"][start : start + window])
            groups[membership]["x"].append(input_field(averaged, case["xy"]))
            groups[membership]["cd"].append(float(case["cd"]))
    data = {name: {key: torch.as_tensor(np.stack(value), dtype=torch.float32, device=device) for key, value in group.items()} for name, group in groups.items()}
    return data, whole


def predict(model: FNO, x: torch.Tensor) -> torch.Tensor:
    return model(x).mean(dim=(1, 2, 3)) * 4.0


@torch.no_grad()
def evaluate(model: FNO, group: dict, batch: int) -> float:
    model.eval()
    errors = []
    for start in range(0, len(group["x"]), batch):
        p = predict(model, group["x"][start : start + batch])
        errors.append((p - group["cd"][start : start + batch]).abs())
    return float(torch.cat(errors).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/piv"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/public_piv_force_observer.pt"))
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    selected = select_cases(manifest)
    split = split_controls(sorted(selected))
    data, whole = load_data(args.data, selected, split, args.window, device)
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=5e-5)
    best, best_epoch = float("inf"), 0
    history = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        ids = torch.randperm(len(data["train"]["x"]), device=device)
        train_errors = []
        for batch_ids in ids.split(args.batch):
            group = data["train"]
            pred = predict(model, group["x"][batch_ids])
            loss = ((pred - group["cd"][batch_ids]) / 4.0).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_errors.append(float(loss.detach()))
        scheduler.step()
        validation = evaluate(model, data["val"], args.batch)
        history.append({"epoch": epoch, "training_mse": float(np.mean(train_errors)), "validation_cd_mae": validation})
        if validation < best:
            best, best_epoch = validation, epoch
            torch.save({"model": model.state_dict(), "split": split, "epoch": epoch, "source": manifest["source"], "model_type": "PhysicsNeMo FNO observation-to-force"}, args.out)
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch {epoch}: train_mse={np.mean(train_errors):.5f} val_cd_mae={validation:.5f}", flush=True)
    model.load_state_dict(torch.load(args.out, map_location=device, weights_only=False)["model"])
    model.eval()
    casewise = {}
    train_p = np.asarray(split["train"])
    train_cd = np.asarray([whole[p]["cd"] for p in train_p])
    for name, controls in split.items():
        casewise[name] = []
        for p in controls:
            x = torch.as_tensor(whole[p]["x"][None], dtype=torch.float32, device=device)
            with torch.no_grad():
                estimate = float(predict(model, x).item())
            measured = whole[p]["cd"]
            casewise[name].append({"p": p, "cd_measured": measured, "cd_predicted": estimate, "absolute_error": abs(estimate - measured), "linear_p_interpolation_baseline": float(np.interp(p, train_p, train_cd))})
    report = {"source": manifest["source"], "split": split, "best_epoch": best_epoch, "history": history, "casewise": casewise, "note": "Uses already observed PIV field; not an action-to-flow or action-to-drag counterfactual model"}
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"validation": casewise["val"], "test": casewise["test"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
