"""Train a PhysicsNeMo FNO mean-flow/drag surrogate on published PIV cases.

Splits are by *entire actuation value*, never by snapshots or windows.
This models steady open-loop actuation; it is not a feedback controller.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from physicsnemo.models.fno import FNO


def build_model() -> FNO:
    return FNO(
        in_channels=6,
        out_channels=3,
        dimension=2,
        latent_channels=16,
        num_fno_layers=3,
        num_fno_modes=[10, 10],
        decoder_layers=1,
        decoder_layer_size=32,
        padding=4,
    )


def p_key(value: float) -> float:
    return round(float(value), 4)


def mean_field(velocity: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = valid.astype(np.float32)
    counts = weights.sum(axis=0)
    mean = (velocity * weights).sum(axis=0) / np.maximum(counts, 1)
    return mean.astype(np.float32), (counts[0] > 0).astype(np.float32)


def select_cases(manifest: dict) -> dict[float, dict]:
    grouped: dict[float, list[dict]] = {}
    for entry in manifest["cases"]:
        grouped.setdefault(p_key(entry["p"]), []).append(entry)
    selected = {}
    for p, entries in grouped.items():
        preferred = [item for item in entries if "_zero" not in item["source"].lower()]
        selected[p] = min(preferred or entries, key=lambda item: item["source"])
    return selected


def split_controls(values: list[float]) -> dict[str, list[float]]:
    available = set(values)
    if len(available) < 12:
        raise ValueError("Too few distinct actuation values for held-out control evaluation")
    desired = {"test": [-1.8, 1.8], "val": [-1.0, 1.0]}
    split = {name: [value for value in controls if value in available] for name, controls in desired.items()}
    for name in ("test", "val"):
        if len(split[name]) < 2:
            raise ValueError(f"Required predeclared {name} actuation values absent: {desired[name]}")
    held = set(split["test"] + split["val"])
    split["train"] = sorted(available - held)
    if 0.0 not in split["train"]:
        raise ValueError("Baseline p=0 must remain in training")
    return split


def features(baseline: np.ndarray, xy: np.ndarray, p: float) -> np.ndarray:
    h, w = baseline.shape[-2:]
    control = np.full((1, h, w), p / 3.0, dtype=np.float32)
    control_sq = np.full((1, h, w), (p / 3.0) ** 2, dtype=np.float32)
    return np.concatenate([baseline, xy[:1] / 8.0, xy[1:] / 3.0, control, control_sq], axis=0)


def load_arrays(root: Path, entry: dict) -> dict:
    with np.load(root / entry["processed"]) as item:
        return {key: item[key] for key in item.files}


def build_examples(root: Path, selected: dict[float, dict], split: dict, window: int) -> tuple[dict, dict]:
    baseline_data = load_arrays(root, selected[0.0])
    baseline, _ = mean_field(baseline_data["velocity"], baseline_data["valid"])
    xy = baseline_data["xy"]
    examples: dict[str, dict[str, list]] = {name: {"x": [], "field": [], "valid": [], "cd": [], "p": []} for name in split}
    full_case = {}
    for p in sorted(selected):
        data = load_arrays(root, selected[p])
        if data["velocity"].shape[2:] != baseline.shape[1:]:
            raise ValueError("Preprocessed PIV spatial shapes do not match")
        whole_mean, whole_valid = mean_field(data["velocity"], data["valid"])
        full_case[p] = {
            "mean": whole_mean,
            "valid": whole_valid,
            "cd": float(data["cd"]),
            "cd_uncertainty": float(data["cd_uncertainty"]) if np.isfinite(data["cd_uncertainty"]) else None,
            "source": selected[p]["source"],
        }
        membership = next(name for name, values in split.items() if p in values)
        group = examples[membership]
        count = len(data["velocity"]) // window
        for index in range(count):
            sl = slice(index * window, (index + 1) * window)
            averaged, valid = mean_field(data["velocity"][sl], data["valid"][sl])
            group["x"].append(features(baseline, xy, p))
            group["field"].append(averaged)
            group["valid"].append(valid)
            group["cd"].append(float(data["cd"]))
            group["p"].append(p)
    return examples, full_case


def to_device(examples: dict, device: str) -> dict:
    result = {}
    for name, group in examples.items():
        result[name] = {key: torch.as_tensor(np.stack(value), dtype=torch.float32, device=device) if key not in ("cd", "p") else torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in group.items()}
    return result


def predict(model: FNO, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    raw = model(x)
    velocity = x[:, :2] + raw[:, :2]
    cd = raw[:, 2].mean(dim=(-2, -1)) * 4.0
    return velocity, cd


def losses(model: FNO, group: dict, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    velocity, cd = predict(model, group["x"][ids])
    target = group["field"][ids]
    mask = group["valid"][ids, None]
    field = ((velocity - target).square() * mask).sum() / (2 * mask.sum().clamp_min(1))
    drag = ((cd - group["cd"][ids]) / 4.0).square().mean()
    return field, drag


@torch.no_grad()
def evaluate(model: FNO, group: dict, batch: int) -> dict[str, float]:
    model.eval()
    predicted_fields, predicted_cd = [], []
    for start in range(0, len(group["x"]), batch):
        velocity, cd = predict(model, group["x"][start : start + batch])
        predicted_fields.append(velocity)
        predicted_cd.append(cd)
    fields = torch.cat(predicted_fields)
    cd = torch.cat(predicted_cd)
    mask = group["valid"][:, None]
    mae = ((fields - group["field"]).abs() * mask).sum() / (2 * mask.sum().clamp_min(1))
    persistence = ((group["x"][:, :2] - group["field"]).abs() * mask).sum() / (2 * mask.sum().clamp_min(1))
    drag_mae = (cd - group["cd"]).abs().mean()
    return {"velocity_mae_u_infty": float(mae), "baseline_field_mae_u_infty": float(persistence), "drag_mae": float(drag_mae)}


def casewise_eval(model: FNO, root: Path, selected: dict, split: dict, full_case: dict, device: str) -> dict:
    baseline_data = load_arrays(root, selected[0.0])
    baseline, _ = mean_field(baseline_data["velocity"], baseline_data["valid"])
    xy = baseline_data["xy"]
    train_p = np.asarray(split["train"])
    train_cd = np.asarray([full_case[p]["cd"] for p in train_p])
    report = {}
    model.eval()
    for name in ("train", "val", "test"):
        report[name] = []
        for p in split[name]:
            x = torch.as_tensor(features(baseline, xy, p)[None], dtype=torch.float32, device=device)
            with torch.no_grad():
                velocity, cd = predict(model, x)
            measured = full_case[p]
            mask = measured["valid"].astype(bool)
            field_mae = float(np.abs(velocity[0].cpu().numpy()[:, mask] - measured["mean"][:, mask]).mean())
            baseline_field_mae = float(np.abs(baseline[:, mask] - measured["mean"][:, mask]).mean())
            insertion = int(np.searchsorted(train_p, p))
            lower = max(0, insertion - 1)
            upper = min(len(train_p) - 1, insertion)
            if lower == upper:
                interpolated_field = full_case[float(train_p[lower])]["mean"]
            else:
                left, right = float(train_p[lower]), float(train_p[upper])
                ratio = (p - left) / (right - left)
                interpolated_field = (1 - ratio) * full_case[left]["mean"] + ratio * full_case[right]["mean"]
            interpolated_field_mae = float(np.abs(interpolated_field[:, mask] - measured["mean"][:, mask]).mean())
            drag_pred = float(cd.item())
            report[name].append({
                "p": p,
                "source": measured["source"],
                "field_mae_u_infty": field_mae,
                "field_baseline_persistence_mae_u_infty": baseline_field_mae,
                "field_linear_interpolation_mae_u_infty": interpolated_field_mae,
                "cd_measured": measured["cd"],
                "cd_uncertainty": measured["cd_uncertainty"],
                "cd_predicted": drag_pred,
                "cd_error": drag_pred - measured["cd"],
                "cd_linear_interpolation_baseline": float(np.interp(p, train_p, train_cd)),
            })
    controls = np.linspace(min(train_p), max(train_p), 109, dtype=np.float32)
    arrays = np.stack([features(baseline, xy, float(p)) for p in controls])
    with torch.no_grad():
        guesses = []
        for start in range(0, len(arrays), 16):
            _, cd = predict(model, torch.as_tensor(arrays[start : start + 16], dtype=torch.float32, device=device))
            guesses.extend(cd.cpu().numpy().tolist())
    predicted_optimum = float(controls[int(np.argmin(guesses))])
    measured_optimum = min(full_case, key=lambda p: full_case[p]["cd"])
    nearest_measured = min(full_case, key=lambda p: abs(p - predicted_optimum))
    baseline_cd = full_case[0.0]["cd"]
    report["open_loop_search"] = {
        "predicted_min_cd_p": predicted_optimum,
        "predicted_min_cd": float(min(guesses)),
        "nearest_measured_p": nearest_measured,
        "nearest_measured_p_was_in_training": nearest_measured in split["train"],
        "nearest_measured_cd": full_case[nearest_measured]["cd"],
        "baseline_measured_cd": baseline_cd,
        "measured_drag_reduction_at_nearest_p_fraction": 1 - full_case[nearest_measured]["cd"] / baseline_cd,
        "measured_min_cd_p": measured_optimum,
        "measured_min_cd": full_case[measured_optimum]["cd"],
        "measured_cd_regret_at_nearest_p": full_case[nearest_measured]["cd"] - full_case[measured_optimum]["cd"],
        "domain": [float(min(train_p)), float(max(train_p))],
        "note": "Steady open-loop parameter search, not closed-loop or CFD-in-the-loop validation",
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/piv"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/public_piv_fno.pt"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--window", type=int, default=20, help="Retained frames per non-overlapping mean window")
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    selected = select_cases(manifest)
    split = split_controls(sorted(selected))
    examples, full_case = build_examples(args.data, selected, split, args.window)
    data = to_device(examples, device)
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=5e-5)
    best = float("inf")
    best_epoch = 0
    history = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(len(data["train"]["x"]), device=device)
        train_losses = []
        for ids in permutation.split(args.batch):
            field, drag = losses(model, data["train"], ids)
            loss = field + 0.3 * drag
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach()))
        scheduler.step()
        validation = evaluate(model, data["val"], args.batch)
        score = validation["velocity_mae_u_infty"] + 0.1 * validation["drag_mae"]
        history.append({"epoch": epoch, "training_loss": float(np.mean(train_losses)), "validation": validation, "score": score})
        if score < best:
            best, best_epoch = score, epoch
            torch.save({"model": model.state_dict(), "split": split, "epoch": epoch, "source": manifest["source"], "seed": args.seed, "model_type": "PhysicsNeMo FNO"}, args.out)
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch {epoch}: loss={np.mean(train_losses):.5f} val_u_mae={validation['velocity_mae_u_infty']:.5f} val_cd_mae={validation['drag_mae']:.5f}", flush=True)
    checkpoint = torch.load(args.out, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    metrics = {"source": manifest["source"], "split": split, "best_epoch": best_epoch, "best_validation_score": best, "window_frames": args.window, "history": history, "casewise": casewise_eval(model, args.data, selected, split, full_case, device)}
    args.out.with_suffix(".json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics["casewise"], indent=2), flush=True)


if __name__ == "__main__":
    main()
