"""Choose controls in the PhysicsNeMo surrogate and verify in CFD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from fluid_control.cfd import CFDConfig, CylinderCFD
from train_surrogate import make_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/cfd_dataset.npz"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/fno_surrogate.pt"))
    parser.add_argument("--out", type=Path, default=Path("results/control_comparison.json"))
    parser.add_argument("--resolution", type=int, default=31)
    parser.add_argument("--allow-development-data", action="store_true", help="Explicitly permit exploratory use of unvalidated CFD")
    args = parser.parse_args()
    torch.set_num_threads(4)
    data = np.load(args.dataset)
    metadata = json.loads(args.dataset.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata.get("status") != "validated" and not args.allow_development_data:
        raise SystemExit("CFD data are not validated; this control result would be exploratory only")
    cfg = CFDConfig(**metadata["config"])
    cfd = CylinderCFD(cfg)
    checkpoint = torch.load(args.model, map_location="cuda", weights_only=False)
    model = make_model().to("cuda")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    q = np.linspace(-2, 2, args.resolution, dtype=np.float32)
    action_grid = np.stack(np.meshgrid(q, q, indexing="ij"), -1).reshape(-1, 2)
    scores = []
    ny, nx = 64, 128
    x = (np.arange(nx) * cfg.lx / nx)[None, :]
    y = ((np.arange(ny) + 0.5) * cfg.ly / ny - cfg.ly / 2)[:, None]
    wake = torch.from_numpy((x >= cfg.main_x + 1.5) & (x <= cfg.main_x + 5.0) & (np.abs(y) <= 1.5)).to("cuda")
    with torch.no_grad():
        for start in range(0, len(action_grid), 32):
            actions = torch.from_numpy(action_grid[start : start + 32]).to("cuda")
            features = cfd.feature_maps(actions, out_size=(64, 128))
            output = model(features)
            energy = output[:, 2, wake] / 4.0
            score = energy.mean(dim=1) + 0.01 * actions.square().sum(dim=1)
            scores.append(score.cpu().numpy())
    scores = np.concatenate(scores)
    predicted_id = int(scores.argmin())
    selected = action_grid[predicted_id]
    baseline = np.array([0.0, 0.0], dtype=np.float32)
    sampled_best_id = int(data["objective"].argmin())
    sampled_best = data["actions"][sampled_best_id]
    check_actions = np.stack([baseline, sampled_best, selected])
    verified = cfd.run(check_actions, progress=True)
    labels = ["uncontrolled", "best_training_action", "physicsnemo_selected"]
    summary = {
        "config": cfg.metadata(),
        "surrogate_checkpoint_epoch": checkpoint["epoch"],
        "training_sample_count": len(data["actions"]),
        "validation_sample_count": len(checkpoint["val_ids"]),
        "predicted_candidate_action": selected.tolist(),
        "predicted_candidate_objective": float(scores[predicted_id]),
        "cases": [
            {"label": label, "action": verified["actions"][i].tolist(), "cfd_wake_error": float(verified["wake_error"][i]), "control_cost": float(verified["control_cost"][i]), "cfd_objective": float(verified["objective"][i])}
            for i, label in enumerate(labels)
        ],
    }
    summary["ai_vs_uncontrolled_fraction"] = float(1 - summary["cases"][2]["cfd_objective"] / summary["cases"][0]["cfd_objective"])
    summary["ai_vs_best_training_fraction"] = float(1 - summary["cases"][2]["cfd_objective"] / summary["cases"][1]["cfd_objective"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    for index, (ax, label) in enumerate(zip(axes, labels)):
        image = ax.imshow(verified["fields"][index, 0], origin="lower", extent=[0, cfg.lx, -cfg.ly / 2, cfg.ly / 2], vmin=0, vmax=1.3, cmap="viridis")
        ax.set_title(f"{label}: q={check_actions[index].round(2)}, J={verified['objective'][index]:.4f}")
        ax.set_xlim(1.5, 9)
        ax.set_ylabel("y/D")
    axes[-1].set_xlabel("x/D")
    fig.colorbar(image, ax=axes, label="mean streamwise velocity / U")
    fig.savefig(args.out.with_suffix(".png"), dpi=180)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
