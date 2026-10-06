#!/usr/bin/env python3
"""Render fixed saved H1/H5 CFD/prediction/error fields without model inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULT_SHA256 = "247af0405d9e622f0b3b3b5dbc64e46c20890439fcd5216682b8d973957fd00d"
INPUTS = {
    "0000": "a49ac2b99230cb92505dcf96331279d22de36af0f3dcef11962d9b81b89de5a1",
    "0700": "dc98eb84b41b9646fd5ce7801151c484d2a80b11096f3534e83d92f02fd9321d",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checked_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    expected = {
        "initial_state": (3, 128, 256), "truth_states": (5, 3, 128, 256),
        "predicted_states": (5, 3, 128, 256), "mask": (1, 128, 256),
        "x": (256,), "y": (128,), "time": (6, 1), "omega": (6, 1),
        "truth_forces": (6, 4), "predicted_forces": (5, 4),
    }
    if set(arrays) != set(expected):
        raise ValueError("unexpected saved-field keys")
    for name, shape in expected.items():
        if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
            raise ValueError(f"invalid saved array {name}")
    if arrays["mask"].dtype != np.uint8 or not set(np.unique(arrays["mask"])).issubset({0, 1}):
        raise ValueError("invalid saved mask")
    if not (np.all(np.diff(arrays["x"]) > 0) and np.all(np.diff(arrays["y"]) > 0)):
        raise ValueError("invalid saved coordinates")
    return arrays


def _quantity(state: np.ndarray, kind: str) -> np.ndarray:
    if kind == "speed":
        return np.sqrt(state[0].astype(np.float64) ** 2 + state[1].astype(np.float64) ** 2)
    if kind == "pressure":
        return state[2].astype(np.float64)
    raise ValueError(kind)


def render_one(source: Path, destination: Path, start: str) -> dict:
    arrays = _checked_arrays(source)
    mask = arrays["mask"][0].astype(bool)
    extent = [float(arrays["x"][0]), float(arrays["x"][-1]),
              float(arrays["y"][0]), float(arrays["y"][-1])]
    fig, axes = plt.subplots(4, 3, figsize=(15, 11.5), constrained_layout=True)
    row = 0
    for lead in (1, 5):
        truth = arrays["truth_states"][lead - 1]
        pred = arrays["predicted_states"][lead - 1]
        for kind, label in (("speed", "|U| (solver units)"),
                            ("pressure", "p′ (ROI mean removed; solver units)")):
            actual = np.where(mask, _quantity(truth, kind), np.nan)
            predicted = np.where(mask, _quantity(pred, kind), np.nan)
            error = np.abs(predicted - actual)
            vmin = float(np.nanmin([np.nanmin(actual), np.nanmin(predicted)]))
            vmax = float(np.nanmax([np.nanmax(actual), np.nanmax(predicted)]))
            if not vmax > vmin:
                raise ValueError("degenerate field range")
            for column, (field, title) in enumerate(((actual, "Saved real CFD"),
                                                     (predicted, "Frozen K1 prediction"),
                                                     (error, "Absolute error"))):
                ax = axes[row, column]
                image = ax.imshow(field, origin="lower", extent=extent, aspect="equal",
                                  cmap="magma" if kind == "speed" else "coolwarm",
                                  vmin=0 if column == 2 else vmin,
                                  vmax=float(np.nanmax(error)) if column == 2 else vmax)
                ax.set_title(f"H{lead} {title}")
                ax.set_xlabel("x (solver coordinates)")
                ax.set_ylabel("y (solver coordinates)")
                error_label = "|ΔU| (solver units)" if kind == "speed" else "|Δp′| (solver units)"
                fig.colorbar(image, ax=ax, shrink=.78,
                             label=label if column < 2 else error_label)
            row += 1
    t0 = float(arrays["time"][0, 0])
    fig.suptitle(
        f"Projected-policy saved replay, controlled branch start {start} (t={t0:.1f})\n"
        "Retrospective realized-action forecast; saved arrays only, no model rerun",
        fontsize=13,
    )
    fig.savefig(destination, dpi=135, metadata={"Software": "matplotlib", "Creation Time": None})
    plt.close(fig)
    return {
        "start": start, "source_npz": source.name, "source_npz_sha256": sha256(source),
        "png": destination.name, "png_sha256": sha256(destination),
        "start_time": t0, "horizons": [1, 5],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_root = args.inference_root.resolve()
    output = args.output
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise SystemExit("output must be a new directory below an existing parent")
    result_path = source_root / "result.json"
    if sha256(result_path) != RESULT_SHA256:
        raise SystemExit("inference result identity mismatch")
    result = json.loads(result_path.read_text())
    if (result.get("status") != "PROJECTED_POLICY_H1_H5_REPLAY_COMPLETE_NOT_ADMISSION"
            or result.get("scientific_admission") is not False
            or result.get("optimizer_steps") != 0):
        raise SystemExit("unexpected inference result contract")
    for start, digest in INPUTS.items():
        if sha256(source_root / f"fields_mpc_{start}.npz") != digest:
            raise SystemExit(f"saved field identity mismatch: {start}")
    output.mkdir(mode=0o755)
    try:
        images = [render_one(source_root / f"fields_mpc_{start}.npz",
                             output / f"mpc_start_{start}_h1_h5.png", start)
                  for start in INPUTS]
        manifest = {
            "status": "PROJECTED_POLICY_H1_H5_FIELD_PREVIEW_COMPLETE_PENDING_REVIEW",
            "source_inference_result_sha256": RESULT_SHA256,
            "saved_predictions_only": True, "model_rerun": False,
            "cfd_rerun": False, "scientific_admission": False, "images": images,
        }
        (output / "result.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.chmod(output / "result.json", 0o444)
        for item in images:
            os.chmod(output / item["png"], 0o444)
    except BaseException:
        for path in output.iterdir():
            path.unlink()
        output.rmdir()
        raise


if __name__ == "__main__":
    main()
