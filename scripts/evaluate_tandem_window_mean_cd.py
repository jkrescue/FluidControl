#!/usr/bin/env python3
"""Validation-only 100-step window-mean drag audit for a trained tandem FNO.

The existing Gate-B evaluator reports the terminal force at step 100. This
independent diagnostic retains every predicted force in the same free rollout
and compares the mean total drag over all 100 steps with real CFD.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path

import h5py
import numpy as np
import torch
from evaluate_tandem_fno import load_composed_config
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import load_checkpoint
from train_tandem_fno import build_model

HORIZON = 100
STRIDE = 25
FORCE_CHANNELS = ["front_cd", "front_cl", "rear_cd", "rear_cl"]
FORCE_INDICES = [0, 1, 2, 3]
GPU_MEMORY_FRACTION = 0.20
MIN_MEM_AVAILABLE_GIB = 20.0
MAX_WINDOW_MEAN_CD_NRMSE = 0.10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pooled_nrmse(cases: list[dict]) -> float:
    numerator = sum(row["segments"] * row["window_mean_cd_rmse"] ** 2 for row in cases)
    denominator = sum(
        row["segments"] * row["window_mean_cd_target_rms"] ** 2 for row in cases
    )
    if denominator <= 0 or not math.isfinite(numerator + denominator):
        raise ValueError("invalid pooled window-mean drag sums")
    return math.sqrt(numerator / denominator)


def build_rollout_inputs(
    state: torch.Tensor,
    mask: torch.Tensor,
    omega_now: torch.Tensor,
    omega_next: torch.Tensor,
) -> torch.Tensor:
    """Build the same six-channel input used by training and Gate-B rollout."""
    return torch.cat((state, mask, omega_now, omega_next), dim=1)


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def decode_attribute(value: object, label: str) -> object:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str) and label == "force_channels":
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("invalid HDF5 force_channels JSON") from error
    return value


def validate_hdf5(path: Path) -> None:
    with h5py.File(path, "r") as handle:
        for name in ("state", "mask", "omega", "force", "time"):
            if name not in handle:
                raise ValueError(f"{path}: missing dataset {name}")
        if handle["force"].ndim != 2 or handle["force"].shape[1] != 4:
            raise ValueError(f"{path}: force must have shape (frames, 4)")
        if handle["state"].ndim != 4 or handle["state"].shape[1] != 3:
            raise ValueError(f"{path}: state must have shape (frames, 3, y, x)")
        if handle["mask"].ndim != 4 or handle["mask"].shape[1] != 1:
            raise ValueError(f"{path}: mask must have shape (frames, 1, y, x)")
        if handle["omega"].ndim != 2 or handle["omega"].shape[1] != 1:
            raise ValueError(f"{path}: omega must have shape (frames, 1)")
        channels = decode_attribute(
            handle.attrs.get("force_channels"), "force_channels"
        )
        if list(channels or ()) != FORCE_CHANNELS:
            raise ValueError(f"{path}: force channel order mismatch")
        split = decode_attribute(handle.attrs.get("split"), "split")
        if split != "validation":
            raise ValueError(f"{path}: split attribute is not validation")
        frame_counts = {len(handle[name]) for name in ("state", "mask", "omega", "force", "time")}
        if len(frame_counts) != 1 or next(iter(frame_counts)) <= HORIZON:
            raise ValueError(f"{path}: inconsistent or insufficient frame counts")
        time = np.asarray(handle["time"][:], dtype=np.float64).reshape(-1)
        if not np.isfinite(time).all() or not np.allclose(np.diff(time), 0.1, atol=1e-6):
            raise ValueError(f"{path}: expected uniform 0.1 time spacing")


def validate_normalization(stats: dict) -> None:
    if stats.get("computed_from") != "train split only":
        raise ValueError("normalization is not declared train-only")
    if stats.get("all_force_channels") != FORCE_CHANNELS:
        raise ValueError("four-force normalization order mismatch")
    for key, size in (
        ("state_mean", 3),
        ("state_std", 3),
        ("all_force_mean", 4),
        ("all_force_std", 4),
    ):
        values = np.asarray(stats.get(key), dtype=np.float64)
        if values.shape != (size,) or not np.isfinite(values).all():
            raise ValueError(f"invalid normalization field: {key}")
        if key.endswith("_std") and np.any(values <= 0):
            raise ValueError(f"non-positive normalization scale: {key}")


def validate_model_config(checkpoint: object, expected: dict) -> None:
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint model_config is missing or invalid")
    for key, expected_value in expected.items():
        actual_value = checkpoint.get(key)
        if key == "num_fno_modes":
            actual_value = None if actual_value is None else list(actual_value)
            expected_value = list(expected_value)
        if actual_value != expected_value:
            raise ValueError(f"checkpoint model config mismatch: {key}")


def mem_available_gib(path: Path = Path("/proc/meminfo")) -> float:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / 1024**2
    raise RuntimeError("MemAvailable is unavailable")


def write_json_exclusive(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--normalization-data", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-normalization-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    if args.batch_size < 1 or args.batch_size > 8:
        parser.error("batch size must lie in [1, 8]")
    if os.environ.get("WINDOW_MEAN_HOST_TRAINING_GUARD") != "passed":
        parser.error("host training guard was not established by the Spark wrapper")
    if args.data.name != "tandem_cylinders_control_gap_v4":
        parser.error("evaluation must use the v4 validation profile")
    if args.normalization_data.name not in {
        "tandem_cylinders_gate_b_aug_v3",
        "tandem_cylinders_control_gap_v4",
    }:
        parser.error("unexpected normalization profile")

    paths = sorted((args.data / "validation").glob("*.h5"))
    if len(paths) != 4:
        raise ValueError("expected exactly four validation trajectories")
    stats_path = args.normalization_data / "normalization.json"
    manifest_path = args.normalization_data / "manifest.json"
    evaluation_manifest = load_json(args.data / "manifest.json")
    stats = load_json(stats_path)
    manifest = load_json(manifest_path)
    if evaluation_manifest.get("profile") != "control_gap_v4":
        raise ValueError("evaluation manifest is not control_gap_v4")
    if evaluation_manifest.get("trajectory_counts", {}).get("validation") != 4:
        raise ValueError("v4 manifest validation count mismatch")
    expected_normalization_profile = (
        "control_gap_v4"
        if args.normalization_data.name == "tandem_cylinders_control_gap_v4"
        else "gate_b_aug_v3"
    )
    if manifest.get("profile") != expected_normalization_profile:
        raise ValueError("normalization manifest profile mismatch")
    validate_normalization(stats)
    for path in paths:
        validate_hdf5(path)
    normalization_sha = sha256(stats_path)
    if normalization_sha != args.expected_normalization_sha256:
        raise ValueError("normalization SHA-256 mismatch")
    action_scale = float(manifest["max_abs_omega"])
    if action_scale != 5.0:
        raise ValueError("unexpected action scale")
    if float(evaluation_manifest.get("max_abs_omega", math.inf)) > action_scale:
        raise ValueError("evaluation actions exceed checkpoint training support")

    model_paths = sorted(args.checkpoint_dir.glob("FNO.*.mdlus"))
    if len(model_paths) != 1:
        raise ValueError("expected one FNO checkpoint model")
    checkpoint_sha = sha256(model_paths[0])
    if checkpoint_sha != args.expected_checkpoint_sha256:
        raise ValueError("checkpoint SHA-256 mismatch")
    if mem_available_gib() < MIN_MEM_AVAILABLE_GIB:
        raise RuntimeError("MemAvailable is below the 20 GiB safety floor")

    DistributedManager.initialize()
    dist = DistributedManager()
    if dist.distributed and dist.rank != 0:
        raise RuntimeError("window-mean evaluation must run as one process")
    if not dist.cuda:
        raise RuntimeError("CUDA is required for the official PhysicsNeMo FNO")
    torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, device=dist.device)
    cfg = load_composed_config(Path("conf/tandem_fno_total_drag.yaml"))
    network = build_model(cfg).to(dist.device)
    metadata: dict = {}
    epoch = load_checkpoint(
        args.checkpoint_dir, models=network, metadata_dict=metadata, device=dist.device
    )
    if epoch <= 0:
        raise ValueError("official PhysicsNeMo checkpoint was not loaded")
    epoch_match = re.fullmatch(r"FNO\.0\.(\d+)\.mdlus", model_paths[0].name)
    if not epoch_match or int(epoch_match.group(1)) != int(epoch):
        raise ValueError("checkpoint filename epoch mismatch")
    if metadata.get("force_channels") != FORCE_CHANNELS:
        raise ValueError("checkpoint force channel order mismatch")
    if metadata.get("force_indices") != FORCE_INDICES:
        raise ValueError("checkpoint force indices mismatch")
    if float(metadata.get("action_scale", float("nan"))) != action_scale:
        raise ValueError("checkpoint action scale mismatch")
    expected_model_config = {
        key: cfg.model[key]
        for key in (
            "in_channels",
            "out_channels",
            "latent_channels",
            "num_fno_layers",
            "num_fno_modes",
            "decoder_layers",
            "decoder_layer_size",
            "padding",
            "coord_features",
        )
    }
    checkpoint_model_config = metadata.get("model_config", {})
    validate_model_config(checkpoint_model_config, expected_model_config)
    network.eval().requires_grad_(False)

    state_mean = torch.as_tensor(stats["state_mean"], device=dist.device)[:, None, None]
    state_std = torch.as_tensor(stats["state_std"], device=dist.device)[:, None, None]
    force_mean = torch.as_tensor(stats["all_force_mean"], device=dist.device)
    force_std = torch.as_tensor(stats["all_force_std"], device=dist.device)
    cases = []
    with torch.inference_mode():
        for path in paths:
            with h5py.File(path) as handle:
                nframes = len(handle["state"])
                starts = list(range(0, nframes - HORIZON, STRIDE))
                if not starts:
                    raise ValueError(f"insufficient frames in {path}")
                omega = torch.from_numpy(handle["omega"][:, 0]).float().to(dist.device)
                force = torch.from_numpy(handle["force"][:]).float().to(dist.device)
                if not torch.isfinite(omega).all() or not torch.isfinite(force).all():
                    raise ValueError(f"non-finite CFD action or force in {path}")
                if float(omega.abs().max().item()) > action_scale + 1.0e-6:
                    raise ValueError(f"observed action exceeds training support in {path}")
                segment_errors = []
                segment_targets = []
                point_errors = []
                for offset in range(0, len(starts), args.batch_size):
                    batch_starts = starts[offset : offset + args.batch_size]
                    if mem_available_gib() < MIN_MEM_AVAILABLE_GIB:
                        raise RuntimeError("MemAvailable fell below the 20 GiB safety floor")
                    indices = torch.as_tensor(batch_starts, dtype=torch.long, device=dist.device)
                    raw_state = torch.from_numpy(handle["state"][batch_starts]).float().to(dist.device)
                    mask = torch.from_numpy(handle["mask"][batch_starts]).float().to(dist.device)
                    if mask.ndim != 4 or mask.shape[1] != 1:
                        raise ValueError(f"invalid spatial mask in {path}")
                    if not torch.isfinite(raw_state).all() or not torch.isfinite(mask).all():
                        raise ValueError(f"non-finite state or mask in {path}")
                    if bool(((mask < 0) | (mask > 1)).any()):
                        raise ValueError(f"spatial mask is outside [0, 1] in {path}")
                    state = ((raw_state - state_mean) / state_std) * mask
                    height, width = mask.shape[-2:]
                    predicted_cd = []
                    target_cd = []
                    for step in range(HORIZON):
                        now = (omega[indices + step] / action_scale).reshape(-1, 1, 1, 1)
                        following = (omega[indices + step + 1] / action_scale).reshape(-1, 1, 1, 1)
                        inputs = build_rollout_inputs(
                            state,
                            mask,
                            now.expand(-1, 1, height, width),
                            following.expand(-1, 1, height, width),
                        )
                        raw = network(inputs)
                        if raw.shape[1] != 7:
                            raise ValueError("expected official seven-output tandem FNO")
                        state = (state + raw[:, :3]) * mask
                        normalized_force = (raw[:, 3:] * mask).sum(dim=(-2, -1)) / (
                            mask.sum(dim=(-2, -1)).clamp_min(1)
                        )
                        physical_force = normalized_force * force_std + force_mean
                        predicted_cd.append(physical_force[:, 0] + physical_force[:, 2])
                        truth = force[indices + step + 1]
                        target_cd.append(truth[:, 0] + truth[:, 2])
                    predicted = torch.stack(predicted_cd, dim=1)
                    target = torch.stack(target_cd, dim=1)
                    if not torch.isfinite(predicted).all():
                        raise ValueError(f"non-finite FNO drag rollout in {path}")
                    mean_error = predicted.mean(dim=1) - target.mean(dim=1)
                    segment_errors.extend(mean_error.cpu().tolist())
                    segment_targets.extend(target.mean(dim=1).cpu().tolist())
                    point_errors.extend((predicted - target).flatten().cpu().tolist())
            errors = np.asarray(segment_errors, dtype=np.float64)
            targets = np.asarray(segment_targets, dtype=np.float64)
            points = np.asarray(point_errors, dtype=np.float64)
            rmse = float(np.sqrt(np.mean(errors**2)))
            target_rms = float(np.sqrt(np.mean(targets**2)))
            if not math.isfinite(rmse + target_rms) or target_rms <= 0:
                raise ValueError(f"invalid window-mean drag metric in {path}")
            cases.append(
                {
                    "case": path.stem,
                    "validation_hdf5_sha256": sha256(path),
                    "segments": len(errors),
                    "window_mean_cd_rmse": rmse,
                    "window_mean_cd_target_rms": target_rms,
                    "window_mean_cd_nrmse": rmse / target_rms,
                    "pointwise_cd_rmse_over_overlapping_windows": float(
                        np.sqrt(np.mean(points**2))
                    ),
                }
            )
    pooled = pooled_nrmse(cases)
    worst = max(cases, key=lambda row: row["window_mean_cd_nrmse"])
    result = {
        "status": (
            "PASS_VALIDATION_WINDOW_MEAN_CD"
            if pooled <= MAX_WINDOW_MEAN_CD_NRMSE
            else "FAIL_VALIDATION_WINDOW_MEAN_CD"
        ),
        "scope": "100-step observed-action free rollout; validation only; not CFD control benefit",
        "metric": "100-step time-window mean system-total Cd NRMSE",
        "window_mean_definition": (
            "arithmetic mean of CFD/FNO system-total Cd at rollout endpoints "
            "1..100; uniform nondimensional dt=0.1"
        ),
        "horizon_steps": HORIZON,
        "segment_stride": STRIDE,
        "acceptance_threshold_nrmse": MAX_WINDOW_MEAN_CD_NRMSE,
        "split": "validation",
        "evaluation_data": str(args.data),
        "normalization_data": str(args.normalization_data),
        "normalization_sha256": normalization_sha,
        "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoint_epoch": int(epoch),
        "checkpoint_model_sha256": checkpoint_sha,
        "cases": cases,
        "pooled_window_mean_cd_nrmse": pooled,
        "macro_window_mean_cd_nrmse": float(
            np.mean([row["window_mean_cd_nrmse"] for row in cases])
        ),
        "worst_case": worst["case"],
        "worst_case_window_mean_cd_nrmse": worst["window_mean_cd_nrmse"],
        "limitations": [
            "Validation trajectories only; test split remains frozen.",
            "100-step windows overlap at stride 25; segments are not independent CFD cases.",
            "Known future observed actions are supplied at each FNO step, as in CEM planning.",
            "This scores window-average drag prediction, not action-selection or lift fidelity.",
        ],
    }
    write_json_exclusive(args.output, result)
    print(json.dumps({key: result[key] for key in (
        "pooled_window_mean_cd_nrmse",
        "macro_window_mean_cd_nrmse",
        "worst_case_window_mean_cd_nrmse",
    )}, indent=2), flush=True)


if __name__ == "__main__":
    main()
