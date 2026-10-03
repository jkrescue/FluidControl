#!/usr/bin/env python3
"""Fixed dynamic6 start-zero force-window diagnostic, not a control gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

MANIFEST_SHA = "bfa49031a1ab2b8e10a5cc5d95f1fbd8713e289c3ac8155838334e3c3d007dae"
NORMALIZATION_SHA = "f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1"
CASES = tuple(
    f"full40_dynamic_validation_b{phase:02d}_{profile}"
    for phase in (1, 5)
    for profile in ("minus", "zero", "plus")
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def window_statistics(times, forces, end_time, duration=6.15):
    """Arithmetic statistics of uniformly sampled causal force observations."""
    if len(times) != len(forces) or len(times) < 2:
        raise ValueError("aligned time/force samples are required")
    if not math.isfinite(end_time) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("finite endpoint and positive duration required")
    if any(not math.isfinite(float(t)) for t in times):
        raise ValueError("nonfinite time")
    spacing = (float(times[-1]) - float(times[0])) / (len(times) - 1)
    if spacing <= 0 or any(
        not math.isclose(float(b) - float(a), spacing, rel_tol=0, abs_tol=2e-5)
        for a, b in zip(times[:-1], times[1:])
    ):
        raise ValueError("uniform strictly increasing samples required")
    if times[0] > end_time - duration or times[-1] < end_time:
        raise ValueError("full causal window is not available")
    indices = [
        i
        for i, t in enumerate(times)
        if end_time - duration - 1e-9 <= t <= end_time + 1e-9
    ]
    rows = [forces[i] for i in indices]
    if not rows or any(
        len(row) != 4 or any(not math.isfinite(float(v)) for v in row) for row in rows
    ):
        raise ValueError("finite four-channel forces required")
    means = [sum(float(r[k]) for r in rows) / len(rows) for k in range(4)]
    rms = math.sqrt(sum((float(r[3]) - means[3]) ** 2 for r in rows) / len(rows))
    return {
        "requested_end_time": end_time,
        "requested_window_D_over_U": duration,
        "first_sample_time": float(times[indices[0]]),
        "last_sample_time": float(times[indices[-1]]),
        "sample_spacing_D_over_U": spacing,
        "float32_time_quantization_tolerance": 2e-5,
        "sample_count": len(rows),
        "mean_total_cd": means[0] + means[2],
        "rear_cl_mean": means[3],
        "rear_cl_fluctuation_rms": rms,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "normalization-data", "config", "checkpoint-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-model-sha", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("diagnostics never overwrite an earlier result")
    if sha256(args.data / "manifest.json") != MANIFEST_SHA:
        raise ValueError("fixed dynamic6 manifest differs")
    normalization = args.normalization_data / "normalization.json"
    if sha256(normalization) != NORMALIZATION_SHA:
        raise ValueError("immutable train20 normalization differs")
    models = list(args.checkpoint_dir.glob("FNO.0.*.mdlus"))
    if len(models) != 1 or sha256(models[0]) != args.expected_model_sha:
        raise ValueError("explicit immutable model identity differs")
    files = sorted((args.data / "validation").glob("*.h5"))
    if {p.stem for p in files} != set(CASES):
        raise ValueError("fixed six-case validation panel differs")

    import h5py
    import numpy as np
    import torch
    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils import load_checkpoint

    from evaluate_tandem_fno import load_composed_config
    from train_tandem_fno import build_model, configured_force_indices

    DistributedManager.initialize()
    dist = DistributedManager()
    if dist.distributed:
        raise ValueError("single-process diagnostic required")
    cfg = load_composed_config(args.config)
    if tuple(configured_force_indices(cfg)) != (0, 1, 2, 3):
        raise ValueError("four force channels required")
    if dist.cuda:
        torch.cuda.set_per_process_memory_fraction(0.15, device=dist.device)
    network = build_model(cfg).to(dist.device)
    epoch = load_checkpoint(args.checkpoint_dir, models=network, device=dist.device)
    if epoch < 1:
        raise ValueError("no checkpoint loaded")
    network.eval()
    stats = json.loads(normalization.read_text())
    if stats["all_force_channels"] != ["front_cd", "front_cl", "rear_cd", "rear_cl"]:
        raise ValueError("force channel order differs")
    manifest = json.loads((args.normalization_data / "manifest.json").read_text())
    action_scale = float(manifest["max_abs_omega"])
    state_mean = torch.tensor(stats["state_mean"], device=dist.device)[
        None, :, None, None
    ]
    state_std = torch.tensor(stats["state_std"], device=dist.device)[
        None, :, None, None
    ]
    force_mean = torch.tensor(stats["all_force_mean"], device=dist.device)
    force_std = torch.tensor(stats["all_force_std"], device=dist.device)
    initial_states = {}
    reports = []
    for path in files:
        with h5py.File(path, "r") as handle:
            initial = handle["state"][0]
            mask_np = handle["mask"][:101]
            times = np.asarray(handle["time"][:101], dtype=np.float64).reshape(-1)
            omega = np.asarray(handle["omega"][:101], dtype=np.float64).reshape(-1)
            truth = np.asarray(handle["force"][:101], dtype=np.float64)
        if len(times) != 101 or not np.allclose(np.diff(times), 0.1, rtol=0, atol=2e-5):
            raise ValueError("requires H100 at dt=0.1D/U")
        if not np.array_equal(mask_np, np.broadcast_to(mask_np[0], mask_np.shape)):
            raise ValueError("fixed geometry mask differs across time")
        if (
            np.max(np.abs(omega)) > 0.75 + 1e-7
            or np.max(np.abs(np.diff(omega))) > 0.1 + 2e-5
        ):
            raise ValueError("action limits differ")
        phase = path.stem.rsplit("_", 1)[0]
        if phase in initial_states and not np.array_equal(
            initial_states[phase], initial
        ):
            raise ValueError("paired action cases do not share identical initial field")
        initial_states[phase] = initial
        mask = torch.as_tensor(mask_np[0:1], dtype=torch.float32, device=dist.device)
        state = torch.as_tensor(initial[None], dtype=torch.float32, device=dist.device)
        predicted = (state - state_mean) / state_std * mask
        predicted_forces = [truth[0].tolist()]
        with torch.no_grad():
            for step in range(100):
                now = torch.full_like(mask, float(omega[step] / action_scale))
                following = torch.full_like(mask, float(omega[step + 1] / action_scale))
                raw = network(torch.cat([predicted, mask, now, following], dim=1))
                if raw.shape[1] != 7 or not bool(torch.isfinite(raw).all()):
                    raise ValueError(
                        f"nonfinite or malformed prediction: {path.stem}/{step}"
                    )
                predicted = (predicted + raw[:, :3]) * mask
                if not bool(torch.isfinite(predicted).all()):
                    raise ValueError(
                        f"nonfinite autoregressive state: {path.stem}/{step}"
                    )
                force = (raw[:, 3:] * mask).sum((-2, -1)) / mask.sum(
                    (-2, -1)
                ).clamp_min(1)
                predicted_forces.append(
                    (force * force_std + force_mean)[0].cpu().tolist()
                )
        actual_stats = window_statistics(
            times.tolist(), truth.tolist(), float(times[-1])
        )
        model_stats = window_statistics(
            times.tolist(), predicted_forces, float(times[-1])
        )
        reports.append(
            {
                "case": path.stem,
                "hdf5_sha256": sha256(path),
                "initial_time": float(times[0]),
                "horizon_steps": 100,
                "truth_window": actual_stats,
                "prediction_window": model_stats,
                "times": times.tolist(),
                "omega_endpoints": omega.tolist(),
                "true_forces": truth.tolist(),
                "predicted_forces": predicted_forces,
            }
        )
    by_case = {r["case"]: r for r in reports}
    pairs = []
    for phase in (1, 5):
        prefix = f"full40_dynamic_validation_b{phase:02d}_"
        zero = by_case[prefix + "zero"]
        for profile in ("minus", "plus"):
            row = by_case[prefix + profile]
            actual = (
                row["truth_window"]["mean_total_cd"]
                - zero["truth_window"]["mean_total_cd"]
            )
            predicted_delta = (
                row["prediction_window"]["mean_total_cd"]
                - zero["prediction_window"]["mean_total_cd"]
            )
            pairs.append(
                {
                    "case": row["case"],
                    "true_delta_mean_cd": actual,
                    "predicted_delta_mean_cd": predicted_delta,
                    "absolute_error": abs(predicted_delta - actual),
                }
            )
    result = {
        "status": "SAMPLED_FORCE_WINDOW_DIAGNOSTIC_COMPLETE",
        "scope": "Fixed validation-only start0 H100; last6.15D/U sampled forces. Not dense CFD 60D/U physical acceptance, not a PPO gate or checkpoint selection.",
        "force_channels": stats["all_force_channels"],
        "model_sha256": args.expected_model_sha,
        "checkpoint_epoch": epoch,
        "normalization_sha256": NORMALIZATION_SHA,
        "manifest_sha256": MANIFEST_SHA,
        "script_sha256": sha256(__file__),
        "autoregression": "100 recursive state predictions, no intermediate CFD state correction; observed action endpoints with linear ramp convention",
        "pairs": pairs,
        "mean_action_delta_error": sum(r["absolute_error"] for r in pairs) / len(pairs),
        "cases": reports,
        "ppo_authorized": False,
        "frozen_test_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
