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
FIXED_SPATIAL_DIAGNOSTIC_STEPS = (1, 10, 50, 100)


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


def field_step_statistics(predicted, truth, mask):
    """Return observational physical-field errors without altering either field."""
    import numpy as np

    predicted = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    mask = np.asarray(mask)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if (
        predicted.shape != truth.shape
        or predicted.ndim != 3
        or predicted.shape[0] != 3
        or mask.shape != predicted.shape[1:]
    ):
        raise ValueError("expected aligned (u,v,p) fields and one spatial mask")
    valid = mask.astype(bool)
    if not valid.any():
        raise ValueError("field diagnostic requires valid fluid cells")
    predicted_valid = predicted[:, valid]
    truth_valid = truth[:, valid]
    if not np.isfinite(predicted_valid).all() or not np.isfinite(truth_valid).all():
        raise ValueError("field diagnostic requires finite valid-cell values")

    predicted_pressure = predicted_valid[2]
    truth_pressure = truth_valid[2]
    predicted_pressure_mean = float(predicted_pressure.mean())
    truth_pressure_mean = float(truth_pressure.mean())
    pressure_error = predicted_pressure - truth_pressure

    def relative_l2(error, reference):
        reference_norm = float(np.sqrt(np.square(reference).sum()))
        if reference_norm == 0.0:
            return None
        return float(np.sqrt(np.square(error).sum()) / reference_norm)

    pressure_raw_relative_l2 = relative_l2(pressure_error, truth_pressure)
    predicted_pressure_gauge = predicted_pressure - predicted_pressure_mean
    truth_pressure_gauge = truth_pressure - truth_pressure_mean
    pressure_demeaned_relative_l2 = relative_l2(
        predicted_pressure_gauge - truth_pressure_gauge,
        truth_pressure_gauge,
    )
    velocity_error = predicted_valid[:2] - truth_valid[:2]
    velocity_relative_l2 = relative_l2(velocity_error, truth_valid[:2])
    return {
        "predicted_pressure_spatial_mean": predicted_pressure_mean,
        "truth_pressure_spatial_mean": truth_pressure_mean,
        "pressure_raw_relative_l2": pressure_raw_relative_l2,
        "pressure_demeaned_relative_l2_diagnostic_only": pressure_demeaned_relative_l2,
        "velocity_uv_relative_l2": velocity_relative_l2,
    }


def summarize_field_diagnostics(rows):
    """Summarize stepwise diagnostics while preserving the complete rows."""
    if not rows:
        raise ValueError("at least one field diagnostic row is required")
    keys = (
        "predicted_pressure_spatial_mean",
        "truth_pressure_spatial_mean",
        "pressure_raw_relative_l2",
        "pressure_demeaned_relative_l2_diagnostic_only",
        "velocity_uv_relative_l2",
    )
    result = {}
    for key in keys:
        values = [None if row[key] is None else float(row[key]) for row in rows]
        defined = [value for value in values if value is not None]
        if any(not math.isfinite(value) for value in defined):
            raise ValueError("nonfinite field diagnostic summary input")
        result[key] = {
            "defined_steps": len(defined),
            "total_steps": len(values),
            "minimum": min(defined) if defined else None,
            "maximum": max(defined) if defined else None,
            "mean": sum(defined) / len(defined) if defined else None,
            "final": values[-1],
        }
    return result


def fixed_spatial_snapshot_diagnostic(
    step, truth, prediction, x, y, mask, *, diagnostic=None
):
    """Return the fixed observational ROI diagnostic, or None for other steps."""
    if step not in FIXED_SPATIAL_DIAGNOSTIC_STEPS:
        return None
    if diagnostic is None:
        from fno_spatial_diagnostics import spatial_snapshot_diagnostics

        diagnostic = spatial_snapshot_diagnostics
    return {"step": int(step), **diagnostic(truth, prediction, x, y, mask)}


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
            states = handle["state"][:101]
            initial = states[0]
            mask_np = handle["mask"][:101]
            x = np.asarray(handle["x"][:], dtype=np.float64)
            y = np.asarray(handle["y"][:], dtype=np.float64)
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
        field_diagnostics = []
        spatial_diagnostics = []
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
                predicted_physical = predicted * state_std + state_mean
                predicted_physical_np = predicted_physical[0].cpu().numpy()
                field_diagnostics.append(
                    {
                        "step": step + 1,
                        "time": float(times[step + 1]),
                        **field_step_statistics(
                            predicted_physical_np,
                            states[step + 1],
                            mask_np[step + 1],
                        ),
                    }
                )
                spatial = fixed_spatial_snapshot_diagnostic(
                    step + 1,
                    states[step + 1],
                    predicted_physical_np,
                    x,
                    y,
                    mask_np[step + 1],
                )
                if spatial is not None:
                    spatial["time"] = float(times[step + 1])
                    spatial_diagnostics.append(spatial)
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
                "field_diagnostics": field_diagnostics,
                "field_diagnostic_summary": summarize_field_diagnostics(
                    field_diagnostics
                ),
                "fixed_spatial_diagnostics": spatial_diagnostics,
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
        "pressure_gauge_diagnostic": (
            "observational only: pressure is not demeaned or projected in the recursive "
            "state, force head, or formal metrics; diagnostic demeaning is applied only "
            "to copied arrays when computing the separately labelled relative L2"
        ),
        "spatial_diagnostic": (
            "observational only at fixed H1/H10/H50/H100 snapshots; finite solid-free "
            "ROI x/D=[17,24], y/D=[5,10], spatial mean removed and a common 2D Hann "
            "window used before orthonormal FFT. This is not homogeneous-turbulence "
            "E(k), defines no scientific pass threshold, does not filter fields, and "
            "does not feed back into autoregression, forces, gates, or model selection"
        ),
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
