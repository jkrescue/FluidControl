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

H1_SHA = "1eacc9219f2f608c54e6ef48d4856624af64b8ad00eb477bbd5772eff8ad61ef"


def precision_flags(torch):
    return {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }


def tensor_digest(network):
    result = {}
    for role in ("flow_model", "aerodynamic_model"):
        digest = hashlib.sha256()
        for name, tensor in sorted(getattr(network, role).state_dict().items()):
            value = tensor.detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(value.numpy().tobytes())
        result[role] = digest.hexdigest()
    return result


def precision_comparison(reports):
    by_case = {r["case"]: r for r in reports}
    if set(by_case) != set(CASES) or len(reports) != 6:
        raise ValueError("exact six cases required")
    pairs = []
    per_case = []
    for name, report in by_case.items():
        rows = report["rows"]
        if len(rows) != 2:
            raise ValueError("two precision rows required")
        high, highest = rows
        for row, label in zip(rows, ("high", "highest")):
            if row["precision"] != {
                "float32_matmul_precision": label,
                "cuda_matmul_allow_tf32": label == "high",
                "cudnn_allow_tf32": label == "high",
            }:
                raise ValueError("effective precision differs")
        for key in ("input_time", "target_time", "omega_s", "omega_s_plus_1", "true_force_s_plus_1"):
            if high[key] != highest[key]:
                raise ValueError("precision pair inputs differ")
        per_case.append(dict(case=name, highest_minus_high=[
            b-a for a,b in zip(high["predicted_force_s_plus_1"], highest["predicted_force_s_plus_1"])]))
        if name.endswith("_zero"):
            continue
        zero = by_case[name.rsplit("_", 1)[0] + "_zero"]["rows"]
        for i, label in enumerate(("high", "highest")):
            row, z = rows[i], zero[i]
            truth = [a-b for a,b in zip(row["true_force_s_plus_1"], z["true_force_s_plus_1"])]
            pred = [a-b for a,b in zip(row["predicted_force_s_plus_1"], z["predicted_force_s_plus_1"])]
            pairs.append(dict(case=name, precision=label, true_delta_force=truth,
                predicted_delta_force=pred, delta_error=[a-b for a,b in zip(pred,truth)],
                true_delta_total_cd=truth[0]+truth[2], predicted_delta_total_cd=pred[0]+pred[2]))
    return dict(per_case=per_case, action_minus_zero=pairs,
        interpretation="Signed small differences only; no zero-tolerance sign gate or automatic model claim.")



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


def teacher_forced_h1_predictions(
    states,
    mask_np,
    omega,
    truth,
    times,
    *,
    action_scale,
    state_mean,
    state_std,
    force_mean,
    force_std,
    network,
    history_k,
    reset_history,
    history_dual_step,
    expected_high,
):
    """Predict signed F(s+1) from true q(s) and recorded endpoint actions."""
    import numpy as np
    import torch

    if history_k is None:
        raise ValueError("teacher-forced H1 requires an explicit history profile")
    if (
        states.shape[0] != 101
        or states.ndim != 4
        or states.shape[1] != 3
        or mask_np.shape != (101, 1, *states.shape[-2:])
        or len(omega) != 101
        or truth.shape != (101, 4)
        or len(times) != 101
        or not np.isfinite(states).all()
        or not np.isfinite(mask_np).all()
        or not np.isfinite(omega).all()
        or not np.isfinite(truth).all()
        or not np.isfinite(times).all()
    ):
        raise ValueError("finite aligned 101-frame H1 inputs are required")
    mask = torch.as_tensor(mask_np[0:1], dtype=torch.float32, device=state_mean.device)
    rows = []
    with torch.no_grad():
        for precision_name in ("high", "highest"):
            torch.set_float32_matmul_precision(precision_name)
            torch.backends.cuda.matmul.allow_tf32 = precision_name == "high"
            torch.backends.cudnn.allow_tf32 = precision_name == "high"
            effective = precision_flags(torch)
            step = 0
            # Only q_s is normalized here. q_(s+1) is used below solely as the
            # force target and is never passed to reset_history or the network.
            state = torch.as_tensor(
                states[step : step + 1], dtype=torch.float32, device=state_mean.device
            )
            normalized = (state - state_mean) / state_std * mask
            now = torch.tensor(
                [float(omega[step])], dtype=normalized.dtype, device=normalized.device
            ) / action_scale
            following = torch.tensor(
                [float(omega[step + 1])], dtype=normalized.dtype, device=normalized.device
            ) / action_scale
            history = reset_history(normalized, now, k=history_k)
            raw, _next_state, _next_history = history_dual_step(
                network, history, mask, following
            )
            if raw.shape != (1, 7, *mask.shape[-2:]) or not bool(torch.isfinite(raw).all()):
                raise ValueError(f"nonfinite or malformed teacher-forced prediction at {step}")
            force = (raw[:, 3:] * mask).sum((-2, -1)) / mask.sum((-2, -1)).clamp_min(1)
            predicted = (force * force_std + force_mean)[0].cpu().tolist()
            target = [float(value) for value in truth[step + 1]]
            if len(target) != 4 or any(not math.isfinite(value) for value in target + predicted):
                raise ValueError("finite signed four-force rows required")
            if precision_name == "high" and predicted != expected_high:
                raise ValueError("high/TF32 first-step exact replay differs; do not relax tolerance")
            rows.append(
                {
                    "precision": effective,
                    "step": step + 1,
                    "input_time": float(times[step]),
                    "target_time": float(times[step + 1]),
                    "omega_s": float(omega[step]),
                    "omega_s_plus_1": float(omega[step + 1]),
                    "true_force_s_plus_1": target,
                    "predicted_force_s_plus_1": predicted,
                }
            )
    return rows


def summarize_signed_force_rows(rows):
    """Return channel-wise signed bias, amplitude, MAE and RMSE."""
    if not rows:
        raise ValueError("signed force rows required")
    channels = ("front_cd", "front_cl", "rear_cd", "rear_cl")
    result = {}
    for index, channel in enumerate(channels):
        truth = [float(row["true_force_s_plus_1"][index]) for row in rows]
        predicted = [float(row["predicted_force_s_plus_1"][index]) for row in rows]
        error = [p - t for p, t in zip(predicted, truth)]
        if any(not math.isfinite(value) for value in truth + predicted):
            raise ValueError("nonfinite signed force summary")
        truth_mean = sum(truth) / len(truth)
        predicted_mean = sum(predicted) / len(predicted)
        result[channel] = {
            "count": len(rows),
            "truth_mean": truth_mean,
            "predicted_mean": predicted_mean,
            "signed_error_mean": sum(error) / len(error),
            "mean_absolute_error": sum(abs(value) for value in error) / len(error),
            "root_mean_square_error": math.sqrt(sum(value * value for value in error) / len(error)),
            "truth_centered_rms": math.sqrt(
                sum((value - truth_mean) ** 2 for value in truth) / len(truth)
            ),
            "predicted_centered_rms": math.sqrt(
                sum((value - predicted_mean) ** 2 for value in predicted) / len(predicted)
            ),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "normalization-data", "config", "checkpoint-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-model-sha", required=True)
    parser.add_argument("--dual-fno-manifest", type=Path, default=None)
    parser.add_argument("--expected-dual-fno-manifest-sha256", default=None)
    parser.add_argument("--dual-training-config", type=Path, default=None,
                        help="immutable training config; evaluation still uses --config")
    parser.add_argument(
        "--fno-history-profile",
        choices=("legacy_k1", "p026_k1", "p026_k4"),
        default="legacy_k1",
        help="explicit caller input contract; legacy behavior remains the default",
    )
    parser.add_argument("--expected-calibrated-state-sha256")
    parser.add_argument("--expected-calibrated-kind", choices=("FC_P008_TRAIN_ONLY_FORCE_ROW_CANDIDATE", "FC_P009_TRAIN_ONLY_JOINT_FORCE_ROW_CANDIDATE"), default="FC_P008_TRAIN_ONLY_FORCE_ROW_CANDIDATE")
    parser.add_argument("--allow-calibrated-epoch-zero", action="store_true")
    parser.add_argument(
        "--mode", choices=("autoregressive", "teacher_forced_h1"), default="autoregressive"
    )
    parser.add_argument("--h1-reference", type=Path, required=True)
    parser.add_argument("--autoregressive-reference", type=Path)
    parser.add_argument("--expected-autoregressive-reference-sha256")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("diagnostics never overwrite an earlier result")
    if sha256(args.data / "manifest.json") != MANIFEST_SHA:
        raise ValueError("fixed dynamic6 manifest differs")
    normalization = args.normalization_data / "normalization.json"
    if sha256(normalization) != NORMALIZATION_SHA:
        raise ValueError("immutable train20 normalization differs")
    if sha256(args.h1_reference) != H1_SHA:
        raise ValueError("E073 reference SHA differs")
    h1_reference = json.loads(args.h1_reference.read_text())
    if args.mode != "teacher_forced_h1":
        raise ValueError("E094 only supports paired first-step precision")
    autoregressive_reference = None
    if args.mode == "teacher_forced_h1":
        if args.autoregressive_reference is None or args.expected_autoregressive_reference_sha256 is None:
            raise ValueError("teacher-forced mode requires the immutable AR reference")
        if (
            not args.autoregressive_reference.is_file()
            or args.autoregressive_reference.is_symlink()
            or sha256(args.autoregressive_reference)
            != args.expected_autoregressive_reference_sha256
        ):
            raise ValueError("autoregressive reference differs")
        autoregressive_reference = json.loads(args.autoregressive_reference.read_text())
        if (
            autoregressive_reference.get("dual_fno_manifest_sha256")
            != args.expected_dual_fno_manifest_sha256
            or autoregressive_reference.get("model_sha256") != args.expected_model_sha
        ):
            raise ValueError("autoregressive reference candidate identity differs")
    from fluid_control.dual_fno import dual_fno_requested, require_dual_training_config

    use_dual_fno = dual_fno_requested(
        args.dual_fno_manifest,
        args.expected_dual_fno_manifest_sha256,
        single_model_calibrated_arguments=(
            args.allow_calibrated_epoch_zero,
            args.expected_calibrated_state_sha256,
        ),
    )
    if args.mode == "teacher_forced_h1" and not use_dual_fno:
        raise ValueError("teacher-forced comparison requires the official dual-FNO candidate")
    require_dual_training_config(use_dual_fno, args.dual_training_config)
    if not use_dual_fno:
        models = list(args.checkpoint_dir.glob("FNO.0.*.mdlus"))
        if len(models) != 1 or sha256(models[0]) != args.expected_model_sha:
            raise ValueError("explicit immutable model identity differs")
    files = sorted((args.data / "validation").glob("*.h5"))
    if {p.stem for p in files} != set(CASES):
        raise ValueError("fixed six-case validation panel differs")
    if autoregressive_reference is not None:
        reference_cases = {
            row.get("case"): row.get("hdf5_sha256")
            for row in autoregressive_reference.get("cases", [])
            if isinstance(row, dict)
        }
        if set(reference_cases) != set(CASES):
            raise ValueError("autoregressive reference six-case inventory differs")
        for path in files:
            if reference_cases[path.stem] != sha256(path):
                raise ValueError(f"autoregressive reference HDF differs: {path.stem}")

    import h5py
    import numpy as np
    import torch
    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils import load_checkpoint

    from evaluate_tandem_fno import (
        history_dual_step,
        history_profile_length,
        load_composed_config,
    )
    from train_tandem_fno import build_model, configured_force_indices

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    DistributedManager.initialize()
    dist = DistributedManager()
    if dist.distributed:
        raise ValueError("single-process diagnostic required")
    cfg = load_composed_config(args.config)
    if tuple(configured_force_indices(cfg)) != (0, 1, 2, 3):
        raise ValueError("four force channels required")
    allocator_fraction = None
    allocator_target_bytes = None
    if dist.cuda:
        if args.mode == "teacher_forced_h1":
            allocator_target_bytes = 6 * 1024**3
            total = int(torch.cuda.get_device_properties(dist.device).total_memory)
            allocator_fraction = allocator_target_bytes / total
            if not 0.0 < allocator_fraction < 1.0:
                raise ValueError("invalid six-GiB allocator contract")
            torch.cuda.set_per_process_memory_fraction(allocator_fraction, device=dist.device)
        else:
            allocator_fraction = 0.15
            torch.cuda.set_per_process_memory_fraction(allocator_fraction, device=dist.device)
    dual_identity = None
    if use_dual_fno:
        from fluid_control.dual_fno import load_dual_fno, validate_dual_runtime_files

        network, dual_identity = load_dual_fno(
            args.dual_fno_manifest,
            cfg,
            dist.device,
            build_model=build_model,
            load_checkpoint=load_checkpoint,
            expected_manifest_sha256=args.expected_dual_fno_manifest_sha256,
        )
        validate_dual_runtime_files(
            dual_identity,
            config_path=args.dual_training_config,
            normalization_path=normalization,
        )
        if args.checkpoint_dir.resolve() != dual_identity.aerodynamic.directory:
            raise ValueError("dual FNO aerodynamic checkpoint directory differs")
        if args.expected_model_sha != dual_identity.aerodynamic.model_sha256:
            raise ValueError("dual FNO expected aerodynamic model SHA differs")
        epoch = dual_identity.aerodynamic.epoch
    else:
        network = build_model(cfg).to(dist.device)
        epoch = load_checkpoint(args.checkpoint_dir, models=network, device=dist.device)
        from fluid_control.calibrated_checkpoint import validate_calibrated_epoch_zero

        validate_calibrated_epoch_zero(
            args.checkpoint_dir,
            epoch,
            allow=args.allow_calibrated_epoch_zero,
            expected_model_sha256=args.expected_model_sha,
            expected_state_sha256=args.expected_calibrated_state_sha256,
            expected_kind=args.expected_calibrated_kind,
        )
    history_k = history_profile_length(
        args.fno_history_profile,
        use_dual_fno=use_dual_fno,
        network=network,
        manifest_payload=None if dual_identity is None else dual_identity.payload,
    )
    network.eval()
    tensors_before = tensor_digest(network)
    if args.mode == "teacher_forced_h1" and history_k != 1:
        raise ValueError("teacher-forced comparison requires the K1 history contract")
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
        if args.mode == "teacher_forced_h1":
            from p026_history_inference import reset_history

            rows = teacher_forced_h1_predictions(
                states,
                mask_np,
                omega,
                truth,
                times,
                action_scale=action_scale,
                state_mean=state_mean,
                state_std=state_std,
                force_mean=force_mean,
                force_std=force_std,
                network=network,
                history_k=history_k,
                reset_history=reset_history,
                history_dual_step=history_dual_step,
                expected_high=next(c for c in h1_reference["cases"] if c["case"] == path.stem)["rows"][0]["predicted_force_s_plus_1"],
            )
            reports.append(
                {
                    "case": path.stem,
                    "hdf5_sha256": sha256(path),
                    "rows": rows,
                    "scope": "two separately labelled precision rows; no mixed-precision summary",
                }
            )
            continue
        state = torch.as_tensor(initial[None], dtype=torch.float32, device=dist.device)
        predicted = (state - state_mean) / state_std * mask
        history_buffer = None
        if history_k is not None:
            from p026_history_inference import reset_history

            history_buffer = reset_history(
                predicted,
                torch.tensor(
                    [omega[0] / action_scale],
                    dtype=predicted.dtype,
                    device=predicted.device,
                ),
                k=history_k,
            )
        predicted_forces = [truth[0].tolist()]
        field_diagnostics = []
        spatial_diagnostics = []
        with torch.no_grad():
            for step in range(100):
                now = torch.full_like(mask, float(omega[step] / action_scale))
                following = torch.full_like(mask, float(omega[step + 1] / action_scale))
                if history_buffer is None:
                    raw = network(torch.cat([predicted, mask, now, following], dim=1))
                else:
                    raw, predicted, history_buffer = history_dual_step(
                        network,
                        history_buffer,
                        mask,
                        torch.tensor(
                            [omega[step + 1] / action_scale],
                            dtype=predicted.dtype,
                            device=predicted.device,
                        ),
                    )
                if raw.shape[1] != 7 or not bool(torch.isfinite(raw).all()):
                    raise ValueError(
                        f"nonfinite or malformed prediction: {path.stem}/{step}"
                    )
                if history_buffer is None:
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
    if args.mode == "teacher_forced_h1":
        all_rows = [row for report in reports for row in report["rows"]]
        tensors_after = tensor_digest(network)
        if tensors_before != tensors_after:
            raise ValueError("model tensors changed")
        result = {
            "status": "P064_FIRST_STEP_PRECISION_COMPLETE_NOT_ADMISSION",
            "scope": (
                "Six fixed dynamic6 initial states, high/TF32 then highest/noTF32 per case; "
                "batch1 two calls per case. Not a precision-based admission or training."
            ),
            "mode": args.mode,
            "batch_size": 1,
            "theoretical_forward_counts": {
                "dual_steps": len(all_rows),
                "flow_submodel_forwards": len(all_rows),
                "aerodynamic_submodel_forwards": len(all_rows),
                "total_submodel_forwards": 2 * len(all_rows),
            },
            "force_channels": stats["all_force_channels"],
            "model_sha256": args.expected_model_sha,
            "checkpoint_epoch": epoch,
            "normalization_sha256": NORMALIZATION_SHA,
            "manifest_sha256": MANIFEST_SHA,
            "script_sha256": sha256(__file__),
            "precision": {
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            },
            "allocator_target_bytes": allocator_target_bytes,
            "allocator_fraction": allocator_fraction,
            "dual_fno_manifest_sha256": dual_identity.manifest_sha256,
            "flow_model_sha256": dual_identity.flow.model_sha256,
            "flow_state_sha256": dual_identity.flow.state_sha256,
            "aerodynamic_state_sha256": dual_identity.aerodynamic.state_sha256,
            "fno_history_profile": args.fno_history_profile,
            "autoregressive_reference": {
                "path": str(args.autoregressive_reference),
                "sha256": args.expected_autoregressive_reference_sha256,
            },
            "h1_reference_sha256": H1_SHA,
            "high_reproduction_tolerance": {"atol": 0, "rtol": 0},
            "high_reproduction_pass": True,
            "tensor_digest_before": tensors_before,
            "tensor_digest_after": tensors_after,
            "precision_comparison": precision_comparison(reports),
            "signed_force_summary_by_precision": {name: summarize_signed_force_rows([r for r in all_rows if r["precision"]["float32_matmul_precision"] == name]) for name in ("high", "highest")},
            "cases": reports,
            "optimizer_steps": 0,
            "model_updated": False,
            "frozen_test_accessed": False,
            "scientific_admission": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2))
        return
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
    if dual_identity is not None:
        result["dual_fno_manifest"] = str(dual_identity.manifest_path)
        result["dual_fno_manifest_sha256"] = dual_identity.manifest_sha256
        result["flow_model_sha256"] = dual_identity.flow.model_sha256
        result["flow_state_sha256"] = dual_identity.flow.state_sha256
        result["aerodynamic_state_sha256"] = dual_identity.aerodynamic.state_sha256
    if history_k is not None:
        result["fno_history_profile"] = args.fno_history_profile
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
