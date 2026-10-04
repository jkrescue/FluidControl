#!/usr/bin/env python3
"""Produce the D012 canonical window and dynamic compatibility receipts.

This is a CPU-only compatibility producer.  It recomputes metrics from the
stepwise force/evaluation evidence and never authorizes or starts PPO.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

CASES = tuple(
    f"full40_dynamic_validation_b{phase:02d}_{profile}"
    for phase in (1, 5)
    for profile in ("minus", "zero", "plus")
)
FORCE_CHANNELS = ["front_cd", "front_cl", "rear_cd", "rear_cl"]
PROFILE = "matched_start_full40_v1"
PROTOCOL_DATE = "2026-10-05"
PROTOCOL_SHA256 = "bab990c3c60e136fa33c300858b9fe95b95353332712db84da1ced57e9ddb078"
DYNAMIC_MANIFEST_SHA256 = "bfa49031a1ab2b8e10a5cc5d95f1fbd8713e289c3ac8155838334e3c3d007dae"
NORMALIZATION_SHA256 = "f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1"
PREDECLARATION_SHA256 = "0478c8532bd2ded504ccd5f89303001eb8359f69b3036f296e31428a085d1272"
ORIGINAL_DYNAMIC_AUDITOR_SHA256 = "4e78d8473d1d0f93b25031a3bf9dcc0582f43604f7b1f67c65e6754f032af100"
MAX_WINDOW_CD_ZERO_FRACTION = 0.01
MAX_WINDOW_LIFT_ZERO_RMS_FRACTION = 0.025
MAX_DYNAMIC_NRMSE = 0.10
MAX_DYNAMIC_DELTA_MAE = 0.023
ACTION_LIMIT = 0.75
DELTA_LIMIT = 0.1
TIME_TOLERANCE = 2.0e-5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def finite(value: object, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {label}")
    return result


def verify_step_receipt(
    path: Path,
    *,
    step: str,
    checkpoint_sha256: str,
    required_files: tuple[Path, ...],
) -> dict:
    receipt = load(path)
    if (
        receipt.get("status") != "CONTROL_TRAIN16_POSTEVAL_STEP_COMPLETE"
        or receipt.get("step") != step
        or receipt.get("checkpoint_sha256") != checkpoint_sha256
    ):
        raise ValueError(f"{step} step receipt identity differs")
    table = receipt.get("sha256")
    if not isinstance(table, dict) or not table:
        raise ValueError(f"{step} step receipt SHA table is absent")
    by_name: dict[str, list[str]] = {}
    for raw_path, digest in table.items():
        by_name.setdefault(Path(raw_path).name, []).append(str(digest))
    for source in required_files:
        values = by_name.get(source.name, [])
        if len(values) != 1 or values[0] != sha256(source):
            raise ValueError(f"{step} step receipt does not bind {source.name}")
    return receipt


def verify_posteval_receipt(
    path: Path,
    *,
    checkpoint_sha256: str,
    required_files: tuple[Path, ...],
) -> dict:
    receipt = load(path)
    if (
        not str(receipt.get("status", "")).endswith("POSTEVAL_COMPLETE")
        or receipt.get("checkpoint_sha256") != checkpoint_sha256
        or receipt.get("frozen_test_accessed") is not False
        or receipt.get("ppo_auto_launched") is not False
    ):
        raise ValueError("posteval receipt identity or scope differs")
    table = receipt.get("sha256")
    if not isinstance(table, dict) or not table:
        raise ValueError("posteval receipt SHA table is absent")
    root = path.resolve().parent
    for relative, expected in table.items():
        source = (root / relative).resolve()
        try:
            source.relative_to(root)
        except ValueError as error:
            raise ValueError("posteval receipt path escapes its root") from error
        if not source.is_file() or sha256(source) != expected:
            raise ValueError(f"posteval receipt file differs: {relative}")
    for source in required_files:
        try:
            relative = str(source.resolve().relative_to(root))
        except ValueError as error:
            raise ValueError(f"required evidence is outside posteval root: {source}") from error
        if table.get(relative) != sha256(source):
            raise ValueError(f"posteval receipt does not bind {relative}")
    return receipt


def sampled_window(times: list, forces: list, duration: float = 6.15) -> dict:
    if len(times) != 101 or len(forces) != 101:
        raise ValueError("canonical window requires H100 plus the initial endpoint")
    time = [finite(item, "time") for item in times]
    if any(
        not math.isclose(b - a, 0.1, rel_tol=0.0, abs_tol=TIME_TOLERANCE)
        for a, b in zip(time[:-1], time[1:], strict=True)
    ):
        raise ValueError("canonical window time grid differs")
    end = time[-1]
    indices = [
        index
        for index, value in enumerate(time)
        if end - duration - 1.0e-9 <= value <= end + 1.0e-9
    ]
    if len(indices) != 62:
        raise ValueError("6.15-D/U sampled window must contain 62 endpoints")
    rows = []
    for index in indices:
        row = [finite(item, "force") for item in forces[index]]
        if len(row) != 4:
            raise ValueError("four force channels required")
        rows.append(row)
    means = [sum(row[channel] for row in rows) / len(rows) for channel in range(4)]
    rear_rms = math.sqrt(sum((row[3] - means[3]) ** 2 for row in rows) / len(rows))
    return {
        "sample_count": 62,
        "first_sample_time": time[indices[0]],
        "last_sample_time": time[indices[-1]],
        "sample_span_D_over_U": time[indices[-1]] - time[indices[0]],
        "requested_window_D_over_U": duration,
        "mean_total_cd": means[0] + means[2],
        "rear_cl_fluctuation_rms": rear_rms,
        "rear_cl_mean": means[3],
    }


def recompute_window(force_window: dict, checkpoint_sha256: str) -> dict:
    if (
        force_window.get("status") != "SAMPLED_FORCE_WINDOW_DIAGNOSTIC_COMPLETE"
        or force_window.get("model_sha256") != checkpoint_sha256
        or force_window.get("manifest_sha256") != DYNAMIC_MANIFEST_SHA256
        or force_window.get("normalization_sha256") != NORMALIZATION_SHA256
        or force_window.get("force_channels") != FORCE_CHANNELS
        or force_window.get("frozen_test_accessed") is not False
        or force_window.get("ppo_authorized") is not False
    ):
        raise ValueError("force-window evidence identity differs")
    source = force_window.get("cases")
    if (
        not isinstance(source, list)
        or len(source) != len(CASES)
        or {row.get("case") for row in source} != set(CASES)
    ):
        raise ValueError("force-window evidence must contain exactly dynamic6")
    rows = {row["case"]: row for row in source}
    windows = {}
    for name in CASES:
        row = rows[name]
        if row.get("horizon_steps") != 100:
            raise ValueError(f"H100 force-window evidence required: {name}")
        windows[name] = {
            "truth": sampled_window(row.get("times", []), row.get("true_forces", [])),
            "prediction": sampled_window(
                row.get("times", []), row.get("predicted_forces", [])
            ),
        }
    branches = []
    metric_all_pass = {
        "total_drag": True,
        "rear_cl_fluctuation_rms": True,
        "rear_cl_mean": True,
    }
    for name in CASES:
        phase = name.split("_")[-2]
        zero = windows[f"full40_dynamic_validation_{phase}_zero"]["truth"]
        truth = windows[name]["truth"]
        prediction = windows[name]["prediction"]
        maximum = {
            "total_drag": MAX_WINDOW_CD_ZERO_FRACTION * abs(zero["mean_total_cd"]),
            "rear_cl_fluctuation_rms": MAX_WINDOW_LIFT_ZERO_RMS_FRACTION
            * zero["rear_cl_fluctuation_rms"],
            "rear_cl_mean": MAX_WINDOW_LIFT_ZERO_RMS_FRACTION
            * zero["rear_cl_fluctuation_rms"],
        }
        if maximum["total_drag"] <= 0 or maximum["rear_cl_fluctuation_rms"] <= 0:
            raise ValueError(f"degenerate same-window zero scale: {phase}")
        errors = {
            "total_drag": abs(prediction["mean_total_cd"] - truth["mean_total_cd"]),
            "rear_cl_fluctuation_rms": abs(
                prediction["rear_cl_fluctuation_rms"]
                - truth["rear_cl_fluctuation_rms"]
            ),
            "rear_cl_mean": abs(prediction["rear_cl_mean"] - truth["rear_cl_mean"]),
        }
        passed = {key: errors[key] <= maximum[key] for key in errors}
        for key, value in passed.items():
            metric_all_pass[key] = metric_all_pass[key] and value
        branches.append(
            {
                "case": name,
                "truth": truth,
                "prediction": prediction,
                "absolute_errors": errors,
                "maximum_errors": maximum,
                "metric_pass": passed,
                "joint_pass": all(passed.values()),
            }
        )
    return {"metric_all_branches_pass": metric_all_pass, "branches": branches}


def audit_actions(data: Path) -> dict:
    import h5py
    import numpy as np

    manifest_path = data / "manifest.json"
    if sha256(manifest_path) != DYNAMIC_MANIFEST_SHA256:
        raise ValueError("dynamic6 manifest SHA differs")
    manifest = load(manifest_path)
    if (
        manifest.get("trajectory_counts")
        != {"train": 0, "validation": 6, "frozen_test": 0}
        or manifest.get("training_access") != "FORBIDDEN"
        or manifest.get("frozen_test_accessed") is not False
        or manifest.get("normalization_sha256") != NORMALIZATION_SHA256
    ):
        raise ValueError("dynamic6 manifest scope differs")
    files = sorted((data / "validation").glob("*.h5"))
    if {path.stem for path in files} != set(CASES) or len(files) != len(CASES):
        raise ValueError("dynamic6 HDF case matrix differs")
    reports = []
    for path in files:
        expected = manifest.get("hdf_sha256", {}).get(path.name)
        if not expected or sha256(path) != expected:
            raise ValueError(f"dynamic6 HDF SHA differs: {path.name}")
        with h5py.File(path, "r") as handle:
            omega = np.asarray(handle["omega"][:101], dtype=np.float64).reshape(-1)
            times = np.asarray(handle["time"][:101], dtype=np.float64).reshape(-1)
        if len(omega) != 101 or len(times) != 101:
            raise ValueError(f"H100 action sequence required: {path.stem}")
        if not np.isfinite(omega).all() or not np.isfinite(times).all():
            raise ValueError(f"non-finite action sequence: {path.stem}")
        if not np.allclose(np.diff(times), 0.1, rtol=0.0, atol=TIME_TOLERANCE):
            raise ValueError(f"action time grid differs: {path.stem}")
        max_abs = float(np.max(np.abs(omega)))
        max_delta = float(np.max(np.abs(np.diff(omega))))
        if max_abs > ACTION_LIMIT + 1.0e-7 or max_delta > DELTA_LIMIT + TIME_TOLERANCE:
            raise ValueError(f"action contract violated: {path.stem}")
        reports.append(
            {
                "case": path.stem,
                "hdf5_sha256": expected,
                "endpoint_count": 101,
                "horizon_steps": 100,
                "max_abs_omega": max_abs,
                "max_abs_delta_omega": max_delta,
            }
        )
    return {"cases": reports, "all_cases_pass": True}


def recompute_dynamic(evaluation: dict, segments: dict) -> dict:
    if (
        evaluation.get("split") != "validation"
        or evaluation.get("action_mode") != "observed"
        or evaluation.get("force_channels") != FORCE_CHANNELS
    ):
        raise ValueError("dynamic evaluation contract differs")
    rows = {row.get("case"): row for row in evaluation.get("cases", [])}
    if len(rows) != len(CASES) or set(rows) != set(CASES):
        raise ValueError("dynamic evaluation case matrix differs")
    horizons = {"1", "10", "50", "100"}
    h100 = []
    for name in CASES:
        metrics = rows[name].get("horizons", {})
        if set(metrics) != horizons:
            raise ValueError(f"dynamic horizons differ: {name}")
        for horizon, metric in metrics.items():
            if metric.get("stable") is not True or metric.get("failed_segments") != 0:
                raise ValueError(f"unstable rollout: {name}/H{horizon}")
        metric = metrics["100"]
        count = int(metric["segments"])
        if count != 101:
            raise ValueError(f"rolling H100 count must be 101: {name}")
        h100.append(
            (
                count,
                finite(metric["total_drag_rmse"], "total drag RMSE"),
                finite(metric["total_drag_target_rms"], "total drag target RMS"),
            )
        )
    denominator = sum(count * target * target for count, _, target in h100)
    if denominator <= 0:
        raise ValueError("rolling H100 reference norm is zero")
    pooled = math.sqrt(
        sum(count * error * error for count, error, _ in h100) / denominator
    )
    source_segments = segments.get("segments")
    if not isinstance(source_segments, list):
        raise ValueError("dynamic segments are absent")
    h100_rows = [row for row in source_segments if row.get("horizon") == 100]
    if len(h100_rows) != 606:
        raise ValueError("dynamic segments must contain exactly 606 rolling H100 rows")
    if any(
        sum(row.get("case") == name for row in h100_rows) != 101 for name in CASES
    ):
        raise ValueError("dynamic segments must contain 101 rolling H100 rows per case")
    start_rows = [row for row in h100_rows if row.get("start") == 0]
    if len(start_rows) != len(CASES):
        raise ValueError("strict start0 H100 rows must contain exactly six records")
    starts = {row.get("case"): row for row in start_rows}
    if len(starts) != len(CASES) or set(starts) != set(CASES):
        raise ValueError("strict start0 H100 matrix differs")
    pairs = []
    for phase in ("b01", "b05"):
        prefix = f"full40_dynamic_validation_{phase}_"
        zero = starts[prefix + "zero"]
        for profile in ("minus", "plus"):
            row = starts[prefix + profile]
            true_delta = finite(row["target_total_drag"], "target total drag") - finite(
                zero["target_total_drag"], "zero target total drag"
            )
            predicted_delta = finite(
                row["predicted_total_drag"], "predicted total drag"
            ) - finite(zero["predicted_total_drag"], "zero predicted total drag")
            pairs.append(
                {
                    "phase": phase,
                    "profile": profile,
                    "true_delta_cd": true_delta,
                    "predicted_delta_cd": predicted_delta,
                    "absolute_error": abs(predicted_delta - true_delta),
                }
            )
    delta_mae = sum(row["absolute_error"] for row in pairs) / len(pairs)
    return {
        "pooled_all_rolling_h100_total_cd_nrmse": pooled,
        "pooled_all_rolling_h100_segment_count": sum(row[0] for row in h100),
        "strict_start0_action_minus_zero_total_cd_mae": delta_mae,
        "strict_start0_pairs": pairs,
        "metric_pass": {
            "pooled_all_rolling_h100_total_cd_nrmse": pooled <= MAX_DYNAMIC_NRMSE,
            "strict_start0_action_minus_zero_total_cd_mae": delta_mae
            <= MAX_DYNAMIC_DELTA_MAE,
        },
    }


def common_receipt(
    *, checkpoint_sha256: str, producer: Path, protocol: Path, evidence: dict
) -> dict:
    return {
        "profile": PROFILE,
        "checkpoint_sha256": checkpoint_sha256,
        "validation_phases": ["b01", "b05"],
        "producer_script": str(producer.resolve()),
        "producer_script_sha256": sha256(producer),
        "protocol_path": str(protocol.resolve()),
        "protocol_sha256": sha256(protocol),
        "protocol_decision": "D012",
        "protocol_completion_date": PROTOCOL_DATE,
        "historical_characterization": (
            "prospective numerical completion of legacy compatibility schema; "
            "not historical preregistration and not an independent experiment"
        ),
        "source_evidence": evidence,
        "frozen_test_accessed": False,
        "ppo_authorized": False,
    }


def build_receipts(args: argparse.Namespace) -> tuple[dict, dict]:
    producer = Path(__file__).resolve()
    protocol = args.protocol.resolve()
    if sha256(protocol) != PROTOCOL_SHA256:
        raise ValueError("D012 protocol SHA differs")
    if sha256(args.predeclaration) != PREDECLARATION_SHA256:
        raise ValueError("dynamic6 predeclaration SHA differs")
    model = args.checkpoint_dir / f"FNO.0.{args.checkpoint_epoch}.mdlus"
    if sha256(model) != args.checkpoint_sha256:
        raise ValueError("candidate model SHA differs")
    force_path = args.force_window.resolve()
    evaluation_path = args.evaluation.resolve()
    segments_path = args.segments.resolve()
    dynamic_diagnostic_path = args.dynamic_diagnostic.resolve()
    physical_path = args.physical_qc.resolve()
    posteval = verify_posteval_receipt(
        args.posteval_receipt.resolve(),
        checkpoint_sha256=args.checkpoint_sha256,
        required_files=(force_path, evaluation_path, segments_path, dynamic_diagnostic_path),
    )
    force_step = verify_step_receipt(
        args.force_step_receipt.resolve(),
        step="force_window",
        checkpoint_sha256=args.checkpoint_sha256,
        required_files=(force_path,),
    )
    force_window = load(force_path)
    window = recompute_window(force_window, args.checkpoint_sha256)
    window_pass = all(window["metric_all_branches_pass"].values())
    window_common = common_receipt(
        checkpoint_sha256=args.checkpoint_sha256,
        producer=producer,
        protocol=protocol,
        evidence={
            "force_window": {"path": str(force_path), "sha256": sha256(force_path)},
            "force_window_step_receipt": {
                "path": str(args.force_step_receipt.resolve()),
                "sha256": sha256(args.force_step_receipt),
                "status": force_step["status"],
            },
            "posteval_receipt": {
                "path": str(args.posteval_receipt.resolve()),
                "sha256": sha256(args.posteval_receipt),
                "status": posteval["status"],
            },
            "candidate_model": {"path": str(model.resolve()), "sha256": sha256(model)},
            "dynamic_manifest": {
                "path": str((args.dynamic_data / "manifest.json").resolve()),
                "sha256": sha256(args.dynamic_data / "manifest.json"),
            },
        },
    )
    window_receipt = {
        **window_common,
        "status": (
            "FULL40_VALIDATION_CANONICAL_WINDOW_FIDELITY_PASS"
            if window_pass
            else "FULL40_VALIDATION_CANONICAL_WINDOW_FIDELITY_FAIL"
        ),
        "evidence_path": str(force_path),
        "evidence_sha256": sha256(force_path),
        "causal_window_seconds": 6.15,
        "sample_count": 62,
        "sample_span_D_over_U": 6.1,
        "normalization_sha256": NORMALIZATION_SHA256,
        "data_manifest_sha256": DYNAMIC_MANIFEST_SHA256,
        "thresholds": {
            "total_drag_error_fraction_of_same_window_zero_cd": 0.01,
            "rear_cl_fluctuation_rms_error_fraction_of_same_window_zero_rms": 0.025,
            "rear_cl_mean_error_fraction_of_same_window_zero_rms": 0.025,
            "all_branches_must_pass": True,
        },
        "total_drag_window_fidelity_pass": window["metric_all_branches_pass"]["total_drag"],
        "rear_cl_fluctuation_window_fidelity_pass": window[
            "metric_all_branches_pass"
        ]["rear_cl_fluctuation_rms"],
        "rear_cl_mean_bias_window_fidelity_pass": window["metric_all_branches_pass"][
            "rear_cl_mean"
        ],
        "branches": window["branches"],
        "shared_evidence_note": (
            "uses the same start0 H100 force trajectories as the development window "
            "audit; this receipt is a compatibility assertion, not new observations"
        ),
    }

    evaluation = load(evaluation_path)
    if int(evaluation.get("checkpoint_epoch", -1)) != args.checkpoint_epoch:
        raise ValueError("dynamic evaluation checkpoint epoch differs")
    dynamic_step = verify_step_receipt(
        args.dynamic_step_receipt.resolve(),
        step="dynamic6",
        checkpoint_sha256=args.checkpoint_sha256,
        required_files=(evaluation_path, segments_path),
    )
    dynamic = recompute_dynamic(evaluation, load(segments_path))
    diagnostic = load(dynamic_diagnostic_path)
    expected_diagnostic_status = (
        "DYNAMIC6_FNO_DIAGNOSTIC_PASS"
        if all(dynamic["metric_pass"].values())
        else "DYNAMIC6_FNO_DIAGNOSTIC_FAIL"
    )
    if (
        diagnostic.get("status") != expected_diagnostic_status
        or diagnostic.get("checkpoint_sha256") != args.checkpoint_sha256
        or diagnostic.get("dataset_manifest_sha256") != DYNAMIC_MANIFEST_SHA256
        or diagnostic.get("frozen_test_accessed") is not False
        or diagnostic.get("ppo_authorized") is not False
        or not math.isclose(
            finite(diagnostic.get("pooled_h100_total_cd_nrmse"), "diagnostic NRMSE"),
            dynamic["pooled_all_rolling_h100_total_cd_nrmse"],
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or not math.isclose(
            finite(
                diagnostic.get("strict_start0_h100_delta_total_cd_mae"),
                "diagnostic delta MAE",
            ),
            dynamic["strict_start0_action_minus_zero_total_cd_mae"],
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
    ):
        raise ValueError("stored dynamic6 diagnostic differs from strict recomputation")
    actions = audit_actions(args.dynamic_data)
    physical = load(physical_path)
    if (
        physical.get("status")
        != "FULL40_DYNAMIC_VALIDATION_REAL_OPENFOAM_QC_COMPLETE"
        or physical.get("predeclaration_sha256") != PREDECLARATION_SHA256
        or physical.get("frozen_test_accessed") is not False
        or set(physical.get("phases", {})) != {"b01", "b05"}
    ):
        raise ValueError("dynamic physical-QC phase scope differs")
    dynamic_pass = all(dynamic["metric_pass"].values()) and actions["all_cases_pass"]
    dynamic_common = common_receipt(
        checkpoint_sha256=args.checkpoint_sha256,
        producer=producer,
        protocol=protocol,
        evidence={
            "evaluation": {"path": str(evaluation_path), "sha256": sha256(evaluation_path)},
            "segments": {"path": str(segments_path), "sha256": sha256(segments_path)},
            "dynamic_step_receipt": {
                "path": str(args.dynamic_step_receipt.resolve()),
                "sha256": sha256(args.dynamic_step_receipt),
                "status": dynamic_step["status"],
            },
            "dynamic_diagnostic": {
                "path": str(dynamic_diagnostic_path),
                "sha256": sha256(dynamic_diagnostic_path),
                "status": diagnostic["status"],
            },
            "posteval_receipt": {
                "path": str(args.posteval_receipt.resolve()),
                "sha256": sha256(args.posteval_receipt),
                "status": posteval["status"],
            },
            "physical_qc": {"path": str(physical_path), "sha256": sha256(physical_path)},
            "predeclaration": {
                "path": str(args.predeclaration.resolve()),
                "sha256": sha256(args.predeclaration),
            },
            "candidate_model": {"path": str(model.resolve()), "sha256": sha256(model)},
            "dynamic_manifest": {
                "path": str((args.dynamic_data / "manifest.json").resolve()),
                "sha256": sha256(args.dynamic_data / "manifest.json"),
            },
        },
    )
    dynamic_receipt = {
        **dynamic_common,
        "status": (
            "FULL40_VALIDATION_DYNAMIC_ACTION_PASS"
            if dynamic_pass
            else "FULL40_VALIDATION_DYNAMIC_ACTION_FAIL"
        ),
        "evidence_path": str(evaluation_path),
        "evidence_sha256": sha256(evaluation_path),
        "max_abs_omega": ACTION_LIMIT,
        "max_delta_omega": DELTA_LIMIT,
        "minimum_horizon_steps": 100,
        "normalization_sha256": NORMALIZATION_SHA256,
        "data_manifest_sha256": DYNAMIC_MANIFEST_SHA256,
        "dynamic_action_validation_pass": dynamic_pass,
        "thresholds": {
            "pooled_all_rolling_h100_total_cd_nrmse_max": MAX_DYNAMIC_NRMSE,
            "strict_start0_action_minus_zero_total_cd_mae_max": MAX_DYNAMIC_DELTA_MAE,
        },
        "metrics": dynamic,
        "action_sequence_audit": actions,
        "original_predeclared_computation": {
            "commit": "72b62ac",
            "auditor": "cfd/tandem_cylinders/audit_full40_dynamic6_fno.py",
            "auditor_sha256_at_protocol_completion": ORIGINAL_DYNAMIC_AUDITOR_SHA256,
        },
        "non_equivalence_note": (
            "all-rolling H100 pooled NRMSE here is not the six-terminal-point NRMSE "
            "or sign/order requirement in the separate development admission"
        ),
    }
    return window_receipt, dynamic_receipt


def write_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-window", type=Path, required=True)
    parser.add_argument("--force-step-receipt", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--dynamic-diagnostic", type=Path, required=True)
    parser.add_argument("--posteval-receipt", type=Path, required=True)
    parser.add_argument("--dynamic-step-receipt", type=Path, required=True)
    parser.add_argument("--dynamic-data", type=Path, required=True)
    parser.add_argument("--physical-qc", type=Path, required=True)
    parser.add_argument("--predeclaration", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-epoch", type=int, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--window-output", type=Path, required=True)
    parser.add_argument("--dynamic-output", type=Path, required=True)
    args = parser.parse_args()
    if args.window_output.exists() or args.dynamic_output.exists():
        raise FileExistsError("canonical compatibility receipts never overwrite")
    if len(args.checkpoint_sha256) != 64:
        parser.error("checkpoint SHA-256 must contain 64 hexadecimal characters")
    window, dynamic = build_receipts(args)
    write_exclusive(args.window_output, window)
    write_exclusive(args.dynamic_output, dynamic)
    print(json.dumps({"window": window["status"], "dynamic": dynamic["status"]}))


if __name__ == "__main__":
    main()
