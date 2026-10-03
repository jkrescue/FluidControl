#!/usr/bin/env python3
"""Recompute approved development window/dynamic gates from stepwise forces.

These 2026-10-04 development-admission thresholds are not the original formal
Gate, a confirmatory paper test, or the final dense-60D/U real-CFD acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path

MANIFEST_SHA = "bfa49031a1ab2b8e10a5cc5d95f1fbd8713e289c3ac8155838334e3c3d007dae"
NORMALIZATION_SHA = "f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1"
FORCE_CHANNELS = ["front_cd", "front_cl", "rear_cd", "rear_cl"]
CASES = tuple(
    f"full40_dynamic_validation_b{phase:02d}_{profile}"
    for phase in (1, 5)
    for profile in ("minus", "zero", "plus")
)
TIE_TOLERANCE = 1.0e-12
MAX_POOLED_ENDPOINT_TOTAL_CD_NRMSE = 0.10
MAX_ZERO_RELATIVE_ENDPOINT_DELTA_CD_MAE = 0.023
MAX_WINDOW_TOTAL_CD_ERROR_ZERO_FRACTION = 0.01
MAX_WINDOW_REAR_CL_RMS_ERROR_ZERO_FRACTION = 0.025
MAX_WINDOW_REAR_CL_MEAN_ERROR_ZERO_RMS_FRACTION = 0.025


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def finite(value, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {label}")
    return result


def sign(value: float) -> int:
    return int(value > TIE_TOLERANCE) - int(value < -TIE_TOLERANCE)


def window(times, forces, duration: float = 6.15) -> dict:
    if len(times) != 101 or len(forces) != 101:
        raise ValueError("development Gate requires exactly H100 plus initial sample")
    values = [finite(item, "time") for item in times]
    if any(
        not math.isclose(b - a, 0.1, rel_tol=0.0, abs_tol=2.0e-5)
        for a, b in zip(values[:-1], values[1:])
    ):
        raise ValueError("force diagnostic time grid differs")
    end = values[-1]
    indices = [
        index
        for index, value in enumerate(values)
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
    rear_rms = math.sqrt(
        sum((row[3] - means[3]) ** 2 for row in rows) / len(rows)
    )
    return {
        "sample_count": len(rows),
        "first_sample_time": values[indices[0]],
        "last_sample_time": values[indices[-1]],
        "sample_span_D_over_U": values[indices[-1]] - values[indices[0]],
        "requested_continuous_window_D_over_U": duration,
        "mean_total_cd": means[0] + means[2],
        "rear_cl_fluctuation_rms": rear_rms,
        "rear_cl_mean": means[3],
    }


def audit(force_window_path: Path, checkpoint_sha256: str) -> dict:
    document = load(force_window_path)
    if (
        document.get("status") != "SAMPLED_FORCE_WINDOW_DIAGNOSTIC_COMPLETE"
        or document.get("model_sha256") != checkpoint_sha256
        or document.get("manifest_sha256") != MANIFEST_SHA
        or document.get("normalization_sha256") != NORMALIZATION_SHA
        or document.get("force_channels") != FORCE_CHANNELS
        or document.get("frozen_test_accessed") is not False
        or document.get("ppo_authorized") is not False
    ):
        raise ValueError("force-window diagnostic identity differs")
    source = document.get("cases")
    if (
        not isinstance(source, list)
        or len(source) != len(CASES)
        or {row.get("case") for row in source} != set(CASES)
    ):
        raise ValueError("force-window diagnostic must contain exactly dynamic6")
    rows = {row["case"]: row for row in source}
    windows = {}
    endpoints = {}
    for name in CASES:
        row = rows[name]
        if row.get("horizon_steps") != 100:
            raise ValueError(f"H100 force diagnostic required: {name}")
        times = row.get("times")
        omega = [finite(item, "omega") for item in row.get("omega_endpoints", [])]
        truth = row.get("true_forces")
        prediction = row.get("predicted_forces")
        if len(omega) != 101 or max(abs(item) for item in omega) > 0.75 + 1.0e-7:
            raise ValueError(f"action magnitude differs: {name}")
        if max(abs(b - a) for a, b in zip(omega[:-1], omega[1:])) > 0.1 + 2.0e-5:
            raise ValueError(f"action slew differs: {name}")
        true_window = window(times, truth)
        predicted_window = window(times, prediction)
        windows[name] = {"truth": true_window, "prediction": predicted_window}
        endpoints[name] = {
            "truth": finite(truth[-1][0], "front Cd") + finite(truth[-1][2], "rear Cd"),
            "prediction": finite(prediction[-1][0], "front Cd")
            + finite(prediction[-1][2], "rear Cd"),
        }

    branch_rows = []
    for name in CASES:
        phase = name.split("_")[-2]
        zero = windows[f"full40_dynamic_validation_{phase}_zero"]["truth"]
        truth = windows[name]["truth"]
        prediction = windows[name]["prediction"]
        scales = {
            "total_cd": MAX_WINDOW_TOTAL_CD_ERROR_ZERO_FRACTION
            * abs(zero["mean_total_cd"]),
            "rear_cl_fluctuation_rms": MAX_WINDOW_REAR_CL_RMS_ERROR_ZERO_FRACTION
            * zero["rear_cl_fluctuation_rms"],
            "rear_cl_mean": MAX_WINDOW_REAR_CL_MEAN_ERROR_ZERO_RMS_FRACTION
            * zero["rear_cl_fluctuation_rms"],
        }
        if scales["total_cd"] <= 0.0 or scales["rear_cl_fluctuation_rms"] <= 0.0:
            raise ValueError(f"same-window zero reference scale is degenerate: {phase}")
        errors = {
            "total_cd": abs(prediction["mean_total_cd"] - truth["mean_total_cd"]),
            "rear_cl_fluctuation_rms": abs(
                prediction["rear_cl_fluctuation_rms"]
                - truth["rear_cl_fluctuation_rms"]
            ),
            "rear_cl_mean": abs(prediction["rear_cl_mean"] - truth["rear_cl_mean"]),
        }
        passes = {key: errors[key] <= scales[key] for key in errors}
        branch_rows.append(
            {
                "case": name,
                "truth": truth,
                "prediction": prediction,
                "absolute_errors": errors,
                "maximum_errors": scales,
                "margin_to_maximum": {
                    key: scales[key] - errors[key] for key in errors
                },
                "metric_pass": passes,
                "joint_pass": all(passes.values()),
            }
        )
    window_pass = all(row["joint_pass"] for row in branch_rows)

    endpoint_error_ss = sum(
        (row["prediction"] - row["truth"]) ** 2 for row in endpoints.values()
    )
    endpoint_reference_ss = sum(row["truth"] ** 2 for row in endpoints.values())
    if endpoint_reference_ss == 0.0:
        raise ValueError("endpoint total-Cd reference norm is zero")
    pooled_nrmse = math.sqrt(endpoint_error_ss / endpoint_reference_ss)
    delta_rows = []
    ordering_rows = []
    for phase in ("b01", "b05"):
        prefix = f"full40_dynamic_validation_{phase}_"
        zero = endpoints[prefix + "zero"]
        phase_rows = []
        for profile in ("minus", "zero", "plus"):
            item = endpoints[prefix + profile]
            phase_rows.append((profile, item))
            if profile != "zero":
                true_delta = item["truth"] - zero["truth"]
                predicted_delta = item["prediction"] - zero["prediction"]
                delta_rows.append(
                    {
                        "phase": phase,
                        "profile": profile,
                        "true_delta_cd": true_delta,
                        "predicted_delta_cd": predicted_delta,
                        "absolute_error": abs(predicted_delta - true_delta),
                        "sign_correct": None
                        if sign(true_delta) == 0
                        else sign(true_delta) == sign(predicted_delta),
                    }
                )
        for left_index, (left_name, left) in enumerate(phase_rows):
            for right_name, right in phase_rows[left_index + 1 :]:
                true_delta = right["truth"] - left["truth"]
                predicted_delta = right["prediction"] - left["prediction"]
                ordering_rows.append(
                    {
                        "phase": phase,
                        "profiles": [left_name, right_name],
                        "true_delta_cd": true_delta,
                        "predicted_delta_cd": predicted_delta,
                        "ordering_correct": None
                        if sign(true_delta) == 0
                        else sign(true_delta) == sign(predicted_delta),
                    }
                )
    delta_mae = sum(row["absolute_error"] for row in delta_rows) / len(delta_rows)
    signs = [row for row in delta_rows if row["sign_correct"] is not None]
    ordering = [row for row in ordering_rows if row["ordering_correct"] is not None]
    if not signs or not ordering:
        raise ValueError("all dynamic action comparisons are true ties")
    sign_accuracy = sum(bool(row["sign_correct"]) for row in signs) / len(signs)
    ordering_accuracy = sum(bool(row["ordering_correct"]) for row in ordering) / len(ordering)
    dynamic_checks = {
        "pooled_total_cd_nrmse": pooled_nrmse <= MAX_POOLED_ENDPOINT_TOTAL_CD_NRMSE,
        "zero_relative_delta_cd_mae": delta_mae
        <= MAX_ZERO_RELATIVE_ENDPOINT_DELTA_CD_MAE,
        "non_tie_sign_accuracy": sign_accuracy == 1.0,
        "non_tie_cross_action_ordering_accuracy": ordering_accuracy == 1.0,
    }
    dynamic_pass = all(dynamic_checks.values())
    return {
        "status": (
            "DYNAMIC_FNO_DEVELOPMENT_ADMISSION_PASS"
            if window_pass and dynamic_pass
            else "DYNAMIC_FNO_DEVELOPMENT_ADMISSION_FAIL"
        ),
        "standard_status": (
            "2026-10-04 development admission; not original predeclaration, "
            "not confirmatory paper Gate, not final physical acceptance"
        ),
        "checkpoint_sha256": checkpoint_sha256,
        "source_force_window_sha256": sha256(force_window_path),
        "sampled_window_definition": (
            "62 uniformly spaced 0.1-D/U endpoints span 6.1 D/U and discretely "
            "represent the requested trailing 6.15-D/U continuous interval"
        ),
        "window_gate": {
            "status": "PASS" if window_pass else "FAIL",
            "thresholds": {
                "total_cd_error_fraction_of_same_window_zero_cd": 0.01,
                "rear_cl_fluctuation_rms_error_fraction_of_same_window_zero_rms": 0.025,
                "rear_cl_mean_error_fraction_of_same_window_zero_rms": 0.025,
            },
            "all_branches_must_pass": True,
            "branches": branch_rows,
        },
        "dynamic_action_gate": {
            "status": "PASS" if dynamic_pass else "FAIL",
            "thresholds": {
                "pooled_endpoint_total_cd_nrmse_max": MAX_POOLED_ENDPOINT_TOTAL_CD_NRMSE,
                "zero_relative_endpoint_delta_cd_mae_max": MAX_ZERO_RELATIVE_ENDPOINT_DELTA_CD_MAE,
                "non_tie_sign_accuracy_min": 1.0,
                "non_tie_cross_action_ordering_accuracy_min": 1.0,
                "tie_tolerance": TIE_TOLERANCE,
            },
            "metrics": {
                "pooled_endpoint_total_cd_nrmse": pooled_nrmse,
                "zero_relative_endpoint_delta_cd_mae": delta_mae,
                "non_tie_sign_accuracy": sign_accuracy,
                "non_tie_sign_pairs": len(signs),
                "non_tie_cross_action_ordering_accuracy": ordering_accuracy,
                "non_tie_cross_action_ordering_pairs": len(ordering),
            },
            "quantitative_margins": {
                "pooled_endpoint_total_cd_nrmse_to_maximum": (
                    MAX_POOLED_ENDPOINT_TOTAL_CD_NRMSE - pooled_nrmse
                ),
                "zero_relative_endpoint_delta_cd_mae_to_maximum": (
                    MAX_ZERO_RELATIVE_ENDPOINT_DELTA_CD_MAE - delta_mae
                ),
                "non_tie_sign_accuracy_above_minimum": sign_accuracy - 1.0,
                "non_tie_cross_action_ordering_accuracy_above_minimum": (
                    ordering_accuracy - 1.0
                ),
            },
            "metric_pass": dynamic_checks,
            "delta_pairs": delta_rows,
            "ordering_pairs": ordering_rows,
        },
        "field_metric_policy": (
            "field relative-L2 remains separately reported; force/drag accuracy is not "
            "described as percent-level full-field accuracy"
        ),
        "final_acceptance_unchanged": (
            "paired real OpenFOAM, dense final 60 D/U: drag reduction >=2%, rear Cl-prime "
            "ratio <=1.05, absolute rear mean Cl / zero rear Cl-prime RMS <=0.10"
        ),
        "ppo_authorized": False,
        "frozen_test_accessed": False,
    }


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
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not re.fullmatch(r"[0-9a-f]{64}", args.checkpoint_sha256):
        parser.error("checkpoint SHA-256 must be 64 lowercase hexadecimal characters")
    payload = audit(args.force_window, args.checkpoint_sha256)
    write_exclusive(args.output, payload)
    print(json.dumps({key: value for key, value in payload.items() if key not in {"window_gate"}}, indent=2))


if __name__ == "__main__":
    main()
