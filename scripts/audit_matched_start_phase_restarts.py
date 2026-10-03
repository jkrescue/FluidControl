#!/usr/bin/env python3
"""Audit real baseline restarts for a predeclared matched-start phase panel.

This script only reads the uncontrolled OpenFOAM baseline.  Linear interpolation
is used solely to locate scalar lift zero crossings; selected CFD states are
always existing on-disk U/p restart files and are never interpolated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

PHASE_WINDOW = (80.0, 160.0)
CANDIDATE_WINDOW = (100.0, 160.0)
RESTART_CADENCE = 2.0
PHASE_BINS = 8
MAX_PHASE_ERROR = math.pi / 16.0
SPLIT_BY_BIN = {
    0: "train",
    1: "validation",
    2: "train",
    3: "frozen_test",
    4: "train",
    5: "validation",
    6: "train",
    7: "frozen_test",
}
COMMISSIONING_BINS = (0, 2, 4)
FULL_ACTIONS = (-0.75, -0.375, 0.0, 0.375, 0.75)
COMMISSIONING_ACTIONS = (-0.75, 0.0, 0.75)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _combined_state_sha256(u_path: Path, p_path: Path) -> str:
    digest = hashlib.sha256()
    for label, path in ((b"U\0", u_path), (b"p\0", p_path)):
        digest.update(label)
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _circular_difference(angle: float, target: float) -> float:
    return math.atan2(math.sin(angle - target), math.cos(angle - target))


def _load_rear_lift(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[1] < 5:
        raise ValueError("rear coefficient file must contain Time and Cl columns")
    time = data[:, 0]
    rear_cl = data[:, 4]
    if len(time) < 3 or not np.all(np.isfinite(time)) or not np.all(np.isfinite(rear_cl)):
        raise ValueError("rear coefficient data are insufficient or non-finite")
    if not np.all(np.diff(time) > 0):
        raise ValueError("rear coefficient timestamps must be strictly increasing")
    return time, rear_cl


def _positive_crossings(time: np.ndarray, signal: np.ndarray) -> np.ndarray:
    centered = signal - float(np.mean(signal))
    indices = np.flatnonzero((centered[:-1] < 0.0) & (centered[1:] >= 0.0))
    crossings = []
    for index in indices:
        dt = time[index + 1] - time[index]
        dy = centered[index + 1] - centered[index]
        if dy <= 0:
            continue
        crossings.append(time[index] - centered[index] * dt / dy)
    return np.asarray(crossings, dtype=float)


def _phase_at(time: float, crossings: np.ndarray, median_period: float) -> float:
    """Piecewise phase between observed positive crossings.

    Candidate states outside the observed crossing bracket are rejected by the
    caller, so this function never extrapolates a restart phase.
    """
    index = int(np.searchsorted(crossings, time, side="right") - 1)
    if index < 0 or index >= len(crossings) - 1:
        raise ValueError("candidate restart is not bracketed by observed crossings")
    local_period = crossings[index + 1] - crossings[index]
    if not math.isfinite(local_period) or local_period <= 0:
        local_period = median_period
    return float(2.0 * math.pi * (time - crossings[index]) / local_period)


def _autocorrelation(signal: np.ndarray, lag: int) -> dict[str, float | int]:
    if lag <= 0 or lag >= len(signal):
        raise ValueError("autocorrelation lag outside signal")
    left = signal[:-lag]
    right = signal[lag:]
    correlation = float(np.corrcoef(left, right)[0, 1])
    scale = float(np.std(left))
    nrmse = float(np.sqrt(np.mean((left - right) ** 2)) / scale)
    return {"lag_samples": lag, "correlation": correlation, "nrmse": nrmse}


def audit_baseline(case_dir: Path) -> dict:
    force_path = case_dir / "postProcessing" / "forceRear" / "0" / "coefficient.dat"
    time, rear_cl = _load_rear_lift(force_path)
    phase_mask = (time >= PHASE_WINDOW[0]) & (time <= PHASE_WINDOW[1])
    phase_time = time[phase_mask]
    phase_cl = rear_cl[phase_mask]
    if len(phase_time) < 3:
        raise ValueError("phase window has insufficient rear-lift samples")
    time_steps = np.diff(phase_time)
    sample_dt = float(np.median(time_steps))
    if np.max(np.abs(time_steps - sample_dt)) > 1.0e-8:
        raise ValueError("phase-window force sampling is not uniform")

    crossings = _positive_crossings(phase_time, phase_cl)
    if len(crossings) < 4:
        raise ValueError("fewer than four positive rear-lift crossings")
    periods = np.diff(crossings)
    median_period = float(np.median(periods))

    restart_rows = []
    for path in case_dir.iterdir():
        if not path.is_dir():
            continue
        try:
            restart_time = float(path.name)
        except ValueError:
            continue
        if not (CANDIDATE_WINDOW[0] <= restart_time <= CANDIDATE_WINDOW[1]):
            continue
        cadence_units = restart_time / RESTART_CADENCE
        if abs(cadence_units - round(cadence_units)) > 1.0e-9:
            continue
        # Fail closed: a real state must exist and its phase must be bracketed by
        # two observed crossings.  No state or phase extrapolation is allowed.
        u_path = path / "U"
        p_path = path / "p"
        if not u_path.is_file() or not p_path.is_file():
            continue
        if not (crossings[0] <= restart_time < crossings[-1]):
            continue
        phase = _phase_at(restart_time, crossings, median_period)
        restart_rows.append(
            {
                "restart_time": restart_time,
                "phase_rad": phase,
                "u_sha256": _sha256(u_path),
                "p_sha256": _sha256(p_path),
                "state_sha256": _combined_state_sha256(u_path, p_path),
            }
        )
    restart_rows.sort(key=lambda row: row["restart_time"])

    selections = []
    coverage_ok = True
    for phase_bin in range(PHASE_BINS):
        target = 2.0 * math.pi * phase_bin / PHASE_BINS
        ranked = []
        for row in restart_rows:
            signed_error = _circular_difference(row["phase_rad"], target)
            ranked.append(
                {
                    **row,
                    "signed_phase_error_rad": signed_error,
                    "absolute_phase_error_rad": abs(signed_error),
                }
            )
        ranked.sort(key=lambda row: (row["absolute_phase_error_rad"], row["restart_time"]))
        selected = ranked[0] if ranked else None
        bin_ok = bool(selected and selected["absolute_phase_error_rad"] <= MAX_PHASE_ERROR)
        coverage_ok = coverage_ok and bin_ok
        selections.append(
            {
                "phase_bin": phase_bin,
                "target_phase_rad": target,
                "split": SPLIT_BY_BIN[phase_bin],
                "commissioning": phase_bin in COMMISSIONING_BINS,
                "coverage_pass": bin_ok,
                "selected": selected,
                "three_nearest_real_restarts": ranked[:3],
            }
        )

    selected_rows = [row["selected"] for row in selections if row["selected"]]
    unique_times = len({row["restart_time"] for row in selected_rows}) == PHASE_BINS
    unique_hashes = len({row["state_sha256"] for row in selected_rows}) == PHASE_BINS
    selected_phases = sorted(row["phase_rad"] for row in selected_rows)
    phase_gaps = []
    if len(selected_phases) == PHASE_BINS:
        wrapped = selected_phases + [selected_phases[0] + 2.0 * math.pi]
        phase_gaps = [wrapped[index + 1] - wrapped[index] for index in range(PHASE_BINS)]

    periodicity = {}
    for multiple in (1, 2, 3):
        lag = int(round(multiple * median_period / sample_dt))
        periodicity[str(multiple)] = _autocorrelation(phase_cl, lag)

    go = coverage_ok and unique_times and unique_hashes and len(restart_rows) >= PHASE_BINS
    return {
        "status": "MATCHED_START_PHASE_RESTART_AUDIT_COMPLETE",
        "decision": "GO_9_CASE_COMMISSIONING" if go else "NO_GO_REGENERATE_BASELINE_RESTARTS",
        "scope": "baseline restart phase feasibility only; no controlled CFD and no frozen-test fields read",
        "algorithm": {
            "phase_signal": "rear-cylinder raw Cl from forceRear coefficient.dat",
            "phase_window": list(PHASE_WINDOW),
            "centering": "arithmetic mean over fixed phase window",
            "phase_origin": "positive-going mean crossing; scalar crossing time linearly interpolated",
            "phase_between_crossings": "piecewise linear fraction between adjacent observed crossings",
            "state_interpolation": False,
            "candidate_window": list(CANDIDATE_WINDOW),
            "restart_cadence": RESTART_CADENCE,
            "observed_crossing_bracket_required": True,
            "phase_bins": PHASE_BINS,
            "max_absolute_phase_error_rad": MAX_PHASE_ERROR,
            "split_by_bin": {str(key): value for key, value in SPLIT_BY_BIN.items()},
        },
        "signal_qc": {
            "force_path": str(force_path),
            "force_sha256": _sha256(force_path),
            "sample_dt": sample_dt,
            "samples": int(len(phase_time)),
            "rear_cl_mean": float(np.mean(phase_cl)),
            "positive_crossing_times": crossings.tolist(),
            "periods": periods.tolist(),
            "median_period": median_period,
            "period_std": float(np.std(periods)),
            "periodicity_at_nearest_sample_lags": periodicity,
        },
        "restart_qc": {
            "eligible_real_restart_count": len(restart_rows),
            "selected_restart_times_unique": unique_times,
            "selected_state_hashes_unique": unique_hashes,
            "selected_phase_gap_min_rad": min(phase_gaps) if phase_gaps else None,
            "selected_phase_gap_max_rad": max(phase_gaps) if phase_gaps else None,
            "all_selected_have_real_u_and_p": len(selected_rows) == PHASE_BINS,
        },
        "selections": selections,
        "commissioning": {
            "phase_bins": list(COMMISSIONING_BINS),
            "actions": list(COMMISSIONING_ACTIONS),
            "case_count": len(COMMISSIONING_BINS) * len(COMMISSIONING_ACTIONS),
        },
        "full_seed_matrix": {
            "actions": list(FULL_ACTIONS),
            "case_count": PHASE_BINS * len(FULL_ACTIONS),
            "authorization": "not authorized by this audit",
        },
        "limitations": [
            "all phase bins lie on one uncontrolled limit cycle and are not independent physical conditions",
            "high periodic autocorrelation means cycle-separated states are near repeats",
            "phase interpolation applies only to the scalar diagnostic; CFD U/p states are existing files",
            "fixed plateau actions do not cover dynamic closed-loop action primitives",
        ],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-case",
        type=Path,
        default=Path("cfd/tandem_cylinders/cases/tandem_backward_dt005"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/tandem_cylinders/"
            "matched_start_phase_restart_predeclared_v3_20261003.json"
        ),
    )
    return parser.parse_args()


def write_audit(result: dict, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    result = audit_baseline(args.baseline_case)
    write_audit(result, args.output)
    print(json.dumps({"decision": result["decision"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
