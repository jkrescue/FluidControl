#!/usr/bin/env python3
"""Read-only physical summary of the frozen nine-case commissioning panel.

This script is deliberately downstream of raw-transfer QC.  It refuses to
write an artifact unless all nine RAW_TRANSFER_VERIFIED receipts and the
complete aggregate QC report are present and consistent.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from itertools import pairwise
from pathlib import Path

import numpy as np

PHASE_MANIFEST_SHA256 = (
    "6279492bd3a79eff868a4333e1f642e3be4a4d58a78e46dc47dde040e4e39603"
)
PHASES = (0, 2, 4)
ACTIONS = (-0.75, 0.0, 0.75)
ACTION_LABELS = {-0.75: "m075", 0.0: "zero", 0.75: "p075"}
FORCE_DT = 0.005
EXPECTED_ANALYSIS_SAMPLES = 12001


def case_name(phase_bin: int, action: float) -> str:
    return f"matched_start_acquisition_train_b{phase_bin:02d}_{ACTION_LABELS[action]}"


def read_force_rows(case: Path, force_name: str) -> np.ndarray:
    samples: dict[float, np.ndarray] = {}
    paths = sorted(case.glob(f"postProcessing/{force_name}/*/coefficient.dat"))
    if not paths:
        raise FileNotFoundError(f"missing {force_name} output: {case}")
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            values = np.asarray([float(value) for value in line.split()], dtype=float)
            if len(values) < 8 or not np.isfinite(values).all():
                raise ValueError(f"invalid force record: {path}")
            key = round(float(values[0]), 8)
            if key in samples and not np.allclose(samples[key], values, rtol=0, atol=1e-10):
                raise ValueError(f"conflicting force record at t={key}: {case}")
            samples[key] = values
    return np.asarray([samples[key] for key in sorted(samples)], dtype=float)


def select_exact_window(rows: np.ndarray, begin: float, end: float) -> np.ndarray:
    selected = rows[(rows[:, 0] >= begin - 1e-9) & (rows[:, 0] <= end + 1e-9)]
    expected = np.linspace(begin, end, EXPECTED_ANALYSIS_SAMPLES)
    if len(selected) != EXPECTED_ANALYSIS_SAMPLES:
        raise ValueError(
            f"analysis window [{begin}, {end}] requires "
            f"{EXPECTED_ANALYSIS_SAMPLES} samples, got {len(selected)}"
        )
    if not np.allclose(selected[:, 0], expected, rtol=0, atol=1e-8):
        raise ValueError("analysis timestamps do not match the frozen 0.005 grid")
    return selected


def action_series(points: list[list[float]], time: np.ndarray) -> np.ndarray:
    point_time = np.asarray([row[0] for row in points], dtype=float)
    point_value = np.asarray([row[1] for row in points], dtype=float)
    if len(points) < 2 or not np.all(np.diff(point_time) > 0):
        raise ValueError("action_points must contain strictly increasing timestamps")
    if time[0] < point_time[0] - 1e-9 or time[-1] > point_time[-1] + 1e-9:
        raise ValueError("analysis window is outside action_points support")
    return np.interp(time, point_time, point_value)


def configured_max_rate(points: list[list[float]]) -> float:
    return max(abs(a1 - a0) / (t1 - t0) for (t0, a0), (t1, a1) in pairwise(points))


def branch_metrics(
    front: np.ndarray, rear: np.ndarray, omega: np.ndarray, action_rate: float
) -> dict[str, float | int]:
    if not np.array_equal(front[:, 0], rear[:, 0]) or len(front) != len(omega):
        raise ValueError("front/rear/action timelines differ")
    # OpenFOAM forceCoeffs columns: Time, Cd, Cd(f), Cd(r), Cl, ... CmPitch.
    cd_front = front[:, 1]
    cd_rear = rear[:, 1]
    cl_rear = rear[:, 4]
    cl_mean = float(np.mean(cl_rear))
    return {
        "sample_count": len(front),
        "mean_cd_front": float(np.mean(cd_front)),
        "mean_cd_rear": float(np.mean(cd_rear)),
        "mean_cd_total": float(np.mean(cd_front + cd_rear)),
        "mean_cl_rear": cl_mean,
        "rms_cl_rear_fluctuation": float(np.sqrt(np.mean((cl_rear - cl_mean) ** 2))),
        "mean_cm_pitch_rear": float(np.mean(rear[:, 7])),
        "mean_omega_times_cm_pitch_rear": float(np.mean(omega * rear[:, 7])),
        "action_mean": float(np.mean(omega)),
        "action_rms": float(np.sqrt(np.mean(omega**2))),
        "action_max_abs": float(np.max(np.abs(omega))),
        "action_configured_max_abs_rate": float(action_rate),
    }


def comparison(control: dict, zero: dict) -> dict[str, float | bool]:
    drag_reduction = 1.0 - control["mean_cd_total"] / zero["mean_cd_total"]
    fluctuation_ratio = (
        control["rms_cl_rear_fluctuation"] / zero["rms_cl_rear_fluctuation"]
    )
    mean_bias_ratio = (
        abs(control["mean_cl_rear"]) / zero["rms_cl_rear_fluctuation"]
    )
    return {
        "total_drag_reduction_fraction_positive_is_better": float(drag_reduction),
        "rear_cl_fluctuation_rms_ratio_to_zero": float(fluctuation_ratio),
        "abs_mean_rear_cl_over_zero_fluctuation_rms": float(mean_bias_ratio),
        "canonical_drag_check_ge_0p02": bool(drag_reduction >= 0.02),
        "canonical_rear_cl_fluctuation_check_le_1p05": bool(fluctuation_ratio <= 1.05),
        "canonical_mean_rear_cl_check_le_0p10_zero_clprime_rms": bool(
            mean_bias_ratio <= 0.10
        ),
        "canonical_joint_diagnostic_pass": bool(
            drag_reduction >= 0.02
            and fluctuation_ratio <= 1.05
            and mean_bias_ratio <= 0.10
        ),
    }


def load_and_validate_prerequisites(
    cases_dir: Path, receipt_dir: Path, aggregate_qc: Path
) -> dict[str, tuple[Path, dict]]:
    if not aggregate_qc.is_file():
        raise FileNotFoundError(f"missing aggregate QC: {aggregate_qc}")
    aggregate = json.loads(aggregate_qc.read_text(encoding="utf-8"))
    if (
        aggregate.get("status") != "MATCHED_START_9_CASE_COMMISSIONING_QC_PASS"
        or aggregate.get("complete_nine_case_panel") is not True
        or aggregate.get("phase_manifest_sha256") != PHASE_MANIFEST_SHA256
    ):
        raise ValueError("aggregate QC is not the complete reviewed nine-case panel")

    expected_names = {
        case_name(phase_bin, action) for phase_bin in PHASES for action in ACTIONS
    }
    aggregate_names = {row.get("case") for row in aggregate.get("cases", [])}
    if aggregate_names != expected_names or set(aggregate.get("transfer", {})) != expected_names:
        raise ValueError("aggregate QC does not enumerate exactly the frozen nine cases")

    panel: dict[str, tuple[Path, dict]] = {}
    for phase_bin in PHASES:
        source_hashes = []
        for action in ACTIONS:
            name = case_name(phase_bin, action)
            receipt_path = receipt_dir / f"{name}.json"
            if not receipt_path.is_file():
                raise FileNotFoundError(f"missing raw-transfer receipt: {receipt_path}")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (
                receipt.get("status") != "RAW_TRANSFER_VERIFIED"
                or receipt.get("case") != name
                or receipt.get("phase_manifest_sha256") != PHASE_MANIFEST_SHA256
                or receipt.get("coverage")
                != "raw OpenFOAM solver files only; VTK is not included"
            ):
                raise ValueError(f"invalid raw-transfer receipt: {name}")
            case = cases_dir / name
            config_path = case / "case_config.json"
            if not config_path.is_file():
                raise FileNotFoundError(f"missing case config: {config_path}")
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if (
                config.get("case") != name
                or config.get("phase_bin") != phase_bin
                or config.get("split") != "train"
                or float(config.get("action_target")) != action
                or config.get("phase_manifest_sha256") != PHASE_MANIFEST_SHA256
            ):
                raise ValueError(f"case config differs from frozen panel: {name}")
            source_hashes.append(config.get("source_state_sha256"))
            panel[name] = case, config
        if source_hashes[1:] != source_hashes[:-1]:
            raise ValueError(f"phase b{phase_bin:02d} branches are not matched-start")
    return panel


def build_report(cases_dir: Path, receipt_dir: Path, aggregate_qc: Path) -> dict:
    panel = load_and_validate_prerequisites(cases_dir, receipt_dir, aggregate_qc)
    phases = {}
    comparisons_by_action: dict[str, list[dict]] = {"m075": [], "p075": []}
    for phase_bin in PHASES:
        branches = {}
        for action in ACTIONS:
            name = case_name(phase_bin, action)
            case, config = panel[name]
            begin, end = (float(value) for value in config["analysis_window"])
            if not np.isclose(end - begin, 60.0, rtol=0, atol=1e-10):
                raise ValueError(f"analysis window is not the frozen final 60D/U: {name}")
            front = select_exact_window(read_force_rows(case, "forceFront"), begin, end)
            rear = select_exact_window(read_force_rows(case, "forceRear"), begin, end)
            omega = action_series(config["action_points"], front[:, 0])
            metrics = branch_metrics(
                front, rear, omega, configured_max_rate(config["action_points"])
            )
            metrics.update(
                {
                    "case": name,
                    "action_target": action,
                    "analysis_window": [begin, end],
                }
            )
            branches[ACTION_LABELS[action]] = metrics
        zero = branches["zero"]
        phase_comparisons = {}
        for label in ("m075", "p075"):
            item = comparison(branches[label], zero)
            phase_comparisons[label] = item
            comparisons_by_action[label].append(item)
        phases[f"b{phase_bin:02d}"] = {
            "split": "train",
            "matched_start_branches": branches,
            "same_phase_zero_comparisons": phase_comparisons,
        }

    descriptive_macro = {}
    for label, rows in comparisons_by_action.items():
        descriptive_macro[label] = {
            "phase_count": len(rows),
            "mean_total_drag_reduction_fraction": float(
                np.mean(
                    [row["total_drag_reduction_fraction_positive_is_better"] for row in rows]
                )
            ),
            "all_three_canonical_joint_diagnostics_pass": all(
                row["canonical_joint_diagnostic_pass"] for row in rows
            ),
            "interpretation": "descriptive train-phase commissioning aggregate only",
        }

    return {
        "status": "MATCHED_START_9_CASE_TRAIN_COMMISSIONING_PHYSICS_SUMMARY",
        "phase_manifest_sha256": PHASE_MANIFEST_SHA256,
        "source_aggregate_qc": str(aggregate_qc),
        "metric_contract": {
            "analysis_window": "predeclared final 60D/U for every branch",
            "force_sample_interval": FORCE_DT,
            "samples_per_branch": EXPECTED_ANALYSIS_SAMPLES,
            "total_drag": "mean(Cd_front + Cd_rear)",
            "rear_lift_fluctuation": "RMS(Cl_rear - mean(Cl_rear))",
            "mean_lift_bias": "abs(mean(Cl_rear)) / zero rear Cl-prime RMS",
            "power_proxy": (
                "mean(omega * CmPitch_rear); sign/normalization not promoted "
                "to physical power"
            ),
        },
        "phases": phases,
        "descriptive_macro_across_three_train_phases": descriptive_macro,
        "interpretation_guards": [
            "All phase bins are train commissioning bins from one baseline limit cycle.",
            "This is not independent generalization or validation evidence.",
            (
                "The actions are prescribed open-loop branches, not learned or "
                "online closed-loop control."
            ),
            "Canonical checks are diagnostics only and do not establish closed-loop success.",
            "No subwindow selection or post-hoc action selection is performed.",
        ],
    }


def write_json_exclusive_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(payload, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise ValueError(f"refusing to overwrite {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases-dir", type=Path, default=repo / "cfd/tandem_cylinders/cases"
    )
    parser.add_argument(
        "--receipt-dir",
        type=Path,
        default=repo / "artifacts/matched_start_acquisition/transfer_verified",
    )
    parser.add_argument(
        "--aggregate-qc",
        type=Path,
        default=repo / "artifacts/matched_start_acquisition/aggregate_qc/result.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo / "artifacts/matched_start_acquisition/physics_summary/result.json",
    )
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to((repo / "artifacts").resolve()):
        parser.error("output must be inside the repository artifacts directory")
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    # Build first: incomplete/invalid inputs cannot create an output or its parent.
    report = build_report(args.cases_dir, args.receipt_dir, args.aggregate_qc)
    write_json_exclusive_atomic(args.output, report)
    print(json.dumps({"status": report["status"], "phases": len(report["phases"])}, indent=2))


if __name__ == "__main__":
    main()
