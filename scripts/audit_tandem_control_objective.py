#!/usr/bin/env python3
"""Audit real-CFD control coverage and reward trade-offs from curated HDF5."""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


SPLITS = ("train", "validation", "test")
FORCE_CHANNELS = ("front_cd", "front_cl", "rear_cd", "rear_cl")
ACTION_BINS = np.linspace(-5.0, 5.0, 11)
ABS_RATE_BINS = np.asarray([0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
PHASE_BINS = np.linspace(0.0, 1.0, 9)
DEFAULT_SHEDDING_PERIOD = 6.154


@dataclass(frozen=True)
class Series:
    """Small physical series read from one curated trajectory."""

    time: np.ndarray
    omega: np.ndarray
    delta_omega: np.ndarray
    action_rate: np.ndarray
    force: np.ndarray


def finite(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"non-finite {name}")
    return result


def distribution(values: np.ndarray) -> dict[str, float | list[float]]:
    """Return compact descriptive statistics with explicit quantiles."""
    values = finite(values, "distribution input").reshape(-1)
    if len(values) == 0:
        raise ValueError("cannot summarize an empty array")
    return {
        "count": int(len(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "rms": float(np.sqrt(np.mean(np.square(values)))),
        "quantiles_0_05_25_50_75_95_100": np.quantile(
            values, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]
        ).tolist(),
    }


def histogram(values: np.ndarray, edges: np.ndarray) -> dict[str, object]:
    """Histogram with under/overflow so coverage accounting is fail-closed."""
    values = finite(values, "histogram input").reshape(-1)
    counts, _ = np.histogram(values, bins=edges)
    return {
        "edges": edges.tolist(),
        "counts": counts.tolist(),
        "underflow": int(np.count_nonzero(values < edges[0])),
        "overflow": int(np.count_nonzero(values > edges[-1])),
    }


def read_series(path: Path) -> tuple[Series, dict, float, float]:
    """Read only low-volume action/force datasets, never the full flow fields."""
    with h5py.File(path, "r") as handle:
        time = finite(handle["time"][:, 0], "time")
        omega = finite(handle["omega"][:, 0], "omega")
        force = finite(handle["force"][:], "force")
        config = json.loads(handle.attrs["config_json"])
    if force.shape != (len(time), 4):
        raise ValueError(
            f"{path}: expected force shape ({len(time)}, 4), got {force.shape}"
        )
    if len(time) < 2 or not np.all(np.diff(time) > 0.0):
        raise ValueError(f"{path}: time must be strictly increasing")
    delta = np.diff(omega)
    rate = delta / np.diff(time)
    start = float(config.get("source_restart_time", config.get("start_time", time[0])))
    return (
        Series(time, omega, delta, rate, force),
        config,
        start,
        float(time[-1] - time[0]),
    )


def load_force_coefficients(path: Path) -> np.ndarray:
    """Load OpenFOAM coefficient.dat columns Time, Cd and Cl."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) >= 5:
            rows.append((float(fields[0]), float(fields[1]), float(fields[4])))
    values = finite(np.asarray(rows), f"force coefficients in {path}")
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 2:
        raise ValueError(f"{path}: expected at least two Time/Cd/Cl rows")
    if not np.all(np.diff(values[:, 0]) > 0.0):
        raise ValueError(f"{path}: coefficient time must be strictly increasing")
    return values


def load_zero_baseline(case: Path) -> dict[str, np.ndarray]:
    return {
        name: load_force_coefficients(
            case / "postProcessing" / object_name / "0" / "coefficient.dat"
        )
        for name, object_name in (("front", "forceFront"), ("rear", "forceRear"))
    }


def phase_matched_baseline(
    time: np.ndarray, baseline: dict[str, np.ndarray]
) -> np.ndarray:
    """Interpolate the uncontrolled front/rear force at exact curated frame times."""
    aligned = []
    for name in ("front", "rear"):
        source = baseline[name]
        if time[0] < source[0, 0] - 1.0e-8 or time[-1] > source[-1, 0] + 1.0e-8:
            raise ValueError(
                f"zero-action {name} baseline does not cover {time[0]}..{time[-1]}"
            )
        aligned.extend(
            np.interp(time, source[:, 0], source[:, column]) for column in (1, 2)
        )
    return np.stack(aligned, axis=1)


def lift_metrics(values: np.ndarray) -> dict[str, float]:
    values = finite(values, "lift")
    mean = float(values.mean())
    return {
        "mean": mean,
        "rms_fluctuation": float(np.sqrt(np.mean(np.square(values - mean)))),
        "root_mean_square": float(np.sqrt(np.mean(np.square(values)))),
        "mean_abs": float(np.mean(np.abs(values))),
    }


def physical_objectives(
    series: Series,
    baseline_force: np.ndarray,
) -> dict[str, float]:
    """Compute the four dimensionless terms used by the Stage-C objective."""
    rear_cl = series.force[:, 3]
    total_cd = series.force[:, 0] + series.force[:, 2]
    baseline_total_cd = float(np.mean(baseline_force[:, 0] + baseline_force[:, 2]))
    baseline_rear_mean_abs_cl = float(np.mean(np.abs(baseline_force[:, 3])))
    if baseline_total_cd <= 0.0 or baseline_rear_mean_abs_cl <= 0.0:
        raise ValueError("phase-matched baseline normalizers must be positive")
    return {
        "total_cd_over_baseline": float(total_cd.mean() / baseline_total_cd),
        "rear_mean_abs_cl_over_baseline": float(
            np.mean(np.abs(rear_cl)) / baseline_rear_mean_abs_cl
        ),
        "omega_sq_over_25": float(np.mean(np.square(series.omega)) / 25.0),
        "delta_omega_sq_over_0p25": float(
            np.mean(np.square(series.delta_omega)) / 0.25
        ),
    }


def case_report(
    path: Path,
    dataset: str,
    split: str,
    zero_baseline: dict[str, np.ndarray],
    phase_reference_time: float,
    shedding_period: float,
) -> tuple[dict, Series]:
    series, config, start_time, duration = read_series(path)
    force = series.force
    rear_cl = force[:, 3]
    front_cl = force[:, 1]
    total_cd = force[:, 0] + force[:, 2]
    baseline_force = phase_matched_baseline(series.time, zero_baseline)
    baseline_total_cd = baseline_force[:, 0] + baseline_force[:, 2]
    baseline_front_lift = lift_metrics(baseline_force[:, 1])
    baseline_rear_lift = lift_metrics(baseline_force[:, 3])
    controlled_front_lift = lift_metrics(front_cl)
    controlled_rear_lift = lift_metrics(rear_cl)
    report = {
        "dataset": dataset,
        "split": split,
        "case": path.stem,
        "schedule_kind": config.get("schedule_kind", "unknown"),
        "frames": int(len(series.omega)),
        "transitions": int(len(series.delta_omega)),
        "start_time": start_time,
        "duration": duration,
        "initial_phase_cycles": float(
            ((start_time - phase_reference_time) / shedding_period) % 1.0
        ),
        "action": distribution(series.omega),
        "delta_omega": distribution(series.delta_omega),
        "action_rate": distribution(series.action_rate),
        "force_mean": {
            name: float(force[:, index].mean())
            for index, name in enumerate(FORCE_CHANNELS)
        },
        "force_std": {
            name: float(force[:, index].std())
            for index, name in enumerate(FORCE_CHANNELS)
        },
        "total_cd": distribution(total_cd),
        "front_cl": controlled_front_lift,
        "rear_cl": controlled_rear_lift,
        "front_cl_mean": controlled_front_lift["mean"],
        "front_cl_rms_fluctuation": controlled_front_lift["rms_fluctuation"],
        "front_mean_abs_cl": controlled_front_lift["mean_abs"],
        "rear_cl_mean": controlled_rear_lift["mean"],
        "rear_cl_rms_fluctuation": controlled_rear_lift["rms_fluctuation"],
        "rear_cl_root_mean_square": controlled_rear_lift["root_mean_square"],
        "rear_mean_abs_cl": controlled_rear_lift["mean_abs"],
        "mean_omega_sq": float(np.mean(np.square(series.omega))),
        "mean_delta_omega_sq": float(np.mean(np.square(series.delta_omega))),
        "phase_matched_zero_action_baseline": {
            "window": [float(series.time[0]), float(series.time[-1])],
            "samples": int(len(series.time)),
            "front_cd_mean": float(baseline_force[:, 0].mean()),
            "rear_cd_mean": float(baseline_force[:, 2].mean()),
            "total_cd_mean": float(baseline_total_cd.mean()),
            "front_cl": baseline_front_lift,
            "rear_cl": baseline_rear_lift,
        },
        "relative_to_phase_matched_zero_action": {
            "total_cd_ratio": float(total_cd.mean() / baseline_total_cd.mean()),
            "total_cd_change_fraction": float(
                total_cd.mean() / baseline_total_cd.mean() - 1.0
            ),
            "front_cl_rms_ratio": float(
                controlled_front_lift["rms_fluctuation"]
                / baseline_front_lift["rms_fluctuation"]
            ),
            "rear_cl_rms_ratio": float(
                controlled_rear_lift["rms_fluctuation"]
                / baseline_rear_lift["rms_fluctuation"]
            ),
        },
        "objectives": physical_objectives(series, baseline_force),
    }
    return report, series


def aggregate_series(series: list[Series]) -> Series:
    if not series:
        raise ValueError("cannot aggregate zero trajectories")
    return Series(
        time=np.concatenate([item.time for item in series]),
        omega=np.concatenate([item.omega for item in series]),
        delta_omega=np.concatenate([item.delta_omega for item in series]),
        action_rate=np.concatenate([item.action_rate for item in series]),
        force=np.concatenate([item.force for item in series]),
    )


def coverage_report(series: Series) -> dict[str, object]:
    return {
        "action": distribution(series.omega),
        "action_histogram": histogram(series.omega, ACTION_BINS),
        "delta_omega": distribution(series.delta_omega),
        "action_rate": distribution(series.action_rate),
        "absolute_action_rate_histogram": histogram(
            np.abs(series.action_rate), ABS_RATE_BINS
        ),
    }


def is_dominated(candidate: dict, cases: list[dict]) -> bool:
    keys = tuple(candidate["objectives"])
    point = np.asarray([candidate["objectives"][key] for key in keys])
    for other in cases:
        if other is candidate:
            continue
        rival = np.asarray([other["objectives"][key] for key in keys])
        if np.all(rival <= point) and np.any(rival < point):
            return True
    return False


def pareto_audit(cases: list[dict]) -> dict[str, object]:
    """Build an offline screening grid; this does not claim policy optimality."""
    weights = itertools.product(
        (0.0, 0.1, 0.2, 0.3, 0.5),
        (0.0, 0.01, 0.02, 0.05),
        (0.0, 0.005, 0.01, 0.02),
    )
    scans = []
    for lift_weight, action_weight, rate_weight in weights:
        ranked = []
        for case in cases:
            terms = case["objectives"]
            cost = (
                terms["total_cd_over_baseline"]
                + lift_weight * terms["rear_mean_abs_cl_over_baseline"]
                + action_weight * terms["omega_sq_over_25"]
                + rate_weight * terms["delta_omega_sq_over_0p25"]
            )
            ranked.append(
                {
                    "dataset": case["dataset"],
                    "split": case["split"],
                    "case": case["case"],
                    "mean_cost": float(cost),
                }
            )
        ranked.sort(key=lambda row: row["mean_cost"])
        scans.append(
            {
                "weights": {
                    "rear_mean_abs_cl": lift_weight,
                    "omega_sq": action_weight,
                    "delta_omega_sq": rate_weight,
                },
                "best_five_open_loop_trajectories": ranked[:5],
            }
        )
    frontier = [
        {
            "dataset": case["dataset"],
            "split": case["split"],
            "case": case["case"],
            "objectives": case["objectives"],
        }
        for case in cases
        if not is_dominated(case, cases)
    ]
    return {
        "interpretation": (
            "Open-loop trajectory screening only; rankings are not policy comparisons "
            "because schedules and initial phases differ."
        ),
        "objective_formula": (
            "mean(total_cd)/baseline + w_lift*mean(abs(rear_cl))/baseline + "
            "w_action*mean((omega/5)^2) + "
            "w_rate*mean((delta_omega/0.5)^2)"
        ),
        "nondominated_open_loop_trajectories": frontier,
        "weight_scan": scans,
    }


def audit(
    roots: list[Path],
    baseline_case: Path,
    phase_reference_time: float = 80.0,
    shedding_period: float = DEFAULT_SHEDDING_PERIOD,
) -> dict:
    if shedding_period <= 0.0:
        raise ValueError("shedding period must be positive")
    zero_baseline = load_zero_baseline(baseline_case)
    cases: list[dict] = []
    raw_by_group: dict[tuple[str, str], list[Series]] = {}
    manifests = {}
    for root in roots:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        dataset = str(manifest["profile"])
        if dataset in manifests:
            raise ValueError(f"duplicate dataset profile: {dataset}")
        manifests[dataset] = manifest
        for split in SPLITS:
            paths = sorted((root / split).glob("*.h5"))
            expected = int(manifest["trajectory_counts"][split])
            if len(paths) != expected:
                raise ValueError(
                    f"{dataset}/{split}: expected {expected}, found {len(paths)}"
                )
            for path in paths:
                row, series = case_report(
                    path,
                    dataset,
                    split,
                    zero_baseline,
                    phase_reference_time,
                    shedding_period,
                )
                cases.append(row)
                raw_by_group.setdefault((dataset, split), []).append(series)

    groups = {}
    for (dataset, split), items in sorted(raw_by_group.items()):
        joined = aggregate_series(items)
        force = joined.force
        front_cl = force[:, 1]
        rear_cl = force[:, 3]
        groups[f"{dataset}/{split}"] = {
            "trajectory_count": len(items),
            "coverage": coverage_report(joined),
            "force_mean": {
                name: float(force[:, index].mean())
                for index, name in enumerate(FORCE_CHANNELS)
            },
            "total_cd": distribution(force[:, 0] + force[:, 2]),
            "front_cl": lift_metrics(front_cl),
            "rear_cl_mean": float(rear_cl.mean()),
            "rear_cl_rms_fluctuation": float(
                np.sqrt(np.mean(np.square(rear_cl - rear_cl.mean())))
            ),
            "rear_mean_abs_cl": float(np.mean(np.abs(rear_cl))),
            "mean_omega_sq": float(np.mean(np.square(joined.omega))),
            "mean_delta_omega_sq": float(np.mean(np.square(joined.delta_omega))),
        }

    phases = np.asarray([case["initial_phase_cycles"] for case in cases])
    return {
        "status": "REAL_CFD_CONTROL_OBJECTIVE_AUDITED",
        "provenance": {
            "source": "curated real OpenFOAM HDF5 force/action/time arrays",
            "flow_fields_loaded": False,
            "force_channels": list(FORCE_CHANNELS),
            "dataset_profiles": list(manifests),
        },
        "normalizers": {
            "method": "per-trajectory phase-matched zero-action window",
            "source_case": str(baseline_case),
            "source_force_objects": ["forceFront", "forceRear"],
            "alignment": "linear interpolation at exact curated HDF5 frame times",
        },
        "phase_definition": {
            "reference_time": phase_reference_time,
            "uncontrolled_shedding_period": shedding_period,
            "formula": "((source_restart_time-reference_time)/period) modulo 1",
            "histogram": histogram(phases, PHASE_BINS),
            "unique_initial_phase_cycles": np.unique(np.round(phases, 8)).tolist(),
        },
        "groups": groups,
        "cases": cases,
        "pareto": pareto_audit(cases),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--baseline-case",
        type=Path,
        default=Path("cfd/tandem_cylinders/cases/tandem_backward_dt005"),
    )
    parser.add_argument("--phase-reference-time", type=float, default=80.0)
    parser.add_argument(
        "--shedding-period", type=float, default=DEFAULT_SHEDDING_PERIOD
    )
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if not output.is_relative_to(project / "artifacts"):
        parser.error("output must be under project artifacts")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    report = audit(
        [path.resolve() for path in args.data],
        args.baseline_case.resolve(),
        args.phase_reference_time,
        args.shedding_period,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(output),
                "cases": len(report["cases"]),
                "groups": len(report["groups"]),
                "pareto_points": len(
                    report["pareto"]["nondominated_open_loop_trajectories"]
                ),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
