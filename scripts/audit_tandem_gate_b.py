#!/usr/bin/env python3
"""Audit the tandem-cylinder surrogate at the autoregressive control gate.

This audit measures model fitness only.  It never interprets prediction accuracy
as a drag-reduction result; that claim requires phase-matched closed-loop CFD.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


HORIZONS = ("1", "10", "50", "100")
COUNTERFACTUAL_MODES = ("zero", "sign_flip", "shuffle")
FORCE_CHANNELS = ("front_cd", "front_cl", "rear_cd", "rear_cl")


def _finite_number(row: dict, key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"missing numeric metric: {key}")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"invalid metric {key}: {value}")
    return result


def _load(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if tuple(report.get("force_channels", ())) != FORCE_CHANNELS:
        raise ValueError(f"{path}: expected all four cylinder-force channels")
    if set(report.get("summary", {})) != set(HORIZONS):
        raise ValueError(f"{path}: expected horizons {HORIZONS}")
    return report


def _check(checks: dict, name: str, passed: bool, **evidence: object) -> None:
    checks[name] = {"passed": bool(passed), **evidence}


def audit_gate_b(
    observed: dict,
    counterfactuals: dict[str, dict],
    independent: dict,
    *,
    full_period_horizon: str = "100",
    max_total_drag_nrmse: float = 0.10,
) -> dict:
    """Return a fail-closed Gate-B decision with metric-level evidence."""
    reports = {"observed": observed, **counterfactuals, "independent": independent}
    epochs = {int(report["checkpoint_epoch"]) for report in reports.values()}
    if len(epochs) != 1:
        raise ValueError(f"checkpoint epoch mismatch: {sorted(epochs)}")
    if observed.get("action_mode") != "observed":
        raise ValueError("observed report has the wrong action mode")
    if independent.get("action_mode") != "observed":
        raise ValueError("independent-phase report must use observed actions")
    for mode in COUNTERFACTUAL_MODES:
        if counterfactuals[mode].get("action_mode") != mode:
            raise ValueError(f"{mode} report has the wrong action mode")
        if counterfactuals[mode].get("normalization_data") != observed.get(
            "normalization_data"
        ):
            raise ValueError(f"{mode} normalization data mismatch")
    if independent.get("normalization_data") != observed.get("normalization_data"):
        raise ValueError("independent-phase normalization data mismatch")

    observed_cases = [case["case"] for case in observed["cases"]]
    for mode in COUNTERFACTUAL_MODES:
        if [case["case"] for case in counterfactuals[mode]["cases"]] != observed_cases:
            raise ValueError(f"{mode} held-out case mismatch")

    checks: dict[str, dict] = {}
    for label, report in reports.items():
        for horizon in HORIZONS:
            row = report["summary"][horizon]
            stable = (
                bool(row.get("stable")) and int(row.get("failed_segments", -1)) == 0
            )
            _check(
                checks,
                f"{label}_{horizon}step_finite",
                stable,
                failed_segments=int(row.get("failed_segments", -1)),
                segments=int(row.get("segments", 0)),
            )

    for label, report in (("observed", observed), ("independent", independent)):
        for horizon in HORIZONS:
            row = report["summary"][horizon]
            drag = _finite_number(row, "total_drag_mae")
            persistence = _finite_number(row, "persistence_total_drag_mae")
            _check(
                checks,
                f"{label}_{horizon}step_drag_beats_persistence",
                drag < persistence,
                total_drag_mae=drag,
                persistence_total_drag_mae=persistence,
            )

    for horizon in HORIZONS:
        actual = _finite_number(observed["summary"][horizon], "total_drag_mae")
        for mode in COUNTERFACTUAL_MODES:
            altered = _finite_number(
                counterfactuals[mode]["summary"][horizon], "total_drag_mae"
            )
            _check(
                checks,
                f"{horizon}step_action_sensitivity_vs_{mode}",
                actual < altered,
                observed_total_drag_mae=actual,
                altered_total_drag_mae=altered,
            )

    for label, report in (("heldout", observed), ("independent_phase", independent)):
        nrmse = _finite_number(
            report["summary"][full_period_horizon], "total_drag_nrmse"
        )
        _check(
            checks,
            f"{label}_full_period_total_drag_nrmse",
            nrmse <= max_total_drag_nrmse,
            horizon_steps=int(full_period_horizon),
            total_drag_nrmse=nrmse,
            maximum=max_total_drag_nrmse,
        )

    passed = all(item["passed"] for item in checks.values())
    return {
        "status": "GATE_B_PASS" if passed else "GATE_B_NEEDS_MULTISTEP_RETRAINING",
        "scientific_scope": "surrogate_fitness_only_not_closed_loop_drag_reduction",
        "checkpoint_epoch": epochs.pop(),
        "objective": "predict system total drag with front/rear lift retained as safety observables",
        "full_period_proxy_horizon_steps": int(full_period_horizon),
        "max_total_drag_nrmse": max_total_drag_nrmse,
        "heldout_cases": observed_cases,
        "independent_phase_cases": [case["case"] for case in independent["cases"]],
        "checks": checks,
        "failed_checks": [name for name, item in checks.items() if not item["passed"]],
    }


def render_markdown(result: dict) -> str:
    decision = result["status"]
    failed = result["failed_checks"]
    lines = [
        "# Tandem-cylinder Gate-B surrogate audit",
        "",
        f"- Decision: `{decision}`",
        f"- Checkpoint epoch: `{result['checkpoint_epoch']}`",
        f"- Scope: `{result['scientific_scope']}`",
        f"- Full-period proxy: `{result['full_period_proxy_horizon_steps']}` steps",
        f"- Total-drag NRMSE ceiling: `{result['max_total_drag_nrmse']:.1%}`",
        "",
        "This result assesses whether the PhysicsNeMo world model is suitable for",
        "control-horizon use. It is not evidence of closed-loop drag reduction.",
        "",
        "## Failed checks",
        "",
    ]
    lines.extend(f"- `{name}`" for name in failed)
    if not failed:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Full-period evidence",
            "",
            "| Dataset | NRMSE | Threshold | Pass |",
            "| --- | ---: | ---: | :---: |",
        ]
    )
    for label in ("heldout", "independent_phase"):
        item = result["checks"][f"{label}_full_period_total_drag_nrmse"]
        lines.append(
            f"| {label} | {item['total_drag_nrmse']:.4f} | "
            f"{item['maximum']:.4f} | {'yes' if item['passed'] else 'no'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed", type=Path, required=True)
    parser.add_argument("--zero", type=Path, required=True)
    parser.add_argument("--sign-flip", type=Path, required=True)
    parser.add_argument("--shuffle", type=Path, required=True)
    parser.add_argument("--independent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--max-total-drag-nrmse", type=float, default=0.10)
    args = parser.parse_args()
    if args.output.exists() or args.markdown.exists():
        raise FileExistsError("refusing to overwrite an existing Gate-B report")
    result = audit_gate_b(
        _load(args.observed),
        {
            "zero": _load(args.zero),
            "sign_flip": _load(args.sign_flip),
            "shuffle": _load(args.shuffle),
        },
        _load(args.independent),
        max_total_drag_nrmse=args.max_total_drag_nrmse,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(result), encoding="utf-8")
    print(render_markdown(result))


if __name__ == "__main__":
    main()
