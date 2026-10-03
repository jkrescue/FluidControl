#!/usr/bin/env python3
"""QC the completed real-CFD full40 dynamic validation panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
CFD = PROJECT / "cfd/tandem_cylinders"
CASES = CFD / "cases"
sys.path.insert(0, str(PROJECT / "scripts"))
from run_tandem_phase_feedback_pair import (
    compare_metrics,
    force_metrics,
    read_force_window,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def window_metrics(case: Path, begin: float, end: float) -> dict:
    # Raw solver force starts at t0+dt; shift the inclusive lower bound by dt
    # so adjacent declared windows do not double-count their shared endpoint.
    front = read_force_window(case, "forceFront", begin + 0.005 - 1e-9, end)
    rear = read_force_window(case, "forceRear", begin + 0.005 - 1e-9, end)
    return force_metrics(front, rear)


def audit(predeclaration: Path) -> dict:
    predecl = load(predeclaration)
    if (
        predecl.get("status") != "FULL40_DYNAMIC_VALIDATION_PREDECLARED_NOT_EXECUTED"
        or predecl.get("frozen_test_accessed") is not False
        or predecl.get("matrix", {}).get("case_count") != 6
    ):
        raise ValueError("dynamic validation predeclaration differs")
    expected_names = {
        f"full40_dynamic_validation_b{phase:02d}_{role}"
        for phase in (1, 5)
        for role in ("minus", "zero", "plus")
    }
    if set(predecl.get("cases", {})) != expected_names:
        raise ValueError("dynamic validation case matrix differs")

    case_results = {}
    for name in sorted(expected_names):
        case = CASES / name
        expected = predecl["cases"][name]
        config = load(case / "case_config.json")
        marker = load(case / "solver_complete.full40_dynamic_validation.json")
        solver_qc_path = case / "solver_log_qc.full40_dynamic_validation.json"
        solver_qc = load(solver_qc_path)
        if (
            config.get("case") != name
            or config.get("panel") != "full40_dynamic_validation_v1"
            or config.get("split") != "validation"
            or marker.get("status")
            != "FULL40_DYNAMIC_VALIDATION_SOLVER_COMPLETED_PENDING_PANEL_QC"
            or marker.get("case") != name
            or marker.get("predeclaration_sha256") != sha256(predeclaration)
            or marker.get("solver_qc_sha256") != sha256(solver_qc_path)
            or solver_qc.get("steps") != 4000
        ):
            raise ValueError(f"case/solver provenance differs: {name}")
        for key, value in expected.items():
            if config.get(key) != value:
                raise ValueError(f"case contract differs: {name}/{key}")
        start, end = map(float, expected["run_window"])
        windows = {}
        for rel_begin, rel_end in expected["fixed_force_windows_elapsed"]:
            label = f"elapsed_{rel_begin:g}_{rel_end:g}"
            windows[label] = window_metrics(case, start + rel_begin, start + rel_end)
        case_results[name] = {
            "phase_bin": expected["phase_bin"],
            "profile": expected["profile"],
            "source_state_sha256": expected["source_state_sha256"],
            "action_metrics": expected["action_metrics"],
            "solver_qc": solver_qc,
            "windows": windows,
            "terminal_time": end,
        }

    phases = {}
    for phase in (1, 5):
        prefix = f"full40_dynamic_validation_b{phase:02d}_"
        rows = {role: case_results[prefix + role] for role in ("minus", "zero", "plus")}
        if len({json.dumps(row["source_state_sha256"], sort_keys=True) for row in rows.values()}) != 1:
            raise ValueError(f"phase b{phase:02d} does not have a byte-identical start")
        comparisons = {}
        rankings = {}
        for window in rows["zero"]["windows"]:
            zero = rows["zero"]["windows"][window]
            comparisons[window] = {
                role: compare_metrics(rows[role]["windows"][window], zero)
                for role in ("minus", "plus")
            }
            rankings[window] = sorted(
                (rows[role]["windows"][window]["total_cd_mean"], role)
                for role in ("minus", "zero", "plus")
            )
        phases[f"b{phase:02d}"] = {
            "strict_common_initial_state": True,
            "profiles": rows,
            "comparisons_to_paired_zero": comparisons,
            "true_cfd_total_cd_ranking": {
                window: [role for _, role in ranking]
                for window, ranking in rankings.items()
            },
        }
    return {
        "status": "FULL40_DYNAMIC_VALIDATION_REAL_OPENFOAM_QC_COMPLETE",
        "scope": "validation b01/b05 prescribed time-varying actions; not closed-loop PPO",
        "predeclaration": str(predeclaration.relative_to(PROJECT)),
        "predeclaration_sha256": sha256(predeclaration),
        "frozen_test_accessed": False,
        "phases": phases,
        "interpretation_guards": [
            "Only frame 0 within each phase is a strict counterfactual common state.",
            "Later equal frame indices are matched elapsed time but action-diverged states.",
            "Physical action benefit and FNO prediction accuracy require separate reports.",
            "This panel cannot authorize PPO without the unchanged formal validation gates.",
        ],
    }


def write_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predeclaration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_exclusive(args.output, audit(args.predeclaration.resolve()))
    print(args.output)


if __name__ == "__main__":
    main()
