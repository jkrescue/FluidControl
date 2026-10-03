#!/usr/bin/env python3
"""Build the fixed-checkpoint low-action validation decision from saved outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LABELS = ("v3_h20_parent", "v4_h20_candidate")
HORIZONS = ("1", "10", "50", "100")


def derive_decision(v3_h100: dict, v4_h100: dict) -> dict:
    checks = {
        "v4_h100_pooled_total_drag_nrmse_at_most_10pct": (
            v4_h100["pooled_total_drag_nrmse"] <= 0.10
        ),
        "v4_h100_beats_persistence_total_drag_mae": (
            v4_h100["total_drag_mae"] < v4_h100["persistence_total_drag_mae"]
        ),
        "v4_h100_pooled_total_drag_nrmse_no_worse_than_v3": (
            v4_h100["pooled_total_drag_nrmse"] <= v3_h100["pooled_total_drag_nrmse"]
        ),
    }
    if all(checks.values()):
        decision = "ELIGIBLE_FOR_PAIRED_ACTION_RANKING_AUDIT"
        reason = (
            "v4 passes the H100 10% NRMSE gate, beats persistence in total-drag "
            "MAE, and is no worse than v3 on the same validation profile; this "
            "only permits paired action drag-difference/ranking audit and does "
            "not pass Gate-C or authorize closed-loop promotion"
        )
    else:
        decision = "DO_NOT_PROMOTE_V4_TO_CLOSED_LOOP"
        failures = [name for name, passed in checks.items() if not passed]
        reason = "failed derived H100 checks: " + ", ".join(failures)
    return {"decision": decision, "decision_checks": checks, "reason": reason}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    models = {}
    for label in LABELS:
        directory = args.evaluations / label
        evaluation = json.loads((directory / "evaluation.json").read_text())
        pooled = json.loads((directory / "pooled_audit.json").read_text())
        provenance = json.loads((directory / "evaluation_provenance.json").read_text())
        if evaluation["split"] != "validation" or provenance["frozen_test_status"] != "NOT_ACCESSED":
            raise ValueError(f"{label}: split/provenance guard failed")
        horizons = {}
        for horizon in HORIZONS:
            strict = pooled["horizons"][horizon]["pooled"]
            summary = evaluation["summary"][horizon]
            if not summary["stable"] or summary["failed_segments"] != 0:
                raise ValueError(f"{label}: unstable horizon {horizon}")
            horizons[horizon] = {
                "segments": strict["segments"],
                "pooled_total_drag_nrmse": strict["total_drag_nrmse_pooled"],
                "state_mae_physical_units": strict["state_mae_physical_units"],
                "total_drag_mae": strict["total_drag_mae"],
                "persistence_total_drag_mae": summary["persistence_total_drag_mae"],
                "beats_persistence_total_drag_mae": strict["total_drag_mae"] < summary["persistence_total_drag_mae"],
                "rear_cd_mae": strict["rear_cd_mae"],
                "rear_cl_mae": strict["rear_cl_mae"],
            }
        models[label] = {
            "checkpoint_epoch": evaluation["checkpoint_epoch"],
            "normalization_data": evaluation["normalization_data"],
            "provenance": provenance,
            "horizons": horizons,
        }
    v3_h100 = models["v3_h20_parent"]["horizons"]["100"]
    v4_h100 = models["v4_h20_candidate"]["horizons"]["100"]
    derived = derive_decision(v3_h100, v4_h100)
    result = {
        "status": "LOW_ACTION_PHASE94_FIXED_CHECKPOINT_DECISION_COMPLETE",
        "profile": "control_gap_low_action_phase94_validation_v1",
        "scope": "independent validation-only OOD diagnostic; not training or model-selection data",
        "models": models,
        "h100_v4_relative_change_vs_v3": (
            v4_h100["pooled_total_drag_nrmse"]
            / v3_h100["pooled_total_drag_nrmse"]
            - 1.0
        ),
        **derived,
        "frozen_test_status": "NOT_ACCESSED",
        "gate_c_status": "NOT_PASSED",
        "next_required_evidence": (
            "paired same-initial-state action drag-difference/ranking audit "
            "followed by matched real-CFD confirmation"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"decision": result["decision"], "h100_v4_relative_change_vs_v3": result["h100_v4_relative_change_vs_v3"]}))


if __name__ == "__main__":
    main()
