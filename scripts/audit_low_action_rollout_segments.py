#!/usr/bin/env python3
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def summarize(rows):
    squared_error = [
        (row["predicted_total_drag"] - row["target_total_drag"]) ** 2
        for row in rows
    ]
    target_squares = [row["target_total_drag"] ** 2 for row in rows]
    return {
        "segments": len(rows),
        "state_mae_physical_units": sum(
            row["state_mae_physical_units"] for row in rows
        )
        / len(rows),
        "total_drag_mae": sum(
            row["total_drag_absolute_error"] for row in rows
        )
        / len(rows),
        "total_drag_rmse": math.sqrt(sum(squared_error) / len(rows)),
        "total_drag_target_rms": math.sqrt(sum(target_squares) / len(rows)),
        "total_drag_nrmse_pooled": math.sqrt(sum(squared_error) / sum(target_squares)),
        "rear_cd_mae": sum(row["rear_cd_mae"] for row in rows) / len(rows),
        "rear_cl_mae": sum(row["rear_cl_mae"] for row in rows) / len(rows),
        "mean_max_abs_omega": sum(row["max_abs_omega"] for row in rows) / len(rows),
    }


def analyze(path):
    document = json.loads(path.read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    for row in document["segments"]:
        grouped[int(row["horizon"])].append(row)

    horizons = {}
    for horizon, rows in sorted(grouped.items()):
        low = [row for row in rows if row["max_abs_omega"] <= 1.0]
        high = [row for row in rows if row["max_abs_omega"] > 1.0]
        per_case = defaultdict(list)
        for row in rows:
            per_case[row["case"]].append(row)
        horizons[str(horizon)] = {
            "pooled": summarize(rows),
            "action_strata": {
                "max_abs_omega_le_1": summarize(low) if low else None,
                "max_abs_omega_gt_1": summarize(high) if high else None,
            },
            "per_case": {
                case: summarize(case_rows)
                for case, case_rows in sorted(per_case.items())
            },
        }
    return {
        "source": str(path),
        "split": document["split"],
        "action_mode": document["action_mode"],
        "horizons": horizons,
        "notes": [
            "Pooled NRMSE is sqrt(sum endpoint squared error / sum endpoint target squared).",
            "Action strata are post-hoc diagnostics from the same inference; no rerun or tuning.",
            "Per-case results are trajectory-conditioned, not proof of independent phase unless the dataset declares an independent restart phase.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("segments", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.segments)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
