"""Grid/time-step sensitivity diagnostic for the developmental CFD solver.

This is not a literature-benchmark or physical-validation substitute.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fluid_control.cfd import CFDConfig, CylinderCFD


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("artifacts/cfd_sensitivity.json"))
    parser.add_argument("--quick", action="store_true", help="Short pipeline check only; no physical interpretation")
    args = parser.parse_args()
    actions = np.array([[0.0, 0.0], [1.0, -1.0]], dtype=np.float32)
    time_end = 4.0 if args.quick else 18.0
    average_from = 2.0 if args.quick else 10.0
    cases = {
        "base": {"nx": 256, "ny": 128, "dt": 0.005},
        "fine_grid": {"nx": 384, "ny": 192, "dt": 0.005},
        "half_dt": {"nx": 256, "ny": 128, "dt": 0.0025},
    }
    results: dict[str, dict] = {}
    for name, values in cases.items():
        cfg = CFDConfig(
            **values,
            steps=round(time_end / values["dt"]),
            average_start=round(average_from / values["dt"]),
        )
        print(f"Running {name}: {cfg.nx}x{cfg.ny}, dt={cfg.dt}", flush=True)
        output = CylinderCFD(cfg).run(actions)
        results[name] = {"config": cfg.metadata(), "wake_error": output["wake_error"].tolist()}
    reference = np.asarray(results["base"]["wake_error"])
    for name in ("fine_grid", "half_dt"):
        values = np.asarray(results[name]["wake_error"])
        results[name]["relative_change_from_base"] = (np.abs(values - reference) / np.maximum(np.abs(reference), 1e-12)).tolist()
    report = {
        "status": "numerical_sensitivity_only_not_validated",
        "quick": args.quick,
        "actions": actions.tolist(),
        "physical_time_end": time_end,
        "physical_average_start": average_from,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({name: results[name].get("relative_change_from_base") for name in ("fine_grid", "half_dt")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
