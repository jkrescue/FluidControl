#!/usr/bin/env python3
"""Package representative OpenFOAM VTK snapshots for ParaView."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


DEFAULT_CASES = (
    "control_small_m100",
    "control_small_m050",
    "control_small_z000",
    "control_small_p050",
    "control_small_p100",
    "dynamic_test_00",
    "dynamic_test_01",
)
CONSTANT_OMEGA = {
    "control_small_m100": -1.0,
    "control_small_m050": -0.5,
    "control_small_z000": 0.0,
    "control_small_p050": 0.5,
    "control_small_p100": 1.0,
}


def write_pvd(path: Path, records: list[dict]) -> None:
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
    ]
    lines.extend(
        f'    <DataSet timestep="{record["time"]:.8g}" group="" part="0" '
        f'file="{record["file"]}"/>'
        for record in records
    )
    lines.extend(("  </Collection>", "</VTKFile>"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-root", type=Path, default=Path("cfd/tandem_cylinders/cases"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    parser.add_argument("--times", type=float, nargs="+", default=(80, 100, 120, 140, 160))
    parser.add_argument("--first-time", type=float, default=80.0)
    parser.add_argument("--frame-dt", type=float, default=0.1)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": str(args.cases_root),
        "selection": {"times": args.times, "first_time": args.first_time, "frame_dt": args.frame_dt},
        "cases": [],
    }
    for case in args.cases:
        source_files = sorted((args.cases_root / case / "VTK_curator").glob("*/internal.vtu"))
        if not source_files:
            raise FileNotFoundError(f"no VTK snapshots found for {case}")
        case_dir = args.output / case
        case_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for requested_time in args.times:
            index = round((requested_time - args.first_time) / args.frame_dt)
            if index < 0 or index >= len(source_files):
                raise IndexError(f"time {requested_time} maps to frame {index}, outside {case}")
            actual_time = args.first_time + index * args.frame_dt
            destination = case_dir / f"time_{actual_time:07.2f}.vtu"
            shutil.copy2(source_files[index], destination)
            records.append({
                "time": actual_time,
                "file": destination.name,
                "source": str(source_files[index]),
            })
        write_pvd(case_dir / f"{case}.pvd", records)
        manifest["cases"].append({
            "case": case,
            "constant_omega": CONSTANT_OMEGA.get(case),
            "snapshots": records,
            "pvd": str(case_dir / f"{case}.pvd"),
        })
        print(f"{case}: copied {len(records)} snapshots")

    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {sum(len(item['snapshots']) for item in manifest['cases'])} VTK files")


if __name__ == "__main__":
    main()
