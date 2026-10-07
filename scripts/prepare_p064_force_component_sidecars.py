"""Prepare compact train-only pressure/viscous label sidecars; no CFD/model."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
from physicsnemo.datapipes.readers.hdf5 import HDF5Reader

from fluid_control.p064_coarse_force_recoverability import (
    component_vectors,
    parse_force_components,
)

TIME_ATOL = 2e-5
SUM_ATOL = 1e-10
FAMILIES = ("base20", "train8", "train16", "controlled_b00")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_load(path: Path):
    return json.loads(path.read_text())


def mem_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    raise ValueError("MemAvailable missing")


def source_entries(spec: dict) -> list[dict]:
    sources = spec["sources"]
    for item in sources.values():
        require(sha(Path(item["manifest_path"])) == item["manifest_sha256"], "manifest SHA")
    base = json_load(Path(sources["base20"]["manifest_path"]))
    train8 = json_load(Path(sources["train8"]["manifest_path"]))
    train16 = json_load(Path(sources["train16"]["manifest_path"]))
    b00 = json_load(Path(sources["controlled_b00"]["manifest_path"]))
    result = []
    definitions = (
        ("base20", base["cases"], base["hdf5_sha256"]),
        ("train8", [Path(k).stem for k in train8["hdf_sha256"]], train8["hdf_sha256"]),
        ("train16", [Path(k).stem for k in train16["hdf_sha256"]], train16["hdf_sha256"]),
        ("controlled_b00", ["case_mpc"], {"b00_projected_ppo_train.h5": b00["hdf"]["sha256"]}),
    )
    for family, cases, hashes in definitions:
        cfg = sources[family]
        for case in cases:
            hdf_name = "b00_projected_ppo_train.h5" if family == "controlled_b00" else case + ".h5"
            expected = hashes[hdf_name] if hdf_name in hashes else hashes[case]
            result.append({
                "family": family,
                "case": case,
                "hdf_path": str(Path(cfg["hdf_root"]) / hdf_name),
                "hdf_sha256": expected,
                "raw_root": str(Path(cfg["raw_root"]) / case),
            })
    require(len(result) == 45, "expected 45 train-only trajectories")
    return result


def property_map(raw_root: Path) -> list[tuple[float, Path]]:
    found = []
    for path in raw_root.glob("*/uniform/functionObjects/functionObjectProperties"):
        try:
            found.append((float(path.parent.parent.parent.name), path))
        except ValueError:
            continue
    require(found, f"no properties under {raw_root}")
    return sorted(found)


def aligned_components(time_value: float, properties: list[tuple[float, Path]]):
    delta, path = min((abs(time_value - t), p) for t, p in properties)
    require(delta <= TIME_ATOL, f"property time mismatch {time_value}: {delta}")
    total, pressure, viscous = component_vectors(parse_force_components(path.read_text()))
    require(float(np.max(np.abs(pressure + viscous - total))) <= SUM_ATOL, "component algebra")
    return path, total, pressure, viscous, delta


def scan_entry(entry: dict, *, include_rows: bool) -> tuple[dict, dict | None]:
    hdf = Path(entry["hdf_path"])
    raw = Path(entry["raw_root"])
    require(hdf.is_file() and raw.is_dir(), "source path missing")
    props = property_map(raw)
    with h5py.File(hdf, "r") as handle:
        require(set(("time", "force")).issubset(handle), "HDF fields")
        times = np.asarray(handle["time"][:])
        hdf_total = np.asarray(handle["force"][:])
    require(times.shape == (len(hdf_total), 1) and hdf_total.shape[1:] == (4,), "label shapes")
    raw_total, pressure, viscous, rows = [], [], [], []
    max_time = max_hdf = max_sum = 0.0
    for index, stored_time in enumerate(times[:, 0]):
        path, total, press, visc, delta = aligned_components(float(stored_time), props)
        max_time = max(max_time, delta)
        max_hdf = max(max_hdf, float(np.max(np.abs(hdf_total[index].astype(np.float64) - total))))
        max_sum = max(max_sum, float(np.max(np.abs(press + visc - total))))
        raw_total.append(total); pressure.append(press); viscous.append(visc)
        if include_rows:
            rows.append({"index": index, "time": float(stored_time), "property_path": str(path), "property_sha256": sha(path)})
    arrays = {
        "time": times,
        "hdf_total": hdf_total,
        "raw_total": np.asarray(raw_total, dtype=np.float64),
        "pressure": np.asarray(pressure, dtype=np.float64),
        "viscous": np.asarray(viscous, dtype=np.float64),
    }
    summary = {**entry, "frames": len(times), "max_time_abs": max_time,
               "max_hdf_total_minus_raw_total_abs": max_hdf,
               "max_pressure_plus_viscous_minus_raw_total_abs": max_sum,
               "property_inventory": rows if include_rows else None}
    return summary, arrays if include_rows else None


def array_moments(chunks: list[np.ndarray]) -> dict:
    values = np.concatenate(chunks, axis=0)
    return {"count": int(values.shape[0]), "mean": values.mean(axis=0).tolist(),
            "std_population": values.std(axis=0).tolist(),
            "dtype": "float64_physical_coefficients"}


def coverage(spec: dict) -> dict:
    summaries = []
    family_values = {family: {"pressure": [], "viscous": []} for family in FAMILIES}
    for entry in source_entries(spec):
        require(mem_available_gib() >= spec["resources"]["runtime_mem_available_gib"], "runtime memory")
        summary, arrays = scan_entry(entry, include_rows=True)
        summary["property_inventory"] = None
        summaries.append(summary)
        family_values[entry["family"]]["pressure"].append(arrays["pressure"])
        family_values[entry["family"]]["viscous"].append(arrays["viscous"])
    families = {}
    for family in FAMILIES:
        rows = [x for x in summaries if x["family"] == family]
        families[family] = {
            "cases": len(rows), "frames": sum(x["frames"] for x in rows),
            "max_time_abs": max(x["max_time_abs"] for x in rows),
            "max_hdf_total_minus_raw_total_abs": max(x["max_hdf_total_minus_raw_total_abs"] for x in rows),
            "max_component_sum_abs": max(x["max_pressure_plus_viscous_minus_raw_total_abs"] for x in rows),
            "pressure_moments": array_moments(family_values[family]["pressure"]),
            "viscous_moments": array_moments(family_values[family]["viscous"]),
        }
    all_pressure = [array for family in FAMILIES for array in family_values[family]["pressure"]]
    all_viscous = [array for family in FAMILIES for array in family_values[family]["viscous"]]
    expected = {"base20": (20, 16020), "train8": (8, 1608),
                "train16": (16, 2064), "controlled_b00": (1, 801)}
    require({family: (families[family]["cases"], families[family]["frames"])
             for family in FAMILIES} == expected, "coverage family counts")
    return {"status": "P064_FORCE_COMPONENT_LABEL_COVERAGE_COMPLETE",
            "cases": len(summaries), "frames": sum(x["frames"] for x in summaries),
            "missing": 0, "families": families,
            "all_train_pressure_moments": array_moments(all_pressure),
            "all_train_viscous_moments": array_moments(all_viscous),
            "base20_is_original20": True,
            "source_summaries": summaries}


def write_sidecar(path: Path, entry: dict, arrays: dict) -> None:
    with h5py.File(path, "w") as handle:
        for name, value in arrays.items():
            handle.create_dataset(name, data=value, compression="gzip", compression_opts=1, shuffle=True)
        for key in ("family", "case", "hdf_path", "hdf_sha256", "raw_root"):
            handle.attrs[key] = entry[key]
        handle.attrs["split"] = "train"
        handle.attrs["project_writer"] = "h5py; official PhysicsNeMo HDF5Reader verified after write"


def verify_official(path: Path, expected: dict) -> None:
    fields = ["time", "hdf_total", "raw_total", "pressure", "viscous"]
    reader = HDF5Reader(path, fields=fields)
    try:
        require(len(reader) == len(expected["time"]), "official reader length")
        for index in (0, len(reader) - 1):
            sample, _ = reader[index]
            for field in fields:
                require(np.array_equal(sample[field].detach().cpu().numpy(), expected[field][index]), "official reader value")
    finally:
        reader.close()


def convert(spec: dict, spec_sha: str) -> dict:
    output = Path(spec["planned_output"])
    require(not output.exists(), "output exists")
    stage = output.with_name(output.name + f".tmp.{os.getpid()}")
    require(not stage.exists(), "temporary output exists")
    stage.mkdir(parents=True)
    manifest_rows = []
    family_values = {family: {"pressure": [], "viscous": []} for family in FAMILIES}
    try:
        for entry in source_entries(spec):
            require(mem_available_gib() >= spec["resources"]["runtime_mem_available_gib"], "runtime memory")
            summary, arrays = scan_entry(entry, include_rows=True)
            family_dir = stage / entry["family"]
            family_dir.mkdir(exist_ok=True)
            sidecar = family_dir / (entry["case"] + ".components.h5")
            write_sidecar(sidecar, entry, arrays)
            verify_official(sidecar, arrays)
            family_values[entry["family"]]["pressure"].append(arrays["pressure"])
            family_values[entry["family"]]["viscous"].append(arrays["viscous"])
            summary["sidecar_path"] = str(output / entry["family"] / sidecar.name)
            summary["sidecar_sha256"] = sha(sidecar)
            manifest_rows.append(summary)
        expected = {"base20": (20, 16020), "train8": (8, 1608),
                    "train16": (16, 2064), "controlled_b00": (1, 801)}
        actual = {family: (sum(row["family"] == family for row in manifest_rows),
                           sum(row["frames"] for row in manifest_rows if row["family"] == family))
                  for family in FAMILIES}
        require(actual == expected and len(manifest_rows) == 45 and
                sum(row["frames"] for row in manifest_rows) == 20493,
                "conversion family/frame counts")
        family_moments = {family: {
            "pressure": array_moments(family_values[family]["pressure"]),
            "viscous": array_moments(family_values[family]["viscous"]),
        } for family in FAMILIES}
        manifest = {"status": "P064_FORCE_COMPONENT_SIDECARS_COMPLETE_NOT_TRAINING",
                    "spec_sha256": spec_sha, "writer": "project h5py",
                    "reader": "official PhysicsNeMo HDF5Reader",
                    "cases": 45, "frames": 20493, "normalization_refit": False,
                    "original_hdf_modified": False, "new_cfd": False,
                    "labels_are_physical_openfoam_coefficients": True,
                    "viscous_is_not_derived_from_hdf_total": True,
                    "family_moments": family_moments,
                    "all_train_pressure_moments": array_moments([
                        value for family in FAMILIES for value in family_values[family]["pressure"]]),
                    "all_train_viscous_moments": array_moments([
                        value for family in FAMILIES for value in family_values[family]["viscous"]]),
                    "rows": manifest_rows}
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        stage.replace(output)
        return {"status": manifest["status"], "manifest_sha256": sha(output / "manifest.json"),
                "cases": 45, "frames": 20493}
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def validate_spec(spec: dict, spec_path: Path, spec_sha: str, output: Path):
    require(sha(spec_path) == spec_sha, "spec SHA")
    require(spec["status"] in {"P064_FORCE_COMPONENT_SIDECARS_PENDING", "P064_FORCE_COMPONENT_SIDECARS_EXECUTION_APPROVED"}, "status")
    require(Path(spec["planned_output"]).resolve() == output.resolve(), "output binding")
    require(not output.exists(), "output exists")
    require(spec["selection"] == {"families": ["base20", "train8", "train16", "controlled_b00"], "cases": 45, "frames": 20493, "split": "train"}, "selection")
    require(spec["resources"] == {"cpu_quota": 1, "memory_max_gib": 8, "memory_swap_max": 0, "runtime_seconds": 120, "startup_mem_available_gib": 50, "runtime_mem_available_gib": 22, "cuda_visible_devices": ""}, "resources")
    require(sha(Path(spec["worker"]["path"])) == spec["worker"]["sha256"], "worker SHA")
    require(sha(Path(spec["helper"]["path"])) == spec["helper"]["sha256"], "helper SHA")
    require(Path(inspect.getfile(parse_force_components)).resolve() ==
            Path(spec["helper"]["path"]).resolve(), "helper import origin")
    require(sha(Path(spec["supervisor"]["path"])) == spec["supervisor"]["sha256"], "supervisor SHA")
    require(len(source_entries(spec)) == 45, "source entries")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coverage-output", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    spec = json_load(args.spec)
    validate_spec(spec, args.spec, args.spec_sha256, args.output)
    if args.coverage_output:
        result = coverage(spec)
        args.coverage_output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"status": result["status"], "cases": result["cases"], "frames": result["frames"], "sha256": sha(args.coverage_output)}))
        return
    if not args.execute:
        print("P064_FORCE_COMPONENT_SIDECAR_PREPARATION_ONLY_NO_CONVERSION")
        return
    require(spec["execution_authorized"] is True and os.environ.get("P064_COMPONENT_SIDECAR_TOKEN") == "APPROVED", "not authorized")
    require(mem_available_gib() >= spec["resources"]["startup_mem_available_gib"], "startup memory")
    print(json.dumps(convert(spec, args.spec_sha256)))


if __name__ == "__main__":
    main()
