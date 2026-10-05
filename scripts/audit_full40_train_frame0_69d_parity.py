#!/usr/bin/env python3
"""Aggregate raw-vs-curated 69D parity for the four train frame-0 resets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np

from fluid_control.openfoam_observation import total_drag_observation_at
from validate_tandem_69d_observation_parity import compare
from validate_tandem_probe_mapping import bilinear_probes


CASES = tuple(
    f"matched_start_acquisition_train_b{phase}_zero"
    for phase in ("00", "02", "04", "06")
)
PROBE_DIAGNOSTIC_LIMIT = 0.0045


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise ValueError(f"refusing to overwrite {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def audit(repo: Path) -> dict:
    repo = repo.resolve()
    data = repo / "data/curated/tandem_cylinders_matched_start_full40_v1"
    cases_root = repo / "cfd/tandem_cylinders/cases"
    manifest_path = data / "manifest.json"
    split_path = data / "splits/train.json"
    manifest = read_json(manifest_path)
    split = read_json(split_path)
    if (
        manifest.get("profile") != "matched_start_full40_v1"
        or manifest.get("split_manifests", {}).get("train", {}).get("sha256")
        != sha256(split_path)
        or split.get("split") != "train"
        or not all(case in split.get("cases", []) for case in CASES)
    ):
        raise ValueError("full40 train manifest identity differs")

    rows = []
    for case in CASES:
        hdf = data / "train" / f"{case}.h5"
        config_path = cases_root / case / "case_config.json"
        config = read_json(config_path)
        with h5py.File(hdf, "r") as handle:
            embedded = json.loads(str(handle.attrs["config_json"]))
            hdf_case = str(handle.attrs["case"])
            hdf_split = str(handle.attrs["split"])
            time = float(handle["time"][0, 0])
            omega = float(handle["omega"][0, 0])
            curated_observation = np.concatenate(
                (
                    bilinear_probes(
                        handle["state"][0], handle["x"][:], handle["y"][:]
                    ).reshape(-1),
                    np.asarray(handle["force"][0], dtype=np.float64),
                    np.asarray(handle["omega"][0], dtype=np.float64),
                )
            )
        if (
            hdf_case != case
            or hdf_split != "train"
            or embedded != config
            or config.get("case") != case
            or config.get("split") != "train"
            or time != float(config["source_restart_time"])
            or omega != 0.0
            or split.get("hdf5_sha256", {}).get(case) != sha256(hdf)
        ):
            raise ValueError(f"frame-0 provenance differs: {case}")
        source = (cases_root / str(config["source_restart_case"])).resolve()
        if not source.is_relative_to(cases_root.resolve()):
            raise ValueError("source restart escapes train cases root")
        force_sha = {}
        for name in ("forceFront", "forceRear"):
            paths = sorted(source.glob(f"postProcessing/{name}/*/coefficient.dat"))
            if len(paths) != 1:
                raise ValueError(f"expected one pinned {name} source")
            force_sha[name] = sha256(paths[0])
        if force_sha != config.get("source_force_sha256"):
            raise ValueError(f"raw force SHA differs: {case}")
        raw_observation, _ = total_drag_observation_at(source, time, omega)
        raw_observation = np.asarray(raw_observation, dtype=np.float64)
        curated_observation = np.asarray(curated_observation, dtype=np.float64)
        if (
            raw_observation.shape != (69,)
            or curated_observation.shape != (69,)
            or not np.isfinite(raw_observation).all()
            or not np.isfinite(curated_observation).all()
        ):
            raise ValueError(f"invalid frame-0 69D observation: {case}")

        parity = compare(data, cases_root, "train", case, [0])
        comparison = parity["comparisons"][0]
        vector_error = np.abs(raw_observation - curated_observation)
        reproduced = {
            "probe_max_abs_error": float(vector_error[:64].max()),
            "front_force_max_abs_error": float(vector_error[64:66].max()),
            "rear_force_max_abs_error": float(vector_error[66:68].max()),
            "omega_abs_error": float(vector_error[68]),
        }
        if any(
            reproduced[key] != float(comparison[key])
            for key in reproduced
        ):
            raise AssertionError("stored observation vectors differ from reused audit")
        diagnostic_pass = bool(
            comparison["probe_max_abs_error"] <= PROBE_DIAGNOSTIC_LIMIT
            and comparison["front_force_max_abs_error"] <= 1e-4
            and comparison["rear_force_max_abs_error"] <= 1e-4
            and comparison["omega_abs_error"] <= 1e-5
        )
        rows.append(
            {
                "case": case,
                "split": "train",
                "frame": 0,
                "source_restart_case": source.name,
                "source_restart_time": time,
                "case_config_sha256": sha256(config_path),
                "hdf5_sha256": sha256(hdf),
                "raw_force_sha256": force_sha,
                "raw_observation_69d": raw_observation.tolist(),
                "raw_observation_float64_sha256": hashlib.sha256(
                    raw_observation.tobytes()
                ).hexdigest(),
                "curated_observation_69d": curated_observation.tolist(),
                "curated_observation_float64_sha256": hashlib.sha256(
                    curated_observation.tobytes()
                ).hexdigest(),
                "comparison": comparison,
                "diagnostic_pass": diagnostic_pass,
            }
        )

    passed = all(row["diagnostic_pass"] for row in rows)
    return {
        "status": (
            "FULL40_TRAIN_FRAME0_69D_PARITY_PASS"
            if passed
            else "FULL40_TRAIN_FRAME0_69D_PARITY_FAIL"
        ),
        "scope": "train-only raw OpenFOAM vs curated frame-0 interface diagnostic",
        "scientific_admission_gate": False,
        "fno_loaded": False,
        "ppo_executed": False,
        "validation_or_frozen_accessed": False,
        "channel_order": "32*(u,v),front_cd,front_cl,rear_cd,rear_cl,omega",
        "criteria": {
            "probe_max_abs_error": PROBE_DIAGNOSTIC_LIMIT,
            "force_max_abs_error": 1e-4,
            "omega_abs_error": 1e-5,
            "note": "0.0045 is a predeclared interface diagnostic criterion, not an FNO formal gate",
        },
        "input_sha256": {
            "manifest": sha256(manifest_path),
            "train_split_manifest": sha256(split_path),
            "producer": sha256(Path(__file__)),
            "reused_compare": sha256(
                Path(__file__).with_name("validate_tandem_69d_observation_parity.py")
            ),
        },
        "cases": rows,
        "maxima": {
            key: max(row["comparison"][key] for row in rows)
            for key in (
                "probe_max_abs_error",
                "front_force_max_abs_error",
                "rear_force_max_abs_error",
                "omega_abs_error",
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.repo)
    write_exclusive(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "FULL40_TRAIN_FRAME0_69D_PARITY_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
