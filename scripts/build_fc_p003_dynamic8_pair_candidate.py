#!/usr/bin/env python3
"""Build a train-only dynamic-action/zero matched-pair candidate manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np

PHASES = (0, 2, 4, 6)
PROFILES = ("multisine", "prbs")
STATE_ATOL = 3e-7


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_prefix(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as source:
        required = {"state", "mask", "omega", "force", "time"}
        if not required.issubset(source):
            raise ValueError(f"missing HDF fields in {path.name}")
        if any(source[key].shape[0] < 101 for key in required):
            raise ValueError(f"{path.name} has fewer than 101 H100 frames")
        return {key: np.asarray(source[key][:101]) for key in required}


def audit_pair(action: dict[str, np.ndarray], zero: dict[str, np.ndarray]) -> dict:
    if action["state"].shape != zero["state"].shape:
        raise ValueError("state shape differs")
    state_max = float(np.max(np.abs(action["state"][0] - zero["state"][0])))
    if not np.allclose(action["state"][0], zero["state"][0], rtol=0, atol=STATE_ATOL):
        raise ValueError(f"state0 exceeds atol: {state_max}")
    if not np.array_equal(action["mask"], zero["mask"]):
        raise ValueError("H100 mask differs")
    if not np.array_equal(action["force"][0], zero["force"][0]):
        raise ValueError("initial force differs")
    if not np.allclose(action["time"], zero["time"], rtol=0, atol=2e-5):
        raise ValueError("H100 time grid differs")
    zero_omega = zero["omega"].reshape(-1)
    if not np.allclose(zero_omega, 0, rtol=0, atol=1e-7):
        raise ValueError("zero reference has nonzero omega")
    omega = action["omega"].reshape(-1)
    if not np.isfinite(omega).all() or np.max(np.abs(omega)) > 0.7500001:
        raise ValueError("dynamic omega violates finite/absolute limit")
    return {
        "state0_max_abs_difference": state_max,
        "state0_rtol": 0.0,
        "state0_atol": STATE_ATOL,
        "h100_mask_exact": True,
        "initial_force_exact": True,
        "h100_time_atol": 2e-5,
        "omega_min": float(omega.min()),
        "omega_max": float(omega.max()),
        "omega_change_count": int(np.count_nonzero(np.abs(np.diff(omega)) > 1e-7)),
    }


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynamic-root", type=Path, required=True)
    parser.add_argument("--static-root", type=Path, required=True)
    parser.add_argument("--dynamic-authorization", type=Path, required=True)
    parser.add_argument("--full40-predeclaration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dynamic_manifest_path = args.dynamic_root / "manifest.json"
    dynamic_manifest = json.loads(dynamic_manifest_path.read_text())
    authorization = json.loads(args.dynamic_authorization.read_text())
    predeclaration = json.loads(args.full40_predeclaration.read_text())
    if dynamic_manifest.get("trajectory_counts") != {"train": 8, "validation": 0, "frozen_test": 0}:
        raise ValueError("dynamic_train8 is not exact train-only 8-trajectory release")
    if authorization.get("validation_or_frozen_accessed") is not False:
        raise ValueError("dynamic authorization split guard differs")

    pairs = []
    for phase in PHASES:
        zero_case = f"matched_start_acquisition_train_b{phase:02d}_zero"
        zero_file = args.static_root / "train" / f"{zero_case}.h5"
        zero_source = predeclaration["cases"][zero_case]["source_state_sha256"]
        zero = load_prefix(zero_file)
        for profile in PROFILES:
            action_case = f"dynamic_train8_b{phase:02d}_{profile}"
            action_file = args.dynamic_root / "train" / f"{action_case}.h5"
            action_source = authorization["cases"][action_case]["source_state_sha256"]
            if action_source != zero_source:
                raise ValueError(f"source restart SHA differs: {action_case}")
            metrics = audit_pair(load_prefix(action_file), zero)
            pairs.append({
                "phase": f"b{phase:02d}", "profile": profile, "split": "train",
                "start": 0, "horizon": 100,
                "action_file": str(action_file.relative_to(args.dynamic_root)),
                "zero_file": str(zero_file.relative_to(args.static_root)),
                "action_hdf_sha256": sha256(action_file),
                "zero_hdf_sha256": sha256(zero_file),
                "source_state_sha256": action_source, "qc": metrics,
            })

    result = {
        "status": "FC_P003_DYNAMIC8_PAIR_CANDIDATE_QC_PASS",
        "scope": "train-only existing-data candidate; not approved for training",
        "split": "train", "pair_count": len(pairs), "sequence_length": 101,
        "horizon": 100, "state0_tolerance": {"rtol": 0.0, "atol": STATE_ATOL},
        "validation_or_frozen_accessed": False,
        "dynamic_manifest_sha256": sha256(dynamic_manifest_path),
        "dynamic_authorization_sha256": sha256(args.dynamic_authorization),
        "full40_predeclaration_sha256": sha256(args.full40_predeclaration),
        "normalization_sha256": dynamic_manifest["normalization_sha256"],
        "interpretation": [
            "Prepared only as the minimum next data intervention if FC-P003 interleaving fails.",
            "Completion of this QC is not evidence of model or control improvement.",
            "Source restart equality is exact; state0 tolerance only accounts for distinct Curator float32 paths.",
        ],
        "pairs": pairs,
    }
    atomic_json(args.output, result)
    print(json.dumps({"status": result["status"], "pairs": len(pairs), "output": str(args.output)}))


if __name__ == "__main__":
    main()
