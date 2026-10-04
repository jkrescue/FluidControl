#!/usr/bin/env python3
"""Build the immutable 16-pair train-only DataPipe manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build(root: Path, audit_path: Path) -> dict:
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "TRAIN20_MATCHED_START_PAIRED_TARGET_AUDIT_PASS":
        raise ValueError("source paired-target audit did not pass")
    if audit.get("validation_opened") is not False:
        raise ValueError("source audit accessed validation")
    if audit.get("frozen_test_opened_or_enumerated") is not False:
        raise ValueError("source audit accessed or enumerated frozen test")
    if sha(root / "manifest.json") != audit.get("data_manifest_sha256"):
        raise ValueError("source data manifest differs from paired-target audit")
    normalization_path = root / "normalization.json"
    normalization = json.loads(normalization_path.read_text())
    channels = ["front_cd", "front_cl", "rear_cd", "rear_cl"]
    if normalization.get("all_force_channels") != channels:
        raise ValueError("unexpected four-force channel order")
    force_norm = {
        "channels": channels,
        "mean": normalization["all_force_mean"],
        "std": normalization["all_force_std"],
    }
    pairs = []
    for phase in ("b00", "b02", "b04", "b06"):
        record = audit["phases"][phase]
        zero = record["actions"]["zero"]
        if not record["matched_start_exact"]:
            raise ValueError(f"audit rejects matched start for {phase}")
        for action in ("m075", "m0375", "p0375", "p075"):
            item = record["actions"][action]
            action_file = f"{item['case']}.h5"
            zero_file = f"{zero['case']}.h5"
            for filename, expected in (
                (action_file, item["hdf5_sha256"]),
                (zero_file, zero["hdf5_sha256"]),
            ):
                path = root / "train" / filename
                if sha(path) != expected:
                    raise ValueError(f"HDF SHA mismatch: {filename}")
            if item["initial_sha256"] != zero["initial_sha256"]:
                raise ValueError(f"initial q mismatch: {phase}/{action}")
            pairs.append(
                {
                    "phase": phase,
                    "action": action,
                    "split": "train",
                    "start": 0,
                    "action_file": action_file,
                    "zero_file": zero_file,
                    "action_hdf_sha256": item["hdf5_sha256"],
                    "zero_hdf_sha256": zero["hdf5_sha256"],
                    "initial_sha256": item["initial_sha256"],
                    "zero_initial_sha256": zero["initial_sha256"],
                    "target_omega": item["target_omega"],
                    "targets": record["action_minus_zero_targets"][action],
                }
            )
    return {
        "schema_version": 1,
        "status": "TRAIN20_MATCHED_PAIR_DATAPIPE_READY",
        "scientific_scope": "train-only paired-stat supervision; not validation, frozen test, or control evidence",
        "split": "train",
        "pair_count": 16,
        "phases": ["b00", "b02", "b04", "b06"],
        "actions": ["m075", "m0375", "p0375", "p075"],
        "start": 0,
        "sequence_length": 101,
        "dt": 0.1,
        "horizons": [20, 50, 100],
        "stat_names": ["mean_total_cd", "mean_rear_cl", "rear_cl_fluctuation_rms"],
        "force_channels": channels,
        "force_normalization_sha256": json_sha(force_norm),
        "normalization_sha256": sha(normalization_path),
        "source_audit_sha256": sha(audit_path),
        "source_data_manifest_sha256": audit["data_manifest_sha256"],
        "max_abs_omega": 0.75,
        "validation_or_frozen_accessed": False,
        "pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.data.resolve(), args.source_audit.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.link(temporary, args.output)
    temporary.unlink()
    print(json.dumps({"status": result["status"], "pairs": len(result["pairs"]), "output": str(args.output), "sha256": sha(args.output)}))


if __name__ == "__main__":
    main()
