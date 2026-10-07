#!/usr/bin/env python3
"""Prepare the fixed b04 120->200 repeated-PRBS acquisition contract.

Preparation only: this module never starts OpenFOAM, Curator, or training.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


CASE = "p064_b04_long_prbs_120_200"
SOURCE_CASE = "matched_start_acquisition_train_b04_zero"
SOURCE_PROFILE = "dynamic_train8_b04_prbs"
START = 120.0
END = 200.0
DT = 0.1
FRAMES = 801
SOURCE_SPLIT_SHA256 = "1eaa84f12ebd56e2fc9c4da3b51fa6392e94276fa44f6d2539db66c325296b89"
VALIDATION_SPLIT_SHA256 = "2bf348b5b7a2d1d902078470ec02065cdc5b794de0d8c16be977e6df54e7d54f"
NORMALIZATION_SHA256 = "f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def repeated_prbs(source_points: list[list[float]]) -> list[list[float]]:
    """Repeat the exact 20 D/U b04 PRBS values four times on one 801-point grid."""
    require(len(source_points) == 201, "source PRBS must contain 201 points")
    expected = [round(START + i * DT, 10) for i in range(201)]
    require([float(row[0]) for row in source_points] == expected, "source time grid differs")
    values = [float(row[1]) for row in source_points]
    require(values[0] == 0.0 and values[-1] == 0.0, "source endpoints must be zero")
    rows: list[list[float]] = []
    for period in range(4):
        first = 0 if period == 0 else 1
        for index in range(first, 201):
            rows.append([round(START + period * 20.0 + index * DT, 10), values[index]])
    require(len(rows) == FRAMES, "long action table must contain 801 points")
    require(rows[0] == [START, 0.0] and rows[-1] == [END, 0.0], "long endpoints differ")
    require([row[0] for row in rows] == [round(START + i * DT, 10) for i in range(FRAMES)], "long grid differs")
    require(max(abs(row[1]) for row in rows) <= 0.75, "omega limit exceeded")
    require(max(abs(rows[i + 1][1] - rows[i][1]) for i in range(FRAMES - 1)) <= 0.1000000001, "rate limit exceeded")
    return rows


def build_contract(repo: Path) -> dict:
    repo = Path(repo).resolve()
    train_split = repo / "data/curated/tandem_cylinders_matched_start_full40_dev30_v1/splits/train.json"
    validation_split = repo / "data/curated/tandem_cylinders_matched_start_full40_dev30_v1/splits/validation.json"
    norm = repo / "data/curated/tandem_cylinders_matched_start_full40_dev30_v1/normalization.json"
    source_cfg = repo / f"cfd/tandem_cylinders/cases/{SOURCE_CASE}/case_config.json"
    prbs_cfg = repo / f"cfd/tandem_cylinders/cases/{SOURCE_PROFILE}/case_config.json"
    for path, digest in ((train_split, SOURCE_SPLIT_SHA256), (validation_split, VALIDATION_SPLIT_SHA256), (norm, NORMALIZATION_SHA256)):
        require(path.is_file() and sha256(path) == digest, f"bound input differs: {path}")
    source = json.loads(source_cfg.read_text())
    prbs = json.loads(prbs_cfg.read_text())
    require(source.get("split") == "train" and source.get("phase_bin") == 4, "source is not b04 train")
    require(source.get("source_restart_time") == START and source.get("end_time") == END, "restart/run interval differs")
    source_payload = source_cfg.parent / "source_restart_provenance"
    for field, digest in source["source_state_sha256"].items():
        require((source_payload / field).is_file() and sha256(source_payload / field) == digest, f"restart payload differs: {field}")
    require(prbs.get("split") == "train" and prbs.get("phase_bin") == 4 and prbs.get("profile") == "prbs", "PRBS source identity differs")
    action_points = repeated_prbs(prbs["action_points"])
    train_names = set(json.loads(train_split.read_text())["cases"])
    validation_names = set(json.loads(validation_split.read_text())["cases"])
    require(SOURCE_CASE in train_names and SOURCE_CASE not in validation_names, "source split binding differs")
    return {
        "status": "P064_B04_LONG_EXCITATION_PREPARED_NOT_AUTHORIZED",
        "case": CASE,
        "scope": "one train-only b04 OpenFOAM trajectory; no execution in preparation",
        "source": {
            "case": SOURCE_CASE,
            "restart_time": START,
            "source_state_sha256": source["source_state_sha256"],
            "case_config": str(source_cfg),
            "case_config_sha256": sha256(source_cfg),
            "restart_payload": str(source_payload),
            "restart_payload_verified": True,
            "train_split": str(train_split),
            "train_split_sha256": SOURCE_SPLIT_SHA256,
            "validation_split_sha256": VALIDATION_SPLIT_SHA256,
        },
        "action": {
            "source_case": SOURCE_PROFILE,
            "source_case_config_sha256": sha256(prbs_cfg),
            "construction": "repeat exact 201-point b04 PRBS four periods, dropping duplicate period starts",
            "points": action_points,
            "points_sha256": hashlib.sha256(json.dumps(action_points, separators=(",", ":")).encode()).hexdigest(),
            "max_abs_omega": max(abs(row[1]) for row in action_points),
            "max_abs_delta": max(abs(action_points[i + 1][1] - action_points[i][1]) for i in range(FRAMES - 1)),
        },
        "solver": {
            "start": START,
            "end": END,
            "delta_t": 0.005,
            "control_interval": DT,
            "expected_steps": 16000,
            "expected_frames": FRAMES,
            "reuse": [
                "cfd/tandem_cylinders/make_dynamic_train8_panel.py",
                "cfd/tandem_cylinders/run_dynamic_train8_case.sh",
                "cfd/tandem_cylinders/run_dynamic_train8_vtk_case.sh",
            ],
        },
        "curation": {
            "reuse": "cfd/tandem_cylinders/curate_dynamic_train8.py with case-local 801-frame contract",
            "official_path": "PhysicsNeMo Curator VTKSource -> Filter -> Sink; official HDF5Reader verification",
            "normalization": str(norm),
            "normalization_sha256": NORMALIZATION_SHA256,
            "normalization_policy": "reuse bytes; do not refit",
            "expected_frames": FRAMES,
        },
        "training_future_only": {
            "original_B_entries": 192,
            "b00_controlled_entries": 32,
            "long_b04_entries": 32,
            "long_b04_starts": [i * 700 // 31 for i in range(32)],
            "updates": 32,
            "accumulation": 8,
            "authorized": False,
        },
        "causal_limit": "relative to B, source profile, phase/action program, and visited states are jointly changed",
        "execution_authorized": False,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_contract(args.repo), indent=2))
