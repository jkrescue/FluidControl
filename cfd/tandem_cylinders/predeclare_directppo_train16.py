#!/usr/bin/env python3
"""Freeze the historical train-only direct-PPO CFD trajectories for curation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CASES = REPO / "cfd/tandem_cylinders/cases"
RUN = REPO / "artifacts/direct_cfd/directppo2048_v1"
OUTPUT = REPO / "artifacts/tandem_cylinders/directppo_train16_predeclared_20261004.json"
NORMALIZATION_SHA = "f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1"
BASE_MANIFEST_SHA = "5213c7bb07c824c6e601636c3cfa974b80ec7c051633b0c8368a045cea41ddd2"
FORCE_SHA = {
    "forceFront": "bce88443ce3d19a6411c31b9266af16a3dd4e992f7adb1ddabc659238a1e88d1",
    "forceRear": "654c5bbf64b758505bdfe4d379e153f460862902309e96d2b01974e45536680b",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def journal_rows(path: Path, env: int) -> dict[int, list[dict]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    steps = [row for row in rows if row.get("event") == "step"]
    grouped = {episode: [] for episode in range(2, 10)}
    for row in steps:
        if row.get("episode") in grouped:
            grouped[row["episode"]].append(row)
    if len(steps) != 1024 or any(len(grouped[episode]) != 128 for episode in grouped):
        raise ValueError(f"env{env}: expected exactly 8x128 historical PPO steps")
    return grouped


def build(repo: Path = REPO) -> dict:
    cases_root = repo / "cfd/tandem_cylinders/cases"
    run = repo / "artifacts/direct_cfd/directppo2048_v1"
    baseline = cases_root / "tandem_backward_dt005/postProcessing"
    for key, digest in FORCE_SHA.items():
        if sha256(baseline / key / "0/coefficient.dat") != digest:
            raise ValueError(f"baseline force SHA differs: {key}")
    cases: dict[str, dict] = {}
    journals: dict[str, str] = {}
    for env, phase, start in ((0, "b00", 148.0), (1, "b02", 106.0)):
        journal = run / f"worker_env{env}.jsonl"
        journals[str(journal.relative_to(repo))] = sha256(journal)
        grouped = journal_rows(journal, env)
        for episode in range(2, 10):
            name = f"direct_cfd_directppo2048_v1_env{env}_ep{episode:04d}_{phase}"
            case = cases_root / name
            config_path = case / "case_config.json"
            config = load_json(config_path)
            expected = {
                "case": name,
                "status": "direct_cfd_ppo_episode_initialized",
                "split": "train",
                "phase": phase,
                "source_restart_time": start,
            }
            if any(config.get(key) != value for key, value in expected.items()):
                raise ValueError(f"case config differs: {name}")
            rows = sorted(grouped[episode], key=lambda row: int(row["step"]))
            if [int(row["step"]) for row in rows] != list(range(1, 129)):
                raise ValueError(f"nonconsecutive journal steps: {name}")
            times = [start] + [float(row["cfd_time"]) for row in rows]
            target = [start + 0.1 * index for index in range(129)]
            if any(not math.isclose(a, b, abs_tol=2e-6) for a, b in zip(times, target, strict=True)):
                raise ValueError(f"field time contract differs: {name}")
            actions = [[start, 0.0]] + [
                [float(row["cfd_time"]), float(row["applied_omega"])] for row in rows
            ]
            if any(
                row.get("case") != name
                or row.get("phase") != phase
                or row.get("solver_health", {}).get("steps") != 20
                or row.get("solver_health", {}).get("solver_ended_cleanly") is not True
                or not math.isfinite(float(row["applied_omega"]))
                or abs(float(row["applied_omega"])) > 0.750001
                or abs(float(row["applied_delta_omega"])) > 0.100001
                for row in rows
            ):
                raise ValueError(f"journal physics contract differs: {name}")
            state_sha = {}
            for field in ("U", "p", "phi", "phi_0", "U_0"):
                path = case / f"{start:g}" / field
                if not path.is_file():
                    raise FileNotFoundError(path)
                state_sha[field] = sha256(path)
            field_dirs = sorted(
                float(path.name)
                for path in case.iterdir()
                if path.is_dir() and path.name.replace(".", "", 1).isdigit()
            )
            if len(field_dirs) != 129 or any(
                not math.isclose(a, b, abs_tol=2e-6)
                for a, b in zip(field_dirs, target, strict=True)
            ):
                raise ValueError(f"129 saved states not exact: {name}")
            if any(not (case / f"{time:g}" / "U").is_file() or not (case / f"{time:g}" / "p").is_file() for time in target):
                raise FileNotFoundError(f"missing saved U/p: {name}")
            cases[name] = {
                "split": "train",
                "source": "historical direct real-CFD PPO training interaction",
                "env_index": env,
                "episode": episode,
                "phase": phase,
                "run_window": [start, start + 12.8],
                "expected_frames": 129,
                "control_dt": 0.1,
                "action_points": actions,
                "case_config_sha256": sha256(config_path),
                "source_state_sha256": state_sha,
                "journal_path": str(journal.relative_to(repo)),
            }
    if len(cases) != 16:
        raise ValueError("expected exactly 16 train-only cases")
    return {
        "status": "DIRECTPPO_TRAIN16_CURATOR_PREDECLARED_NOT_EXECUTED",
        "scope": (
            "reuse of real OpenFOAM trajectories generated during PPO training; "
            "actions came from changing exploratory policies and are not a final-policy on-policy dataset"
        ),
        "cases_root": "cfd/tandem_cylinders/cases",
        "excluded_reset_only_episodes": [1, 10],
        "split_contract": {"train": 16, "validation": 0, "frozen_test": 0},
        "field_contract": {
            "frames_per_case": 129,
            "spacing_D_over_U": 0.1,
            "channels": ["u", "v", "gauge_pressure"],
            "pressure_gauge": "subtract valid-domain spatial mean per frame",
            "grid": {"nx": 256, "ny": 128, "x": [8.0, 25.0], "y": [4.0, 11.0]},
        },
        "force_contract": {
            "channels": ["front_cd", "front_cl", "rear_cd", "rear_cl"],
            "source_t0_sha256": FORCE_SHA,
            "policy": "unique exact t0 from pinned baseline plus restart-split real force rows; no synthetic or interpolated state",
        },
        "normalization_contract": {
            "policy": "byte-identical immutable full40 train20 transform",
            "normalization_sha256": NORMALIZATION_SHA,
            "base_manifest_sha256": BASE_MANIFEST_SHA,
        },
        "journal_sha256": journals,
        "validation_or_frozen_accessed": False,
        "cases": cases,
    }


def exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError(f"refusing overwrite: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    exclusive(args.output.resolve(), build())
    print(args.output)


if __name__ == "__main__":
    main()
