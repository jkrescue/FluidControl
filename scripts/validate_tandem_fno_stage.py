#!/usr/bin/env python3
"""Fail closed on incomplete or non-finite Spark FNO stage results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


TRAIN_METRICS = (
    "train_loss",
    "state_mae_physical_units",
    "state_rmse_physical_units",
    "force_mae_normalized",
)
EVAL_METRICS = (
    "state_mae_physical_units",
    "persistence_state_mae_physical_units",
    "rear_force_mae",
    "rear_cd_mae",
    "rear_cl_mae",
    "persistence_rear_force_mae",
)
HORIZONS = ("1", "10", "50")


def nonnegative_finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}: expected a numeric metric, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{label}: expected a finite nonnegative metric, got {value!r}")
    return number


def verify_training(path: Path, min_epoch: int) -> dict:
    history = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(history, list) or not history:
        raise ValueError(f"{path}: missing training history")
    epochs = [row.get("epoch") for row in history]
    if any(type(epoch) is not int for epoch in epochs):
        raise ValueError(f"{path}: non-integer epoch")
    if epochs != list(range(1, epochs[-1] + 1)) or epochs[-1] < min_epoch:
        raise ValueError(f"{path}: incomplete or nonconsecutive epochs {epochs}")
    for row in history:
        for key in TRAIN_METRICS:
            nonnegative_finite(row.get(key), f"epoch {row['epoch']} {key}")
    return {"stage": "training", "epochs": len(history), "last_epoch": epochs[-1]}


def verify_evaluation(path: Path, action_mode: str) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("split") != "test" or result.get("action_mode") != action_mode:
        raise ValueError(f"{path}: wrong split or action mode")
    if not isinstance(result.get("checkpoint_epoch"), int) or result["checkpoint_epoch"] < 1:
        raise ValueError(f"{path}: no loaded PhysicsNeMo checkpoint")
    cases = result.get("cases")
    if not isinstance(cases, list) or len(cases) != 4:
        raise ValueError(f"{path}: expected four held-out CFD cases")
    summary = result.get("summary")
    if not isinstance(summary, dict) or set(summary) != set(HORIZONS):
        raise ValueError(f"{path}: expected 1/10/50-step summaries")
    ratios = {}
    for horizon in HORIZONS:
        row = summary[horizon]
        if row.get("stable") is not True or row.get("failed_segments") != 0:
            raise ValueError(f"{path}: non-finite rollout at horizon {horizon}")
        if type(row.get("segments")) is not int or row["segments"] < 4:
            raise ValueError(f"{path}: too few evaluated segments at horizon {horizon}")
        for key in EVAL_METRICS:
            nonnegative_finite(row.get(key), f"horizon {horizon} {key}")
        channels = row.get("state_channel_mae_u_v_p")
        if not isinstance(channels, list) or len(channels) != 3:
            raise ValueError(f"{path}: missing per-channel errors at horizon {horizon}")
        for index, value in enumerate(channels):
            nonnegative_finite(value, f"horizon {horizon} channel {index}")
        persistence = float(row["persistence_state_mae_physical_units"])
        ratios[horizon] = (
            float(row["state_mae_physical_units"]) / persistence
            if persistence > 0 else None
        )
        for case in cases:
            case_row = case.get("horizons", {}).get(horizon, {})
            if case_row.get("stable") is not True or case_row.get("segments", 0) < 1:
                raise ValueError(f"{path}: invalid case {case.get('case')} horizon {horizon}")
    return {
        "stage": "heldout_evaluation",
        "action_mode": action_mode,
        "checkpoint_epoch": result["checkpoint_epoch"],
        "cases": len(cases),
        "state_mae_to_persistence_ratio": ratios,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--training-history", type=Path)
    group.add_argument("--evaluation", type=Path)
    parser.add_argument("--min-epoch", type=int, default=1)
    parser.add_argument("--action-mode", choices=("observed", "zero", "sign_flip"), default="observed")
