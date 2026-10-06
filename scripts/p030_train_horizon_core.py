"""Pure project glue for the bounded FC-P030 train-only H100 diagnostic.

This module does not build or load a model or dataset.  The reviewed execution
layer supplies the existing official ``predict``/input helpers and the canonical
field aggregation functions from ``evaluate_tandem_fno.py``.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import torch

HORIZON = 100
REPORT_LEADS = (1, 10, 25, 50, 100)
ACTION_SCALE = 0.75
EXPECTED_FAMILY_COUNTS = {"base": 20, "train8": 8, "train16": 16}
EXPECTED_PHASE_COUNTS = {"b00": 15, "b02": 15, "b04": 7, "b06": 7}
EXPECTED_FRAMES = {"base": 801, "train8": 201, "train16": 129}
EXPECTED_DATASET_INDEX = {"base": 0, "train8": 1, "train16": 2}
EXPECTED_FAMILY_PHASE_COUNTS = {
    "base": {"b00": 5, "b02": 5, "b04": 5, "b06": 5},
    "train8": {"b00": 2, "b02": 2, "b04": 2, "b06": 2},
    "train16": {"b00": 8, "b02": 8},
}


def _finite(*values: torch.Tensor) -> None:
    if any(not bool(torch.isfinite(value).all()) for value in values):
        raise FloatingPointError("nonfinite P030 tensor")


def _sha_tensor(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def validate_time_grid(times: np.ndarray | torch.Tensor) -> float:
    """Apply the existing P028 whole-grid float32-quantization check."""
    array = np.asarray(times.detach().cpu() if torch.is_tensor(times) else times)
    array = np.asarray(array, dtype=np.float64).reshape(-1)
    if array.shape != (HORIZON + 1,) or not np.isfinite(array).all():
        raise ValueError("P030 requires 101 finite timestamps")
    if not np.all(np.diff(array) > 0.0):
        raise ValueError("P030 timestamps must be strictly increasing")
    tolerance = max(
        2.0 * float(np.max(np.spacing(np.abs(array).astype(np.float32)))),
        1.0e-7,
    )
    expected = array[0] + 0.1 * np.arange(HORIZON + 1, dtype=np.float64)
    if float(np.max(np.abs(array - expected))) > tolerance:
        raise ValueError("P030 timestamp grid differs from t0+0.1*arange(101)")
    return tolerance


def validate_selection(
    rows: Sequence[Mapping[str, Any]],
    original_membership: Iterable[tuple[str, int, int]],
) -> dict[str, Any]:
    """Validate the pre-metric earliest-window selection without opening HDF."""
    if len(rows) != 44:
        raise ValueError("P030 requires exactly 44 selected trajectories")
    membership = set(original_membership)
    identities: set[tuple[str, int, int]] = set()
    families: Counter[str] = Counter()
    phases: Counter[str] = Counter()
    family_phases: dict[str, Counter[str]] = defaultdict(Counter)
    cases: set[str] = set()
    for row in rows:
        required = {
            "family", "dataset_index", "case", "start", "rollout_steps",
            "canonical_phase", "action_profile", "source_manifest_sha256",
            "hdf_sha256", "split", "frames",
        }
        if set(row) != required:
            raise ValueError("P030 selection row schema differs")
        family = str(row["family"])
        case = str(row["case"])
        for key in ("dataset_index", "start", "rollout_steps", "frames"):
            if type(row[key]) is not int:
                raise ValueError(f"P030 {key} must be an exact int")
        dataset_index = row["dataset_index"]
        identity = (case, row["start"], dataset_index)
        if (
            family not in EXPECTED_FAMILY_COUNTS
            or dataset_index != EXPECTED_DATASET_INDEX[family]
            or row["split"] != "train"
            or row["start"] != 0
            or row["rollout_steps"] != HORIZON
            or row["frames"] != EXPECTED_FRAMES[family]
            or row["frames"] < HORIZON + 1
            or identity not in membership
            or identity in identities
            or case in cases
        ):
            raise ValueError("P030 selected-window identity differs")
        phase = str(row["canonical_phase"])
        if phase not in EXPECTED_PHASE_COUNTS or not str(row["action_profile"]):
            raise ValueError("P030 phase/action metadata differs")
        for key in ("source_manifest_sha256", "hdf_sha256"):
            value = str(row[key])
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise ValueError(f"invalid {key}")
        identities.add(identity)
        cases.add(case)
        families[family] += 1
        phases[phase] += 1
        family_phases[family][phase] += 1
    actual_family_phases = {
        family: dict(counts) for family, counts in family_phases.items()
    }
    if (dict(families) != EXPECTED_FAMILY_COUNTS
            or dict(phases) != EXPECTED_PHASE_COUNTS
            or actual_family_phases != EXPECTED_FAMILY_PHASE_COUNTS):
        raise ValueError("P030 family/phase coverage differs")
    return {
        "windows": 44,
        "family_counts": dict(families),
        "phase_counts": dict(phases),
        "family_phase_counts": actual_family_phases,
        "start": 0,
        "rollout_steps": HORIZON,
    }


def _window_tensors(sample: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    required = {"state", "target_state", "mask", "omega", "target_force", "time"}
    if set(sample) != required:
        raise ValueError("P030 sample schema differs")
    state = sample["state"]
    target = sample["target_state"]
    mask = sample["mask"]
    omega = sample["omega"]
    force = sample["target_force"]
    time = sample["time"]
    if state.ndim != 4 or state.shape[0] != 1 or state.shape[1] != 3:
        raise ValueError("state must be [1,3,H,W]")
    _, _, height, width = state.shape
    expected = {
        "target_state": ((1, HORIZON, 3, height, width), target),
        "mask": ((1, 1, height, width), mask),
        "omega": ((1, HORIZON + 1, 1), omega),
        "target_force": ((1, HORIZON, 4), force),
        "time": ((1, HORIZON + 1, 1), time),
    }
    for name, (shape, value) in expected.items():
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} shape differs")
        if value.device != state.device or value.dtype != state.dtype:
            raise ValueError(f"{name} dtype/device differs")
    if state.requires_grad or any(value.requires_grad for _, value in expected.values()):
        raise ValueError("P030 input tensors must not require gradients")
    _finite(state, target, mask, omega, force, time)
    if not bool(torch.all((mask == 0) | (mask == 1))) or not bool(mask.sum() > 0):
        raise ValueError("finite nonempty binary mask required")
    return state, target, mask, omega, force, time


def _validated_record_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    if set(identity) != {"case", "start", "dataset_index"}:
        raise ValueError("P030 record identity schema differs")
    if not isinstance(identity["case"], str) or not identity["case"]:
        raise ValueError("P030 record case differs")
    if type(identity["start"]) is not int or type(identity["dataset_index"]) is not int:
        raise ValueError("P030 record start/dataset_index must be exact ints")
    if identity["start"] != 0 or identity["dataset_index"] not in (0, 1, 2):
        raise ValueError("P030 record identity differs")
    return dict(identity)


def rollout_window(
    flow_model: torch.nn.Module,
    aerodynamic_model: torch.nn.Module,
    sample: Mapping[str, torch.Tensor],
    *,
    predict_fn: Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    make_inputs_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    field_error_sums_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    force_mean: torch.Tensor,
    force_std: torch.Tensor,
    identity: Mapping[str, Any],
    action_scale: float = ACTION_SCALE,
    guard: Callable[[], None] = lambda: None,
) -> dict[str, Any]:
    """Roll one batch-one H100 once and retain only small sufficient statistics."""
    state, targets, mask, omega, forces, times = _window_tensors(sample)
    if float(action_scale) != ACTION_SCALE:
        raise ValueError("P030 action scale must remain 0.75")
    if flow_model.training or aerodynamic_model.training:
        raise ValueError("P030 candidates must be eval")
    if any(parameter.requires_grad or parameter.grad is not None
           for model in (flow_model, aerodynamic_model) for parameter in model.parameters()):
        raise ValueError("P030 candidates must be frozen with no parameter gradients")
    stats = (state_mean, state_std, force_mean, force_std)
    if any(value.device != state.device or value.dtype != state.dtype for value in stats):
        raise ValueError("normalization dtype/device differs")
    if state_mean.shape != (3, 1, 1) or state_std.shape != (3, 1, 1):
        raise ValueError("state normalization shape differs")
    if force_mean.shape != (4,) or force_std.shape != (4,):
        raise ValueError("force normalization shape differs")
    _finite(*stats)
    if not bool(torch.all(state_std > 0)) or not bool(torch.all(force_std > 0)):
        raise ValueError("normalization standard deviations must be positive")
    tolerance = validate_time_grid(times[0, :, 0])

    current = state * mask
    field_sums: list[list[list[float]]] = []
    predicted_forces: list[list[float]] = []
    target_forces: list[list[float]] = []
    state_digests: list[str] = []
    state_channel_sums: list[list[float]] = []
    with torch.no_grad():
        for step in range(HORIZON):
            guard()
            inputs = make_inputs_fn(current, mask, omega[:, step], omega[:, step + 1])
            _, force_normalized = predict_fn(aerodynamic_model, inputs, mask)
            delta, unused_force = predict_fn(flow_model, inputs, mask)
            if force_normalized.shape != (1, 4) or unused_force.shape != (1, 4):
                raise ValueError("official force output shape differs")
            if delta.shape != current.shape:
                raise ValueError("official flow delta shape differs")
            current = (current + delta) * mask
            physical_error = (current - targets[:, step]) * state_std
            physical_target = targets[:, step] * state_std + state_mean
            sums = field_error_sums_fn(physical_error, physical_target, mask)
            if sums.shape != (2, 3):
                raise ValueError("canonical field sums shape differs")
            physical_force = force_normalized * force_std + force_mean
            physical_target_force = forces[:, step] * force_std + force_mean
            _finite(current, sums, physical_force, physical_target_force)
            field_sums.append(sums.detach().cpu().double().tolist())
            predicted_forces.append(physical_force[0].detach().cpu().double().tolist())
            target_forces.append(physical_target_force[0].detach().cpu().double().tolist())
            state_digests.append(_sha_tensor(current))
            state_channel_sums.append(
                (current.double() * mask.double()).sum((0, 2, 3)).detach().cpu().tolist()
            )
    return {
        "identity": _validated_record_identity(identity),
        "rollout_steps": HORIZON,
        "time_tolerance": tolerance,
        "field_sums_by_lead": field_sums,
        "predicted_force_physical_by_lead": predicted_forces,
        "target_force_physical_by_lead": target_forces,
        "predicted_state_sha256_by_lead": state_digests,
        "predicted_state_channel_sum_by_lead": state_channel_sums,
        "flow_transition_count": HORIZON,
        "aerodynamic_evaluation_count": HORIZON,
        "full_field_history_retained": False,
    }


def _relative(
    sums: np.ndarray,
    relative_field_metrics_fn: Callable[[np.ndarray], Mapping[str, Any]],
) -> dict[str, Any]:
    result = dict(relative_field_metrics_fn(sums))
    required = {
        "field_squared_error_sums_u_v_p",
        "field_reference_squared_sums_u_v_p",
        "field_relative_l2_u_v_p",
        "velocity_relative_l2",
    }
    if set(result) != required:
        raise ValueError("canonical relative-field metric schema differs")
    return result


def _force_row(predicted: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    if predicted.shape != (4,) or target.shape != (4,):
        raise ValueError("four physical force channels required")
    absolute = np.abs(predicted - target)
    total_cd = abs((predicted[0] + predicted[2]) - (target[0] + target[2]))
    return {
        "force_channel_absolute_error": absolute.tolist(),
        "rear_cl_absolute_error": float(absolute[3]),
        "total_cd_absolute_error": float(total_cd),
    }


def summarize_records(
    records: Sequence[Mapping[str, Any]],
    *,
    relative_field_metrics_fn: Callable[[np.ndarray], Mapping[str, Any]],
    leads: Sequence[int] = REPORT_LEADS,
) -> dict[str, Any]:
    """Summarize at-lead endpoints and distinctly named cumulative prefixes."""
    if not records:
        raise ValueError("P030 records are empty")
    for row in records:
        _validated_record_identity(row["identity"])
        if row["rollout_steps"] != HORIZON:
            raise ValueError("P030 record horizon differs")
        field = np.asarray(row["field_sums_by_lead"], dtype=np.float64)
        predicted = np.asarray(row["predicted_force_physical_by_lead"], dtype=np.float64)
        target = np.asarray(row["target_force_physical_by_lead"], dtype=np.float64)
        if field.shape != (HORIZON, 2, 3) or predicted.shape != (HORIZON, 4) \
                or target.shape != (HORIZON, 4):
            raise ValueError("P030 record sufficient-statistic shape differs")
        if (not np.isfinite(field).all() or not np.isfinite(predicted).all()
                or not np.isfinite(target).all() or np.any(field < 0)):
            raise FloatingPointError("invalid P030 record sufficient statistics")
    output: dict[str, Any] = {"at_lead": {}, "cumulative_prefix_1_to_H": {}}
    for lead in leads:
        if lead < 1 or lead > HORIZON:
            raise ValueError("invalid P030 report lead")
        index = lead - 1
        endpoint_sums = np.sum(
            [np.asarray(row["field_sums_by_lead"][index], dtype=np.float64) for row in records],
            axis=0,
        )
        endpoint_force = [
            _force_row(
                np.asarray(row["predicted_force_physical_by_lead"][index], dtype=np.float64),
                np.asarray(row["target_force_physical_by_lead"][index], dtype=np.float64),
            )
            for row in records
        ]
        prefix_sums = np.sum(
            [np.asarray(row["field_sums_by_lead"][:lead], dtype=np.float64).sum(0)
             for row in records],
            axis=0,
        )
        prefix_force = [
            _force_row(np.asarray(predicted), np.asarray(target))
            for row in records
            for predicted, target in zip(
                row["predicted_force_physical_by_lead"][:lead],
                row["target_force_physical_by_lead"][:lead],
                strict=True,
            )
        ]
        output["at_lead"][str(lead)] = _metric_block(
            endpoint_sums, endpoint_force, relative_field_metrics_fn, len(records)
        )
        output["cumulative_prefix_1_to_H"][str(lead)] = _metric_block(
            prefix_sums, prefix_force, relative_field_metrics_fn, len(prefix_force)
        )
    return output


def _metric_block(sums, force_rows, relative_fn, count):
    if not np.isfinite(sums).all() or not force_rows:
        raise FloatingPointError("invalid P030 sufficient statistics")
    channels = np.asarray([row["force_channel_absolute_error"] for row in force_rows])
    rear_cl = np.asarray([row["rear_cl_absolute_error"] for row in force_rows])
    total_cd = np.asarray([row["total_cd_absolute_error"] for row in force_rows])
    if not (np.isfinite(channels).all() and np.isfinite(rear_cl).all()
            and np.isfinite(total_cd).all()):
        raise FloatingPointError("invalid P030 force errors")
    return {
        **_relative(np.asarray(sums), relative_fn),
        "force_channel_mae": channels.mean(0).tolist(),
        "rear_cl_mae": float(rear_cl.mean()),
        "total_cd_mae": float(total_cd.mean()),
        "count": int(count),
    }


def grouped_and_paired(
    k1_rows: Sequence[Mapping[str, Any]],
    p029_rows: Sequence[Mapping[str, Any]],
    metadata: Sequence[Mapping[str, Any]],
    *,
    relative_field_metrics_fn: Callable[[np.ndarray], Mapping[str, Any]],
) -> dict[str, Any]:
    """Report full rows, exact groups, and descriptive paired delta signs."""
    if not (len(k1_rows) == len(p029_rows) == len(metadata) == 44):
        raise ValueError("P030 paired rows must contain the same 44 cases")
    groups: dict[str, dict[str, list[int]]] = {
        "family": defaultdict(list), "canonical_phase": defaultdict(list),
        "action_profile": defaultdict(list),
    }
    seen: set[str] = set()
    for index, row in enumerate(metadata):
        case = str(row["case"])
        if case in seen:
            raise ValueError("duplicate P030 paired case")
        seen.add(case)
        expected_identity = {
            "case": case, "start": row["start"], "dataset_index": row["dataset_index"]
        }
        left_identity = _validated_record_identity(k1_rows[index]["identity"])
        right_identity = _validated_record_identity(p029_rows[index]["identity"])
        if left_identity != expected_identity or right_identity != expected_identity:
            raise ValueError("P030 paired record identity/order differs")
        left_reference = np.asarray(k1_rows[index]["field_sums_by_lead"], dtype=np.float64)[:, 1]
        right_reference = np.asarray(p029_rows[index]["field_sums_by_lead"], dtype=np.float64)[:, 1]
        left_targets = np.asarray(k1_rows[index]["target_force_physical_by_lead"], dtype=np.float64)
        right_targets = np.asarray(p029_rows[index]["target_force_physical_by_lead"], dtype=np.float64)
        if (left_reference.shape != (HORIZON, 3)
                or left_targets.shape != (HORIZON, 4)
                or not np.array_equal(left_reference, right_reference)
                or not np.array_equal(left_targets, right_targets)):
            raise ValueError("P030 paired physical truth/reference differs")
        for key in groups:
            groups[key][str(row[key])].append(index)
    report: dict[str, Any] = {
        "all": {
            "k1": summarize_records(k1_rows, relative_field_metrics_fn=relative_field_metrics_fn),
            "p029": summarize_records(p029_rows, relative_field_metrics_fn=relative_field_metrics_fn),
        },
        "groups": {},
        "paired_at_lead": {},
    }
    for group_key, values in groups.items():
        report["groups"][group_key] = {}
        for label, indices in sorted(values.items()):
            report["groups"][group_key][label] = {
                "count": len(indices),
                "k1": summarize_records(
                    [k1_rows[i] for i in indices], relative_field_metrics_fn=relative_field_metrics_fn
                ),
                "p029": summarize_records(
                    [p029_rows[i] for i in indices], relative_field_metrics_fn=relative_field_metrics_fn
                ),
                "paired_at_lead": _paired_at_lead(
                    [k1_rows[i] for i in indices],
                    [p029_rows[i] for i in indices],
                    [metadata[i] for i in indices],
                    relative_field_metrics_fn,
                ),
            }
    report["paired_at_lead"] = _paired_at_lead(
        k1_rows, p029_rows, metadata, relative_field_metrics_fn
    )
    return report


def _paired_at_lead(k1_rows, p029_rows, metadata, relative_field_metrics_fn):
    result = {}
    for lead in REPORT_LEADS:
        index = lead - 1
        metric_deltas: dict[str, list[float]] = defaultdict(list)
        cases = []
        for meta, left, right in zip(metadata, k1_rows, p029_rows, strict=True):
            left_field = _relative(np.asarray(left["field_sums_by_lead"][index]), relative_field_metrics_fn)
            right_field = _relative(np.asarray(right["field_sums_by_lead"][index]), relative_field_metrics_fn)
            left_force = _force_row(
                np.asarray(left["predicted_force_physical_by_lead"][index]),
                np.asarray(left["target_force_physical_by_lead"][index]),
            )
            right_force = _force_row(
                np.asarray(right["predicted_force_physical_by_lead"][index]),
                np.asarray(right["target_force_physical_by_lead"][index]),
            )
            left_velocity = left_field["velocity_relative_l2"]
            right_velocity = right_field["velocity_relative_l2"]
            if left_velocity is None or right_velocity is None:
                raise FloatingPointError("undefined paired P030 field metric")
            values = {
                "velocity_relative_l2": right_velocity - left_velocity,
                "rear_cl_absolute_error": right_force["rear_cl_absolute_error"] - left_force["rear_cl_absolute_error"],
                "total_cd_absolute_error": right_force["total_cd_absolute_error"] - left_force["total_cd_absolute_error"],
            }
            if any(not math.isfinite(value) for value in values.values()):
                raise FloatingPointError("undefined/nonfinite paired P030 metric")
            for key, value in values.items():
                metric_deltas[key].append(float(value))
            cases.append({"case": meta["case"], "deltas_p029_minus_k1": values})
        result[str(lead)] = {
            "cases": cases,
            "summary": {
                key: {
                    "count": len(values),
                    "negative": sum(value < 0 for value in values),
                    "zero": sum(value == 0 for value in values),
                    "positive": sum(value > 0 for value in values),
                    "mean_delta": math.fsum(values) / len(values),
                }
                for key, values in metric_deltas.items()
            },
        }
    return result


def assert_h1_force_identity(k1: Mapping[str, Any], p029: Mapping[str, Any]) -> None:
    left = np.asarray(k1["predicted_force_physical_by_lead"][0], dtype=np.float64)
    right = np.asarray(p029["predicted_force_physical_by_lead"][0], dtype=np.float64)
    if not np.array_equal(left, right):
        raise ValueError("P030 lead1 force identity failed")
