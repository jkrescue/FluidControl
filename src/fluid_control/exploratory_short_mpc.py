"""Bounded exploratory H2 selector around the existing official K1 dual FNO.

This project glue does not alter either official model.  It evaluates five
rate-limited commands from one newly observed CFD field, holds each command for
the second predicted transition, and executes only the selected first command.
It is not a long-autoregressive admission result or a CFD-control result.
"""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


K1_MANIFEST_SHA256 = "7adca21e3a75691b10f164c342ea91995cc38060e7416dd217b8bd8e5feeacc7"
K1_KIND = "FC_P026_K1_HISTORY_FORCE_FNO"
K1_FLOW_MODEL_SHA256 = "dc41fc91d42476e052970b39fc66aed22fa72aa8b6f218a341a3abb095f42e31"
K1_FLOW_STATE_SHA256 = "4998e534d4b82b17393c217357ed18220fb8e739166a88147483bb9cc5fb771e"
K1_AERO_MODEL_SHA256 = "e2f67dbde0ab28ccd7aa46b34ee3904178c7549cd1539f4a3ae40e2bd17e67b5"
K1_AERO_STATE_SHA256 = "ab2fe103bca0a8c84156e2c9fd7ded5336f2d4236436b593fc414e564d7e92d3"
BASELINE_SHA256 = "b5b7923f30600eba25c837f3b8d6781f37afbb6fcb1649e0c0d410f32a101ed7"
K1_FORMAL_RECEIPT_SHA256 = "f2f7a50a26177c65ee048b58fb20df0aee7f4cfa42aed0edd857f08d911ef948"

ACTION_SCALE = 0.75
MAX_DELTA = 0.10
INCREMENTS = (-0.10, -0.05, 0.0, 0.05, 0.10)
HORIZON = 2


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_bound_b00_baseline(path: Path) -> tuple[float, float]:
    """Read only the pinned train b00 zero reference used by the fixed cost."""
    path = Path(path)
    _require(path.is_file() and not path.is_symlink() and _sha(path) == BASELINE_SHA256,
             "exact train baseline evidence required")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload.get("status") == "FULL40_TRAIN20_OPEN_LOOP_PHYSICS_SUMMARY",
             "train baseline status differs")
    row = payload["phases"]["b00"]["same_phase_zero"]["metrics"]
    drag = row["mean_cd_total"]
    rear_rms = row["rms_cl_rear_fluctuation"]
    _require(all(type(value) in (int, float) and math.isfinite(value) and value > 0
                 for value in (drag, rear_rms)), "train baseline values differ")
    return float(drag), float(rear_rms)


def validate_recorded_k1_failure(path: Path) -> dict:
    """Require the unchanged formal receipt and its explicit non-admission state."""
    path = Path(path)
    _require(path.is_file() and not path.is_symlink()
             and _sha(path) == K1_FORMAL_RECEIPT_SHA256,
             "exact K1 formal failure evidence required")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload.get("status") == "FC_P026_ORIGINAL_FORMAL_COMPLETE_NOT_ADMISSION"
             and payload.get("history_k") == 1
             and payload.get("scientific_admission") is False
             and payload.get("ppo_auto_launched") is False,
             "K1 formal receipt must remain non-admitted")
    return payload


def validate_k1_identity(identity) -> None:
    """Bind a loader-returned identity to the failed-long-AR K1 reference."""
    payload = identity.payload
    _require(payload.get("kind") == K1_KIND, "exact exploratory K1 kind required")
    exact = (
        (identity.flow.model_sha256, K1_FLOW_MODEL_SHA256),
        (identity.flow.state_sha256, K1_FLOW_STATE_SHA256),
        (identity.aerodynamic.model_sha256, K1_AERO_MODEL_SHA256),
        (identity.aerodynamic.state_sha256, K1_AERO_STATE_SHA256),
    )
    _require(all(actual == expected for actual, expected in exact),
             "exact exploratory K1 checkpoint pair required")


def load_bound_k1(manifest, cfg, device, *, load_dual_fno, build_model):
    """Use the reviewed role-aware loader; never treat K1 as scientifically passed."""
    adapter, identity = load_dual_fno(
        manifest, cfg, device, build_model=build_model,
        expected_manifest_sha256=K1_MANIFEST_SHA256,
    )
    validate_k1_identity(identity)
    return adapter.flow_model, adapter.aerodynamic_model, identity


def five_hold_sequences(current_omega: float) -> np.ndarray:
    """Return five H2 sequences: one bounded increment, then hold."""
    current = float(current_omega)
    _require(math.isfinite(current) and abs(current) <= ACTION_SCALE,
             "current action outside exact bound")
    next_actions = np.clip(current + np.asarray(INCREMENTS), -ACTION_SCALE, ACTION_SCALE)
    _require(bool(np.all(np.abs(next_actions - current) <= MAX_DELTA + 1e-12)),
             "candidate rate bound differs")
    return np.repeat(next_actions[:, None], HORIZON, axis=1)


def _packed(q, mask, now, nxt, build_input):
    action = torch.tensor([now], dtype=q.dtype, device=q.device) / ACTION_SCALE
    following = torch.tensor(nxt, dtype=q.dtype, device=q.device) / ACTION_SCALE
    packed = build_input(q[0:1], mask[0], action, following)[None]
    _require(packed.shape == (1, 6, *q.shape[-2:]), "exact K1 packed input required")
    return packed


def _physical_force(raw, mask, force_mean, force_std):
    _require(raw.shape == (1, 7, *mask.shape[-2:]), "official raw seven-channel output required")
    normalized = (raw[:, 3:7] * mask).sum((-2, -1)) / mask.sum((-2, -1)).clamp_min(1)
    force = normalized * force_std + force_mean
    if not bool(torch.isfinite(force).all()):
        raise FloatingPointError("nonfinite physical force")
    return force[0]


def rollout_five_h2(
    flow,
    aerodynamic,
    current_state: torch.Tensor,
    mask: torch.Tensor,
    current_omega: float,
    force_mean: torch.Tensor,
    force_std: torch.Tensor,
    *,
    build_input: Callable,
    state_abs_limit: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict five H2 force sequences from only the newly observed CFD field."""
    _require(current_state.ndim == 4 and current_state.shape[:2] == (1, 3),
             "one normalized current field required")
    _require(mask.shape == (1, 1, *current_state.shape[-2:]), "mask shape differs")
    _require(current_state.dtype == mask.dtype and current_state.device == mask.device,
             "state/mask device or dtype differs")
    _require(force_mean.shape == force_std.shape == (4,), "four-force normalization required")
    _require(force_mean.device == force_std.device == current_state.device,
             "force normalization device differs")
    _require(not current_state.requires_grad and not mask.requires_grad,
             "observed current field must not carry gradients")
    _require(bool(torch.isfinite(current_state).all() and torch.isfinite(mask).all()
                  and torch.isfinite(force_mean).all() and torch.isfinite(force_std).all()),
             "nonfinite current inference input")
    _require(bool(((mask == 0) | (mask == 1)).all() and mask.any()),
             "binary nonempty mask required")
    _require(bool((current_state * (1 - mask) == 0).all()),
             "current field must be masked")
    _require(type(state_abs_limit) in (int, float) and math.isfinite(state_abs_limit)
             and state_abs_limit > 0, "positive finite state bound required")
    for model in (flow, aerodynamic):
        _require(not model.training and all(not p.requires_grad and p.grad is None
                                            for p in model.parameters()),
                 "official models must be frozen eval")

    sequences = five_hold_sequences(current_omega)
    forces, bounds = [], []
    with torch.inference_mode():
        for sequence in sequences:
            q = current_state.clone()
            previous = float(current_omega)
            rows = []
            maximum = float(q.abs().amax().cpu())
            for following in sequence:
                packed = _packed(q, mask, previous, float(following), build_input)
                flow_raw = flow(packed)
                aero_raw = aerodynamic(packed)
                if flow_raw.shape != (1, 7, *q.shape[-2:]):
                    raise ValueError("official flow raw seven-channel output required")
                if not bool(torch.isfinite(flow_raw).all()):
                    raise FloatingPointError("nonfinite flow output")
                rows.append(_physical_force(aero_raw, mask, force_mean, force_std).cpu().numpy())
                q = (q + flow_raw[:, :3]) * mask
                if not bool(torch.isfinite(q).all()):
                    raise FloatingPointError("nonfinite predicted state")
                maximum = max(maximum, float(q.abs().amax().cpu()))
                previous = float(following)
            forces.append(np.stack(rows))
            bounds.append(maximum)
    return np.stack(forces), np.asarray(bounds, dtype=np.float64)


def cost_components(
    forces: np.ndarray,
    actions: np.ndarray,
    *,
    current_omega: float,
    baseline_total_drag: float,
    baseline_rear_cl_rms: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Fixed H2 drag/lift-energy plus small action/rate penalties."""
    values = np.asarray(forces, dtype=np.float64)
    control = np.asarray(actions, dtype=np.float64)
    _require(values.ndim == 3 and values.shape[0] == 5 and values.shape[2] == 4
             and values.shape[1] in (1, 2) and control.shape == values.shape[:2],
             "exact five-by-H1/H2 prediction contract required")
    _require(np.isfinite(values).all() and np.isfinite(control).all(),
             "finite cost inputs required")
    for value in (baseline_total_drag, baseline_rear_cl_rms):
        _require(type(value) in (int, float) and math.isfinite(value) and value > 0,
                 "positive finite train baseline required")
    total_cd = values[:, :, 0] + values[:, :, 2]
    rear_cl = values[:, :, 3]
    deltas = np.diff(np.concatenate((np.full((5, 1), current_omega), control), 1), axis=1)
    components = {
        "mean_total_drag": total_cd.mean(1) / baseline_total_drag,
        "rear_cl_variance": rear_cl.var(1) / baseline_rear_cl_rms**2,
        "rear_cl_mean_squared": rear_cl.mean(1) ** 2 / baseline_rear_cl_rms**2,
        "actuation": 0.01 * np.mean((control / ACTION_SCALE) ** 2, axis=1),
        "rate": 0.01 * np.mean((deltas / MAX_DELTA) ** 2, axis=1),
    }
    return np.sum(np.stack(tuple(components.values())), axis=0), components


def select_exploratory_action(
    forces: np.ndarray,
    state_bounds: np.ndarray,
    *,
    current_omega: float,
    state_abs_limit: float,
    baseline_total_drag: float,
    baseline_rear_cl_rms: float,
) -> dict:
    """Select by H2 only; expose H1 costs as diagnostics, execute first action."""
    actions = five_hold_sequences(current_omega)
    total, components = cost_components(
        forces, actions, current_omega=current_omega,
        baseline_total_drag=baseline_total_drag,
        baseline_rear_cl_rms=baseline_rear_cl_rms,
    )
    bounds = np.asarray(state_bounds, dtype=np.float64)
    _require(bounds.shape == (5,) and np.isfinite(bounds).all(), "state bounds differ")
    feasible = bounds <= state_abs_limit
    ranked = np.where(feasible, total, np.inf)
    _require(bool(np.isfinite(ranked).any()), "no finite state-bounded candidate")
    # Exact ties prefer the smallest first rate change, then the fixed candidate order.
    order = np.lexsort((np.arange(5), np.abs(actions[:, 0] - current_omega), ranked))
    selected = int(order[0])
    h1_total, h1_components = cost_components(
        forces[:, :1], actions[:, :1],
        current_omega=current_omega, baseline_total_drag=baseline_total_drag,
        baseline_rear_cl_rms=baseline_rear_cl_rms,
    )
    return {
        "status": "EXPLORATORY_SHORT_H2_SELECTION_NOT_ADMISSION",
        "selected_index": selected,
        "selected_action": float(actions[selected, 0]),
        "execute_only_first_action": True,
        "actions": actions.tolist(),
        "predicted_forces_h2": np.asarray(forces).tolist(),
        "h2_cost": total.tolist(),
        "h2_components": {key: value.tolist() for key, value in components.items()},
        "h1_diagnostic_cost": h1_total.tolist(),
        "h1_diagnostic_components": {
            key: value.tolist() for key, value in h1_components.items()
        },
        "state_bounds": bounds.tolist(),
        "state_feasible": feasible.tolist(),
        "original_long_ar_gate_passed": False,
        "scientific_admission": False,
    }


def plan_from_current_observation(
    flow,
    aerodynamic,
    current_state,
    mask,
    current_omega,
    force_mean,
    force_std,
    *,
    build_input,
    state_abs_limit,
    baseline_total_drag,
    baseline_rear_cl_rms,
) -> dict:
    """Stateless replan entry: every call starts from a fresh observed CFD field."""
    forces, bounds = rollout_five_h2(
        flow, aerodynamic, current_state, mask, current_omega,
        force_mean, force_std, build_input=build_input,
        state_abs_limit=state_abs_limit,
    )
    return select_exploratory_action(
        forces, bounds, current_omega=current_omega,
        state_abs_limit=state_abs_limit,
        baseline_total_drag=baseline_total_drag,
        baseline_rear_cl_rms=baseline_rear_cl_rms,
    )
