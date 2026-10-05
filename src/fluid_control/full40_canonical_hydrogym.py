"""HydroGym adapter for the isolated full40 ``canonical_joint_v1`` objective.

The flow dynamics are the existing PhysicsNeMo FNO surrogate.  This module
does not alter the historical Stage-C adapter and makes no real-CFD claim.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from hydrogym import FlowEnv

from .canonical_joint_v1 import (
    ACTION_LIMIT,
    CONTROL_DT,
    DEFAULT_WINDOW_SECONDS,
    MAX_DELTA_OMEGA,
    MIN_EPISODE_STEPS,
    canonical_joint_cost_components,
    causal_window_ledger,
    validate_baseline,
)
from .tandem_hydrogym import TandemFNOStepper, TandemSurrogateFlow


class Full40CanonicalSurrogateFlow(TandemSurrogateFlow):
    """Four-force, 69-observation FNO flow with the canonical joint objective."""

    def __init__(
        self,
        *,
        canonical_baseline: Mapping[str, float | str],
        shedding_period: float = DEFAULT_WINDOW_SECONDS,
        **kwargs,
    ) -> None:
        # PDEBase.__init__ invokes reset virtually before the raw prehistory can
        # be bound.  Only that construction-time reset may use the base path.
        self._canonical_initializing = True
        super().__init__(
            reward_mode="legacy_rear",
            phase_baseline=None,
            max_delta_omega=MAX_DELTA_OMEGA,
            shedding_period=shedding_period,
            **kwargs,
        )
        if len(self.force_channels) != 4 or self.num_outputs != 69:
            raise ValueError("full40 canonical PPO requires four-force 69D observations")
        if not np.isclose(self.MAX_CONTROL, ACTION_LIMIT, rtol=0.0, atol=1e-12):
            raise ValueError("full40 FNO action normalization must be exactly 0.75")
        self.require_actual_causal_prehistory()
        self.reward_mode = "canonical_joint_v1"
        self.canonical_baseline = validate_baseline(canonical_baseline)
        self._canonical_initializing = False
        # FlowEnv snapshots the already-constructed flow without resetting it.
        # Finish construction in the same window-ready state as every Gym reset.
        self.reset()

    def reset(self, q0: dict | None = None, t: float = 0.0) -> None:
        if self._canonical_initializing:
            super().reset(q0=q0, t=t)
            return
        if self._initial_reward_history is None:
            raise RuntimeError("canonical PPO requires real causal force prehistory")
        if not (
            np.isclose(t, 0.0, rtol=0.0, atol=1e-12)
            or np.isclose(t, self.initial_cfd_time, rtol=0.0, atol=1e-12)
        ):
            raise ValueError("canonical PPO reset time differs from restart")
        if q0 is not None:
            field = q0.get("field")
            force = np.asarray(q0.get("force"), dtype=np.float32)
            if (
                not isinstance(field, torch.Tensor)
                or not torch.equal(field.to(self.device), self.initial_state["field"])
                or not np.isclose(
                    float(q0.get("omega", np.nan)),
                    float(self.initial_state["omega"]),
                    rtol=0.0,
                    atol=1e-8,
                )
                or force.shape != self.initial_state["force"].shape
                or not np.array_equal(force, self.initial_state["force"])
            ):
                raise ValueError("causal force prehistory is valid only for initial q0")
        super().reset(q0=q0, t=self.initial_cfd_time)
        history = deque(
            (timestamp, force.copy())
            for timestamp, force in self._initial_reward_history
        )
        if (
            len(history) != 62
            or not np.isclose(history[-1][0], self.t, rtol=0.0, atol=1e-8)
            or not np.array_equal(history[-1][1].astype(np.float32), self.force)
        ):
            raise AssertionError("canonical reset prehistory is incomplete")
        self._reward_history = history

    def canonical_ledger(self) -> dict:
        times = np.asarray([row[0] for row in self._reward_history], dtype=np.float64)
        forces = np.stack([row[1] for row in self._reward_history])
        return causal_window_ledger(
            times,
            forces,
            self.canonical_baseline,
            window_seconds=self.shedding_period,
        )

    def objective_terms(self) -> dict[str, float]:
        return canonical_joint_cost_components(
            self.canonical_ledger(),
            omega=self.omega,
            delta_omega=self.applied_delta,
            action_scale=self.MAX_CONTROL,
            max_delta_omega=self.max_delta_omega,
        )


class Full40CanonicalRewardAudit(gym.Wrapper):
    """Audit reward terms, actions, four forces, and the causal gate ledger."""

    def __init__(self, env, max_abs_normalized_state: float | None = None):
        super().__init__(env)
        flow: Full40CanonicalSurrogateFlow = env.flow
        if env.max_steps < MIN_EPISODE_STEPS:
            raise ValueError(
                f"canonical PPO episodes require at least {MIN_EPISODE_STEPS} steps"
            )
        selected = (
            flow.max_abs_normalized_state_guard
            if max_abs_normalized_state is None
            else max_abs_normalized_state
        )
        self.max_abs_normalized_state = float(selected)
        if not np.isfinite(self.max_abs_normalized_state) or self.max_abs_normalized_state <= 0:
            raise ValueError("state guard must be positive and finite")

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        flow: Full40CanonicalSurrogateFlow = self.env.flow
        components = {
            f"reward_{key}": -float(self.env.solver.dt) * float(value)
            for key, value in flow.objective_terms().items()
        }
        if not np.isclose(reward, sum(components.values()), rtol=0.0, atol=1e-7):
            raise AssertionError("HydroGym reward differs from canonical components")
        state_bound = float(flow.q.abs().amax())
        if not np.isfinite(state_bound):
            raise FloatingPointError("non-finite normalized FNO state")
        if state_bound > self.max_abs_normalized_state:
            terminated = True
            info["termination_reason"] = "normalized_state_divergence_guard"
        truncated = bool(truncated or self.env.iter >= self.env.max_steps)
        ledger = flow.canonical_ledger()
        info.update(
            **components,
            **{
                f"predicted_{channel}": float(value)
                for channel, value in zip(flow.force_channels, flow.force, strict=True)
            },
            requested_omega=float(flow.requested_action),
            applied_omega=float(flow.omega),
            applied_delta_omega=float(flow.applied_delta),
            applied_abs_rate=abs(float(flow.applied_delta)) / CONTROL_DT,
            rate_limited=bool(flow.rate_limited),
            max_abs_normalized_state=state_bound,
            max_abs_normalized_state_guard=self.max_abs_normalized_state,
            checkpoint_epoch=flow.checkpoint_epoch,
            source_case=flow.case_path.stem,
            initial_frame=flow.frame,
            observation_dimension=flow.num_outputs,
            canonical_joint_ledger=ledger,
            backend="physicsnemo_full40_fno_surrogate_not_cfd",
        )
        return observation, reward, terminated, truncated, info


def make_full40_canonical_env(
    *,
    data: Path,
    split: str,
    case: str,
    frame: int,
    network: torch.nn.Module,
    checkpoint_epoch: int,
    baseline: Mapping[str, float | str],
    episode_steps: int,
    device: torch.device | str,
    cases_root: Path | None = None,
):
    """Construct the official HydroGym FlowEnv around the fixed FNO stepper."""
    if split not in {"train", "validation"}:
        raise ValueError("canonical PPO adapter permits only train or validation")
    raw = FlowEnv(
        {
            "flow": Full40CanonicalSurrogateFlow,
            "flow_config": {
                "data_root": data,
                "split": split,
                "case": case,
                "frame": frame,
                "network": network,
                "device": device,
                "checkpoint_epoch": checkpoint_epoch,
                "canonical_baseline": baseline,
                "cases_root": cases_root,
            },
            "solver": TandemFNOStepper,
            "solver_config": {"dt": CONTROL_DT},
            "max_steps": episode_steps,
        }
    )
    return Full40CanonicalRewardAudit(raw)
