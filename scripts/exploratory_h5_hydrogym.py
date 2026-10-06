"""Explicit exploratory H5 interface; never the canonical 100-step admission path.

The actual HydroGym flow, stepper, observations, prehistory and cost are reused.
Only episode truncation is new. Importing this module does not load a model/data.
"""

from __future__ import annotations

import math
from pathlib import Path

HORIZON = 5
TRAIN_CASES = tuple(
    f"matched_start_acquisition_train_b{p}_zero" for p in ("00", "02", "04", "06")
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_transition(
    *,
    step,
    reward,
    components,
    state_bound,
    state_limit,
    terminated=False,
    truncated=False,
):
    """Pure checks shared by the actual wrapper and synthetic unit fixtures."""
    require(type(step) is int and 1 <= step <= HORIZON, "H5 step outside episode")
    require(
        components and all(math.isfinite(float(x)) for x in components.values()),
        "nonfinite reward components",
    )
    require(
        math.isfinite(reward)
        and math.isclose(reward, sum(components.values()), rel_tol=0, abs_tol=1e-7),
        "HydroGym reward differs from canonical components",
    )
    require(
        math.isfinite(state_bound) and math.isfinite(state_limit) and state_limit > 0,
        "invalid normalized-state bound",
    )
    terminated = bool(terminated or state_bound > state_limit)
    # A divergence is a true termination, NOT a timeout eligible for bootstrap.
    truncated = bool(not terminated and (truncated or step == HORIZON))
    return terminated, truncated


def wrap_exploratory_h5(raw):
    """Attach the explicit H5 audit to an actual HydroGym FlowEnv."""
    import gymnasium as gym

    class ExploratoryH5Audit(gym.Wrapper):
        """Not Full40CanonicalRewardAudit and not an override of its gate."""

        def reset(self, *, seed=None, options=None):
            observation, info = self.env.reset(seed=seed, options=options)
            flow = self.env.flow
            require(
                len(flow._reward_history) == 62, "real 62-sample prehistory required"
            )
            require(
                flow.t == flow.initial_cfd_time, "reset must restore actual start time"
            )
            require(
                flow.frame == 0 and flow.num_outputs == 69, "reset identity differs"
            )
            return observation, {
                **info,
                "exploratory_horizon": HORIZON,
                "scientific_admission": False,
            }

        def step(self, action):
            observation, reward, terminated, truncated, info = self.env.step(action)
            flow = self.env.flow
            components = {
                f"reward_{k}": -float(self.env.solver.dt) * float(v)
                for k, v in flow.objective_terms().items()
            }
            bound = float(flow.q.abs().amax())
            terminated, truncated = audit_transition(
                step=self.env.iter,
                reward=float(reward),
                components=components,
                state_bound=bound,
                state_limit=flow.max_abs_normalized_state_guard,
                terminated=terminated,
                truncated=truncated,
            )
            if bound > flow.max_abs_normalized_state_guard:
                info["termination_reason"] = "normalized_state_divergence_guard"
            info.update(
                **components,
                applied_omega=float(flow.omega),
                requested_omega=float(flow.requested_action),
                applied_delta_omega=float(flow.applied_delta),
                rate_limited=bool(flow.rate_limited),
                max_abs_normalized_state=bound,
                source_case=flow.case_path.stem,
                cfd_time=float(flow.t),
                initial_cfd_time=float(flow.initial_cfd_time),
                episode_step=self.env.iter,
                observation_dimension=69,
                predicted_force=[float(x) for x in flow.force],
                predicted_samples_in_reward_window=self.env.iter,
                measured_samples_in_reward_window=62 - self.env.iter,
                canonical_joint_ledger=flow.canonical_ledger(),
                exploratory_horizon=HORIZON,
                scientific_admission=False,
                backend="actual_hydrogym_frozen_official_fno_not_cfd",
            )
            return observation, reward, terminated, truncated, info

    require(
        raw.max_steps == HORIZON and raw.flow.num_outputs == 69,
        "environment schema differs",
    )
    return ExploratoryH5Audit(raw)


def make_exploratory_env(
    *,
    data,
    case,
    network,
    checkpoint_epoch,
    baseline,
    device,
    cases_root,
    fno_history_runtime,
):
    """Actual official HydroGym FlowEnv, with an explicit project H5 wrapper."""
    require(case in TRAIN_CASES, "only four fixed train-zero starts permitted")
    require(fno_history_runtime["profile"] == "p026_k1", "fixed K1 profile required")
    require(
        all(not p.requires_grad and p.grad is None for p in network.parameters()),
        "FNO must be frozen before environment construction",
    )
    from fluid_control.full40_canonical_hydrogym import Full40CanonicalSurrogateFlow
    from fluid_control.tandem_hydrogym import TandemFNOStepper
    from hydrogym import FlowEnv

    raw = FlowEnv(
        {
            "flow": Full40CanonicalSurrogateFlow,
            "flow_config": {
                "data_root": Path(data),
                "split": "train",
                "case": case,
                "frame": 0,
                "network": network,
                "device": device,
                "checkpoint_epoch": checkpoint_epoch,
                "canonical_baseline": baseline,
                "cases_root": Path(cases_root),
                "fno_history_runtime": fno_history_runtime,
            },
            "solver": TandemFNOStepper,
            "solver_config": {"dt": 0.1},
            "max_steps": HORIZON,
        }
    )
    return wrap_exploratory_h5(raw)
