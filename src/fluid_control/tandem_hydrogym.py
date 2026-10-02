"""HydroGym core adapter backed by the existing action-conditioned PhysicsNeMo FNO.

This is a surrogate environment, not an OpenFOAM or official HydroGym CFD solver.
"""

from __future__ import annotations

import json
import math
from collections import deque
from collections.abc import Mapping
from pathlib import Path

import h5py
import gymnasium as gym
import numpy as np
import torch
from hydrogym import PDEBase, TransientSolver

from .openfoam_observation import observation_at
from .stage_c_objective import (
    stage_c_cost_components,
    stage_c_force_ledger,
    validate_stage_c_baseline,
)


class TandemSurrogateFlow(PDEBase):
    """Expose one real CFD restart frame and FNO state through HydroGym's PDE API."""

    DEFAULT_DT = 0.1
    MAX_CONTROL = 5.0

    def __init__(
        self,
        *,
        data_root: str | Path,
        split: str,
        case: str,
        frame: int,
        network: torch.nn.Module,
        device: torch.device | str,
        drag_weight: float = 1.0,
        lift_weight: float = 0.2,
        action_weight: float = 0.01,
        rate_weight: float = 0.001,
        checkpoint_epoch: int | None = None,
        max_delta_omega: float = 0.5,
        reward_mode: str = "legacy_rear",
        phase_baseline: Mapping[str, float | str] | None = None,
        shedding_period: float = 6.15,
    ) -> None:
        self.data_root = Path(data_root).resolve()
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"invalid split: {split}")
        self.split = split
        self.case_path = (self.data_root / split / f"{case}.h5").resolve()
        if (
            self.case_path.parent != self.data_root / split
            or not self.case_path.is_file()
        ):
            raise FileNotFoundError(f"missing curated case in split {split}: {case}")
        self.frame = int(frame)
        self.device = torch.device(device)
        self.network = network.to(self.device).eval()
        self.checkpoint_epoch = checkpoint_epoch
        self.weights = tuple(
            float(x) for x in (drag_weight, lift_weight, action_weight, rate_weight)
        )
        if not all(np.isfinite(x) and x >= 0 for x in self.weights):
            raise ValueError("reward weights must be finite and nonnegative")
        self.max_delta_omega = float(max_delta_omega)
        if not np.isfinite(self.max_delta_omega) or self.max_delta_omega <= 0:
            raise ValueError("max_delta_omega must be positive and finite")
        if reward_mode not in {"legacy_rear", "stage_c_total_drag"}:
            raise ValueError(f"invalid reward_mode: {reward_mode}")
        self.reward_mode = reward_mode
        self.shedding_period = float(shedding_period)
        if not np.isfinite(self.shedding_period) or self.shedding_period <= 0:
            raise ValueError("shedding_period must be positive and finite")
        self.phase_baseline = validate_stage_c_baseline(phase_baseline)
        if self.reward_mode == "stage_c_total_drag" and self.phase_baseline is None:
            raise ValueError("stage_c_total_drag requires an audited phase_baseline")

        manifest = json.loads((self.data_root / "manifest.json").read_text())
        stats = json.loads((self.data_root / "normalization.json").read_text())
        self.MAX_CONTROL = float(manifest["max_abs_omega"])
        if not np.isfinite(self.MAX_CONTROL) or self.MAX_CONTROL <= 0:
            raise ValueError("invalid action scale")
        self.state_mean = torch.tensor(
            stats["state_mean"], dtype=torch.float32, device=self.device
        )[:, None, None]
        self.state_std = torch.tensor(
            stats["state_std"], dtype=torch.float32, device=self.device
        )[:, None, None]
        self.training_state_bound = float(stats["state_abs_normalized_max_train"])
        if not np.isfinite(self.training_state_bound) or self.training_state_bound <= 0:
            raise ValueError("invalid train-split normalized state support")
        # The numerical-divergence guard must cover every real train state.
        # A fixed 20 threshold rejected 4.9% of valid train frames.
        self.max_abs_normalized_state_guard = 1.25 * self.training_state_bound
        if "all_force_mean" in stats:
            self.force_channels = tuple(stats["all_force_channels"])
            self.force_mean = np.asarray(stats["all_force_mean"], dtype=np.float32)
            self.force_std = np.asarray(stats["all_force_std"], dtype=np.float32)
            self.force_indices = (0, 1, 2, 3)
            expected_channels = ("front_cd", "front_cl", "rear_cd", "rear_cl")
        else:
            self.force_channels = tuple(stats["force_channels"])
            self.force_mean = np.asarray(stats["force_mean"], dtype=np.float32)
            self.force_std = np.asarray(stats["force_std"], dtype=np.float32)
            self.force_indices = (2, 3)
            expected_channels = ("rear_cd", "rear_cl")
        if self.force_channels != expected_channels:
            raise ValueError(
                f"unsupported force normalization schema: {self.force_channels}"
            )
        if self.reward_mode == "stage_c_total_drag" and len(self.force_channels) != 4:
            raise ValueError(
                "stage_c_total_drag requires all four cylinder-force channels"
            )
        if np.any(self.force_std <= 0) or torch.any(self.state_std <= 0):
            raise ValueError("nonpositive normalization scale")
        with h5py.File(self.case_path, "r") as handle:
            count = len(handle["state"])
            if not 0 <= self.frame < count:
                raise IndexError(f"frame {self.frame} outside 0..{count - 1}")
            physical = torch.as_tensor(
                handle["state"][self.frame], dtype=torch.float32, device=self.device
            )
            self.mask = torch.as_tensor(
                handle["mask"][self.frame], dtype=torch.float32, device=self.device
            )
            self.x = np.asarray(handle["x"][:], dtype=np.float64)
            self.y = np.asarray(handle["y"][:], dtype=np.float64)
            initial_omega = float(handle["omega"][self.frame, 0])
            initial_force = np.asarray(
                handle["force"][self.frame, list(self.force_indices)], dtype=np.float32
            )
            self.initial_cfd_time = float(handle["time"][self.frame, 0])
        self.initial_force_source = "curated_hdf5"
        if self.frame == 0:
            # The expanded CFD run starts at the copied t=80 state, but its
            # force function object first writes at t=80.005. The curated
            # frame-0 force is therefore a nearest-time sample, not t=80.
            case_root = (
                Path(__file__).resolve().parents[2]
                / "cfd"
                / "tandem_cylinders"
                / "cases"
            )
            case_config = json.loads(
                (case_root / case / "case_config.json").read_text(encoding="utf-8")
            )
            if (
                case_config.get("source_restart_case") != "tandem_backward_dt005"
                or not math.isclose(
                    float(case_config["source_restart_time"]),
                    self.initial_cfd_time,
                    abs_tol=1e-8,
                )
                or not math.isclose(initial_omega, 0.0, abs_tol=1e-8)
            ):
                raise ValueError("frame-0 CFD restart provenance mismatch")
            source_case = case_root / case_config["source_restart_case"]
            raw_initial, _ = observation_at(source_case, self.initial_cfd_time, 0.0)
            if len(self.force_channels) == 2:
                initial_force = raw_initial[64:66].copy()
                self.initial_force_source = "raw_openfoam_source_restart_t80"
            else:
                initial_force[2:4] = raw_initial[64:66]
                self.initial_force_source = (
                    "curated_front_raw_openfoam_rear_source_restart_t80"
                )
        if (
            physical.ndim != 3
            or physical.shape[0] != 3
            or self.mask.shape != (1, *physical.shape[1:])
        ):
            raise ValueError("unexpected curated state/mask dimensions")
        if not np.all(np.diff(self.x) > 0) or not np.all(np.diff(self.y) > 0):
            raise ValueError("curated coordinates must increase")
        if (
            not np.isfinite(initial_omega)
            or abs(initial_omega) > self.MAX_CONTROL + 1e-5
        ):
            raise ValueError("initial action outside declared support")
        self.initial_state = {
            "field": (
                (physical - self.state_mean) / self.state_std * self.mask
            ).detach(),
            "omega": initial_omega,
            "force": initial_force,
        }
        self.initial_state_bound = float(self.initial_state["field"].abs().amax())
        self._probe_indices = self._build_probe_indices()
        self.requested_action = initial_omega
        self.applied_delta = 0.0
        self.rate_limited = False
        self._reward_history: deque[tuple[float, np.ndarray]] = deque()
        super().__init__()

    @property
    def num_inputs(self) -> int:
        return 1

    @property
    def num_outputs(self) -> int:
        return 65 + len(self.force_channels)

    def load_mesh(self, name: str):
        return {"x": self.x, "y": self.y, "mask": self.mask}

    def initialize_state(self) -> None:
        self.set_state(self.initial_state)

    def init_bcs(self) -> None:
        pass  # FNO evolves the CFD field; it has no PDE boundary solver.

    def set_state(self, q: dict) -> None:
        field = q["field"].to(self.device).clone()
        if field.shape != (3, *self.mask.shape[1:]):
            raise ValueError("invalid surrogate field shape")
        self.q = field
        self.omega = float(q["omega"])
        self.force = np.asarray(q["force"], dtype=np.float32).copy()
        if self.force.shape != (len(self.force_channels),):
            raise ValueError(f"force must follow {self.force_channels}")

    def copy_state(self, deepcopy: bool = True) -> dict:
        return {
            "field": self.q.clone() if deepcopy else self.q,
            "omega": self.omega,
            "force": self.force.copy() if deepcopy else self.force,
        }

    def reset(self, q0: dict | None = None, t: float = 0.0) -> None:
        super().reset(q0=self.initial_state if q0 is None else q0, t=t)
        self.set_control([self.omega])
        self.requested_action = self.omega
        self.applied_delta = 0.0
        self.rate_limited = False
        self._reward_history.clear()
        self.record_reward_sample()

    def save_checkpoint(self, filename: str) -> None:
        raise NotImplementedError("surrogate restart uses curated CFD HDF5 frames")

    def load_checkpoint(self, filename: str) -> None:
        raise NotImplementedError("surrogate restart uses curated CFD HDF5 frames")

    def _build_probe_indices(self) -> list[tuple[int, int, float, float]]:
        indices = []
        for py in np.linspace(6.0, 9.0, 32):
            px = 17.0
            ix = int(np.searchsorted(self.x, px) - 1)
            iy = int(np.searchsorted(self.y, py) - 1)
            if not (0 <= ix < len(self.x) - 1 and 0 <= iy < len(self.y) - 1):
                raise ValueError("probe outside curated grid")
            if not bool(torch.all(self.mask[0, iy : iy + 2, ix : ix + 2] > 0)):
                raise ValueError("probe interpolation touches a solid cell")
            wx = float((px - self.x[ix]) / (self.x[ix + 1] - self.x[ix]))
            wy = float((py - self.y[iy]) / (self.y[iy + 1] - self.y[iy]))
            indices.append((ix, iy, wx, wy))
        return indices

    def get_observations(self) -> np.ndarray:
        physical = self.q * self.state_std + self.state_mean
        values = []
        for ix, iy, wx, wy in self._probe_indices:
            corners = physical[:2, iy : iy + 2, ix : ix + 2]
            interpolated = (
                (1 - wx) * (1 - wy) * corners[:, 0, 0]
                + wx * (1 - wy) * corners[:, 0, 1]
                + (1 - wx) * wy * corners[:, 1, 0]
                + wx * wy * corners[:, 1, 1]
            )
            values.extend(interpolated.detach().cpu().tolist())
        values.extend(float(value) for value in self.force)
        values.append(self.omega)
        observation = np.asarray(values, dtype=np.float32)
        if (
            observation.shape != (self.num_outputs,)
            or not np.isfinite(observation).all()
        ):
            raise FloatingPointError("non-finite surrogate observation")
        return observation

    def objective_terms(self) -> dict[str, float]:
        if self.reward_mode == "stage_c_total_drag":
            ledger = self.stage_c_ledger()
            return stage_c_cost_components(
                ledger,
                omega=self.omega,
                delta_omega=self.applied_delta,
                action_scale=self.MAX_CONTROL,
                max_delta_omega=self.max_delta_omega,
            )
        cd, cl = map(float, self.force[-2:])
        w_cd, w_cl, w_u, w_rate = self.weights
        return {
            "drag": w_cd * cd,
            "lift": w_cl * cl * cl,
            "actuation": w_u * self.omega * self.omega,
            "rate": w_rate * self.applied_delta * self.applied_delta,
        }

    def record_reward_sample(self) -> None:
        """Append the current force to the causal Stage-C window."""
        sample = np.asarray(self.force, dtype=np.float64).copy()
        if sample.shape != (len(self.force_channels),) or not np.isfinite(sample).all():
            raise FloatingPointError("cannot record non-finite force in reward window")
        now = float(self.t)
        self._reward_history.append((now, sample))
        cutoff = now - self.shedding_period
        while len(self._reward_history) > 1 and self._reward_history[0][0] < cutoff:
            self._reward_history.popleft()

    def stage_c_ledger(self) -> dict[str, float | int | bool | str]:
        """Return auditable one-shedding-period reward inputs and normalization."""
        if self.phase_baseline is None:
            raise RuntimeError("Stage-C ledger requires phase_baseline")
        if len(self.force_channels) != 4:
            raise RuntimeError("Stage-C ledger requires four force channels")
        times = np.asarray([item[0] for item in self._reward_history], dtype=np.float64)
        forces = np.stack([item[1] for item in self._reward_history])
        coverage = float(times[-1] - times[0])
        force_ledger = stage_c_force_ledger(forces, self.phase_baseline)
        return {
            "reward_mode": self.reward_mode,
            "window_seconds": self.shedding_period,
            "window_coverage": coverage,
            "window_samples": len(self._reward_history),
            "window_ready": coverage >= self.shedding_period - 1e-8,
            **force_ledger,
            "normalized_action": self.omega / self.MAX_CONTROL,
            "normalized_action_delta": self.applied_delta / self.max_delta_omega,
        }

    def evaluate_objective(self, q=None) -> float:
        if q is not None:
            raise NotImplementedError("objective of another state is not used")
        return float(sum(self.objective_terms().values()))

    def render(self, **kwargs):
        raise NotImplementedError("use the existing CFD/FNO plotting scripts")


class TandemFNOStepper(TransientSolver):
    """Advance one HydroGym control interval using the frozen PhysicsNeMo FNO."""

    def step(self, iter: int, control=None, **kwargs) -> TandemSurrogateFlow:
        flow: TandemSurrogateFlow = self.flow
        action = np.asarray(control, dtype=np.float64).reshape(-1)
        if action.shape != (1,) or not np.isfinite(action).all():
            raise ValueError("expected one finite rear-cylinder rotation action")
        requested = float(action[0])
        if abs(requested) > flow.MAX_CONTROL + 1e-6:
            raise ValueError("requested rotation outside declared action space")
        previous = flow.omega
        applied = float(
            np.clip(
                requested,
                previous - flow.max_delta_omega,
                previous + flow.max_delta_omega,
            )
        )
        flow.rate_limited = not np.isclose(applied, requested)
        applied = float(np.clip(applied, -flow.MAX_CONTROL, flow.MAX_CONTROL))
        height, width = flow.mask.shape[-2:]
        omega_now = torch.full(
            (1, 1, height, width),
            previous / flow.MAX_CONTROL,
            dtype=torch.float32,
            device=flow.device,
        )
        omega_next = torch.full_like(omega_now, applied / flow.MAX_CONTROL)
        inputs = torch.cat(
            (flow.q[None], flow.mask[None], omega_now, omega_next), dim=1
        )
        with torch.inference_mode():
            raw = flow.network(inputs)
            expected_outputs = 3 + len(flow.force_channels)
            if raw.ndim != 4 or raw.shape[1] != expected_outputs:
                raise ValueError(
                    f"checkpoint produced {raw.shape[1] if raw.ndim >= 2 else 'invalid'} "
                    f"channels; force schema requires {expected_outputs}"
                )
            next_field = (flow.q + raw[0, :3]) * flow.mask
            normalized_force = (raw[0, 3:expected_outputs] * flow.mask).sum(
                dim=(-2, -1)
            ) / flow.mask.sum(dim=(-2, -1)).clamp_min(1)
        if (
            not torch.isfinite(next_field).all()
            or not torch.isfinite(normalized_force).all()
        ):
            raise FloatingPointError("non-finite FNO rollout")
        predicted_force = (
            normalized_force.detach().cpu().numpy() * flow.force_std + flow.force_mean
        )
        flow.q = next_field.detach()
        flow.force = predicted_force
        flow.omega = applied
        flow.set_control([applied])
        flow.requested_action = requested
        flow.applied_delta = applied - previous
        flow.t += self.dt
        flow.record_reward_sample()
        return flow


class TandemRewardAudit(gym.Wrapper):
    """Report every physical reward term and applied action from official FlowEnv."""

    def __init__(self, env, max_abs_normalized_state: float | None = None):
        super().__init__(env)
        flow: TandemSurrogateFlow = env.flow
        if flow.reward_mode == "stage_c_total_drag":
            episode_duration = float(env.max_steps) * float(env.solver.dt)
            minimum_duration = 1.5 * flow.shedding_period
            if episode_duration + 1e-8 < minimum_duration:
                raise ValueError(
                    "Stage-C episode must cover at least 1.5 shedding periods "
                    f"({episode_duration:.3f} < {minimum_duration:.3f}); "
                    "otherwise PPO sees no useful full-window drag reward"
                )
        selected = (
            flow.max_abs_normalized_state_guard
            if max_abs_normalized_state is None
            else max_abs_normalized_state
        )
        self.max_abs_normalized_state = float(selected)
        if (
            not np.isfinite(self.max_abs_normalized_state)
            or self.max_abs_normalized_state <= 0
        ):
            raise ValueError("max_abs_normalized_state must be positive and finite")

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        flow: TandemSurrogateFlow = self.env.flow
        terms = flow.objective_terms()
        components = {
            f"reward_{key}": -float(self.env.solver.dt) * value
            for key, value in terms.items()
        }
        if abs(reward - sum(components.values())) > 1e-7:
            raise AssertionError("HydroGym reward differs from physical component sum")
        state_bound = float(flow.q.abs().amax())
        if not np.isfinite(state_bound):
            raise FloatingPointError("non-finite normalized FNO state")
        if state_bound > self.max_abs_normalized_state:
            terminated = True
            info["termination_reason"] = "normalized_state_divergence_guard"
        # HydroGym's core check_complete uses '>'; enforce exactly max_steps here.
        truncated = truncated or self.env.iter >= self.env.max_steps
        info.update(
            **components,
            **{
                f"predicted_{channel}": float(value)
                for channel, value in zip(flow.force_channels, flow.force, strict=True)
            },
            requested_omega=float(flow.requested_action),
            applied_omega=float(flow.omega),
            applied_delta_omega=float(flow.applied_delta),
            rate_limited=bool(flow.rate_limited),
            max_abs_normalized_state=state_bound,
            max_abs_normalized_state_guard=self.max_abs_normalized_state,
            training_state_bound=flow.training_state_bound,
            initial_state_bound=flow.initial_state_bound,
            checkpoint_epoch=flow.checkpoint_epoch,
            source_case=flow.case_path.stem,
            initial_force_source=flow.initial_force_source,
            initial_frame=flow.frame,
            backend="physicsnemo_fno_surrogate_not_cfd",
        )
        if flow.reward_mode == "stage_c_total_drag":
            info["stage_c_ledger"] = flow.stage_c_ledger()
        return observation, reward, terminated, truncated, info
