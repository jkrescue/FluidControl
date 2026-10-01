"""HydroGym core adapter backed by the existing action-conditioned PhysicsNeMo FNO.

This is a surrogate environment, not an OpenFOAM or official HydroGym CFD solver.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import gymnasium as gym
import numpy as np
import torch
from hydrogym import PDEBase, TransientSolver


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
    ) -> None:
        self.data_root = Path(data_root).resolve()
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"invalid split: {split}")
        self.split = split
        self.case_path = (self.data_root / split / f"{case}.h5").resolve()
        if self.case_path.parent != self.data_root / split or not self.case_path.is_file():
            raise FileNotFoundError(f"missing curated case in split {split}: {case}")
        self.frame = int(frame)
        self.device = torch.device(device)
        self.network = network.to(self.device).eval()
        self.checkpoint_epoch = checkpoint_epoch
        self.weights = tuple(float(x) for x in (
            drag_weight, lift_weight, action_weight, rate_weight
        ))
        if not all(np.isfinite(x) and x >= 0 for x in self.weights):
            raise ValueError("reward weights must be finite and nonnegative")
        self.max_delta_omega = float(max_delta_omega)
        if not np.isfinite(self.max_delta_omega) or self.max_delta_omega <= 0:
            raise ValueError("max_delta_omega must be positive and finite")

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
        self.force_mean = np.asarray(stats["force_mean"], dtype=np.float32)
        self.force_std = np.asarray(stats["force_std"], dtype=np.float32)
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
            initial_force = np.asarray(handle["force"][self.frame, 2:4], dtype=np.float32)
            self.initial_cfd_time = float(handle["time"][self.frame, 0])
        if physical.shape[0] != 3 or self.mask.shape != physical.shape[1:]:
            # Curator mask is normally [1,H,W], checked below instead.
            if self.mask.shape != (1, *physical.shape[1:]):
                raise ValueError("unexpected curated state/mask dimensions")
        if not np.all(np.diff(self.x) > 0) or not np.all(np.diff(self.y) > 0):
            raise ValueError("curated coordinates must increase")
        if not np.isfinite(initial_omega) or abs(initial_omega) > self.MAX_CONTROL + 1e-5:
            raise ValueError("initial action outside declared support")
        self.initial_state = {
            "field": ((physical - self.state_mean) / self.state_std * self.mask).detach(),
            "omega": initial_omega,
            "force": initial_force,
        }
        self._probe_indices = self._build_probe_indices()
        self.requested_action = initial_omega
        self.applied_delta = 0.0
        self.rate_limited = False
        super().__init__()

    @property
    def num_inputs(self) -> int:
        return 1

    @property
    def num_outputs(self) -> int:
        return 67  # 32 velocity probes x 2 + rear Cd/Cl + applied omega

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
        if self.force.shape != (2,):
            raise ValueError("rear force must be [Cd, Cl]")

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
            if not bool(torch.all(self.mask[0, iy:iy+2, ix:ix+2] > 0)):
                raise ValueError("probe interpolation touches a solid cell")
            wx = float((px - self.x[ix]) / (self.x[ix+1] - self.x[ix]))
            wy = float((py - self.y[iy]) / (self.y[iy+1] - self.y[iy]))
            indices.append((ix, iy, wx, wy))
        return indices

    def get_observations(self) -> np.ndarray:
        physical = self.q * self.state_std + self.state_mean
        values = []
        for ix, iy, wx, wy in self._probe_indices:
            corners = physical[:2, iy:iy+2, ix:ix+2]
            interpolated = (
                (1-wx)*(1-wy)*corners[:, 0, 0]
                + wx*(1-wy)*corners[:, 0, 1]
                + (1-wx)*wy*corners[:, 1, 0]
                + wx*wy*corners[:, 1, 1]
            )
            values.extend(interpolated.detach().cpu().tolist())
        values.extend((float(self.force[0]), float(self.force[1]), self.omega))
        observation = np.asarray(values, dtype=np.float32)
        if observation.shape != (self.num_outputs,) or not np.isfinite(observation).all():
            raise FloatingPointError("non-finite surrogate observation")
        return observation

    def objective_terms(self) -> dict[str, float]:
        cd, cl = map(float, self.force)
        w_cd, w_cl, w_u, w_rate = self.weights
        return {
            "drag": w_cd * cd,
            "lift": w_cl * cl * cl,
            "actuation": w_u * self.omega * self.omega,
            "rate": w_rate * self.applied_delta * self.applied_delta,
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
        applied = float(np.clip(
            requested, previous - flow.max_delta_omega, previous + flow.max_delta_omega
        ))
        flow.rate_limited = not np.isclose(applied, requested)
        applied = float(np.clip(applied, -flow.MAX_CONTROL, flow.MAX_CONTROL))
        height, width = flow.mask.shape[-2:]
        omega_now = torch.full(
            (1, 1, height, width), previous / flow.MAX_CONTROL,
            dtype=torch.float32, device=flow.device,
        )
        omega_next = torch.full_like(omega_now, applied / flow.MAX_CONTROL)
        inputs = torch.cat((
            flow.q[None], flow.mask[None], omega_now, omega_next
        ), dim=1)
        with torch.inference_mode():
            raw = flow.network(inputs)
            next_field = (flow.q + raw[0, :3]) * flow.mask
            normalized_force = (
                (raw[0, 3:5] * flow.mask).sum(dim=(-2, -1))
                / flow.mask.sum(dim=(-2, -1)).clamp_min(1)
            )
        if not torch.isfinite(next_field).all() or not torch.isfinite(normalized_force).all():
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
        return flow


class TandemRewardAudit(gym.Wrapper):
    """Report every physical reward term and applied action from official FlowEnv."""

    def __init__(self, env, max_abs_normalized_state: float = 20.0):
        super().__init__(env)
        self.max_abs_normalized_state = float(max_abs_normalized_state)
        if not np.isfinite(self.max_abs_normalized_state) or self.max_abs_normalized_state <= 0:
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
            predicted_cd=float(flow.force[0]),
            predicted_cl=float(flow.force[1]),
            requested_omega=float(flow.requested_action),
            applied_omega=float(flow.omega),
            applied_delta_omega=float(flow.applied_delta),
            rate_limited=bool(flow.rate_limited),
            max_abs_normalized_state=state_bound,
            checkpoint_epoch=flow.checkpoint_epoch,
            source_case=flow.case_path.stem,
            initial_frame=flow.frame,
            backend="physicsnemo_fno_surrogate_not_cfd",
        )
        return observation, reward, terminated, truncated, info
