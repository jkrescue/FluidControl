"""Official HydroGym adapter for direct, segmented real-OpenFOAM control.

The CFD process intentionally lives outside the policy container. Each
environment talks to one run-owned host worker over an AF_UNIX JSON-lines
socket. This module contains no surrogate model and never opens curated HDF5.
"""

from __future__ import annotations

import json
import socket
from collections import deque
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol

import gymnasium as gym
import numpy as np
from hydrogym import FlowEnv, PDEBase, TransientSolver

from .canonical_joint_v1 import (
    ACTION_LIMIT,
    CONTROL_DT,
    DEFAULT_WINDOW_SECONDS,
    MAX_DELTA_OMEGA,
    canonical_joint_cost_components,
    causal_window_ledger,
    validate_baseline,
)


class WorkerTransport(Protocol):
    """Minimal synchronous transport used by a single environment."""

    def request(self, payload: Mapping[str, object]) -> dict: ...

    def close(self) -> None: ...


class UnixJSONClient:
    """One-request/one-response JSON-lines client with bounded messages."""

    def __init__(self, path: str | Path, *, timeout: float = 120.0) -> None:
        self.path = str(path)
        self.timeout = float(timeout)
        if not np.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("socket timeout must be positive and finite")
        self._socket: socket.socket | None = None
        self._reader = None

    def _connect(self) -> None:
        if self._socket is not None:
            return
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(self.timeout)
        connection.connect(self.path)
        self._socket = connection
        self._reader = connection.makefile("rb")

    def request(self, payload: Mapping[str, object]) -> dict:
        self._connect()
        assert self._socket is not None and self._reader is not None
        encoded = json.dumps(dict(payload), separators=(",", ":")).encode() + b"\n"
        if len(encoded) > 1024 * 1024:
            raise ValueError("worker request exceeds 1 MiB")
        self._socket.sendall(encoded)
        line = self._reader.readline(8 * 1024 * 1024 + 1)
        if not line:
            raise ConnectionError("direct-CFD worker closed the socket")
        if len(line) > 8 * 1024 * 1024:
            raise ValueError("worker response exceeds 8 MiB")
        response = json.loads(line)
        if not isinstance(response, dict):
            raise TypeError("worker response must be a JSON object")
        if response.get("ok") is not True:
            raise RuntimeError(f"direct-CFD worker failed: {response.get('error')}")
        return response

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        if self._socket is not None:
            self._socket.close()
            self._socket = None


class DirectOpenFOAMFlow(PDEBase):
    """Canonical 69D tandem-cylinder flow state backed by a host CFD worker."""

    DEFAULT_DT = CONTROL_DT
    MAX_CONTROL = ACTION_LIMIT

    def __init__(
        self,
        *,
        transport_factory: Callable[[], WorkerTransport],
        baseline: Mapping[str, float | str],
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
    ) -> None:
        self.transport = transport_factory()
        self.baseline = validate_baseline(baseline)
        self.window_seconds = float(window_seconds)
        if not np.isfinite(self.window_seconds) or self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive and finite")
        self._history: deque[tuple[float, np.ndarray]] = deque()
        self.observation = np.zeros(69, dtype=np.float32)
        self.omega = 0.0
        self.requested_omega = 0.0
        self.applied_delta = 0.0
        self.rate_limited = False
        self.worker_info: dict = {}
        self.reset_count = 0
        super().__init__()

    @property
    def num_inputs(self) -> int:
        return 1

    @property
    def num_outputs(self) -> int:
        return 69

    def load_mesh(self, name: str):
        return {"backend": "host_openfoam_segment_worker"}

    def initialize_state(self) -> None:
        self.q = {"backend": "host_openfoam_segment_worker"}

    def init_bcs(self) -> None:
        return None

    def set_state(self, q: dict) -> None:
        self.q = dict(q)

    def copy_state(self, deepcopy: bool = True) -> dict:
        # FlowEnv retains this token, but reset always asks the host worker to
        # materialize a new immutable episode generation on disk.
        return {"backend": "host_openfoam_segment_worker"}

    def reset(self, q0: dict | None = None, t: float = 0.0) -> None:
        response = self.transport.request({"command": "reset"})
        observation = np.asarray(response["observation"], dtype=np.float32)
        times = np.asarray(response["prehistory_times"], dtype=np.float64)
        forces = np.asarray(response["prehistory_forces"], dtype=np.float64)
        if observation.shape != (69,) or not np.isfinite(observation).all():
            raise ValueError("worker reset returned an invalid 69D observation")
        if forces.ndim != 2 or forces.shape[1] != 4 or len(times) != len(forces):
            raise ValueError("worker reset returned invalid force prehistory")
        if len(times) < 2 or not np.isfinite(times).all() or not np.isfinite(forces).all():
            raise ValueError("worker reset prehistory is incomplete or non-finite")
        if np.any(np.diff(times) <= 0):
            raise ValueError("worker reset prehistory times must increase")
        restart_time = float(response["time"])
        if times[-1] > restart_time + 1e-8:
            raise ValueError("force prehistory contains future information")
        if not np.isclose(observation[-1], 0.0, rtol=0.0, atol=1e-8):
            raise ValueError("direct-CFD episodes must restart at zero rotation")
        self.observation = observation
        self.omega = 0.0
        self.requested_omega = 0.0
        self.applied_delta = 0.0
        self.rate_limited = False
        self.t = restart_time
        self._history.clear()
        cutoff = restart_time - self.window_seconds
        for sample_time, sample_force in zip(times, forces, strict=True):
            if sample_time >= cutoff - CONTROL_DT - 1e-8:
                self._history.append((float(sample_time), sample_force.copy()))
        self._trim_history()
        if not self.canonical_ledger()["window_ready"]:
            raise ValueError("actual causal prehistory does not fill the reward window")
        self.worker_info = dict(response.get("worker_info", {}))
        self.reset_count += 1
        self.q = {"episode": self.worker_info.get("episode")}
        self.reset_controls()

    def save_checkpoint(self, filename: str) -> None:
        raise NotImplementedError("direct CFD checkpoints are host-owned episode cases")

    def load_checkpoint(self, filename: str) -> None:
        raise NotImplementedError("direct CFD checkpoints are host-owned episode cases")

    def get_observations(self) -> np.ndarray:
        return self.observation.copy()

    def _trim_history(self) -> None:
        cutoff = self.t - self.window_seconds
        while len(self._history) > 1 and self._history[0][0] < cutoff - 1e-10:
            self._history.popleft()

    def accept_step(self, response: Mapping[str, object], requested: float) -> None:
        observation = np.asarray(response["observation"], dtype=np.float32)
        if observation.shape != (69,) or not np.isfinite(observation).all():
            raise FloatingPointError("worker step returned invalid 69D observation")
        applied = float(response["applied_omega"])
        previous = self.omega
        delta = applied - previous
        if abs(applied) > ACTION_LIMIT + 1e-8 or abs(delta) > MAX_DELTA_OMEGA + 1e-8:
            raise ValueError("worker violated canonical action/rate contract")
        step_time = float(response["time"])
        if not np.isclose(step_time, self.t + CONTROL_DT, rtol=0.0, atol=2e-6):
            raise ValueError("worker violated the 0.1-D/U control clock")
        if not np.isclose(observation[-1], applied, rtol=0.0, atol=1e-6):
            raise ValueError("worker observation/action mismatch")
        force = observation[64:68].astype(np.float64)
        self.observation = observation
        self.requested_omega = float(requested)
        self.omega = applied
        self.applied_delta = delta
        self.rate_limited = bool(response["rate_limited"])
        self.t = step_time
        self._history.append((step_time, force))
        self._trim_history()
        self.worker_info = dict(response.get("worker_info", {}))

    def canonical_ledger(self) -> dict:
        times = np.asarray([row[0] for row in self._history], dtype=np.float64)
        forces = np.stack([row[1] for row in self._history])
        return causal_window_ledger(
            times, forces, self.baseline, window_seconds=self.window_seconds
        )

    def objective_terms(self) -> dict[str, float]:
        return canonical_joint_cost_components(
            self.canonical_ledger(),
            omega=self.omega,
            delta_omega=self.applied_delta,
            action_scale=ACTION_LIMIT,
            max_delta_omega=MAX_DELTA_OMEGA,
        )

    def evaluate_objective(self, q=None) -> float:
        if q is not None:
            raise NotImplementedError("another direct-CFD state cannot be evaluated")
        return float(sum(self.objective_terms().values()))

    def render(self, **kwargs):
        raise NotImplementedError("inspect the preserved OpenFOAM episode case")

    def close_transport(self) -> None:
        self.transport.close()


class DirectOpenFOAMStepper(TransientSolver):
    """Advance exactly one 0.1-D/U host OpenFOAM segment."""

    def step(self, iter: int, control=None, **kwargs) -> DirectOpenFOAMFlow:
        flow: DirectOpenFOAMFlow = self.flow
        action = np.asarray(control, dtype=np.float64).reshape(-1)
        if action.shape != (1,) or not np.isfinite(action).all():
            raise ValueError("expected one finite rear-cylinder rotation action")
        requested = float(action[0])
        if abs(requested) > ACTION_LIMIT + 1e-6:
            raise ValueError("requested action lies outside canonical support")
        response = flow.transport.request(
            {"command": "step", "requested_omega": requested, "iteration": int(iter)}
        )
        flow.accept_step(response, requested)
        flow.set_control([flow.omega])
        return flow


class DirectCFDRewardAudit(gym.Wrapper):
    """Make every real-CFD reward component and safety record observable."""

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        flow: DirectOpenFOAMFlow = self.env.flow
        components = {
            f"reward_{key}": -CONTROL_DT * float(value)
            for key, value in flow.objective_terms().items()
        }
        if not np.isclose(reward, sum(components.values()), rtol=0.0, atol=1e-7):
            raise AssertionError("HydroGym reward differs from canonical components")
        truncated = bool(truncated or self.env.iter >= self.env.max_steps)
        ledger = flow.canonical_ledger()
        info.update(
            **components,
            requested_omega=flow.requested_omega,
            applied_omega=flow.omega,
            applied_delta_omega=flow.applied_delta,
            applied_abs_rate=abs(flow.applied_delta) / CONTROL_DT,
            rate_limited=flow.rate_limited,
            canonical_joint_ledger=ledger,
            backend="real_openfoam_host_worker_not_surrogate",
            observation_dimension=69,
            cfd_time=flow.t,
            worker_info=flow.worker_info,
        )
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        try:
            self.env.unwrapped.flow.close_transport()
        finally:
            super().close()


def make_direct_cfd_env(
    *,
    transport_factory: Callable[[], WorkerTransport],
    baseline: Mapping[str, float | str],
    episode_steps: int = 128,
):
    if episode_steps != 128:
        raise ValueError("first direct-CFD PPO protocol fixes 128-step episodes")
    raw = FlowEnv(
        {
            "flow": DirectOpenFOAMFlow,
            "flow_config": {
                "transport_factory": transport_factory,
                "baseline": baseline,
            },
            "solver": DirectOpenFOAMStepper,
            "solver_config": {"dt": CONTROL_DT},
            "max_steps": episode_steps,
        }
    )
    return DirectCFDRewardAudit(raw)
