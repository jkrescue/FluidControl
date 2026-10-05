"""HydroGym core adapter backed by the existing action-conditioned PhysicsNeMo FNO.

This is a surrogate environment, not an OpenFOAM or official HydroGym CFD solver.
"""

from __future__ import annotations

import json
import importlib
import math
from collections import deque
from collections.abc import Mapping
from pathlib import Path

import h5py
import gymnasium as gym
import numpy as np
import torch
from hydrogym import PDEBase, TransientSolver

from .openfoam_observation import observation_at, total_drag_observation_at
from .openfoam_force_history import actual_causal_prehistory, sha256_file
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
        cases_root: str | Path | None = None,
        fno_history_runtime: Mapping[str, object] | None = None,
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
        self.fno_history_runtime = self._validate_history_runtime(
            fno_history_runtime, network
        )
        if self.fno_history_runtime is not None and self.frame != 0:
            raise ValueError("P026 HydroGym reset semantics require trajectory frame 0")
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
        self.cases_root = (
            Path(cases_root).resolve()
            if cases_root is not None
            else Path(__file__).resolve().parents[2]
            / "cfd"
            / "tandem_cylinders"
            / "cases"
        )

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
            case_root = self.cases_root
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
            if len(self.force_channels) == 2:
                raw_initial, _ = observation_at(source_case, self.initial_cfd_time, 0.0)
                initial_force = raw_initial[64:66].copy()
                self.initial_force_source = "raw_openfoam_source_restart_t80"
            else:
                raw_initial, _ = total_drag_observation_at(
                    source_case, self.initial_cfd_time, 0.0
                )
                initial_force[:] = raw_initial[64:68]
                self.initial_force_source = "raw_openfoam_both_cylinders_restart_t80"
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
        self.fno_history = None
        if self.fno_history_runtime is not None:
            history = self._history_module()
            self.initial_state["fno_history_runtime"] = dict(
                self.fno_history_runtime
            )
            self.initial_state["fno_history"] = history.reset_history(
                self.initial_state["field"][None],
                torch.tensor(
                    [initial_omega / self.MAX_CONTROL],
                    dtype=self.initial_state["field"].dtype,
                    device=self.device,
                ),
                k=int(self.fno_history_runtime["history_length"]),
            )
        self.initial_state_bound = float(self.initial_state["field"].abs().amax())
        self._probe_indices = self._build_probe_indices()
        self.requested_action = initial_omega
        self.applied_delta = 0.0
        self.rate_limited = False
        self._reward_history: deque[tuple[float, np.ndarray]] = deque()
        self._initial_reward_history: tuple[tuple[float, np.ndarray], ...] | None = None
        self.initial_reward_history_sources: dict | None = None
        super().__init__()

    @staticmethod
    def _validate_history_runtime(value, network):
        network_k = getattr(network, "history_length", None)
        if value is None:
            if network_k is not None:
                raise ValueError("P026 network requires an explicit audited history runtime")
            return None
        if not isinstance(value, Mapping):
            raise TypeError("history runtime identity must be a mapping")
        profile = value.get("profile")
        expected = {
            "p026_k1": (1, "FC_P026_K1_HISTORY_FORCE_FNO"),
            "p026_k4": (4, "FC_P026_K4_HISTORY_FORCE_FNO"),
        }
        if profile not in expected:
            raise ValueError("unsupported FNO history runtime profile")
        k, kind = expected[profile]
        required = {
            "profile": profile,
            "history_length": k,
            "manifest_kind": kind,
            "flow_input_channels": 6,
            "aerodynamic_input_channels": 6 if k == 1 else 18,
            "left_padding": "trajectory_frame0",
            "autoregressive_state_source": "frozen_flow_prediction",
            "future_state_inputs": False,
            "future_force_inputs": False,
        }
        if any(value.get(key) != expected_value for key, expected_value in required.items()):
            raise ValueError("P026 FNO history runtime identity differs")
        if network_k != k:
            raise ValueError("P026 network and runtime history lengths differ")
        for key in (
            "dual_manifest_sha256",
            "flow_model_sha256",
            "flow_state_sha256",
            "aerodynamic_model_sha256",
            "aerodynamic_state_sha256",
            "history_state_module_sha256",
            "history_inference_module_sha256",
            "training_protocol_sha256",
            "config_sha256",
            "normalization_sha256",
        ):
            digest = value.get(key)
            if not isinstance(digest, str) or len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"P026 history runtime {key} is invalid")
        return dict(value)

    @staticmethod
    def _history_module():
        return importlib.import_module("p026_history_inference")

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

    def _validated_state(self, q: dict) -> dict:
        field = q["field"].to(self.device).clone()
        if field.shape != (3, *self.mask.shape[1:]):
            raise ValueError("invalid surrogate field shape")
        omega = float(q["omega"])
        force = np.asarray(q["force"], dtype=np.float32).copy()
        if force.shape != (len(self.force_channels),):
            raise ValueError(f"force must follow {self.force_channels}")
        if not torch.isfinite(field).all() or not np.isfinite(omega) or not np.isfinite(force).all():
            raise FloatingPointError("nonfinite surrogate state")
        result = {"field": field, "omega": omega, "force": force}
        if self.fno_history_runtime is not None:
            if q.get("fno_history_runtime") != self.fno_history_runtime:
                raise ValueError("P026 state history runtime identity differs")
            if "fno_history" not in q:
                raise ValueError("P026 state restore requires explicit FNO history")
            history = self._history_module().clone_history(q["fno_history"])
            if history.k != self.fno_history_runtime["history_length"]:
                raise ValueError("restored FNO history length differs")
            if not torch.equal(history.states[:, -1], field[None]) or not torch.equal(
                history.actions[:, -1],
                torch.tensor(
                    [omega / self.MAX_CONTROL],
                    dtype=field.dtype,
                    device=field.device,
                ),
            ):
                raise ValueError("restored FNO history current state/action differs")
            result["fno_history"] = history
        elif "fno_history" in q:
            raise ValueError("legacy surrogate state cannot contain P026 history")
        return result

    def set_state(self, q: dict) -> None:
        if self.fno_history_runtime is None:
            # Preserve the reviewed legacy state path byte-for-byte in behavior.
            field = q["field"].to(self.device).clone()
            if field.shape != (3, *self.mask.shape[1:]):
                raise ValueError("invalid surrogate field shape")
            self.q = field
            self.omega = float(q["omega"])
            self.force = np.asarray(q["force"], dtype=np.float32).copy()
            if self.force.shape != (len(self.force_channels),):
                raise ValueError(f"force must follow {self.force_channels}")
            self.fno_history = None
            return
        state = self._validated_state(q)
        self.q = state["field"]
        self.omega = state["omega"]
        self.force = state["force"]
        self.fno_history = state.get("fno_history")

    def copy_state(self, deepcopy: bool = True) -> dict:
        result = {
            "field": self.q.clone() if deepcopy else self.q,
            "omega": self.omega,
            "force": self.force.copy() if deepcopy else self.force,
        }
        if self.fno_history_runtime is not None:
            # HistoryBuffer is immutable but owns mutable tensors. Always clone;
            # even a shallow PDE state must not alias another environment.
            result["fno_history"] = self._history_module().clone_history(
                self.fno_history
            )
            result["fno_history_runtime"] = dict(self.fno_history_runtime)
        return result

    def copy_runtime_snapshot(self) -> dict:
        """Copy the complete surrogate runtime; unlike copy_state this includes reward/time."""
        return {
            "state": self.copy_state(deepcopy=True),
            "history_runtime": (
                None
                if self.fno_history_runtime is None
                else dict(self.fno_history_runtime)
            ),
            "time": float(self.t),
            "requested_action": float(self.requested_action),
            "applied_delta": float(self.applied_delta),
            "rate_limited": bool(self.rate_limited),
            "reward_history": tuple(
                (float(sample_time), np.asarray(force).copy())
                for sample_time, force in self._reward_history
            ),
        }

    def restore_runtime_snapshot(self, snapshot: Mapping[str, object]) -> None:
        """Validate first, then restore a complete snapshot without reset padding."""
        if not isinstance(snapshot, Mapping) or set(snapshot) != {
            "state", "history_runtime", "time", "requested_action",
            "applied_delta", "rate_limited", "reward_history",
        }:
            raise ValueError("surrogate runtime snapshot schema differs")
        if snapshot["history_runtime"] != self.fno_history_runtime:
            raise ValueError("surrogate runtime snapshot identity differs")
        state = self._validated_state(snapshot["state"])
        values = [
            float(snapshot["time"]),
            float(snapshot["requested_action"]),
            float(snapshot["applied_delta"]),
        ]
        if not np.isfinite(values).all():
            raise FloatingPointError("nonfinite runtime snapshot")
        reward_history = deque()
        previous_time = None
        for sample_time, force in snapshot["reward_history"]:
            copied = np.asarray(force, dtype=np.float64).copy()
            current_time = float(sample_time)
            if (
                copied.shape != (len(self.force_channels),)
                or not np.isfinite(current_time)
                or not np.isfinite(copied).all()
                or (previous_time is not None and current_time <= previous_time)
            ):
                raise FloatingPointError("nonfinite reward-history snapshot")
            reward_history.append((current_time, copied))
            previous_time = current_time
        if (
            not reward_history
            or reward_history[-1][0] != values[0]
            or not np.array_equal(
                reward_history[-1][1].astype(np.float32), state["force"]
            )
        ):
            raise ValueError("reward-history snapshot does not end at runtime time")
        self.q = state["field"]
        self.omega = state["omega"]
        self.force = state["force"]
        self.fno_history = state.get("fno_history")
        self.t, self.requested_action, self.applied_delta = values
        self.rate_limited = bool(snapshot["rate_limited"])
        self._reward_history = reward_history
        self.set_control([self.omega])

    def reset(self, q0: dict | None = None, t: float = 0.0) -> None:
        selected = self.initial_state if q0 is None else q0
        if self.fno_history_runtime is None:
            # Do not silently strengthen or otherwise change the legacy reset.
            super().reset(q0=selected, t=t)
            self.set_control([self.omega])
            self.requested_action = self.omega
            self.applied_delta = 0.0
            self.rate_limited = False
            self._reward_history.clear()
            self.record_reward_sample()
            return
        # PDEBase.reset writes t before set_state. Validate first so a malformed
        # P026 snapshot cannot leak a new clock value into the live environment.
        if not np.isfinite(t):
            raise FloatingPointError("nonfinite P026 reset time")
        self._validated_state(selected)
        super().reset(q0=selected, t=t)
        self.set_control([self.omega])
        self.requested_action = self.omega
        self.applied_delta = 0.0
        self.rate_limited = False
        self._reward_history.clear()
        self.record_reward_sample()

    def require_actual_causal_prehistory(self) -> None:
        """Bind this frame-0 state to its exact raw-CFD causal force history."""
        if self.frame != 0:
            raise ValueError("canonical PPO causal prehistory requires frame 0")
        project = Path(__file__).resolve().parents[2]
        cases = self.cases_root
        case_config_path = (
            cases / self.case_path.stem / "case_config.json"
        ).resolve()
        if not case_config_path.is_relative_to(cases) or not case_config_path.is_file():
            raise ValueError("canonical PPO case config escapes cases root")
        case_config = json.loads(case_config_path.read_text(encoding="utf-8"))
        manifest = json.loads((self.data_root / "manifest.json").read_text())
        split_entry = manifest.get("split_manifests", {}).get(self.split, {})
        split_path = (self.data_root / str(split_entry.get("path", ""))).resolve()
        if (
            manifest.get("profile") != "matched_start_full40_v1"
            or not split_path.is_relative_to(self.data_root)
            or not split_path.is_file()
            or split_entry.get("sha256") != sha256_file(split_path)
        ):
            raise ValueError("canonical PPO dataset manifest differs")
        split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
        if (
            split_manifest.get("split") != self.split
            or self.case_path.stem not in split_manifest.get("cases", [])
            or split_manifest.get("hdf5_sha256", {}).get(self.case_path.stem)
            != sha256_file(self.case_path)
        ):
            raise ValueError("canonical PPO split/HDF5 manifest differs")
        with h5py.File(self.case_path, "r") as handle:
            if (
                handle.attrs.get("case") != self.case_path.stem
                or handle.attrs.get("split") != self.split
            ):
                raise ValueError("curated HDF5 case/split identity differs")
            curated_config = json.loads(str(handle.attrs["config_json"]))
        restart_time = float(case_config.get("source_restart_time", math.nan))
        source_name = str(case_config.get("source_restart_case", ""))
        source_case = (cases / source_name).resolve()
        if (
            case_config.get("case") != self.case_path.stem
            or case_config.get("split") != self.split
            or not math.isclose(restart_time, self.initial_cfd_time, abs_tol=1e-8)
            or not source_name
            or curated_config != case_config
            or float(case_config.get("action_target", math.nan)) != 0.0
            or not source_case.is_relative_to(cases)
        ):
            raise ValueError("canonical PPO restart provenance differs")
        times, forces, sources = actual_causal_prehistory(
            source_case,
            restart_time,
            provenance_root=project if cases.is_relative_to(project) else cases.parent,
            control_dt=self.DEFAULT_DT,
        )
        values = np.asarray(forces, dtype=np.float64)
        if values.shape != (62, 4) or not np.array_equal(
            values[-1].astype(np.float32), self.initial_state["force"]
        ):
            raise ValueError("causal force history does not match frame-0 force")
        absolute = np.asarray(times, dtype=np.float64)
        if np.any(np.diff(absolute) <= 0.0) or not np.isclose(
            absolute[-1], restart_time
        ):
            raise ValueError("causal force history clock differs")
        expected_source_sha = case_config.get("source_force_sha256")
        actual_source_sha = {
            key: rows[0]["sha256"] if len(rows) == 1 else None
            for key, rows in sources.items()
        }
        if expected_source_sha != actual_source_sha:
            raise ValueError("causal force-history source SHA differs")
        self._initial_reward_history = tuple(
            (float(timestamp), force.copy())
            for timestamp, force in zip(absolute, values, strict=True)
        )
        self.initial_reward_history_sources = {
            "case_config": str(
                case_config_path.relative_to(project)
                if case_config_path.is_relative_to(project)
                else case_config_path.relative_to(cases.parent)
            ),
            "source_restart_case": source_name,
            "source_restart_time": restart_time,
            "force_sources": sources,
        }

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

    def _history_step(
        self, flow: TandemSurrogateFlow, requested: float
    ) -> TandemSurrogateFlow:
        """Advance P026 atomically with explicit per-environment history."""
        previous = flow.omega
        applied = float(
            np.clip(
                requested,
                previous - flow.max_delta_omega,
                previous + flow.max_delta_omega,
            )
        )
        rate_limited = not np.isclose(applied, requested)
        applied = float(np.clip(applied, -flow.MAX_CONTROL, flow.MAX_CONTROL))
        height, width = flow.mask.shape[-2:]
        omega_now = torch.full(
            (1, 1, height, width),
            previous / flow.MAX_CONTROL,
            dtype=flow.q.dtype,
            device=flow.device,
        )
        omega_next = torch.full_like(omega_now, applied / flow.MAX_CONTROL)
        flow_inputs = torch.cat(
            (flow.q[None], flow.mask[None], omega_now, omega_next), dim=1
        )
        history_module = flow._history_module()
        aerodynamic_inputs = history_module.pack_history_input(
            flow.fno_history,
            flow.mask[None],
            torch.tensor(
                [applied / flow.MAX_CONTROL],
                dtype=flow.q.dtype,
                device=flow.device,
            ),
        )
        with torch.inference_mode():
            raw = flow.network(flow_inputs, aerodynamic_inputs)
            expected_outputs = 3 + len(flow.force_channels)
            if raw.shape != (1, expected_outputs, height, width):
                raise ValueError("P026 checkpoint output schema differs")
            next_field = (flow.q + raw[0, :3]) * flow.mask
            normalized_force = (raw[0, 3:expected_outputs] * flow.mask).sum(
                dim=(-2, -1)
            ) / flow.mask.sum(dim=(-2, -1)).clamp_min(1)
        if (
            not torch.isfinite(next_field).all()
            or not torch.isfinite(normalized_force).all()
        ):
            raise FloatingPointError("non-finite P026 FNO rollout")
        # Preserve the legacy NumPy float32 affine arithmetic exactly.  The
        # explicit errstate/check catches the case where finite normalized
        # outputs overflow only after conversion to physical force.
        with np.errstate(over="ignore", invalid="ignore"):
            predicted_force = (
                normalized_force.detach().cpu().numpy() * flow.force_std
                + flow.force_mean
            )
        if not np.isfinite(predicted_force).all():
            raise FloatingPointError("non-finite physical P026 force")
        next_history = history_module.advance_history(
            flow.fno_history,
            next_field.detach()[None],
            torch.tensor(
                [applied / flow.MAX_CONTROL],
                dtype=flow.q.dtype,
                device=flow.device,
            ),
        )
        next_time = float(flow.t + self.dt)
        if not np.isfinite(next_time):
            raise FloatingPointError("non-finite prospective P026 time")
        before = flow.copy_runtime_snapshot()
        try:
            flow.q = next_field.detach()
            flow.force = predicted_force
            flow.omega = applied
            flow.fno_history = next_history
            flow.set_control([applied])
            flow.requested_action = requested
            flow.applied_delta = applied - previous
            flow.rate_limited = rate_limited
            flow.t = next_time
            flow.record_reward_sample()
        except Exception:
            flow.restore_runtime_snapshot(before)
            raise
        return flow

    def step(self, iter: int, control=None, **kwargs) -> TandemSurrogateFlow:
        flow: TandemSurrogateFlow = self.flow
        action = np.asarray(control, dtype=np.float64).reshape(-1)
        if action.shape != (1,) or not np.isfinite(action).all():
            raise ValueError("expected one finite rear-cylinder rotation action")
        requested = float(action[0])
        if abs(requested) > flow.MAX_CONTROL + 1e-6:
            raise ValueError("requested rotation outside declared action space")
        if flow.fno_history_runtime is not None:
            return TandemFNOStepper._history_step(self, flow, requested)
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
