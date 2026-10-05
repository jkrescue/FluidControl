"""CPU software fixtures for P026 HydroGym routing; no model/CFD/PPO."""

from __future__ import annotations

import importlib
import sys
import types
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT
sys.path[:0] = [
    str(STAGE / "src"),
    str(STAGE / "scripts"),
    str(ROOT / "src"),
    str(ROOT / "scripts"),
]

# Another collected test may already have imported this package. Reload it from
# the canonical repository before installing the fake HydroGym dependency.
for _name in tuple(sys.modules):
    if _name == "fluid_control" or _name.startswith("fluid_control."):
        sys.modules.pop(_name)


def _install_hydrogym_fixture():
    module = types.ModuleType("hydrogym")

    class PDEBase:
        def reset(self, q0=None, t=0.0):
            self.t = t
            self.set_state(q0)

    class TransientSolver:
        pass

    class FlowEnv:
        def __init__(self, config):
            self.config = config

    module.PDEBase = PDEBase
    module.TransientSolver = TransientSolver
    module.FlowEnv = FlowEnv
    sys.modules["hydrogym"] = module


_install_hydrogym_fixture()
tandem = importlib.import_module("fluid_control.tandem_hydrogym")
history = importlib.import_module("p026_history_inference")
contract = importlib.import_module("fluid_control.dual_control_contract")
canonical = importlib.import_module("fluid_control.full40_canonical_hydrogym")


class DualFixture(torch.nn.Module):
    def __init__(self, force=0.0, *, k=4):
        super().__init__()
        self.history_length = k
        self.force = force
        self.inputs = []

    def forward(self, flow_inputs, aerodynamic_inputs):
        self.inputs.append((flow_inputs.clone(), aerodynamic_inputs.clone()))
        value = torch.zeros(
            (len(flow_inputs), 7, *flow_inputs.shape[-2:]),
            dtype=flow_inputs.dtype,
        )
        value[:, 0] = 1.0
        value[:, 3:] = float(self.force)
        return value


def runtime(k=4):
    return {
        "schema_version": 1,
        "profile": f"p026_k{k}",
        "history_length": k,
        "manifest_kind": f"FC_P026_K{k}_HISTORY_FORCE_FNO",
        "flow_input_channels": 6,
        "aerodynamic_input_channels": 6 if k == 1 else 18,
        "left_padding": "trajectory_frame0",
        "autoregressive_state_source": "frozen_flow_prediction",
        "future_state_inputs": False,
        "future_force_inputs": False,
        **{
            key: character * 64
            for key, character in zip(
                (
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
                ),
                "abcdef0123",
                strict=True,
            )
        },
    }


def flow_fixture(force=0.0, *, k=4):
    flow = object.__new__(tandem.TandemSurrogateFlow)
    flow.fno_history_runtime = runtime(k)
    flow.device = torch.device("cpu")
    flow.q = torch.zeros(3, 2, 2)
    flow.mask = torch.ones(1, 2, 2)
    flow.omega = 0.0
    flow.force = np.zeros(4, dtype=np.float32)
    flow.force_channels = ("front_cd", "front_cl", "rear_cd", "rear_cl")
    flow.force_mean = np.zeros(4, dtype=np.float32)
    flow.force_std = np.ones(4, dtype=np.float32)
    flow.MAX_CONTROL = 0.75
    flow.max_delta_omega = 0.1
    flow.network = DualFixture(force, k=k)
    flow.fno_history = history.reset_history(
        flow.q[None], torch.tensor([0.0]), k=k
    )
    flow.t = 148.0
    flow.requested_action = 0.0
    flow.applied_delta = 0.0
    flow.rate_limited = False
    flow._reward_history = deque([(148.0, flow.force.copy())])
    flow.set_control = lambda value: setattr(flow, "control", list(value))

    def record():
        flow._reward_history.append((flow.t, flow.force.copy()))

    flow.record_reward_sample = record
    return flow


def test_history_step_uses_applied_action_and_predicted_state_only():
    flow = flow_fixture()
    stepper = SimpleNamespace(flow=flow, dt=0.1)
    tandem.TandemFNOStepper.step(stepper, 0, control=[0.75])
    assert flow.omega == pytest.approx(0.1)
    assert flow.requested_action == pytest.approx(0.75)
    assert flow.fno_history.source == "autoregressive_prediction"
    torch.testing.assert_close(flow.fno_history.states[0, -1], flow.q)
    assert flow.fno_history.actions[0, -1].item() == pytest.approx(0.1 / 0.75)
    flow_inputs, aero_inputs = flow.network.inputs[0]
    assert flow_inputs.shape[1] == 6 and aero_inputs.shape[1] == 18
    torch.testing.assert_close(aero_inputs[:, 9:12], flow_inputs[:, :3])


def test_k1_also_uses_explicit_history_and_matches_six_channel_input():
    flow = flow_fixture(k=1)
    flow.force_mean[:] = np.asarray([0.2, -0.3, 0.4, -0.5], dtype=np.float32)
    flow.force_std[:] = np.asarray([1.1, 1.2, 1.3, 1.4], dtype=np.float32)
    expected = np.asarray(0.0, dtype=np.float32) * flow.force_std + flow.force_mean
    tandem.TandemFNOStepper.step(
        SimpleNamespace(flow=flow, dt=0.1), 0, control=[0.05]
    )
    flow_inputs, aero_inputs = flow.network.inputs[0]
    torch.testing.assert_close(flow_inputs, aero_inputs)
    assert flow.fno_history.k == 1
    assert np.array_equal(flow.force, expected)
    assert flow.force.dtype == np.float32


def test_nonfinite_physical_force_rolls_back_complete_runtime():
    flow = flow_fixture(force=np.finfo(np.float32).max)
    flow.force_std[:] = np.finfo(np.float32).max
    before = flow.copy_runtime_snapshot()
    with pytest.raises(FloatingPointError):
        tandem.TandemFNOStepper.step(
            SimpleNamespace(flow=flow, dt=0.1), 0, control=[0.1]
        )
    after = flow.copy_runtime_snapshot()
    assert after["time"] == before["time"]
    torch.testing.assert_close(after["state"]["field"], before["state"]["field"])
    torch.testing.assert_close(
        after["state"]["fno_history"].states,
        before["state"]["fno_history"].states,
    )
    assert len(after["reward_history"]) == len(before["reward_history"])


def test_runtime_snapshot_is_nonaliasing_and_restores_reward_and_time():
    flow = flow_fixture()
    snapshot = flow.copy_runtime_snapshot()
    snapshot["state"]["field"].add_(4)
    snapshot["state"]["fno_history"].states.add_(4)
    assert torch.count_nonzero(flow.q) == 0
    assert torch.count_nonzero(flow.fno_history.states) == 0
    clean = flow.copy_runtime_snapshot()
    tandem.TandemFNOStepper.step(
        SimpleNamespace(flow=flow, dt=0.1), 0, control=[0.1]
    )
    flow.restore_runtime_snapshot(clean)
    assert flow.t == 148.0 and len(flow._reward_history) == 1
    torch.testing.assert_close(flow.q, clean["state"]["field"])


def test_reward_record_failure_restores_complete_runtime():
    flow = flow_fixture()
    before = flow.copy_runtime_snapshot()

    def fail():
        flow._reward_history.append((flow.t, flow.force.copy()))
        raise RuntimeError("fixture reward failure")

    flow.record_reward_sample = fail
    with pytest.raises(RuntimeError, match="fixture reward failure"):
        tandem.TandemFNOStepper.step(
            SimpleNamespace(flow=flow, dt=0.1), 0, control=[0.1]
        )
    after = flow.copy_runtime_snapshot()
    assert after["time"] == before["time"]
    assert len(after["reward_history"]) == len(before["reward_history"])
    torch.testing.assert_close(
        after["state"]["fno_history"].states,
        before["state"]["fno_history"].states,
    )


def test_malformed_state_prevalidation_prevents_pdebase_time_leak():
    flow = flow_fixture()
    invalid = flow.copy_state()
    invalid["fno_history"] = history.reset_history(
        torch.ones_like(flow.q)[None], torch.tensor([0.0]), k=4
    )
    with pytest.raises(ValueError, match="current state/action"):
        flow.reset(q0=invalid, t=999.0)
    assert flow.t == 148.0


def test_p026_rejects_nonfinite_reset_and_prospective_step_time():
    flow = flow_fixture()
    before = flow.copy_runtime_snapshot()
    with pytest.raises(FloatingPointError, match="reset time"):
        flow.reset(q0=flow.copy_state(), t=float("nan"))
    assert flow.t == before["time"]
    with pytest.raises(FloatingPointError, match="prospective"):
        tandem.TandemFNOStepper.step(
            SimpleNamespace(flow=flow, dt=float("inf")), 0, control=[0.1]
        )
    assert flow.t == before["time"]


def test_snapshot_rejects_wrong_identity_and_malformed_reward_history():
    flow = flow_fixture()
    wrong = flow.copy_runtime_snapshot()
    wrong["history_runtime"] = {**wrong["history_runtime"], "manifest_kind": "other"}
    with pytest.raises(ValueError, match="identity"):
        flow.restore_runtime_snapshot(wrong)
    foreign_state = flow.copy_state()
    foreign_state["fno_history_runtime"] = {
        **foreign_state["fno_history_runtime"],
        "dual_manifest_sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="runtime identity"):
        flow.set_state(foreign_state)
    malformed = flow.copy_runtime_snapshot()
    malformed["reward_history"] = ((148.0, np.zeros(3)),)
    with pytest.raises(FloatingPointError, match="reward-history"):
        flow.restore_runtime_snapshot(malformed)
    empty = flow.copy_runtime_snapshot()
    empty["reward_history"] = ()
    with pytest.raises(ValueError, match="does not end"):
        flow.restore_runtime_snapshot(empty)
    inconsistent = flow.copy_runtime_snapshot()
    inconsistent["reward_history"] = (
        (148.0, np.ones(4, dtype=np.float64)),
    )
    with pytest.raises(ValueError, match="does not end"):
        flow.restore_runtime_snapshot(inconsistent)


def test_history_step_rejects_broadcastable_wrong_output_shape():
    flow = flow_fixture()

    def wrong_shape(flow_inputs, aerodynamic_inputs):
        del flow_inputs, aerodynamic_inputs
        return torch.zeros(2, 7, 1, 1)

    flow.network.forward = wrong_shape
    with pytest.raises(ValueError, match="output schema"):
        tandem.TandemFNOStepper.step(
            SimpleNamespace(flow=flow, dt=0.1), 0, control=[0.1]
        )


def test_legacy_state_path_retains_original_nonfinite_behavior():
    flow = flow_fixture()
    flow.fno_history_runtime = None
    state = {
        "field": torch.full_like(flow.q, float("nan")),
        "omega": float("nan"),
        "force": np.full(4, np.nan, dtype=np.float32),
    }
    # This deliberately documents compatibility, not desirable new behavior:
    # the old path did not add a finite-state gate.
    flow.set_state(state)
    assert torch.isnan(flow.q).all() and np.isnan(flow.force).all()


def test_legacy_rejects_history_network_without_explicit_identity():
    with pytest.raises(ValueError, match="explicit audited history"):
        tandem.TandemSurrogateFlow._validate_history_runtime(None, DualFixture())


def test_p026_binding_is_derived_from_validated_identity_only(monkeypatch):
    k = 4
    payload = {
        "kind": contract.P026_K4_SYSTEM_KIND,
        "history_input": {
            "schema_version": 1,
            "profile": "p026_k4",
            "history_length": 4,
            "flow_input_channels": 6,
            "aerodynamic_input_channels": 18,
            "left_padding": "trajectory_frame0",
            "autoregressive_state_source": "frozen_flow_prediction",
            "future_state_inputs": False,
            "future_force_inputs": False,
        },
        "history_state_module_sha256": contract.P026_HISTORY_STATE_SHA256,
        "history_inference_module_sha256": contract.P026_HISTORY_INFERENCE_SHA256,
        "training_protocol_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "normalization_sha256": "3" * 64,
    }
    identity = SimpleNamespace(
        payload=payload,
        manifest_sha256="4" * 64,
        flow=SimpleNamespace(model_sha256="5" * 64, state_sha256="6" * 64),
        aerodynamic=SimpleNamespace(
            model_sha256="7" * 64, state_sha256="8" * 64
        ),
    )
    binding = contract.p026_runtime_binding(identity)
    assert binding["profile"] == f"p026_k{k}"
    payload["history_input"] = {**payload["history_input"], "profile": "p026_k1"}
    with pytest.raises(ValueError, match="history input differs"):
        contract.p026_runtime_binding(identity)


def test_legacy_step_source_remains_distinct_from_history_branch():
    source = (STAGE / "src/fluid_control/tandem_hydrogym.py").read_text()
    assert "if flow.fno_history_runtime is not None:" in source
    assert "return TandemFNOStepper._history_step(self, flow, requested)" in source
    assert "raw = flow.network(inputs)" in source


def test_canonical_env_propagates_exact_runtime_without_touching_reward_math(
    monkeypatch,
):
    monkeypatch.setattr(canonical, "Full40CanonicalRewardAudit", lambda value: value)
    value = runtime()
    env = canonical.make_full40_canonical_env(
        data=Path("/fixture/data"),
        split="train",
        case="fixture",
        frame=0,
        network=DualFixture(),
        checkpoint_epoch=1,
        baseline={"fixture": True},
        episode_steps=100,
        device="cpu",
        fno_history_runtime=value,
    )
    assert env.config["flow_config"]["fno_history_runtime"] == value
    assert env.config["solver"] is tandem.TandemFNOStepper
    assert env.config["max_steps"] == 100


def test_ppo_trainer_propagates_binding_to_every_env_and_audit():
    source = (STAGE / "scripts/train_full40_hydrogym_ppo_canonical.py").read_text()
    assert "fno_history_runtime = p026_runtime_binding(dual_identity)" in source
    assert "runtime_dual.get(\"fno_history_runtime\") != fno_history_runtime" in source
    assert "fno_history_runtime=fno_history_runtime" in source
    assert '{"fno_history_runtime": fno_history_runtime}' in source
