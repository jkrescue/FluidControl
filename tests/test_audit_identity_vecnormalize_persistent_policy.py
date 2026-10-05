import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/audit_identity_vecnormalize_persistent_policy.py"
SPEC = importlib.util.spec_from_file_location("interface_fixture", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_identity_wrapper_preserves_float32_exactly():
    observations = [np.linspace(-2.0, 2.0, 69, dtype=np.float32)]
    transformed = MODULE.identity_observations(observations)
    assert np.array_equal(transformed[0], observations[0])


def test_fixture_env_refuses_steps():
    env = MODULE.ObservationFixture()
    with pytest.raises(RuntimeError, match="must not execute"):
        env.step(np.zeros(1, dtype=np.float32))


def test_reconstruct_rejects_nonpassing_parity(tmp_path):
    with pytest.raises(ValueError, match="status/scope"):
        MODULE.reconstruct_observations(tmp_path, {"status": "FAIL"})


def test_persistent_metadata_mismatch_rejected(monkeypatch):
    class FakeProcess:
        returncode = 0

        def communicate(self, input=None, timeout=None):
            return (
                'CANONICAL_POLICY_ACTION_JSON={"requested_omega":0.1,'
                '"observation_channels":68,"deterministic":true,'
                '"policy_sha256":"sha"}\n',
                "",
            )

    monkeypatch.setattr(MODULE.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    with pytest.raises(ValueError, match="metadata differs"):
        MODULE.persistent_actions(
            ["python"], [np.zeros(69, dtype=np.float32)], "sha"
        )


def test_persistent_timeout_kills_and_drains(monkeypatch):
    class FakeProcess:
        returncode = None

        def __init__(self):
            self.calls = 0
            self.killed = False

        def communicate(self, input=None, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise MODULE.subprocess.TimeoutExpired(["python"], timeout)
            self.returncode = -9
            return "", "timed out"

        def kill(self):
            self.killed = True

    process = FakeProcess()
    monkeypatch.setattr(MODULE.subprocess, "Popen", lambda *args, **kwargs: process)
    with pytest.raises(TimeoutError, match="exceeded 30 seconds"):
        MODULE.persistent_actions(
            ["python"], [np.zeros(69, dtype=np.float32)], "sha"
        )
    assert process.killed is True
    assert process.calls == 2


def test_persistent_missing_response_rejected(monkeypatch):
    class FakeProcess:
        returncode = 0

        def communicate(self, input=None, timeout=None):
            return "", ""

    monkeypatch.setattr(MODULE.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    with pytest.raises(RuntimeError, match="response count/marker differs"):
        MODULE.persistent_actions(
            ["python"], [np.zeros(69, dtype=np.float32)], "sha"
        )


def test_negative_shape_and_nonfinite_checks(monkeypatch, tmp_path):
    class Model:
        pass

    class Helper:
        @staticmethod
        def predict(model, payload, policy_sha):
            value = np.asarray(payload, dtype=np.float32)
            if value.shape != (69,) or not np.isfinite(value).all():
                raise ValueError("expected 69 finite")

    class Completed:
        returncode = 1
        stderr = "ValueError: PPO checkpoint SHA differs"

    monkeypatch.setattr(MODULE.subprocess, "run", lambda *args, **kwargs: Completed())
    result = MODULE.negative_contract_checks(
        Helper(), Model(), tmp_path / "infer.py", tmp_path / "policy.zip", "a" * 64
    )
    assert result == {
        "all_rejected": True,
        "cases": ["68_channels", "70_channels", "nonfinite", "wrong_policy_sha256"],
    }
