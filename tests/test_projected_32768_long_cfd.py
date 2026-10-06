import ast
import importlib.util
import inspect
import os
from pathlib import Path
import sys

import numpy as np
import pytest


STAGE = Path(__file__).parents[1]
SCRIPT = STAGE / "scripts" / "run_exploratory_projected_32768_ppo_long_cfd.py"
BASE = Path(os.environ.get(
    "PROJECTED_32768_BASE_DRIVER",
    "/workspace/fluid_control/artifacts/"
    "exploratory_diverse_32768_ppo_long_cfd_source_20261006_immutable/"
    "run_exploratory_diverse_32768_ppo_long_cfd.py",
))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MODULE = load("projected_long_driver", SCRIPT)
ORIGINAL = load("original_long_driver", BASE)


def physical69(omega=.125):
    result = np.arange(69, dtype=np.float32) / 100
    result[64:69] = [.3, -.4, .5, -.6, omega]
    return result


def test_exact_physical69_reflection_and_involution():
    source = physical69()
    reflected = MODULE.reflect_physical69(source)
    probes = source[:64].reshape(32, 2)
    assert np.array_equal(reflected[:64].reshape(32, 2)[:, 0], probes[::-1, 0])
    assert np.array_equal(reflected[:64].reshape(32, 2)[:, 1], -probes[::-1, 1])
    assert np.array_equal(reflected[64:], np.array([.3, .4, .5, .6, -.125], np.float32))
    assert np.array_equal(MODULE.reflect_physical69(reflected), source)


def test_projected_request_is_exactly_odd_and_calls_policy_twice():
    calls = []

    def policy(obs):
        calls.append(obs.copy())
        return float(.2 + .5 * obs[68])

    source = physical69(.1)
    result = MODULE.projected_request(policy, source)
    mirrored_result = MODULE.projected_request(policy, MODULE.reflect_physical69(source))
    assert len(calls) == 4
    assert result == pytest.approx({"raw_policy_request": .25,
                                    "reflected_policy_request": .15,
                                    "projected_policy_request": .05})
    assert mirrored_result["projected_policy_request"] == -result["projected_policy_request"]


def test_execute_calls_existing_action_filter_once_after_projection():
    tree = ast.parse(SCRIPT.read_text())
    execute = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == "execute")
    calls = [node for node in ast.walk(execute) if isinstance(node, ast.Call)]
    filters = [node for node in calls if isinstance(node.func, ast.Attribute)
               and node.func.attr == "apply_action_rate_limit"]
    projections = [node for node in calls if isinstance(node.func, ast.Name)
                   and node.func.id == "projected_request"]
    assert len(filters) == 1 and len(projections) == 1
    assert filters[0].lineno > projections[0].lineno
    assert isinstance(filters[0].args[0], ast.Subscript)
    assert filters[0].args[0].slice.value == "projected_policy_request"


def test_per_step_record_persists_all_three_requests_before_progress():
    source = SCRIPT.read_text()
    assert "'raw_policy_request': raw" in source
    assert "'reflected_policy_request': reflected" in source
    assert "'projected_policy_request': projected" in source
    assert "**requests, **action" in source
    assert source.index("rows.append(") < source.index("base.atomic_json(output/'progress.json'")


def test_original_numerical_helpers_and_source_closure_are_unchanged():
    assert MODULE.SOURCE_PATHS == ORIGINAL.SOURCE_PATHS
    for name in ("observation", "validate_training", "training_bindings", "validate_vec",
                 "predict", "fixed_window", "summarize"):
        assert inspect.getsource(getattr(MODULE, name)) == inspect.getsource(getattr(ORIGINAL, name))


def test_fixed_six_window_definitions_unchanged(monkeypatch, tmp_path):
    # Intercept the metric requests without reading CFD files. The window list is
    # also source-identical via the preceding helper regression.
    source = inspect.getsource(MODULE.summarize)
    for text in ("('early_12p4',148.,160.4,False)",
                 "('early_first_6p2',148.,154.2,False)",
                 "('early_trailing_6p2',154.2,160.4,False)",
                 "('primary_final_60',168.,228.,False)",
                 "('historical_inclusive_final_60',168.,228.,True)",
                 "('full_80',148.,228.,False)"):
        assert text in source


def test_comparison_rejects_wrong_window_or_policy_identity():
    base = {"status": MODULE.ORIGINAL_STATUS, "cycles": 800,
            "policy_sha256": "wrong", "vecnormalize_sha256": "wrong", "windows": {}}
    with pytest.raises(ValueError, match="same final32768"):
        MODULE.compare_original({}, base)


def test_projected_result_status_distinct_from_original():
    source = SCRIPT.read_text()
    assert "EXPLORATORY_PROJECTED_32768_PPO_LONG_CFD_COMPLETE_NOT_ADMISSION" in source
    assert "not a concurrent third branch" in source
    assert MODULE.STATUS != ORIGINAL.STATUS
