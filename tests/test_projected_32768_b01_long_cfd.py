import ast
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest


STAGE = Path(__file__).parents[1]
SCRIPT = STAGE / "scripts" / "run_exploratory_projected_32768_ppo_b01_long_cfd.py"
PENDING_NAME = "EXPLORATORY_PROJECTED_32768_PPO_B01_LONG_CFD_PENDING_20261006.json"
PENDING = next(path for path in (STAGE / "docs" / PENDING_NAME, STAGE / PENDING_NAME)
               if path.exists())
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


MODULE = load("projected_b01_long_driver", SCRIPT)
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
                 "predict", "fixed_window"):
        assert inspect.getsource(getattr(MODULE, name)) == inspect.getsource(getattr(ORIGINAL, name))


def test_fixed_six_windows_are_exact_relative_b01_replication():
    assert MODULE.declared_windows() == [
        ('early_12p4', 130., 142.4, False),
        ('early_first_6p2', 130., 136.2, False),
        ('early_trailing_6p2', 136.2, 142.4, False),
        ('primary_final_60', 150., 210., False),
        ('historical_inclusive_final_60', 150., 210., True),
        ('full_80', 130., 210., False),
    ]


def test_comparison_requires_successful_b00_projection_and_same_identity():
    base = {"status": MODULE.B00_PROJECTED_STATUS, "cycles": 800,
            "policy_sha256": "wrong", "vecnormalize_sha256": "wrong",
            "reflection_projection": {"definition": "wrong"}, "windows": {}}
    with pytest.raises(ValueError, match="identical policy"):
        MODULE.compare_b00_projection({}, base)


def test_projected_result_status_distinct_from_original():
    source = SCRIPT.read_text()
    assert "EXPLORATORY_PROJECTED_32768_PPO_B01_LONG_CFD_COMPLETE_NOT_ADMISSION" in source
    assert "not statistically independent or a fresh final-test split" in source
    assert MODULE.STATUS != ORIGINAL.STATUS


def test_copy_and_execute_use_real_130_restart_not_base_148_helper(tmp_path):
    source = tmp_path / 'source'
    output = tmp_path / 'output'
    output.mkdir()
    for part in ('constant', 'system', '130'):
        (source / part).mkdir(parents=True)
        (source / part / 'identity').write_text(part)
    cases = MODULE.build_pair_at(source, output)
    assert set(cases) == {'mpc', 'zero'}
    for case in cases.values():
        assert (case/'130/identity').read_text() == '130'
        assert not (case/'148').exists()
    execute = inspect.getsource(MODULE.execute)
    assert "('130', 'constant', 'system')" in execute
    assert 'total_drag_observation_at(source, START, 0.)' in execute
    assert 'round(START+.1*(step-1), 10)' in execute


def test_execute_binds_predeclared_b01_manifest_and_matched_provenance():
    execute = inspect.getsource(MODULE.execute)
    assert "inputs['phase_manifest']" in execute
    assert '6279492bd3a79eff868a4333e1f642e3be4a4d58a78e46dc47dde040e4e39603' in execute
    assert "phase['split'] == 'validation'" in execute
    assert "phase['selected']['restart_time'] == START" in execute
    assert "inputs['b01_case_config']" in execute
    assert '0b59387acf6365700a4a1b2e5b00f63f3c71abb6efcfcd10850ccd5ba4df1f7c' in execute


def test_b00_terminal_gate_precedes_model_output_and_solver_work():
    execute = inspect.getsource(MODULE.execute)
    gate = execute.index('validate_b00_projection(b00_projected_result)')
    assert gate < execute.index("pickle.loads(inputs['vecnormalize'].read_bytes())")
    assert gate < execute.index("output.mkdir()")
    assert gate < execute.index("build_pair_at(source, output)")
    assert gate < execute.index("with base.PairSolvers")
    failed = {
        'status': MODULE.B00_PROJECTED_STATUS, 'cycles': 800,
        'policy_sha256': '5ab92ebe04459419bc724b48c6e20bde2464d7b6d880396e504406aa08806d4a',
        'vecnormalize_sha256': '3161ba46c65bac3bc23fa4ccb300c52c95fda63ee4190d9f30d2f0bd4b9040ec',
        'reflection_projection': {'definition':
            '0.5 * (pi(o) - pi(R(o))) before the existing single amplitude/rate filter'},
        'windows': {'primary_final_60': {
            'paired_drag_reduction': .019,
            'paired_rear_cl_fluctuation_rms_ratio': .9,
            'absolute_mean_rear_cl_over_paired_zero_rms': .05}},
    }
    with pytest.raises(ValueError, match='b00 projected primary'):
        MODULE.validate_b00_projection(failed)


def test_pending_metadata_is_real_b01_but_cannot_authorize_execution():
    pending = json.loads(PENDING.read_text())
    assert pending['status'].endswith('PREPARATION_ONLY_NOT_APPROVED')
    assert pending['execution_authorized'] is False
    assert pending['reviewed_by_lead'] is False
    assert pending['start_time'] == 130. and pending['end_time'] == 210.
    assert pending['primary_window'] == [150., 210.]
    assert pending['driver_sha256'] == '8b653f43bd1ffc69b6285dd10523199898d88d74aebe4279f65a66fb4807c741'
    assert set(pending['source_restart_tree_sha256']) == {'130', 'constant', 'system'}
    assert pending['inputs']['b00_projected_result']['sha256'] == 'PENDING_ACTUAL_FC_E058_TERMINAL_SHA256'
    with pytest.raises(ValueError, match='explicit approval'):
        MODULE.execute(pending, PENDING)
