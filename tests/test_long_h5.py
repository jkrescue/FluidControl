"""Synthetic sequence and source-contract tests, not scientific execution."""
import ast
import importlib.util
from pathlib import Path

import numpy as np


def load(name):
    here = Path(__file__).resolve()
    path = here.parents[1] / 'scripts' / (name + '.py')
    if not path.is_file():
        path = here.with_name(name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact124_cycles_and_three_metric_windows(tmp_path):
    module = load('long_h5_sequence')
    cases = {role: tmp_path / role for role in ('mpc', 'zero')}
    times = {case: 148. for case in cases.values()}
    summaries = []
    plans = []
    def observe(role, case, stamp, omega):
        return {'time': stamp, 'role': role, 'sampled_before_action': True,
                'sample_sha256': 'synthetic'}
    def plan(packet, previous):
        plans.append(packet['time'])
        return {'status': 'EXPLORATORY_CANONICAL_HISTORY_H5_SELECTION_NOT_ADMISSION',
                'execute_only_first_action': True, 'selected_action': 0.,
                'selected_index': 2, 'predicted_forces_h5': np.zeros((5, 5, 4)).tolist()}
    def solve(step, stamp):
        for case in cases.values():
            times[case] = stamp
        return {'mpc': {}, 'zero': {}}
    def summarize(role, case, begin, end):
        summaries.append((role, begin, end))
        return {'samples': round((end-begin)/.005)}
    result = module.run_paired_long_h5(
        cases=cases, output=tmp_path, latest_time=lambda case: times[case],
        observe_current=observe, plan_action=plan, configure_interval=lambda *a: None,
        solve_pair=solve, observe_forces=lambda *a: np.zeros(4),
        summarize_actual=summarize, guard=lambda: None, identity={'fixture': True})
    assert len(plans) == len(result['rows']) == 124
    assert result['rows'][-1]['end_time'] == 160.4
    assert result['actual_force_metrics']['mpc']['samples'] == 2480
    assert result['trailing_6p2_force_metrics']['zero']['samples'] == 1240
    assert set((x[1], x[2]) for x in summaries) == {(148., 160.4), (148., 154.2), (154.2, 160.4)}
    assert result['scientific_admission'] is False


def test_old_sequence_unchanged_except_profile_and_summaries():
    module = load('long_h5_sequence')
    assert module.STEPS == 124 and module.CONTROL_INTERVAL == .1
    assert module.START == 148. and module.ACTION_LIMIT == .75 and module.RATE_LIMIT == .1


def test_actual_driver_precision_and_sampling_calls():
    source = Path(load('run_accelerated_long_h5').__file__).read_text()
    assert source.index('torch.set_float32_matmul_precision("high")') < source.index('flow, aero, identity = load_bound_k1')
    assert source.index('flow, aero, identity = load_bound_k1') < source.index('torch.set_float32_matmul_precision("highest")')
    assert 'sample_frame(root, sample)' in source
    assert 'str(sampler)' not in source
    assert 'decision["inference_precision_override"] = effective' in source
    assert 'process.wait(timeout=120)' in source
    assert source.index('process.wait(timeout=120)') < source.index('base.stop(process)')
    assert 'time.sleep(.5)' in source and 'row["MemAvailable"] >= 22 * 2**30' in source
    assert 'row["elapsed"] < 1800' in source
    assert 'torch.cuda.set_per_process_memory_fraction(.06, 0)' in source
    tree = ast.parse(source)
    execute = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'execute')
    calls = [x for x in ast.walk(execute) if isinstance(x, ast.Call)]
    assert sum(isinstance(x.func, ast.Name) and x.func.id == 'run_paired_long_h5' for x in calls) == 1


def test_pending_precision_review_cannot_validate(tmp_path):
    module = load('run_accelerated_long_h5')
    import json
    import pytest
    payload = {'status': module.STATUS, 'execution_authorized': True,
               'steps': 124, 'start_time': 148., 'planning_horizon': 5,
               'inference_device': 'cuda:0', 'state_abs_limit': module.STATE_ABS_LIMIT,
               'deadline_seconds': 1800, 'gpu_precision_equivalence_reviewed': False}
    path = tmp_path / 'spec.json'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='precision review'):
        module.validate_spec(payload, path, module.sha(path))
