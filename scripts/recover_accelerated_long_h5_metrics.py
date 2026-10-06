"""Offline fixed-window recovery; no CFD/model/action execution or original-result rewrite."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

REPO = Path('/workspace/fluid_control')
ROOT = REPO / 'artifacts/exploratory_accelerated_long_h5_real_cfd_20261006'
APPROVAL = REPO / 'docs/EXPLORATORY_ACCELERATED_LONG_H5_APPROVAL_20261006.json'
APPROVAL_SHA = '03e2bac8f55c4bbd09e377b60bfef849515b53a6d418d490ec682b2cde95bc75'
PROGRESS_SHA = 'c3b004251d6528bb5d65a194d1ad1f3cba1b03e4d8e41555b7c493c1c3337f43'
OLD = REPO / 'artifacts/exploratory_causal_history_h5_real_cfd_20261006'
OLD_SHA = 'd4c3ad8198f69199606c0fa7a6c1a668c9a0f581e3b52bbeca99e2b0bd902c5e'
METRICS = REPO / 'scripts/run_tandem_phase_feedback_pair.py'
METRICS_SHA = '866b4dce33c7401eb447e897641d734828e04b10b38ccde8a0b724e8b5e65d3d'
UNIT = 'fluid-control-accelerated-long-h5-20261006.service'
INVOCATION = 'a601eec2da7649b4af6f9354a4deb470'


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_force_metrics():
    require(sha(METRICS) == METRICS_SHA, 'metric source changed')
    # Execute only the reviewed pure numerical function, not the transport imports.
    tree = ast.parse(METRICS.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'force_metrics')
    namespace = {'np': np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(METRICS), 'exec'), namespace)
    return namespace['force_metrics']


def read_raw(root, role, object_name, provenance):
    paths = sorted((root / f'case_{role}/postProcessing/{object_name}').glob('*/coefficient.dat'))
    require(bool(paths), 'missing raw coefficients')
    arrays = []
    for path in paths:
        require(path.is_file() and not path.is_symlink(), 'nonregular raw source')
        before = sha(path)
        data = np.loadtxt(path, ndmin=2)
        require(data.shape[1] >= 5 and np.isfinite(data).all(), 'invalid raw coefficients')
        require(sha(path) == before, 'raw changed during read')
        provenance[str(path.relative_to(REPO))] = before
        arrays.append(data[:, [0, 1, 4]])
    data = np.concatenate(arrays)
    data = data[np.argsort(data[:, 0])]
    require(np.all(np.diff(data[:, 0]) > 0), 'duplicate or nonmonotonic raw timestamps')
    return data


def window(data, begin, end):
    """Only the predeclared open-left/closed-right interval; no interpolation."""
    selected = data[(data[:, 0] > begin) & (data[:, 0] <= end)]
    count = round((end - begin) / .005)
    require(len(selected) == count, 'fixed-window count differs')
    require(np.allclose(selected[:, 0], begin + .005 * np.arange(1, count + 1),
                        rtol=0, atol=1e-8), 'fixed-window grid differs')
    return selected


def recover():
    require(not (ROOT / 'result.json').exists(), 'original unexpected result exists')
    require(sha(APPROVAL) == APPROVAL_SHA and sha(ROOT / 'progress.json') == PROGRESS_SHA,
            'approval/progress identity changed')
    approval = json.loads(APPROVAL.read_text())
    progress = json.loads((ROOT / 'progress.json').read_text())
    rows = progress['rows']
    require(progress['completed_cycles'] == 124 and len(rows) == 124, 'incomplete124 cycles')
    for i, row in enumerate(rows):
        require(row['step'] == i + 1 and row['start_time'] == round(148 + .1 * i, 10)
                and row['end_time'] == round(148 + .1 * (i + 1), 10), 'cycle grid differs')
        require(all(h['solver_ended_cleanly'] is True and h['steps'] == 20
                    for h in row['solver_health'].values()), 'solver health differs')
    properties = subprocess.check_output(['systemctl', '--user', 'show', UNIT,
        '-p', 'InvocationID', '-p', 'MainPID', '-p', 'ActiveState', '-p', 'SubState',
        '-p', 'Result', '-p', 'ExecMainStatus', '-p', 'ExecMainStartTimestamp',
        '-p', 'ExecMainExitTimestamp'], text=True)
    unit = dict(line.split('=', 1) for line in properties.splitlines())
    require(unit['InvocationID'] == INVOCATION and unit['MainPID'] == '0'
            and unit['Result'] == 'exit-code' and unit['ExecMainStatus'] == '1',
            'preserved failed unit differs')
    sources = {}
    for relative, expected in approval['source_files'].items():
        require(sha(REPO / relative) == expected, 'source identity differs')
        sources[relative] = expected
    immutable = REPO / approval['immutable_driver']
    require(sha(immutable) == sources['scripts/run_accelerated_long_h5.py'], 'immutable driver differs')
    restart = REPO / approval['source_restart']
    for part, expected in approval['source_restart_tree_sha256'].items():
        files = list((restart / part).rglob('*'))
        require(not any(p.is_symlink() for p in files), 'restart symlink')
        actual = {str(p.relative_to(restart / part)): sha(p) for p in files if p.is_file()}
        require(actual == expected, 'original restart changed')
    provenance = {}
    raw = {role: [read_raw(ROOT, role, obj, provenance) for obj in ('forceFront', 'forceRear')]
           for role in ('mpc', 'zero')}
    require(all(len(a) == 2480 for arrays in raw.values() for a in arrays), 'extra/missing raw rows')
    metric = load_force_metrics()
    statistics = {}
    for label, begin, end in [('full', 148., 160.4), ('first_6p2', 148., 154.2),
                              ('trailing_6p2', 154.2, 160.4)]:
        branch = {role: metric(*(window(a, begin, end) for a in arrays))
                  for role, arrays in raw.items()}
        m, z = branch['mpc'], branch['zero']
        statistics[label] = {'interval_open_left_closed_right': [begin, end], 'branches': branch,
            'paired_drag_reduction': 1 - m['total_cd_mean'] / z['total_cd_mean'],
            'paired_rear_cl_fluctuation_rms_ratio': m['rear_cl_fluctuation_rms'] / z['rear_cl_fluctuation_rms'],
            'absolute_mean_rear_cl_over_paired_zero_rms': abs(m['rear_cl_mean']) / z['rear_cl_fluctuation_rms']}
    require(sha(OLD / 'result.json') == OLD_SHA, 'old CPU result changed')
    old = json.loads((OLD / 'result.json').read_text())
    prefix_equal = [r['selected_omega'] for r in rows[:10]] == [r['selected_omega'] for r in old['rows']]
    prefix = {}
    for role, arrays in raw.items():
        for obj, array in zip(('forceFront', 'forceRear'), arrays, strict=True):
            previous = read_raw(OLD, role, obj, provenance)
            prefix[f'{role}/{obj}'] = bool(np.array_equal(window(array, 148, 149), previous))
    require(prefix_equal and all(prefix.values()), 'first10 CPU prefix differs')
    existing = subprocess.check_output(['docker', 'container', 'ls', '-aq', '--no-trunc'], text=True).split()
    terminals = {}
    paths = list(ROOT.glob('container_terminal_*.json'))
    require(len(paths) == 2, 'two container terminal records required')
    for path in paths:
        record = json.loads(path.read_text())
        require(record['Id'] not in existing and record['State']['Running'] is False
                and record['State']['OOMKilled'] is False, 'container cleanup differs')
        terminals[path.name] = {'sha256': sha(path), 'id': record['Id'],
                                'exit_code': record['State']['ExitCode'], 'absent': True}
    supervision = ROOT.with_name(ROOT.name + '_supervision')
    observations = [json.loads(l) for l in (supervision / 'memory.jsonl').read_text().splitlines()]
    require(all(r['MemAvailable'] >= 22 * 2**30 for r in observations), 'resource floor failed')
    keys = ('front_cd', 'front_cl', 'rear_cd', 'rear_cl')
    errors = np.asarray([[r['selected_prediction_minus_actual_next_force'][k] for k in keys] for r in rows])
    return {'status': 'OFFLINE_METRICS_RECOVERED_FROM_FAILED_POSTPROCESSING_NOT_ADMISSION',
        'original_unit': unit, 'original_result_written': False, 'scientific_admission': False,
        'new_cfd_or_model_execution': False, 'selection': '(begin,end]; no interpolation or other exclusions',
        'cycles': 124, 'approval_sha256': APPROVAL_SHA, 'progress_sha256': PROGRESS_SHA,
        'recovery_source_sha256': sha(Path(__file__)), 'metric_source_sha256': METRICS_SHA,
        'source_sha256': sources, 'immutable_driver_sha256': sha(immutable),
        'original_restart_rehashed_unchanged': True, 'raw_file_sha256': provenance,
        'windows': statistics, 'first10_actions_exact_cpu': prefix_equal,
        'first10_raw_force_exact_cpu': prefix, 'prior_cpu_result_sha256': OLD_SHA,
        'selected_one_step_force_mae': dict(zip(keys, np.abs(errors).mean(axis=0).tolist(), strict=True)),
        'owned_container_terminal': terminals, 'supervisor_log_sha256': sha(supervision / 'run.log'),
        'memory_log_sha256': sha(supervision / 'memory.jsonl'),
        'memory_observations': len(observations),
        'minimum_mem_available_bytes': min(r['MemAvailable'] for r in observations),
        'minimum_mem_free_observational_bytes': min(r['MemFree'] for r in observations)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print('PREPARATION_ONLY_NO_RECOVERY_WRITTEN')
        return
    output = ROOT / 'recovered_metrics.json'
    require(not output.exists() and not output.is_symlink(), 'exclusive recovery output required')
    result = recover()
    with output.open('x') as stream:
        json.dump(result, stream, allow_nan=False, indent=2)
        stream.write('\n')
    print(sha(output))


if __name__ == '__main__':
    main()
