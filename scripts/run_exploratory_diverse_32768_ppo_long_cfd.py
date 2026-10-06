"""Fixed800-cycle final32768-policy paired CFD; early and final60 windows, no FNO."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import time

import numpy as np

STATUS = 'EXPLORATORY_DIVERSE_32768_PPO_LONG_CFD_EXECUTION_APPROVED'
TRAIN_UNIT = 'fluid-control-exploratory-diverse-h5-32768-ppo-r2-20261006.service'
PAYLOAD = 'artifacts/exploratory_diverse_h5_32768_ppo_training_20261006_r2/payload'
SOURCE_PATHS = {
    'scripts/run_exploratory_causal_history_h5_feedback.py',
    'scripts/run_full40_canonical_ppo_openfoam_feedback.py',
    'scripts/run_tandem_phase_feedback_pair.py',
    'cfd/tandem_cylinders/analyze_baseline.py',
    'cfd/tandem_cylinders/make_expanded_control_dataset.py',
    'cfd/tandem_cylinders/make_probe_feedback_case.py',
    'src/fluid_control/__init__.py', 'src/fluid_control/canonical_joint_v1.py',
    'src/fluid_control/openfoam_observation.py',
}


def require(ok, why):
    if not ok:
        raise ValueError(why)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def observation(value):
    value = np.asarray(value, dtype=np.float32)
    require(value.shape == (69,) and np.isfinite(value).all(), 'finite physical69 required')
    require(abs(float(value[-1])) <= .75 + 1e-7, 'actual applied omega outside support')
    return value


def validate_training(result, policy_sha, vec_sha):
    require(result['status'] == 'EXPLORATORY_DIVERSE_H5_32768_PPO_TRAINING_COMPLETE_NOT_ADMISSION', 'training status')
    require(result['timesteps'] == 32768 and result['ppo_n_updates'] == 256
            and [r['optimizer_step'] for r in result['optimizer_steps']] == list(range(1, 513))
            and result['fno_tensors_unchanged'] is True,
            'actual completed training counters/frozen FNO')
    require(result['artifacts']['ppo_final.zip'] == policy_sha
            and result['artifacts']['vecnormalize.pkl'] == vec_sha, 'final artifact identities')
    p = result['protocol']
    require(p['observation_dimension'] == 69 and p['action_limit'] == .75
            and p['action_delta_limit'] == .1 and p['control_dt'] == .1
            and p['norm_obs'] is False and p['norm_reward'] is False, 'policy observation/action contract')
    require(result['scientific_admission'] is False, 'exploratory scope')
    require(p['reset_count'] == 24 and p['episode_steps'] == 5
            and p['reset_selection'] == 'deterministic_phase_cycle_no_reward_selection', 'diverse reset protocol')
    require(result['reset_counts_by_phase'] == {phase: [274,273,273,273,273,273]
            for phase in ('00','02','04','06')}, 'actual six-slot reset coverage')


def training_bindings(spec):
    binding = spec['training_execution']
    require(binding['unit'] == TRAIN_UNIT and isinstance(binding['invocation_id'], str)
            and len(binding['invocation_id']) == 32
            and all(c in '0123456789abcdef' for c in binding['invocation_id']), 'actual training invocation required')
    for key, filename in [('training_result','result.json'), ('policy','ppo_final.zip'), ('vecnormalize','vecnormalize.pkl')]:
        row = spec['inputs'][key]
        require(row['path'] == PAYLOAD+'/'+filename and isinstance(row['sha256'], str)
                and len(row['sha256']) == 64 and all(c in '0123456789abcdef' for c in row['sha256']),
                'actual diverse final artifact binding: '+key)
    return tuple(spec['inputs'][key]['sha256'] for key in ('training_result','policy','vecnormalize'))


def validate_vec(vec, vec_type):
    require(type(vec) is vec_type and vec.norm_obs is False and vec.norm_reward is False,
            'identity VecNormalize required')
    require(vec.observation_space.shape == (69,) and vec.action_space.shape == (1,)
            and np.array_equal(vec.action_space.low, np.array([-.75], np.float32))
            and np.array_equal(vec.action_space.high, np.array([.75], np.float32)), 'VecNormalize spaces')
    probe = np.linspace(-20, 20, 69, dtype=np.float32)[None]
    require(np.array_equal(vec.normalize_obs(probe), probe), 'nonidentity normalization')
    vec.training = False


def predict(model, vec, obs):
    physical = observation(obs)
    normalized = vec.normalize_obs(physical.copy())
    require(np.array_equal(normalized, physical), 'physical observation altered')
    action, _ = model.predict(normalized, deterministic=True)
    action = np.asarray(action)
    require(action.shape == (1,) and np.isfinite(action).all(), 'invalid policy action')
    require(abs(float(action[0])) <= .75 + 1e-7, 'policy action outside trained space')
    return float(action[0])


def fixed_window(data, begin, end, inclusive=False):
    selected = data[((data[:, 0] >= begin) if inclusive else (data[:, 0] > begin)) & (data[:, 0] <= end)]
    n = round((end - begin) / .005)
    expected = begin + .005 * np.arange(0 if inclusive else 1, n+1)
    require(len(selected) == len(expected) and np.allclose(selected[:, 0], expected,
            rtol=0, atol=1e-8), 'exact declared force grid')
    return selected


def summarize(cases, metric):
    raw, hashes = {}, {}
    for role, case in cases.items():
        raw[role] = []
        for body in ('forceFront', 'forceRear'):
            paths = sorted((case / 'postProcessing' / body).glob('*/coefficient.dat'))
            require(bool(paths), 'missing raw force stream')
            arrays = []
            for path in paths:
                digest = sha(path)
                data = np.loadtxt(path, ndmin=2)
                require(data.shape[1] >= 5 and np.isfinite(data).all(), 'invalid raw forces')
                require(sha(path) == digest, 'raw force file changed')
                hashes[str(path)] = digest
                arrays.append(data[:, [0, 1, 4]])
            data = np.concatenate(arrays)
            data = data[np.argsort(data[:, 0])]
            require(len(data) == 16000 and np.all(np.diff(data[:, 0]) > 0), 'raw count/duplicate')
            fixed_window(data, 148., 228.)
            raw[role].append(data)
    windows = {}
    for name, begin, end, inclusive in [('early_12p4',148.,160.4,False),
            ('early_first_6p2',148.,154.2,False),('early_trailing_6p2',154.2,160.4,False),
            ('primary_final_60',168.,228.,False),('historical_inclusive_final_60',168.,228.,True),
            ('full_80',148.,228.,False)]:
        branches = {role: metric(*(fixed_window(a, begin, end, inclusive) for a in arrays))
                    for role, arrays in raw.items()}
        for role, arrays in raw.items():
            branches[role]['rear_cl_abs_peak'] = float(np.max(np.abs(fixed_window(arrays[1], begin, end, inclusive)[:, 2])))
        p, z = branches['ppo'], branches['zero']
        windows[name] = {'interval': [begin, end], 'left_endpoint_included': inclusive, 'branches': branches,
            'paired_drag_reduction': 1-p['total_cd_mean']/z['total_cd_mean'],
            'paired_rear_cl_fluctuation_rms_ratio': p['rear_cl_fluctuation_rms']/z['rear_cl_fluctuation_rms'],
            'absolute_mean_rear_cl_over_paired_zero_rms': abs(p['rear_cl_mean'])/z['rear_cl_fluctuation_rms'],
            'original_mean_bias_reference': .10, 'mean_bias_sensitivity_reference': .20}
    return windows, hashes


def execute(spec, spec_path):
    require(spec['status'] == STATUS and spec['execution_authorized'] is True, 'explicit approval required')
    RESULT_SHA, POLICY_SHA, VEC_SHA = training_bindings(spec)
    require(spec['steps'] == 800 and spec['start_time'] == 148. and spec['deadline_seconds'] == 3600,
            'fixed paired800 protocol')
    require(spec['scientific_admission'] is False and spec['inference_device'] == 'cpu', 'scope')
    require(sha(__file__) == spec['driver_sha256'] and set(spec['source_files']) == SOURCE_PATHS,
            'exact driver/import source closure')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only policy inference')
    repo = Path(spec['repo']).resolve()
    require(Path(sys.prefix).resolve() == repo / '.venv-curator-py312', 'exact Python environment')
    require({k: importlib.metadata.version(k) for k in ('stable_baselines3', 'gymnasium', 'torch', 'numpy')}
            == {'stable_baselines3': '2.7.1', 'gymnasium': '1.2.3', 'torch': '2.14.1', 'numpy': '2.5.3'},
            'runtime package versions')
    for path, digest in spec['source_files'].items():
        require(sha(repo / path) == digest, 'source bytes differ: '+path)
    inputs = {k: repo / v['path'] for k, v in spec['inputs'].items()}
    for k, path in inputs.items():
        require(path.resolve().is_relative_to(repo) and sha(path) == spec['inputs'][k]['sha256'], 'input identity: '+k)
    for key, relative in [('pair_driver', 'scripts/run_exploratory_causal_history_h5_feedback.py'),
                          ('transport', 'scripts/run_full40_canonical_ppo_openfoam_feedback.py')]:
        require(inputs[key] == repo/relative and spec['inputs'][key]['sha256'] == spec['source_files'][relative],
                'imported helper must match source closure')
    training_approval = json.loads(inputs['training_approval'].read_text())
    require(training_approval['status'] == 'EXPLORATORY_DIVERSE_H5_32768_PPO_EXECUTION_APPROVED'
            and training_approval['execution_authorized'] is True, 'actual diverse training approval')
    overlay = repo/'.runtime/exploratory-h5-ppo-py312'
    for path, digest in training_approval['runtime_sources'].items():
        if Path(path).is_relative_to(overlay):
            require(sha(path) == digest, 'pinned policy runtime source differs')
    require(sha(inputs['training_result']) == RESULT_SHA and sha(inputs['policy']) == POLICY_SHA
            and sha(inputs['vecnormalize']) == VEC_SHA, 'fixed actual final policy')
    training_result = json.loads(inputs['training_result'].read_text())
    validate_training(training_result, POLICY_SHA, VEC_SHA)
    require(training_result['protocol'] == training_approval['protocol']
            and training_result['reset_packet_verification'] == training_approval['packet_verification'],
            'training protocol/packet proof differs from actual approval')
    props = subprocess.check_output(['systemctl', '--user', 'show', TRAIN_UNIT, '-p', 'InvocationID',
             '-p', 'MainPID', '-p', 'Result', '-p', 'ExecMainStatus'], text=True)
    unit = dict(line.split('=', 1) for line in props.splitlines())
    require(unit == {'InvocationID': spec['training_execution']['invocation_id'], 'MainPID': '0', 'Result': 'success', 'ExecMainStatus': '0'},
            'same actual training invocation must have exited successfully')
    sys.path[:0] = [str(repo / 'scripts'), str(repo / 'src')]
    base = load_module('reviewed_pair_transport', inputs['pair_driver'])
    transport = load_module('reviewed_canonical_transport', inputs['transport'])
    base.host_guard(startup=True)
    cgroup = Path('/sys/fs/cgroup') / next(line.split('::',1)[1] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::')).lstrip('/')
    require(0 < int((cgroup/'memory.max').read_text()) <= 8*2**30
            and int((cgroup/'memory.swap.max').read_text()) == 0, 'outer8GiB/noSwap cgroup required')
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize
    import stable_baselines3
    import gymnasium
    require(all(Path(module.__file__).resolve().is_relative_to(overlay)
                for module in (stable_baselines3, gymnasium)), 'policy runtime import origin')
    # Only explicitly approved, hashed training artifacts may be deserialized.
    vec = pickle.loads(inputs['vecnormalize'].read_bytes())
    validate_vec(vec, VecNormalize)
    model = PPO.load(inputs['policy'], device='cpu')
    require(model.observation_space == vec.observation_space and model.action_space == vec.action_space,
            'policy spaces differ from saved normalizer')
    model.policy.set_training_mode(False)
    require(all(p.device.type == 'cpu' for p in model.policy.parameters()), 'policy must stay CPU')
    output = repo / spec['output']
    require(output.resolve().is_relative_to(repo) and not output.exists() and not output.is_symlink(), 'exclusive output')
    output.mkdir()
    source = repo / spec['source_restart']
    original = {part: base.tree(source / part) for part in ('148', 'constant', 'system')}
    require(original == spec['source_restart_tree_sha256'], 'original restart identity')
    # Reuse byte-for-byte copy helper; rename the role, never substitute MPC decisions.
    copied = base.build_pair(source, output)
    cases = {'ppo': copied['mpc'], 'zero': copied['zero']}
    for case in cases.values():
        require({part: base.tree(case / part) for part in original} == original, 'paired initial state differs')
        transport.substitute(case/'system/controlDict', 'writeInterval', .1)
    image = subprocess.check_output(['docker', 'image', 'inspect', base.IMAGE, '--format', '{{.Id}}'], text=True, timeout=10).strip()
    require(image == spec['openfoam_image_id'], 'solver image identity')
    obs, initial = transport.total_drag_observation_at(source, 148., 0.)
    obs = observation(obs)
    rows, previous, started = [], 0., time.monotonic()
    def guard():
        mem = base.host_guard()
        with (output/'resources.jsonl').open('a') as stream:
            stream.write(json.dumps({'elapsed': time.monotonic()-started, **mem})+'\n')
        require(time.monotonic()-started < 3600, 'internal deadline')
    def terminated(*_):
        raise RuntimeError('termination requested; cleaning owned solver containers')
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, terminated)
    with base.PairSolvers(cases, output, image) as solvers:
        for step in range(1, 801):
            guard()
            begin, end = round(148+.1*(step-1), 10), round(148+.1*step, 10)
            require(all(abs(transport.latest_time(case)-begin) < 2e-6 for case in cases.values()), 'stale paired state')
            input_obs = obs.copy()
            requested = predict(model, vec, input_obs)
            action = transport.apply_action_rate_limit(requested, previous)
            applied = float(action['applied_omega'])
            transport.configure_interval(cases['ppo'], begin, end, previous, applied)
            transport.configure_interval(cases['zero'], begin, end, 0., 0.)
            health = solvers.pair(step, end, transport.check_segment, guard)
            obs, sources = transport.total_drag_observation_at(cases['ppo'], end, applied)
            zero, zero_sources = transport.total_drag_observation_at(cases['zero'], end, 0.)
            obs, zero = observation(obs), observation(zero)
            rows.append({'step': step, 'start_time': begin, 'end_time': end, **action,
                'input_observation': input_obs.tolist(), 'output_observation': obs.tolist(),
                'zero_observation': zero.tolist(), 'observation_sources': sources,
                'zero_sources': zero_sources, 'solver_health': health})
            base.atomic_json(output/'progress.json', {'completed_cycles': step, 'rows': rows})
            previous = applied
    windows, raw_sha = summarize(cases, transport.force_metrics)
    require({part: base.tree(source/part) for part in original} == original, 'original restart changed')
    require(sha(inputs['policy']) == POLICY_SHA and sha(inputs['vecnormalize']) == VEC_SHA, 'policy artifacts changed')
    result = {'status': 'EXPLORATORY_DIVERSE_32768_PPO_LONG_CFD_COMPLETE_NOT_ADMISSION',
        'cycles': 800, 'windows': windows, 'rows': rows, 'raw_file_sha256': raw_sha,
        'policy_sha256': POLICY_SHA, 'vecnormalize_sha256': VEC_SHA, 'training_result_sha256': RESULT_SHA,
        'training_unit': unit, 'approval_sha256': sha(spec_path), 'initial_observation_sources': initial,
        'observation_contract': 'same physical69 locations/order; CFD raw probes vs training grid interpolation, not asserted equal',
        'action_summary': {
            'max_abs_applied_omega': max(abs(r['applied_omega']) for r in rows),
            'max_abs_applied_delta': max(abs(r['applied_omega']-(rows[i-1]['applied_omega'] if i else 0.)) for i,r in enumerate(rows)),
            'saturated_endpoints': sum(abs(r['applied_omega']) >= .75-1e-12 for r in rows),
            'rate_limited_endpoints': sum(abs(r['requested_omega']-r['applied_omega']) > 1e-12 for r in rows)},
        'source_restart_unchanged': True, 'owned_containers_cleaned': True,
        'fno_inference': False, 'mpc_action_selection': False, 'scientific_admission': False,
        'wall_seconds': time.monotonic()-started}
    base.atomic_json(output/'result.json', result, exclusive=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--spec-sha256', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    require(sha(args.spec) == args.spec_sha256, 'approval digest')
    require(args.execute, 'no execution without explicit flag')
    execute(json.loads(args.spec.read_text()), args.spec)
