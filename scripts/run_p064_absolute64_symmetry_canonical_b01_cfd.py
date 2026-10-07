"""Fixed800-cycle G-surrogate-trained canonical PPO paired b01 CFD; no online FNO."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import pickle
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

START, END, PRIMARY_START = 130., 210., 150.

STATUS = 'P064_ABSOLUTE64_SYMMETRY_CANONICAL_32768_PPO_B01_LONG_CFD_EXECUTION_APPROVED'
TRAINING_STATUS = 'P064_ABSOLUTE64_SYMMETRY_CANONICAL_H5_32768_PPO_TRAINING_COMPLETE_NOT_ADMISSION'
SOURCE_PATHS = {
    'scripts/run_exploratory_causal_history_h5_feedback.py',
    'scripts/run_full40_canonical_ppo_openfoam_feedback.py',
    'scripts/run_tandem_phase_feedback_pair.py',
    'cfd/tandem_cylinders/analyze_baseline.py',
    'cfd/tandem_cylinders/make_expanded_control_dataset.py',
    'cfd/tandem_cylinders/make_probe_feedback_case.py',
    'src/fluid_control/__init__.py', 'src/fluid_control/canonical_joint_v1.py',
    'src/fluid_control/openfoam_observation.py',
    'artifacts/p064_symmetry_canonical_policy_source_20261007_immutable/src/fluid_control/symmetry_canonical_wrapper.py',
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


def validate_training(result, policy_sha, vec_sha, arm, manifest_sha):
    require(result['status'] == TRAINING_STATUS, 'training status')
    require(result['candidate_arm'] == arm
            and result['candidate_manifest_sha256'] == manifest_sha,
            'P064 PPO candidate identity')
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
    require(p['seed'] == 20261007
            and p['symmetry_adapter'] == 'physical69_max_abs_first_tie_fixed_plus_v1'
            and result['symmetry_adapter'] == p['symmetry_adapter'],
            'symmetry-canonical training identity')
    require(p['reset_count'] == 24 and p['episode_steps'] == 5
            and p['reset_selection'] == 'deterministic_phase_cycle_no_reward_selection', 'diverse reset protocol')
    require(result['reset_counts_by_phase'] == {phase: [274,273,273,273,273,273]
            for phase in ('00','02','04','06')}, 'actual six-slot reset coverage')


def training_bindings(spec):
    binding = spec['training_execution']
    require(isinstance(binding['unit'], str) and binding['unit'].startswith('fluid-control-p064-')
            and isinstance(binding['invocation_id'], str)
            and len(binding['invocation_id']) == 32
            and all(c in '0123456789abcdef' for c in binding['invocation_id']), 'actual training invocation required')
    payload = spec['training_payload'].rstrip('/')
    for key, filename in [('training_result','result.json'), ('policy','ppo_final.zip'), ('vecnormalize','vecnormalize.pkl')]:
        row = spec['inputs'][key]
        require(row['path'] == payload+'/'+filename and isinstance(row['sha256'], str)
                and len(row['sha256']) == 64 and all(c in '0123456789abcdef' for c in row['sha256']),
                'actual diverse final artifact binding: '+key)
    return tuple(spec['inputs'][key]['sha256'] for key in ('training_result','policy','vecnormalize'))


def validate_symmetry_binding(training_approval, spec, inputs):
    require(training_approval['status'] == 'P064_ABSOLUTE64_SYMMETRY_CANONICAL_H5_32768_PPO_EXECUTION_APPROVED'
            and training_approval['execution_authorized'] is True,
            'actual symmetry-canonical training approval')
    training_path = training_approval['import_bindings']['symmetry_canonical_wrapper']
    require(Path(training_path).resolve() == inputs['symmetry_adapter'].resolve(),
            'training/deployment adapter path differs')
    require(training_approval['source_files'][training_path]
            == spec['inputs']['symmetry_adapter']['sha256'],
            'training/deployment adapter SHA differs')
    require(training_approval['protocol']['symmetry_training_and_deployment_same_bytes'] is True,
            'training approval lacks same-adapter contract')


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


def build_pair_at(source, output):
    """Copy the reviewed pair layout at the predeclared real b01 restart."""
    cases = {}
    for role in ('mpc', 'zero'):
        case = output / ('case_'+role)
        case.mkdir()
        for part in ('constant', 'system', str(int(START))):
            shutil.copytree(source / part, case / part)
        cases[role] = case
    return cases


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
            fixed_window(data, 130., 210.)
            raw[role].append(data)
    windows = {}
    for name, begin, end, inclusive in [('early_12p4',130.,142.4,False),
            ('early_first_6p2',130.,136.2,False),('early_trailing_6p2',136.2,142.4,False),
            ('primary_final_60',150.,210.,False),('historical_inclusive_final_60',150.,210.,True),
            ('full_80',130.,210.,False)]:
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


def values_summary(values):
    array = np.asarray(values, dtype=np.float64)
    require(array.ndim == 1 and len(array) == 800 and np.isfinite(array).all(), 'action summary')
    return {'min': float(array.min()), 'max': float(array.max()),
            'mean': float(array.mean()), 'rms': float(np.sqrt(np.mean(array * array)))}


def validate_spec(spec):
    require(spec['status'] == STATUS and spec['execution_authorized'] is True, 'explicit approval required')
    require(spec['steps'] == 800 and spec['start_time'] == START and spec['end_time'] == END and spec['primary_window'] == [PRIMARY_START, END] and spec['deadline_seconds'] == 3600,
            'fixed paired800 protocol')
    require(spec['scientific_admission'] is False and spec['inference_device'] == 'cpu', 'scope')
    require(spec['candidate_arm'] == 'ABSOLUTE64'
            and isinstance(spec['candidate_manifest_sha256'], str)
            and len(spec['candidate_manifest_sha256']) == 64,
            'explicit reviewed P064 candidate required')
    require(sha(__file__) == spec['driver_sha256'] and set(spec['source_files']) == SOURCE_PATHS,
            'exact driver/import source closure')
    training_bindings(spec)
    return spec


def execute(spec, spec_path):
    spec = validate_spec(spec)
    RESULT_SHA, POLICY_SHA, VEC_SHA = training_bindings(spec)
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
    require(inputs['phase_manifest'] == repo/'artifacts/tandem_cylinders/matched_start_phase_restart_predeclared_v3_20261003.json'
            and spec['inputs']['phase_manifest']['sha256']
            == '6279492bd3a79eff868a4333e1f642e3be4a4d58a78e46dc47dde040e4e39603',
            'fixed predeclared phase manifest')
    manifest = json.loads(inputs['phase_manifest'].read_text())
    phase = next(row for row in manifest['selections'] if row['phase_bin'] == 1)
    require(phase['split'] == 'validation' and phase['selected']['restart_time'] == START
            and phase['selected']['phase_rad'] == .6991961542646722
            and phase['selected']['u_sha256'] == 'ac412e9e3de151253dd3006a70ab195ef8f49f9c81edeeef28086b808fe9d230'
            and phase['selected']['p_sha256'] == '1ddc27110bacd814e31b3425e3550927ba7fee3f57112e62a4cb2a18288aa52c',
            'b01 phase/restart identity differs')
    require(inputs['b01_case_config']
            == repo/'cfd/tandem_cylinders/cases/matched_start_acquisition_validation_b01_zero/case_config.json'
            and spec['inputs']['b01_case_config']['sha256']
            == '0b59387acf6365700a4a1b2e5b00f63f3c71abb6efcfcd10850ccd5ba4df1f7c',
            'actual matched b01 provenance config')
    b01_config = json.loads(inputs['b01_case_config'].read_text())
    require(b01_config['source_restart_time'] == START and b01_config['start_time'] == START
            and b01_config['end_time'] == END and b01_config['analysis_window'] == [PRIMARY_START, END]
            and b01_config['action_target'] == 0. and b01_config['delta_t'] == .005,
            'matched b01 provenance semantics differ')
    require(inputs['symmetry_adapter'] == repo/'artifacts/p064_symmetry_canonical_policy_source_20261007_immutable/src/fluid_control/symmetry_canonical_wrapper.py'
            and spec['inputs']['symmetry_adapter']['sha256'] == spec['source_files']['artifacts/p064_symmetry_canonical_policy_source_20261007_immutable/src/fluid_control/symmetry_canonical_wrapper.py'],
            'training/deployment symmetry adapter bytes differ')
    training_approval = json.loads(inputs['training_approval'].read_text())
    validate_symmetry_binding(training_approval, spec, inputs)
    require(training_approval['candidate_arm'] == spec['candidate_arm']
            and training_approval['candidate_manifest_sha256'] == spec['candidate_manifest_sha256'],
            'P064 PPO approval candidate identity differs')
    overlay = repo/'.runtime/exploratory-h5-ppo-py312'
    for path, digest in training_approval['runtime_sources'].items():
        if Path(path).is_relative_to(overlay):
            require(sha(path) == digest, 'pinned policy runtime source differs')
    require(sha(inputs['training_result']) == RESULT_SHA and sha(inputs['policy']) == POLICY_SHA
            and sha(inputs['vecnormalize']) == VEC_SHA, 'fixed actual final policy')
    training_result = json.loads(inputs['training_result'].read_text())
    validate_training(training_result, POLICY_SHA, VEC_SHA,
                      spec['candidate_arm'], spec['candidate_manifest_sha256'])
    require(training_result['protocol'] == training_approval['protocol']
            and training_result['reset_packet_verification'] == training_approval['packet_verification'],
            'training protocol/packet proof differs from actual approval')
    props = subprocess.check_output(['systemctl', '--user', 'show', spec['training_execution']['unit'], '-p', 'InvocationID',
             '-p', 'MainPID', '-p', 'Result', '-p', 'ExecMainStatus'], text=True)
    unit = dict(line.split('=', 1) for line in props.splitlines())
    require(unit == {'InvocationID': spec['training_execution']['invocation_id'], 'MainPID': '0', 'Result': 'success', 'ExecMainStatus': '0'},
            'same actual training invocation must have exited successfully')
    sys.path[:0] = [str(repo / 'scripts'), str(repo / 'src')]
    base = load_module('reviewed_pair_transport', inputs['pair_driver'])
    transport = load_module('reviewed_canonical_transport', inputs['transport'])
    symmetry = load_module('reviewed_symmetry_canonical_adapter', inputs['symmetry_adapter'])
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
    original = {part: base.tree(source / part) for part in ('130', 'constant', 'system')}
    require(original == spec['source_restart_tree_sha256'], 'original restart identity')
    # Reuse byte-for-byte copy helper; rename the role, never substitute MPC decisions.
    copied = build_pair_at(source, output)
    cases = {'ppo': copied['mpc'], 'zero': copied['zero']}
    for case in cases.values():
        require({part: base.tree(case / part) for part in original} == original, 'paired initial state differs')
        transport.substitute(case/'system/controlDict', 'writeInterval', .1)
    image = subprocess.check_output(['docker', 'image', 'inspect', base.IMAGE, '--format', '{{.Id}}'], text=True, timeout=10).strip()
    require(image == spec['openfoam_image_id'], 'solver image identity')
    obs, initial = transport.total_drag_observation_at(source, START, 0.)
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
            begin, end = round(START+.1*(step-1), 10), round(START+.1*step, 10)
            require(all(abs(transport.latest_time(case)-begin) < 2e-6 for case in cases.values()), 'stale paired state')
            input_obs = obs.copy()
            canonical = symmetry.canonicalize_physical69(input_obs)
            canonical_action = predict(model, vec, canonical.value)
            physical_request = float(symmetry.restore_physical_action(
                np.asarray([canonical_action], np.float32), canonical.orientation
            )[0])
            request_audit = {
                'canonical_observation': canonical.value.tolist(),
                'canonical_orientation_applied': canonical.orientation,
                'canonical_pivot_index': canonical.pivot_index,
                'canonical_odd_margin': canonical.odd_margin,
                'canonical_reflection_fixed': canonical.reflection_fixed,
                'canonical_policy_request': canonical_action,
                'physical_requested_omega_before_filter': physical_request,
            }
            # One policy call, invertible sign restore, then the inherited
            # physical amplitude/rate filter exactly once.
            action = transport.apply_action_rate_limit(physical_request, previous)
            applied = float(action['applied_omega'])
            transport.configure_interval(cases['ppo'], begin, end, previous, applied)
            transport.configure_interval(cases['zero'], begin, end, 0., 0.)
            health = solvers.pair(step, end, transport.check_segment, guard)
            obs, sources = transport.total_drag_observation_at(cases['ppo'], end, applied)
            zero, zero_sources = transport.total_drag_observation_at(cases['zero'], end, 0.)
            obs, zero = observation(obs), observation(zero)
            next_canonical = symmetry.canonicalize_physical69(obs)
            rows.append({'step': step, 'start_time': begin, 'end_time': end, **request_audit, **action,
                'input_observation': input_obs.tolist(), 'output_observation': obs.tolist(),
                'next_canonical_orientation': next_canonical.orientation,
                'zero_observation': zero.tolist(), 'observation_sources': sources,
                'zero_sources': zero_sources, 'solver_health': health})
            base.atomic_json(output/'progress.json', {'completed_cycles': step, 'rows': rows})
            previous = applied
    windows, raw_sha = summarize(cases, transport.force_metrics)
    require({part: base.tree(source/part) for part in original} == original, 'original restart changed')
    require(sha(inputs['policy']) == POLICY_SHA and sha(inputs['vecnormalize']) == VEC_SHA, 'policy artifacts changed')
    result = {'status': 'P064_ABSOLUTE64_SYMMETRY_CANONICAL_32768_PPO_LONG_CFD_COMPLETE_NOT_ADMISSION',
        'cycles': 800, 'windows': windows, 'rows': rows, 'raw_file_sha256': raw_sha,
        'candidate_arm': spec['candidate_arm'],
        'candidate_manifest_sha256': spec['candidate_manifest_sha256'],
        'policy_sha256': POLICY_SHA, 'vecnormalize_sha256': VEC_SHA, 'training_result_sha256': RESULT_SHA,
        'training_unit': unit, 'approval_sha256': sha(spec_path), 'initial_observation_sources': initial,
        'observation_contract': 'same physical69 locations/order; CFD raw probes vs training grid interpolation, not asserted equal',
        'symmetry_canonical_adapter': {
            'definition': 'one pi(C(o)) call; physical request = orientation(o) * canonical request; existing filter once',
            'profile': 'physical69_max_abs_first_tie_fixed_plus_v1',
            'canonical_policy_request': values_summary([r['canonical_policy_request'] for r in rows]),
            'physical_requested_omega_before_filter': values_summary([r['physical_requested_omega_before_filter'] for r in rows]),
            'reflection_fixed_steps': sum(r['canonical_reflection_fixed'] for r in rows),
            'minimum_odd_margin': min(r['canonical_odd_margin'] for r in rows),
        },
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
