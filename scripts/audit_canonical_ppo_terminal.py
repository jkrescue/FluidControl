"""Read-only saved-evidence audit. No torch, model load, forward or GPU.

Orientations are checked for recorded consistency only: physical observations
are not saved in PPO rows, so their geometric canonicalization is not replayed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

PROFILES = {
    'B': ('P064_B_SYMMETRY_CANONICAL_H5_32768_PPO_TRAINING_COMPLETE_NOT_ADMISSION',
          'FC_P064_ARM_B_CONTROLLED_AERO_FORCE_FNO',
          '92766915cb11ca75d313608a5f75e61a218371dcc789f5b44725f0a8260e7891'),
    'G': ('P064_G_SYMMETRY_CANONICAL_H5_32768_PPO_TRAINING_COMPLETE_NOT_ADMISSION',
          'FC_P064_AR5_RESET_K1_FRESH_FORCE_FNO',
          '6123b587065ad46104c328f276949ba07d3d7b61266b6c781cb5e9a538cfd681'),
}
REWARDS = ('reward_drag_screen', 'reward_drag_gate_violation',
           'reward_rear_cl_fluctuation_gate_violation',
           'reward_rear_cl_mean_bias_gate_violation', 'reward_actuation', 'reward_rate')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def finite(value):
    if isinstance(value, dict):
        for v in value.values(): finite(v)
    elif isinstance(value, list):
        for v in value: finite(v)
    elif isinstance(value, float):
        assert math.isfinite(value), 'nonfinite saved numeric value'


def close(a, b, tol=1e-7):
    assert math.isclose(a, b, rel_tol=tol, abs_tol=tol), (a, b)


def terminal(state, invocation):
    assert state['InvocationID'] == invocation
    assert state['MainPID'] == '0' and state['Result'] == 'success'
    assert state['ExecMainStatus'] == '0' and state['SubState'] in ('exited', 'dead')


def identity(result, spec, profile):
    status, kind, manifest = PROFILES[profile]
    assert result['status'] == status and result['candidate_arm'] == profile
    for obj in (result, spec):
        assert obj['candidate_manifest_kind'] == kind
        assert obj['candidate_manifest_sha256'] == manifest
    assert spec['inputs']['manifest']['sha256'] == manifest
    assert result['fno_runtime']['profile'] == 'p026_k1'
    assert result['fno_runtime']['manifest_kind'] == kind
    assert result['fno_runtime']['dual_manifest_sha256'] == manifest


def optimizer(result):
    assert result['timesteps'] == 32768 and result['ppo_n_updates'] == 256
    steps = result['optimizer_steps']
    assert len(steps) == 512
    for i, row in enumerate(steps):
        assert row == {'optimizer_step': i+1, 'num_timesteps': (i//8+1)*512}


def check_rows(rows, packets):
    assert len(packets) == 24
    previous = [None]*4
    counts = [[0]*6 for _ in range(4)]
    reward_sums = {k: 0.0 for k in REWARDS}
    n = episodes = switches = rate_limited = 0
    omega_sum = omega_sq = 0.0
    omega_min, omega_max = math.inf, -math.inf
    for i, row in enumerate(rows):
        finite(row)
        env, step = i % 4, (i//4) % 5 + 1
        panel = ((i//4)//5) % 6
        packet = packets[env*6+panel]
        assert row['env_index'] == env and row['num_timesteps'] == (i//4+1)*4
        assert row['episode_step'] == step and row['exploratory_horizon'] == 5
        assert row['TimeLimit.truncated'] is (step == 5)
        assert row['reset_panel_index'] == panel
        assert row['reset_case'] == row['source_case'] == packet['case']
        assert row['reset_frame'] == packet['frame'] == (0 if panel == 0 else 62)
        close(row['initial_cfd_time'], packet['time'])
        close(row['cfd_time'], packet['time'] + step*.1)
        assert row['observation_dimension'] == 69 and len(row['predicted_force']) == 4
        assert row['predicted_samples_in_reward_window'] == step
        assert row['measured_samples_in_reward_window'] == 62-step
        assert not row['scientific_admission']
        assert row['backend'] == 'actual_hydrogym_frozen_official_fno_not_cfd'
        s, nxt = row['symmetry_applied_orientation'], row['symmetry_next_orientation']
        assert s in (-1, 1) and nxt in (-1, 1) and row['symmetry_orientation'] == nxt
        assert 0 <= row['symmetry_pivot_index'] < 69
        assert row['symmetry_odd_margin'] >= 0 and isinstance(row['symmetry_reflection_fixed'], bool)
        canonical, physical = row['symmetry_canonical_action_received'], row['symmetry_physical_requested_action']
        assert len(canonical) == len(physical) == 1 and abs(canonical[0]) <= .75+1e-7
        assert physical[0] == s*canonical[0] == row['requested_omega']
        prev = packet['omega'] if step == 1 else previous[env]['applied_omega']
        if step > 1:
            assert s == previous[env]['symmetry_next_orientation']
        else:
            counts[env][panel] += 1
        requested = max(-.75, min(.75, physical[0]))
        expected = max(-.75, min(.75, max(prev-.1, min(prev+.1, requested))))
        close(row['applied_omega'], expected)
        close(row['applied_delta_omega'], row['applied_omega']-prev)
        assert abs(row['applied_omega']) <= .75+1e-7 and abs(row['applied_delta_omega']) <= .1+1e-7
        close(row['reward_actuation'], -.001*(row['applied_omega']/.75)**2)
        close(row['reward_rate'], -.001*(row['applied_delta_omega']/.1)**2)
        for k in REWARDS: reward_sums[k] += row[k]
        omega = row['applied_omega']; omega_sum += omega; omega_sq += omega*omega
        omega_min = min(omega_min, omega); omega_max = max(omega_max, omega)
        rate_limited += int(row['rate_limited']); switches += int(s != nxt)
        episodes += int(step == 5); n += 1; previous[env] = row
    assert n == 32768 and episodes == 6552
    assert counts == [[274,273,273,273,273,273]]*4
    return dict(transitions=n, episodes=episodes, reset_counts=counts,
                orientation_switches=switches, reward_means={k:v/n for k,v in reward_sums.items()},
                omega_mean=omega_sum/n, omega_rms=math.sqrt(omega_sq/n),
                omega_min=omega_min, omega_max=omega_max, rate_limited_fraction=rate_limited/n)


def audit(args):
    assert sha(args.approval) == args.approval_sha256
    spec = json.loads(args.approval.read_text())
    state = dict(x.split('=',1) for x in subprocess.check_output([
        'systemctl','--user','show',args.unit,'-p','InvocationID','-p','MainPID',
        '-p','SubState','-p','Result','-p','ExecMainStatus'], text=True).splitlines() if '=' in x)
    terminal(state, args.invocation)  # Must precede reading terminal artifacts.
    out = Path(spec['output']); result = json.loads((out/'result.json').read_text())
    assert json.loads((out/'source_spec.json').read_text()) == spec
    identity(result, spec, args.profile); optimizer(result)
    assert result['protocol'] == spec['protocol']
    for k,v in dict(seed=20261007,episode_steps=5,environments=4,n_steps=128,batch_size=256,
                    n_epochs=4,observation_dimension=69,force_window_samples=62,
                    action_limit=.75,action_delta_limit=.1,control_dt=.1,
                    norm_obs=False,norm_reward=False,final_policy_only=True).items():
        assert spec['protocol'][k] == v
    for group in ('source_files','runtime_sources'):
        for path, digest in spec[group].items(): assert sha(path) == digest, path
    for item in spec['inputs'].values(): assert sha(item['path']) == item['sha256']
    for name,digest in result['artifacts'].items():
        assert Path(name).name == name and sha(out/name) == digest
    assert result['fno_tensors_unchanged'] and not result['scientific_admission'] and not result['cfd_executed']
    assert result['policy_tensor_sha256_before'] != result['policy_tensor_sha256_after']
    assert result['precision_effective'] == dict(matmul='highest',cuda_tf32=False,cudnn_tf32=False)
    packets = json.loads((out/'reset_packets.json').read_text())['packets']
    assert [{'case':p['case'],'frame':p['frame']} for p in packets] == spec['reset_panel']
    with (out/'transitions.jsonl').open() as f:
        computed = check_rows((json.loads(line) for line in f), packets)
    diagnostic = result['diagnostics']
    for k,v in computed['reward_means'].items(): close(v, diagnostic['canonical_reward_component_mean_per_environment_step'][k], 1e-12)
    assert diagnostic['environment_steps'] == computed['transitions']
    assert diagnostic['episodes_completed'] == computed['episodes']
    assert list(result['reset_counts_by_phase'].values()) == computed['reset_counts']
    for k, value in [('mean',computed['omega_mean']),('rms',computed['omega_rms']),('minimum',computed['omega_min']),('maximum',computed['omega_max'])]:
        close(value, diagnostic['applied_omega'][k], 1e-12)
    close(computed['rate_limited_fraction'], diagnostic['rate_limited_fraction'], 1e-12)
    logs=[json.loads(x) for x in (out/'progress.json').read_text().splitlines()]
    finite(logs); assert len(logs)==65 and logs[-1]['train/n_updates']==256
    assert [x['time/total_timesteps'] for x in logs if 'time/total_timesteps' in x] == list(range(512,32769,512))
    sup=json.loads((out.parent/'supervisor_result.json').read_text())
    assert sup['error'] is None and sup['returncode']==0 and sup['approval_sha256']==args.approval_sha256
    assert sup['result_sha256']==sha(out/'result.json')
    assert sup['limits']['memory_max']==12*2**30 and sup['limits']['memory_swap_max']==0
    memory=[json.loads(x) for x in (out.parent/'memory.jsonl').read_text().splitlines()]
    minimum=min(x['MemAvailable'] for x in memory); assert minimum>=22*2**30
    return dict(status='CANONICAL_PPO_SAVED_EVIDENCE_VERIFIED_NOT_ADMISSION',
                profile=args.profile,unit=args.unit,invocation=args.invocation,
                approval_sha256=args.approval_sha256,result_sha256=sha(out/'result.json'),
                artifacts=result['artifacts'],computed=computed,optimizer_hooks=512,ppo_epochs=256,
                source_count=len(spec['source_files']),runtime_count=len(spec['runtime_sources']),
                minimum_available_bytes=minimum,model_deserialized=False,model_forward=False,
                physical_observation_canonicalization_independently_recomputed=False,
                dataset_payloads_rehashed=False,fno_unchanged_evidence='bound producer tensor digest; no independent model load')


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--approval',type=Path,required=True)
    p.add_argument('--approval-sha256',required=True);p.add_argument('--unit',required=True)
    p.add_argument('--invocation',required=True);p.add_argument('--profile',choices=PROFILES,required=True)
    print(json.dumps(audit(p.parse_args()),indent=2))
