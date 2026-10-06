import importlib.util
from pathlib import Path
import pytest

p=Path(__file__).with_name('audit_canonical_ppo_terminal.py')
if not p.exists():
    p=Path(__file__).resolve().parents[1]/'scripts'/'audit_canonical_ppo_terminal.py'
s=importlib.util.spec_from_file_location('audit',p);a=importlib.util.module_from_spec(s);s.loader.exec_module(a)


def packets():
    return [dict(case=f'phase{e}_{k}',frame=0 if k==0 else 62,time=100.,omega=0.) for e in range(4) for k in range(6)]


def rows(change=None):
    ps=packets()
    for i in range(32768):
        env=i%4;step=(i//4)%5+1;panel=(i//20)%6;packet=ps[env*6+panel]
        r=dict(num_timesteps=(i//4+1)*4,env_index=env,episode_step=step,exploratory_horizon=5,
               reset_panel_index=panel,reset_case=packet['case'],source_case=packet['case'],reset_frame=packet['frame'],
               initial_cfd_time=100.,cfd_time=100.+.1*step,observation_dimension=69,predicted_force=[0.]*4,
               predicted_samples_in_reward_window=step,measured_samples_in_reward_window=62-step,
               scientific_admission=False,backend='actual_hydrogym_frozen_official_fno_not_cfd',
               symmetry_applied_orientation=1,symmetry_next_orientation=1,symmetry_orientation=1,
               symmetry_pivot_index=1,symmetry_odd_margin=1.,symmetry_reflection_fixed=False,
               symmetry_canonical_action_received=[0.],symmetry_physical_requested_action=[0.],
               requested_omega=0.,applied_omega=0.,applied_delta_omega=0.,rate_limited=False)
        r['TimeLimit.truncated']=step==5
        r.update({k:0. for k in a.REWARDS})
        if change and i==4:r.update(change)
        yield r


def test_full_counts():
    r=a.check_rows(rows(),packets())
    assert r['transitions']==32768 and r['episodes']==6552
    assert r['reward_means']==dict.fromkeys(a.REWARDS,0.)


@pytest.mark.parametrize('change',[
    {'episode_step':3},{'env_index':2},{'cfd_time':101.},
    {'symmetry_physical_requested_action':[.1]}, {'symmetry_applied_orientation':-1},
    {'applied_omega':.2}, {'reward_rate':float('nan')}, {'measured_samples_in_reward_window':62},
])
def test_bad_transition_rejected(change):
    with pytest.raises(AssertionError):a.check_rows(rows(change),packets())


def test_optimizer_hooks():
    r=dict(timesteps=32768,ppo_n_updates=256,optimizer_steps=[dict(optimizer_step=i+1,num_timesteps=(i//8+1)*512) for i in range(512)])
    a.optimizer(r);r['optimizer_steps'][3]['num_timesteps']=1024
    with pytest.raises(AssertionError):a.optimizer(r)


def test_terminal_must_be_finished():
    s=dict(InvocationID='expected',MainPID='0',Result='success',ExecMainStatus='0',SubState='exited')
    a.terminal(s,'expected');s['MainPID']='22'
    with pytest.raises(AssertionError):a.terminal(s,'expected')


def test_g_cannot_impersonate_b():
    status,kind,sha=a.PROFILES['G']
    r=dict(status=status,candidate_arm='G',candidate_manifest_kind=kind,candidate_manifest_sha256=sha,
           fno_runtime=dict(profile='p026_k1',manifest_kind=kind,dual_manifest_sha256=sha))
    spec=dict(candidate_manifest_kind=kind,candidate_manifest_sha256=sha,inputs={'manifest':{'sha256':sha}})
    a.identity(r,spec,'G')
    with pytest.raises(AssertionError):a.identity(r,spec,'B')
