import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

path = Path(__file__).with_name('run_exploratory_ppo_real_cfd.py')
if not path.exists():
    path = Path(__file__).parents[1]/'scripts/run_exploratory_ppo_real_cfd.py'
spec = importlib.util.spec_from_file_location('direct_ppo_test', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_observation():
    x = np.arange(69, dtype=np.float32); x[-1] = .2
    assert np.array_equal(m.observation(x), x)
    for bad in (x[:68], np.full(69, np.nan), np.ones(69)):
        with pytest.raises(ValueError): m.observation(bad)


class Vec:
    norm_obs = False
    norm_reward = False
    training = True
    observation_space = SimpleNamespace(shape=(69,))
    action_space = SimpleNamespace(shape=(1,), low=np.array([-.75], np.float32), high=np.array([.75], np.float32))
    def normalize_obs(self, x): return x.copy()


def test_identity_vec():
    v = Vec(); m.validate_vec(v, Vec); assert v.training is False
    v.norm_obs = True
    with pytest.raises(ValueError): m.validate_vec(v, Vec)


def test_deterministic_direct_policy_no_selection():
    class Model:
        def predict(self, obs, deterministic):
            assert deterministic is True and obs.shape == (69,)
            return np.array([.72], np.float32), None
    assert m.predict(Model(), Vec(), np.zeros(69)) == float(np.float32(.72))


def test_normalizer_mutation_rejected():
    v = Vec(); v.normalize_obs = lambda x: x+1
    with pytest.raises(ValueError): m.predict(None, v, np.zeros(69))


def test_intended_openleft_counts():
    t = 148+.005*np.arange(1, 2481)
    a = np.column_stack([t, t*0+1, t*0])
    assert len(m.fixed_window(a, 148, 160.4)) == 2480
    assert len(m.fixed_window(a, 148, 154.2)) == 1240
    assert len(m.fixed_window(a, 154.2, 160.4)) == 1240
    with pytest.raises(ValueError): m.fixed_window(a[:-1], 154.2, 160.4)


def test_training_artifact_binding():
    r = dict(status='EXPLORATORY_H5_PPO_TRAINING_COMPLETE_NOT_ADMISSION', timesteps=4096,
             ppo_n_updates=32, optimizer_steps=[{'optimizer_step':i} for i in range(1,65)], fno_tensors_unchanged=True,
             artifacts={'ppo_final.zip':m.POLICY_SHA,'vecnormalize.pkl':m.VEC_SHA},
             protocol=dict(observation_dimension=69,action_limit=.75,action_delta_limit=.1,
                           control_dt=.1,norm_obs=False,norm_reward=False),scientific_admission=False)
    m.validate_training(r)
    r['optimizer_steps'] = []
    with pytest.raises(ValueError): m.validate_training(r)


def test_no_new_model_or_control_route():
    source = path.read_text()
    assert 'PPO.load' in source and "device='cpu'" in source
    assert 'load_bound_k1' not in source and 'foamToVTK' not in source
    assert source.count('transport.apply_action_rate_limit(requested, previous)') == 1
    assert 'transport.configure_interval(cases[\'zero\'], begin, end, 0., 0.)' in source


def test_summarize_actual_all_windows(tmp_path):
    cases = {role:tmp_path/role for role in ('ppo','zero')}
    t = 148+.005*np.arange(1,2481)
    for role,case in cases.items():
        for body in ('forceFront','forceRear'):
            p = case/'postProcessing'/body/'148'
            p.mkdir(parents=True)
            np.savetxt(p/'coefficient.dat', np.column_stack([t,t*0+1,t*0,t*0,np.sin(t)]))
    def metric(front,rear):
        return {'total_cd_mean':float(np.mean(front[:,1]+rear[:,1])),
                'rear_cl_fluctuation_rms':float(np.std(rear[:,2])),
                'rear_cl_mean':float(np.mean(rear[:,2])), 'samples':len(rear)}
    windows, hashes = m.summarize(cases, metric)
    assert len(hashes)==4
    assert [w['branches']['ppo']['samples'] for w in windows.values()] == [2480,1240,1240]
    assert all(w['paired_drag_reduction']==0 and w['paired_rear_cl_fluctuation_rms_ratio']==1
               and w['branches']['ppo']['rear_cl_abs_peak']>0 for w in windows.values())


def test_actual_saved_training_schema_when_available():
    import json
    p = Path('/workspace/fluid_control/artifacts/exploratory_h5_ppo_training_20261006/payload/result.json')
    if not p.exists(): pytest.skip('actual metadata not present')
    assert m.sha(p)==m.RESULT_SHA
    m.validate_training(json.loads(p.read_text()))


def test_actual_transport_import_without_model_or_solver():
    repo = Path('/workspace/fluid_control')
    p = repo/'scripts/run_full40_canonical_ppo_openfoam_feedback.py'
    if not p.exists(): pytest.skip('actual canonical source not present')
    transport = m.load_module('actual_transport_test', p)
    action = transport.apply_action_rate_limit(.72, 0.)
    assert action['applied_omega']==.1
    assert callable(transport.total_drag_observation_at) and callable(transport.force_metrics)
