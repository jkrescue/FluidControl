import copy
import importlib.util
from pathlib import Path
import pytest

source=Path(__file__).with_name('compare_p064_fixed_lags.py')
if not source.exists():source=Path(__file__).resolve().parents[1]/'scripts/compare_p064_fixed_lags.py'
s=importlib.util.spec_from_file_location('lag',source)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def fixture():
    h={'batch_size':1,'cases':[]};a={'cases':[]}
    for k in ('model_sha256','normalization_sha256','dual_fno_manifest_sha256','flow_model_sha256','flow_state_sha256','aerodynamic_state_sha256'):h[k]=a[k]='same'
    for name in sorted(m.CASES):
        y=[[t,2*t,-t,3*t] for t in range(101)]
        ar=dict(case=name,hdf5_sha256=name,times=[.1*t for t in range(101)],omega_endpoints=[0.]*101,true_forces=y,predicted_forces=copy.deepcopy(y))
        rows=[dict(step=t,input_time=ar['times'][t-1],target_time=ar['times'][t],omega_s=0.,omega_s_plus_1=0.,true_force_s_plus_1=y[t],predicted_force_s_plus_1=y[t]) for t in range(1,101)]
        a['cases'].append(ar);h['cases'].append(dict(case=name,hdf5_sha256=name,rows=rows))
    return h,a

def test_three_lags_common_denominator_and_signed_total():
    r=m.analyze(*fixture())
    for stream in ('h1','ar'):
        for lag in (-1,0,1):
            x=r['results'][stream][str(lag)]['pooled']
            assert x['rear_cl']['count']==588
            assert x['rear_cl']['mae']==abs(lag)*3
            assert x['rear_cl']['signed_bias']==-lag*3
            assert x['total_cd']['mae']==0
    assert r['targets']==list(range(2,100))

def test_response_and_first_step_are_explicit():
    r=m.analyze(*fixture())
    assert len(r['first_step_action_minus_zero'])==4
    for row in r['results']['h1']['0']['action_minus_zero'].values():
        assert len(row['truth'])==98
        assert row['metrics']['rear_cl']['mae']==0

@pytest.mark.parametrize('bad',['time','truth','duplicate','nonfinite','identity'])
def test_reject_unmatched_inputs(bad):
    h,a=fixture()
    if bad=='time':h['cases'][0]['rows'][0]['input_time']=-1
    if bad=='truth':h['cases'][0]['rows'][0]['true_force_s_plus_1']=[9]*4
    if bad=='duplicate':h['cases'][0]['rows'][1]['step']=1
    if bad=='nonfinite':h['cases'][0]['rows'][0]['predicted_force_s_plus_1']=[float('nan')]*4
    if bad=='identity':h['model_sha256']='different'
    with pytest.raises(ValueError):m.analyze(h,a)
