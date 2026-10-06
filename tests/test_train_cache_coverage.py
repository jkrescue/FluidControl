import copy
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_train_cache_coverage import analyze, actions, error

def fixture():
    rows=[];files={};mapping=[]
    for i in range(44):
        name='case'+str(i);files[name+'.h5']='unused'
        mapping.append(dict(file=name+'.h5',canonical_physical_phase='b00'))
        metrics={m:dict(prediction=[[1.,2.,3.,4.]]*10,target=[[0.]*4]*10) for m in ('k1_h1','k1_ar','persistence')}
        rows.append(dict(case=name,family='base',phase='b00',origin=51,times=[j*.1 for j in range(62)],stored_physical_actions=[0.]*62,metrics=metrics))
    return dict(rows=rows,source_spec={'data':{'base':{'train_files':files}}}),dict(trajectories=mapping),{'base':{'frames_per_trajectory':801}}

def test_total_drag_cancellation():
    assert error([1,0,-1,0],[0]*4)==[1,0,1,0,0]

def test_action_clock():
    a=[0.]*62;a[52:57]=[.1,0,-.1,-.1,0]
    out=actions(a)
    assert out['current']==0 and out['changed']==4 and out['reversals']==2
    assert out['next']==a[52:57]

def test_all44_offsets_and_duration():
    out=analyze(*fixture());assert len(out['rows'])==44
    assert out['rows'][0]['nominal_duration']==80
    assert len(out['groups']['all']['mean_abs_error_by_offset']['k1_h1'])==5

@pytest.mark.parametrize('mutation',['duplicate','phase','target','persistence','nonfinite'])
def test_reject(mutation):
    c,p,m=copy.deepcopy(fixture());r=c['rows'][0]
    if mutation=='duplicate':c['rows'][0]=c['rows'][1]
    if mutation=='phase':r['phase']='b02'
    if mutation=='target':r['metrics']['k1_ar']['target']=[[9.]*4]*10
    if mutation=='persistence':r['metrics']['persistence']['prediction']=[[9.]*4]+[[1.,2.,3.,4.]]*9
    if mutation=='nonfinite':r['stored_physical_actions'][51]=float('nan')
    with pytest.raises(ValueError):analyze(c,p,m)
