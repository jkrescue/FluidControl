import copy
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from development_phase_selection import make_selection


def progress(base):
    rows=[]
    prior=0.0
    for i in range(800):
        applied=.1 if i%2 == 0 else 0.0
        before=[0.0]*64+[float(i+k) for k in range(4)]+[prior]
        after=[0.0]*64+[float(i+1+k) for k in range(4)]+[applied]
        rows.append(dict(step=i+1,start_time=base+.1*i,end_time=base+.1*(i+1),
                         applied_omega=applied,requested_omega=applied,
                         input_observation=before,output_observation=after))
        prior=applied
    return dict(completed_cycles=800,rows=rows)


@pytest.mark.parametrize('phase,base',[('b01',130),('b03',144)])
def test_fixed_selection_clock_and_force(phase,base):
    doc=progress(base); saved=copy.deepcopy(doc)
    out=make_selection(doc,'/unused',phase,require_files=False)
    assert doc == saved
    assert len(out['records'])==8 and out['frames']==48 and out['endpoints']==40
    indices=[f['global_index'] for r in out['records'] for f in r['frames']]
    assert len(set(indices))==48 and indices[-1]==705
    for r in out['records']:
        j=r['start_index']
        assert r['frames'][0]['time']==base+.1*j
        assert r['initial_force']==doc['rows'][j]['input_observation'][64:68]
        for k,tr in enumerate(r['transitions']):
            assert tr['omega_now']==(doc['rows'][j+k-1]['applied_omega'] if j+k else 0)
            assert tr['omega_next']==doc['rows'][j+k]['applied_omega']
            assert tr['truth_force']==doc['rows'][j+k]['output_observation'][64:68]
        assert all(f['files']['U'].startswith('case_mpc/') for f in r['frames'])


@pytest.mark.parametrize('mutation',['phase','clock','action','count'])
def test_reject(mutation):
    doc=progress(130);phase='b01'
    if mutation=='phase':phase='b00'
    if mutation=='clock':doc['rows'][500]['end_time']+=.1
    if mutation=='action':doc['rows'][100]['input_observation'][68]=.7
    if mutation=='count':doc['completed_cycles']=799
    with pytest.raises(ValueError):make_selection(doc,'/unused',phase,require_files=False)
