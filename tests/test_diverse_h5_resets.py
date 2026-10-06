import copy
import numpy as np
import pytest
import torch
import exploratory_diverse_h5_resets as m


def fixture():
    t=154.2
    frame={'state':torch.ones(3,128,256),'mask':torch.ones(1,128,256),
           'omega':torch.tensor([.75]),'time':torch.tensor([t]),'force':torch.zeros(4)}
    times=np.round(t-.1*np.arange(61,-1,-1),8)
    forces=np.zeros((62,4)); forces[:,0]=1;forces[:,2]=1
    norm={'state_mean':[0,0,0],'state_std':[1,1,1]}
    return frame,t,.75,times,forces,norm


def test_fixed_panel():
    allrows=[s for phase in m.PHASES for s in m.starts(phase)]
    assert len(allrows)==len(set(allrows))==24
    assert sum(i==0 for _,i in allrows)==4
    with pytest.raises(ValueError):m.starts('01')


def test_real_current_packet_not_curated_force():
    args=fixture();p=m.checked_arrays(*args)
    assert p['omega']==.75 and np.array_equal(p['force'],args[4][-1].astype(np.float32))
    assert not np.array_equal(p['force'],args[0]['force'])
    args[0]['state'].add_(100);args[4][:]=100
    assert torch.all(p['field']==1) and p['history_forces'][0,0]==1


@pytest.mark.parametrize('change', ['future','gap','nonfinite','omega','time','mask','norm'])
def test_reject_corrupted_packet(change):
    a=list(fixture())
    if change=='future':a[3][-1]+=.1
    if change=='gap':a[3][0]-=.1
    if change=='nonfinite':a[4][2,1]=np.nan
    if change=='omega':a[0]['omega']=torch.tensor([.5])
    if change=='time':a[0]['time']=torch.tensor([155.])
    if change=='mask':a[0]['mask'][0,0,0]=.5
    if change=='norm':a[5]['state_std'][0]=0
    with pytest.raises(ValueError):m.checked_arrays(*a)


def test_action_schedule_actual_values():
    points=[[148,0],[148.75,.75],[228,.75]]
    assert m.scheduled_omega(points,154.2)==.75
    assert m.scheduled_omega(points,148)==0
    with pytest.raises(ValueError):m.scheduled_omega(points,240)
