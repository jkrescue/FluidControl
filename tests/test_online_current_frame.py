import copy
import hashlib
import json
import numpy as np
import pytest
import torch
from fluid_control.online_current_frame import normalize_current, build_current_input


def fixture():
    packet = dict(state=np.ones((3,128,256),np.float32), mask=np.ones((1,128,256),np.uint8),
                  time=np.array([12.3]), x=np.linspace(8,25,256,dtype=np.float32),
                  y=np.linspace(4,11,128,dtype=np.float32))
    packet["mask"][0,0,0] = 0
    packet["state"][:,0,0] = 0
    raw=json.dumps(dict(state_channels=["u","v","gauge_pressure"],state_mean=[.1,.2,.3],state_std=[.5,2,3])).encode()
    args=dict(expected_time=12.3, expected_mask=packet["mask"].copy(), normalization_bytes=raw,
              expected_normalization_sha256=hashlib.sha256(raw).hexdigest())
    return packet,args


def test_exact_formula_and_real_helper_packing_no_mutation():
    p, args=fixture(); before=copy.deepcopy(p)
    current=normalize_current(p, **args)
    expected=(torch.from_numpy(p["state"])-torch.tensor([.1,.2,.3])[:,None,None])/torch.tensor([.5,2,3])[:,None,None]
    expected*=torch.from_numpy(p["mask"])
    assert torch.equal(current["state"][0],expected)
    packed=build_current_input(current,applied_omega_now=.15,constrained_omega_next=.225)
    assert packed.shape==(1,6,128,256)
    assert torch.equal(packed[:,:3],current["state"])
    assert torch.equal(packed[:,3:4],current["mask"])
    assert torch.all(packed[:,4]==torch.tensor(.15,dtype=torch.float32)/.75)
    assert torch.all(packed[:,5]==torch.tensor(.225,dtype=torch.float32)/.75)
    assert all(np.array_equal(p[k],before[k]) for k in p)


def test_physical_actions_cast_to_fp32_before_division_exactly_like_datapipe():
    p,a=fixture(); current=normalize_current(p,**a)
    packed=build_current_input(current,applied_omega_now=.01,constrained_omega_next=.02)
    expected_now=torch.tensor(.01,dtype=torch.float32)/.75
    expected_next=torch.tensor(.02,dtype=torch.float32)/.75
    assert torch.all(packed[:,4]==expected_now)
    assert torch.all(packed[:,5]==expected_next)
    assert expected_now != torch.tensor(.01/.75,dtype=torch.float32)


@pytest.mark.parametrize("kind",["sha","grid","mask","time","nan","shape","solid","future_key"])
def test_bad_packets(kind):
    p,a=fixture()
    if kind=="sha": a["expected_normalization_sha256"]="0"*64
    if kind=="grid": p["x"][0]+=1
    if kind=="mask": p["mask"][0,1,1]=0
    if kind=="time": p["time"][0]+=.1
    if kind=="nan": p["state"][0,1,1]=np.nan
    if kind=="shape": p["state"]=p["state"][:2]
    if kind=="solid": p["state"][:,0,0]=1
    if kind=="future_key": p["target_state"]=p["state"].copy()
    with pytest.raises(ValueError): normalize_current(p,**a)


@pytest.mark.parametrize("now,nxt",[(True,0),(0,float("nan")),(.75,.8),(0,.2)])
def test_illegal_commands_not_clamped(now,nxt):
    p,a=fixture(); c=normalize_current(p,**a)
    with pytest.raises(ValueError): build_current_input(c,applied_omega_now=now,constrained_omega_next=nxt)
