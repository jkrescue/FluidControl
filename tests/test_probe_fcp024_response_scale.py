import copy
import hashlib
import inspect
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import probe_fcp024_response_scale as probe


class Frozen(torch.nn.Module):
    def __init__(self):
        super().__init__();self.block=torch.nn.Parameter(torch.zeros(24,4,1,1),requires_grad=False)
        self.old=torch.nn.Parameter(torch.ones(2),requires_grad=False)
    def diagnostic_identity(self):
        return {'block':hashlib.sha256(self.block.detach().numpy().tobytes()).hexdigest(),'old':self.old.tolist()}


def panel(value):
    first=dict(rows=[dict(global_index=816,normalized_predictions=dict(h1=[value],ar=[-value]),panel=dict(mean=value))],aggregate=dict(mean=value))
    return dict(**first,repeat=copy.deepcopy(first))


def setup_scales():
    model=Frozen();vector=torch.arange(96,dtype=torch.float32).reshape(24,4,1,1)*1e-6
    prior=dict(expanded_initial_identity=model.diagnostic_identity())
    probe.apply_scale(model,vector,1)
    high=dict(terminal_effective_identity=model.diagnostic_identity(),initial=panel(0),terminal=panel(1))
    return model,vector,prior,high


def test_fresh_scales_not_cumulative_and_no_grad():
    model,vector,_,_=setup_scales()
    for scale in (64,0,8,-1,1):
        record=probe.apply_scale(model,vector,scale)
        assert torch.equal(model.block,vector*scale)
        assert record['values']==(vector*scale).flatten().tolist()
        assert model.block.grad is None and not model.block.requires_grad


def test_zero_scale_canonical_bytes_with_negative_coefficients():
    model=Frozen();initial=model.diagnostic_identity()
    vector=torch.linspace(-1,1,96,dtype=torch.float32).reshape(24,4,1,1)
    assert torch.signbit(vector*0).any()
    probe.apply_scale(model,vector,1)
    probe.apply_scale(model,vector,0)
    assert not torch.signbit(model.block).any()
    assert model.diagnostic_identity()==initial


def test_scale_order_and_reproduction_before_extrapolation():
    model,vector,prior,high=setup_scales();seen=[]
    def evaluate():
        scale=probe.PROTOCOL['scales'][len(seen)];seen.append(scale);return panel(scale)
    result=probe.run_scales(model,vector,prior,high,evaluate)
    assert seen==[0,1,-1,8,64]
    assert [r['exact_prior_reproduction'] for r in result]==[True,True,None,None,None]


@pytest.mark.parametrize('which',[0,1])
def test_reproduction_failure_stops_before_later_scales(which):
    model,vector,prior,high=setup_scales();seen=[]
    def evaluate():
        scale=probe.PROTOCOL['scales'][len(seen)];seen.append(scale);result=panel(scale)
        if scale==which:result['repeat']['rows'][0]['normalized_predictions']['ar'][0]+=1e-12
        return result
    with pytest.raises(RuntimeError,match='reproduction'):probe.run_scales(model,vector,prior,high,evaluate)
    assert seen==([0] if which==0 else [0,1])


def test_full_effective_identity_gate_precedes_panel():
    model,vector,prior,high=setup_scales();high['terminal_effective_identity']['effective_lifting_sha256']='wrong';seen=[]
    def evaluate():seen.append(True);return panel(0)
    with pytest.raises(RuntimeError,match='identity'):probe.run_scales(model,vector,prior,high,evaluate)
    assert len(seen)==1


@pytest.mark.parametrize('part',['rows','aggregate'])
def test_exact_metric_not_tolerance(part):
    original=panel(1);changed=copy.deepcopy(original)
    if part=='rows':changed[part][0]['panel']['mean']+=1e-12
    else:changed[part]['mean']+=1e-12
    with pytest.raises(RuntimeError):probe.exact_reproduction(changed,original)


def test_finite_failure_stops_without_skipping_scale():
    model,vector,prior,high=setup_scales();seen=[]
    def evaluate():
        scale=probe.PROTOCOL['scales'][len(seen)];seen.append(scale);return panel(float('nan') if scale==-1 else scale)
    with pytest.raises(FloatingPointError):probe.run_scales(model,vector,prior,high,evaluate)
    assert seen==[0,1,-1]


@pytest.mark.parametrize('mutation',['weights','grad','mode','torch_rng','numpy_rng','python_rng'])
def test_panel_nonmutation(mutation):
    model=Frozen()
    def run():
        if mutation=='weights':
            with torch.no_grad():model.block.add_(1)
        elif mutation=='grad':model.block.grad=torch.ones_like(model.block)
        elif mutation=='mode':model.eval()
        elif mutation=='torch_rng':torch.rand(1)
        elif mutation=='numpy_rng':np.random.rand()
        else:random.random()
        return [dict(value=1.)]
    with pytest.raises(RuntimeError):probe.immutable_panel(model,run,lambda rows:dict(value=1.))


def test_panel_unchanged_no_backward():
    model=Frozen();before=model.diagnostic_identity()
    result=probe.immutable_panel(model,lambda:[dict(value=1.)],lambda rows:dict(value=1.))
    assert result['aggregate']==dict(value=1.) and model.diagnostic_identity()==before


def test_prior_sha_fp32_reconstruction(tmp_path,monkeypatch):
    values=(torch.arange(96,dtype=torch.float32)*1e-6).double().tolist()
    payload=dict(status='FC_P023_INPUT_BLOCK_COMPARISON_COMPLETE_NOT_ADMISSION',arms=[dict(arm='HIGH',records=[dict(block_update=dict(values=values))])])
    path=tmp_path/'prior.json';path.write_text(json.dumps(payload));monkeypatch.setattr(probe,'PRIOR_SHA',probe.sha(path))
    _,_,vector=probe.read_prior(path)
    assert vector.dtype==torch.float32 and vector.flatten().double().tolist()==values
    path.write_text(json.dumps(payload)+' ')
    with pytest.raises(ValueError,match='SHA'):probe.read_prior(path)


def test_budget_and_no_optimizer_in_execution():
    probe.limits(dict(MemFree=30,MemAvailable=50),0,startup=True)
    for memory,elapsed,startup in [(dict(MemFree=19,MemAvailable=50),1,False),(dict(MemFree=30,MemAvailable=49),1,True),(dict(MemFree=40,MemAvailable=80),901,False)]:
        with pytest.raises(RuntimeError):probe.limits(memory,elapsed,startup)
    source=inspect.getsource(probe.execute)
    assert 'torch.optim' not in source and '.backward(' not in source
    assert 'model.restore_zero()' in source and 'final zero restoration failed' in source
    assert probe.PROTOCOL['optimizer_steps']==probe.PROTOCOL['backwards']==0
