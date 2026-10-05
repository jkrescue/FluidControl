import copy
import importlib.util
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import probe_fcp016_fixed_panel_fit as p


def test_fixed_protocol():
    assert (p.UPDATES,p.WINDOWS,p.PANELS)==(32,6,(0,8,16,32))


def test_amplitude_not_residual_rmse():
    t=np.sin(np.arange(100)/5)
    stats=p.rms_statistics(t+4,t,2)
    assert stats['absolute_tail62_rms_error'] < 1e-14
    assert stats['target_tail62_rms'] > 1
    stats=p.rms_statistics(2*t,t,2)
    assert stats['signed_tail62_rms_error']==pytest.approx(stats['target_tail62_rms'])


@pytest.mark.parametrize('kind',['shape','nan','std'])
def test_invalid_rms(kind):
    x=np.zeros(100)
    if kind=='shape': x=x[:99]
    if kind=='nan': x[0]=np.nan
    with pytest.raises(ValueError): p.rms_statistics(x,np.zeros(100),0 if kind=='std' else 1)


def test_six_raw_gradients_then_one_clip_equivalence(monkeypatch):
    model=torch.nn.Linear(2,1,bias=True,dtype=torch.float64)
    other=copy.deepcopy(model)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-4)
    reference=torch.optim.AdamW(other.parameters(),lr=1e-5,weight_decay=1e-4)
    xs=[torch.tensor([[i+1.,(-1.)**i]],dtype=torch.float64) for i in range(6)]
    def run(x):
        loss=(model(x)-10).square().mean();loss.backward();return float(loss.detach())
    reference.zero_grad()
    for x in xs: ((other(x)-10).square().mean()/6).backward()
    expected_norm=torch.nn.utils.clip_grad_norm_(other.parameters(),1.0)
    reference.step()
    original=torch.nn.utils.clip_grad_norm_
    calls=[]
    def clipped(params,*a,**kw):
        calls.append(1);return original(params,*a,**kw)
    monkeypatch.setattr(torch.nn.utils,'clip_grad_norm_',clipped)
    result=p.update(model,opt,xs,run)
    assert calls==[1]
    assert result['preclip_mean_gradient_norm']==pytest.approx(float(expected_norm),rel=1e-14)
    for a,b in zip(model.parameters(),other.parameters()): torch.testing.assert_close(a,b,rtol=1e-14,atol=1e-14)
    assert all(int(s['step'])==1 for s in opt.state.values())


@pytest.mark.parametrize('count',[0,5,7])
def test_wrong_window_count(count):
    model=torch.nn.Linear(1,1)
    with pytest.raises(ValueError): p.update(model,torch.optim.AdamW(model.parameters()),[0]*count,lambda x:None)


def test_missing_gradient_no_update():
    model=torch.nn.Linear(1,1)
    opt=torch.optim.AdamW(model.parameters())
    before=copy.deepcopy(model.state_dict())
    with pytest.raises(FloatingPointError): p.update(model,opt,[0]*6,lambda x:None)
    assert not opt.state
    for k,v in model.state_dict().items(): assert torch.equal(v,before[k])


def rows():
    return [dict(global_index=index,panel=dict(objective=dict(h1_balanced=i,ar_balanced=2*i),
        domains={d:dict(tail62=dict(bias_mse=i,centered_residual_mse=2*i,absolute_tail62_rms_error=3*i))
                 for d in ('h1','ar')})) for i,index in enumerate((160,816,923,975,1077,1233))]


def test_aggregation_excludes_zero_only_for_five_window_statistics():
    result=p.aggregate(rows())
    assert result['h1']['six_window_macro_objective']==2.5
    assert result['ar']['six_window_macro_objective']==5
    assert result['h1']['five_nonzero_macro']==dict(bias_mse=3,centered_residual_mse=6,absolute_tail62_rms_error=9)


def test_terminal_requires_both_domains_all_metrics():
    initial=p.aggregate(rows());terminal=copy.deepcopy(initial)
    for r in terminal.values():
        r['six_window_macro_objective']/=2
        for k in r['five_nonzero_macro']: r['five_nonzero_macro'][k]/=2
    assert p.terminal_comparison(initial,terminal)['local_support']
    terminal['ar']['five_nonzero_macro']['absolute_tail62_rms_error']=9
    assert not p.terminal_comparison(initial,terminal)['local_support']
    assert not p.terminal_comparison(initial,terminal)['scientific_admission']


def test_actual_pinned_chunk_capture_cpu():
    root=Path(__file__).resolve().parents[1]
    diagnostic,objective=p.load_dependencies(root/'artifacts/fcp013_training_source_1634c05_immutable',root/'scripts/diagnose_fcp014_train_objective.py')
    model=torch.nn.Linear(1,4)
    model.eval()
    state=torch.zeros(1,1,1,1)
    states=torch.linspace(0,1,100).reshape(1,100,1,1,1)
    batch=dict(state=state,mask=torch.ones(1,1,1,1),omega=torch.zeros(1,101),target_force=torch.zeros(1,100,4))
    sizes=[]
    def predict(network,inputs,mask):
        sizes.append(len(inputs))
        return inputs[:,0:1],network(inputs[:,0,0,0,None])
    result=p.capture_panel(diagnostic,objective,model,states,states,batch,torch.ones(4),predict)
    assert sizes==[20]*10
    assert not model.training
    assert all(x.grad is None for x in model.parameters())
    assert result['domains']['h1']['tail62']['target_tail62_rms']==0
    assert len(result['domains']['ar']['four_force_physical_mae'])==4
    assert result['objective']['h1_balanced']==result['objective']['ar_balanced']


def hashing():
    def digest(t): return t.detach().numpy().tobytes()
    return SimpleNamespace(tensor_sha256=digest,tensor_state_sha256=lambda model:tuple(digest(t) for t in model.state_dict().values()))


def test_panel_preserves_rng_grad_mode():
    model=torch.nn.Linear(1,1)
    model(torch.ones(1,1)).sum().backward()
    gradients=[x.grad.clone() for x in model.parameters()]
    random.seed(3); np.random.seed(4); torch.manual_seed(5)
    states=(random.getstate(),np.random.get_state(),torch.random.get_rng_state())
    with p.readonly_panel(model,hashing(),torch.device('cpu')):
        model.eval();random.random();np.random.rand();torch.rand(2)
        assert not torch.is_grad_enabled()
    assert model.training and torch.is_grad_enabled()
    assert random.getstate()==states[0]
    assert np.array_equal(np.random.get_state()[1],states[1][1])
    assert torch.equal(torch.random.get_rng_state(),states[2])
    assert all(torch.equal(x.grad,g) for x,g in zip(model.parameters(),gradients))


def test_panel_detects_gradient_mutation():
    model=torch.nn.Linear(1,1)
    with pytest.raises(RuntimeError):
        with p.readonly_panel(model,hashing(),torch.device('cpu')):
            model.weight.grad=torch.ones_like(model.weight)
