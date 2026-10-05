from pathlib import Path
from types import SimpleNamespace
import copy
import pytest
import torch
import probe_fcp022_causal_conditioning as p

def models():
    return torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(.1,dtype=torch.float64)) for _ in range(28)])

def optimizer(model):return torch.optim.AdamW(model.parameters(),lr=p.PROTOCOL['learning_rate'],weight_decay=p.PROTOCOL['weight_decay'],betas=tuple(p.PROTOCOL['betas']),eps=p.PROTOCOL['eps'])

def test_accumulation_matches_loss_mean_then_single_clip():
    a=models();b=copy.deepcopy(a);oa=optimizer(a);ob=optimizer(b);calls=[]
    def run(coefficient):
        loss=sum(x*coefficient for x in a);loss.backward();return {'training_total':float(loss.detach())}
    result=p.accumulation_step(a,oa,list(range(1,7)),run,lambda:calls.append(len(oa.state)) or {'states':len(oa.state)})
    (sum(sum(x*i for x in b) for i in range(1,7))/6).backward()
    norm=torch.nn.utils.clip_grad_norm_(list(b.parameters()),1.);ob.step()
    for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)
    assert result['preclip_mean_gradient_norm']==pytest.approx(float(norm))
    assert calls==[0,28]
    assert result['actual_adam_moment_bytes']==28*2*8
    p.check_optimizer(oa,1)

def test_exact_six_required():
    a=models()
    with pytest.raises(ValueError):p.accumulation_step(a,optimizer(a),[1]*5,lambda _:None,lambda:{})

def test_missing_gradients_fail_before_step():
    a=models();o=optimizer(a)
    with pytest.raises(FloatingPointError):p.accumulation_step(a,o,[1]*6,lambda _:dict(training_total=0),lambda:{})
    assert not o.state

def test_nonfinite_raw_gradient_rejected():
    a=models();o=optimizer(a)
    def run(_):
        (sum(a)*float('nan')).backward();return {'training_total':float('nan')}
    with pytest.raises(FloatingPointError):p.accumulation_step(a,o,[1]*6,run,lambda:{})
    assert not o.state

def test_fresh_optimizer_after_independent_restore():
    a=models();initial=copy.deepcopy(a.state_dict());o=optimizer(a)
    for x in a:x.grad=torch.ones_like(x)
    o.step();p.check_optimizer(o,1)
    a.load_state_dict(initial);a.zero_grad(set_to_none=True);fresh=optimizer(a)
    assert len(fresh.state)==0
    assert all(x.grad is None for x in a)
    for n,v in a.state_dict().items():torch.testing.assert_close(v,initial[n],rtol=0,atol=0)

@pytest.mark.parametrize('key,value',[('lr',1e-5),('weight_decay',0.),('eps',1e-7),('betas',(.8,.999))])
def test_optimizer_knob_mismatch(key,value):
    a=models();o=optimizer(a)
    for x in a:x.grad=torch.ones_like(x)
    o.step();o.param_groups[0][key]=value
    with pytest.raises(ValueError):p.check_optimizer(o,1)

def test_nonfinite_moment_rejected():
    a=models();o=optimizer(a)
    for x in a:x.grad=torch.ones_like(x)
    o.step();next(iter(o.state.values()))['exp_avg'].fill_(float('nan'))
    with pytest.raises(FloatingPointError):p.check_optimizer(o,1)

@pytest.mark.parametrize('memory,elapsed,startup',[({'MemFree':29.9,'MemAvailable':60},0,True),({'MemFree':31,'MemAvailable':49},0,True),({'MemFree':19.9,'MemAvailable':100},0,False),({'MemFree':31,'MemAvailable':100},1800.01,False)])
def test_resource_stops(memory,elapsed,startup):
    with pytest.raises(RuntimeError):p.limits(memory,elapsed,startup)

def test_valid_resource_edges():
    p.limits({'MemFree':30,'MemAvailable':50},0,True);p.limits({'MemFree':20,'MemAvailable':20},1800)

def panel(value):
    stats={k:value for k in ['bias_mse','rms_error_mse','centered_residual_mse']}
    x={'aggregate':{d:{'six_window_original_objective':value,'five_nonzero':dict(stats)} for d in ['h1','ar']},'rows':[{'global_index':i,'panel':{'value':value}} for i in [160,816,923,975,1077,1233]]}
    x['repeat']=copy.deepcopy(x);return x

@pytest.fixture
def helper():
    path=Path('/workspace/fluid_control/scripts/probe_fcp020_symmetric_statistics.py')
    if not path.exists():pytest.skip('Main pinned P020 fixture absent')
    return p.load(path,p.P020_SHA,'test_p020')

def test_objective_must_improve_control_and_resolve_repeat(helper):
    initial=panel(3.);a=panel(2.);b=panel(1.)
    assert p.comparison(helper,initial,a,b)['strict_local_conditions']
    assert p.repeat_resolution(helper,initial,a,initial,b)['numerically_resolved']
    b['aggregate']['h1']['six_window_original_objective']=2.
    assert not p.comparison(helper,initial,a,b)['strict_local_conditions']
    assert not p.repeat_resolution(helper,initial,a,initial,b)['numerically_resolved']

def test_unresolved_objective_vs_control_noise(helper):
    initial=panel(3.);a=panel(2.);b=panel(1.)
    a['repeat']['aggregate']['h1']['six_window_original_objective']=3.1
    assert not p.repeat_resolution(helper,initial,a,initial,b)['numerically_resolved']

def test_protocol_identity_fixed():
    assert p.PROTOCOL['updates_per_arm']*p.PROTOCOL['windows_per_update']*2==192
    assert p.PROTOCOL['allocator_fraction']==.06 and p.PROTOCOL['learning_rate']==1.5625e-7
    assert len(p.protocol_sha())==64

def test_no_candidate_save_api():
    import ast
    tree=ast.parse(Path(p.__file__).read_text());attrs=[x.attr for x in ast.walk(tree) if isinstance(x,ast.Attribute)]
    assert 'save_checkpoint' not in attrs and 'save' not in attrs
