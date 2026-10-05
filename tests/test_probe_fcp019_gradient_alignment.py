import ast
from pathlib import Path

import pytest
import torch

import probe_fcp019_gradient_alignment as p


@pytest.mark.parametrize('kind',['mean','rms'])
def test_physical_tail_cotangent_matches_full_autograd(kind):
    torch.manual_seed(2)
    x=torch.randn(100,dtype=torch.float64,requires_grad=True);y=torch.randn(100,dtype=torch.float64)
    std=2.7
    if kind=='mean': loss=(std*(x[38:]-y[38:]).mean()).square()
    else: loss=(std*(x[38:].std(correction=0)-y[38:].std(correction=0))).square()
    loss.backward();value,cot=p.statistic_cotangent(x,y,std,kind)
    assert value==pytest.approx(float(loss.detach()),abs=1e-14)
    torch.testing.assert_close(cot,x.grad,rtol=1e-12,atol=1e-14)
    assert torch.count_nonzero(cot[:38])==0


def test_zero_rms_is_undefined_not_epsilon_regularized():
    value,cot=p.statistic_cotangent(torch.ones(100),torch.arange(100),1.,'rms')
    assert value>0 and cot is None


def test_rms_invariant_to_bias_but_mean_not():
    x=torch.arange(100,dtype=torch.float64);y=x*2
    a,ga=p.statistic_cotangent(x,y,3.,'rms');b,gb=p.statistic_cotangent(x+20,y,3.,'rms')
    assert a==b;torch.testing.assert_close(ga,gb)
    assert p.statistic_cotangent(x,y,3.,'mean')[0]!=p.statistic_cotangent(x+20,y,3.,'mean')[0]


class Objective:
    def chunk_force_objective(self,model,states,h1,mask,omega,target,predict,chunk_size,backward):
        total=0.
        for start in range(0,100,10):
            _,f=predict(model,torch.cat((h1[:,start:start+10],states[:,start:start+10])).reshape(20,3),mask)
            truth=target[:,start:start+10].reshape(10,4)
            weights=torch.tensor([.125,.125,.125,.625],dtype=f.dtype)
            loss=.05*((f[:10]-truth).square().mean(0)*weights).sum()+.05*((f[10:]-truth).square().mean(0)*weights).sum()
            if backward: loss.backward()
            total+=float(loss.detach())
        return {'total':total}


@pytest.mark.parametrize('domain',['h1','ar'])
@pytest.mark.parametrize('kind',['mean','rms'])
def test_streaming_vjp_matches_full_graph_and_preserves_model(domain,kind):
    torch.manual_seed(3)
    model=torch.nn.Linear(3,4,dtype=torch.float64)
    before={n:x.detach().clone() for n,x in model.named_parameters()}
    states,h1=torch.randn(1,100,3,dtype=torch.float64),torch.randn(1,100,3,dtype=torch.float64)
    batch=dict(mask=None,omega=None,target_force=torch.randn(1,100,4,dtype=torch.float64))
    predict=lambda model,x,mask:(None,model(x))
    original,out,loss=p.gradient_pass(Objective(),model,states,h1,batch,predict)
    value,cot=p.statistic_cotangent(out[domain][:,3],batch['target_force'][0,:,3],2.,kind)
    observed,replay,_=p.gradient_pass(Objective(),model,states,h1,batch,predict,cotangent=cot,domain=domain)
    full=model((h1 if domain=='h1' else states)[0])[38:,3]
    target=batch['target_force'][0,38:,3]
    reference=(2*(full-target).mean()).square() if kind=='mean' else (2*(full.std(correction=0)-target.std(correction=0))).square()
    reference.backward()
    for name,parameter in model.named_parameters():
        torch.testing.assert_close(observed[name],parameter.grad,rtol=1e-11,atol=1e-12)
        torch.testing.assert_close(before[name],parameter,rtol=0,atol=0)
    assert value==pytest.approx(float(reference.detach()))
    for name in out: torch.testing.assert_close(out[name],replay[name],rtol=0,atol=0)


def test_original_streaming_gradient_exact_mean_weighting():
    torch.manual_seed(4)
    model=torch.nn.Linear(3,4,dtype=torch.float64)
    states,h1=torch.randn(1,100,3,dtype=torch.float64),torch.randn(1,100,3,dtype=torch.float64)
    target=torch.randn(1,100,4,dtype=torch.float64)
    g,_,_=p.gradient_pass(Objective(),model,states,h1,dict(mask=None,omega=None,target_force=target),lambda m,x,k:(None,m(x)))
    weights=torch.tensor([.125,.125,.125,.625],dtype=torch.float64)
    loss=sum(.5*((model(x)-target).square().mean((0,1))*weights).sum() for x in (states,h1))
    loss.backward()
    for n,parameter in model.named_parameters(): torch.testing.assert_close(g[n],parameter.grad)


def test_complex_real_geometry_and_repeat_observational():
    a={'complex':torch.tensor([1+2j,3-4j]),'real':torch.tensor([2.])}
    b={'complex':torch.tensor([3-1j,2+5j]),'real':torch.tensor([-1.])}
    expected=float((a['complex'].conj()*b['complex']).real.sum())-2
    result=p.alignment(a,b)
    assert p.dot(a,b)==expected and result['negative_original_directional_derivative']==-expected
    assert 'NOT AdamW' in result['direction']
    repeat=p.repeat_difference(a,a)
    assert repeat['l2_difference']==0 and 'not rigorous' in repeat['interpretation']


def test_cpu_double_mean_and_zero_norm():
    total={}
    for _ in range(5): p.add_mean(total,{'g':torch.tensor([1+2j])},5)
    assert total['g'].dtype==torch.complex128
    torch.testing.assert_close(total['g'],torch.tensor([1+2j],dtype=torch.complex128))
    assert p.alignment({'g':torch.zeros(1)},{'g':torch.ones(1)})['cosine'] is None


def test_no_optimizer_or_model_save_calls():
    tree=ast.parse(Path(p.__file__).read_text())
    calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
    assert not any('optim.' in name or name.endswith(('save_checkpoint','torch.save','.step')) for name in calls)


def test_immutable_scope_and_gradient_cleanup(monkeypatch):
    import hashlib
    from types import SimpleNamespace
    flow,aero=torch.nn.Linear(2,2),torch.nn.Linear(2,2)
    tensor_sha=lambda x:hashlib.sha256(x.detach().numpy().tobytes()).hexdigest()
    state_sha=lambda m:hashlib.sha256(''.join(tensor_sha(x) for x in m.state_dict().values()).encode()).hexdigest()
    objective=SimpleNamespace(tensor_sha256=tensor_sha,tensor_state_sha256=state_sha)
    monkeypatch.setattr(p,'TENSORS',dict(flow=state_sha(flow),aerodynamic=state_sha(aero)))
    frozen={'bias':tensor_sha(aero.bias)}
    p.verify_unchanged(flow,aero,objective,frozen)
    aero.weight.grad=torch.ones_like(aero.weight)
    with pytest.raises(RuntimeError,match='cleared'):p.verify_unchanged(flow,aero,objective,frozen)
    aero.zero_grad(set_to_none=True)
    with torch.no_grad(): aero.bias.add_(1)
    with pytest.raises(RuntimeError,match='changed'):p.verify_unchanged(flow,aero,objective,frozen)


def test_actual_pinned_objective_mixed20_streaming_vjp():
    import importlib.util
    source=Path('/workspace/fluid_control/artifacts/fcp013_training_source_1634c05_immutable/scripts/train_fcp013_independent_force_fno.py')
    if not source.exists(): pytest.skip('immutable P013 artifact only present on Main')
    assert p.sha(source)=='f3cf4b9a745cc0cbee39db9385cbfc398e4834e25bc487d8fb2d6eca07b483d7'
    spec=importlib.util.spec_from_file_location('p019_test_immutable_objective',source)
    objective=importlib.util.module_from_spec(spec);spec.loader.exec_module(objective)
    torch.manual_seed(7)
    model=torch.nn.Linear(6,4,dtype=torch.float64)
    states,h1=torch.randn(1,100,3,1,1,dtype=torch.float64),torch.randn(1,100,3,1,1,dtype=torch.float64)
    batch=dict(mask=torch.ones(1,1,1,1,dtype=torch.float64),omega=torch.randn(1,101,1,dtype=torch.float64),target_force=torch.randn(1,100,4,dtype=torch.float64))
    predict=lambda m,x,mask:(None,m(x[:,:,0,0]))
    g,outputs,metrics=p.gradient_pass(objective,model,states,h1,batch,predict)
    inputs=[objective.make_inputs(x[0],batch['mask'].expand(100,-1,-1,-1),batch['omega'][0,:100],batch['omega'][0,1:]) for x in (h1,states)]
    forces=[predict(model,x,None)[1] for x in inputs]
    for domain,force in zip(('h1','ar'),forces): torch.testing.assert_close(outputs[domain],force.detach())
    loss=sum(.5*objective.balanced_force_objective(force[None],batch['target_force'])['balanced'] for force in forces)
    loss.backward()
    assert metrics['total']==pytest.approx(float(loss.detach()))
    for n,param in model.named_parameters():torch.testing.assert_close(g[n],param.grad)
    value,cot=p.statistic_cotangent(outputs['ar'][:,3],batch['target_force'][0,:,3],1.3,'rms')
    observed,_,extra=p.gradient_pass(objective,model,states,h1,batch,predict,cotangent=cot,domain='ar')
    force=predict(model,inputs[1],None)[1][38:,3]
    target=batch['target_force'][0,38:,3]
    stat=(1.3*(force.std(correction=0)-target.std(correction=0))).square();stat.backward()
    for n,param in model.named_parameters():torch.testing.assert_close(observed[n],param.grad)
    assert extra['cotangent_cast_max_absolute_difference']==0
