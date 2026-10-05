import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import probe_fcp025_isolated_statistics as p

REPO=Path('/workspace/fluid_control')

def dependency(relative,digest,name):return p.load(REPO/relative,digest,name)

@pytest.fixture
def helpers():
    return (dependency('scripts/p021_causal_force.py',p.CAUSAL_SHA,'test_causal'),
            dependency('scripts/p023_force_block.py',p.BLOCK_SHA,'test_block'),
            dependency('scripts/probe_fcp019_gradient_alignment.py',p.P019_SHA,'test_p019'),
            dependency('artifacts/fcp013_training_source_1634c05_immutable/scripts/train_fcp013_independent_force_fno.py','f3cf4b9a745cc0cbee39db9385cbfc398e4834e25bc487d8fb2d6eca07b483d7','test_objective'))


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_four_values_and_cotangents_match_pinned_p019(helpers,dtype):
    _,_,old,_=helpers;torch.manual_seed(7)
    target=torch.randn(100,4,dtype=dtype)
    out={d:torch.randn(100,4,dtype=dtype,requires_grad=True) for d in ('h1','ar')};out['total']=torch.tensor(.012,dtype=dtype,requires_grad=True)
    augmented,terms=p.statistical_objective(out,target)
    assert augmented.dtype==torch.float64 and len(terms)==4
    for d in ('h1','ar'):
        for kind,key in [('mean','mean_error_squared'),('rms','rms_error_squared')]:
            value,cotangent=old.statistic_cotangent(out[d][:,3],target[:,3],1.,kind)
            assert float(terms[d+'_'+key].detach())==value
            gradient=torch.autograd.grad(terms[d+'_'+key],out[d],retain_graph=True)[0]
            torch.testing.assert_close(gradient[:,3],cotangent.to(dtype),rtol=1e-12 if dtype==torch.float64 else 1e-6,atol=1e-15 if dtype==torch.float64 else 1e-9)
            assert torch.count_nonzero(gradient[:38])==0 and torch.count_nonzero(gradient[:,:3])==0
    allgrads=torch.autograd.grad(augmented,[out['total'],out['h1'],out['ar']])
    assert allgrads[0].item()==1
    for d,g in zip(('h1','ar'),allgrads[1:]):
        expected=sum(old.statistic_cotangent(out[d][:,3],target[:,3],1.,k)[1] for k in ('mean','rms'))*(5/16)
        torch.testing.assert_close(g[:,3],expected.to(dtype),rtol=1e-6,atol=1e-9)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.lift=torch.nn.Conv2d(12,24,1,dtype=torch.float64)
        with torch.no_grad():self.lift.weight.mul_(.2);self.lift.weight[:,6:10].zero_()
    def forward(self,x):return self.lift(x).tanh()[:,:4]


def fixture(helpers):
    causal,block,_,obj=helpers;torch.manual_seed(19)
    model=block.FrozenForceBlock(Toy(),lift_name='lift.weight')
    with torch.no_grad():model.block.copy_(torch.linspace(-.04,.06,96,dtype=torch.float64).reshape_as(model.block))
    inputs=[torch.randn(100,6,1,1,dtype=torch.float64),torch.randn(100,6,1,1,dtype=torch.float64),torch.ones(1,1,1,1,dtype=torch.float64),torch.randn(101,4,dtype=torch.float64),torch.randn(100,4,dtype=torch.float64)]
    def predict(m,x,mask):
        coords=torch.zeros(x.shape[0],2,*x.shape[-2:],dtype=x.dtype)
        return None,m(torch.cat((x,coords),1)).mean((-2,-1))
    def run(checkpointed):return causal.recurrent_objective(model,*inputs,predict,obj.balanced_force_objective,checkpointed=checkpointed)
    return model,inputs,run


def test_full_h100_checkpoint_and_independent_four_gradient_sum(helpers):
    model,inputs,run=fixture(helpers);before=model.diagnostic_identity()
    full=run(False);total,terms=p.statistical_objective(full,inputs[-1])
    components=[torch.autograd.grad(x,model.block,retain_graph=True)[0] for x in [full['total'],*terms.values()]]
    assert all(x.norm()>0 for x in components)
    gradient=torch.autograd.grad(total,model.block)[0]
    torch.testing.assert_close(gradient,components[0]+(5/16)*sum(components[1:]),rtol=1e-11,atol=1e-13)
    checkpoint=run(True);checkpoint_total,_=p.statistical_objective(checkpoint,inputs[-1]);checkpoint_total.backward()
    torch.testing.assert_close(checkpoint_total,total,rtol=0,atol=0)
    torch.testing.assert_close(model.block.grad,gradient,rtol=1e-11,atol=1e-13)
    assert model.diagnostic_identity()==before
    assert all(not param.requires_grad and param.grad is None for param in model.base.parameters())


@pytest.mark.parametrize('domain',['h1','ar'])
def test_no_zero_rms_smoothing(domain):
    out={d:torch.arange(400,dtype=torch.float32).reshape(100,4) for d in ('h1','ar')};out['total']=torch.tensor(1.)
    out[domain]=torch.ones(100,4)
    with pytest.raises(FloatingPointError,match='zero'):p.statistical_objective(out,torch.randn(100,4))


def test_nonfinite_target_rejected():
    out=dict(h1=torch.randn(100,4),ar=torch.randn(100,4),total=torch.tensor(1.));target=torch.randn(100,4);target[50,3]=float('nan')
    with pytest.raises(FloatingPointError):p.statistical_objective(out,target)


def test_six_average_before_clip_single96_adam(helpers):
    _,block,_,_=helpers;model,_,_=fixture(helpers);optimizer=torch.optim.AdamW([model.block],lr=1e-5,weight_decay=1e-4)
    block.assert_optimizer_scope(model,optimizer);before=model.block.detach().clone();raws=[]
    def run(i):
        gradient=torch.ones_like(model.block)*(i+1);raws.append(gradient)
        (model.block*gradient).sum().backward();return dict(training_total=float(i))
    record=p.accumulation_step(model,optimizer,list(range(6)),run,lambda:{})
    reference=torch.nn.Parameter(before.clone());other=torch.optim.AdamW([reference],lr=1e-5,weight_decay=1e-4)
    reference.grad=sum(raws)/6;norm=torch.nn.utils.clip_grad_norm_([reference],1.);other.step()
    assert record['preclip_mean_gradient_norm']==float(norm)
    torch.testing.assert_close(model.block,reference,rtol=0,atol=0)
    p.check_optimizer(model,optimizer,1,1e-5,block);model.verify_frozen()
    assert len(optimizer.state)==1 and model.block.numel()==96


def test_exact_initial_includes_both_repeats_raw_and_aggregate():
    row=dict(rows=[dict(normalized_predictions=dict(h1=[1.],ar=[2.]))],aggregate=dict(mean=1.))
    reference=dict(**row,repeat=copy.deepcopy(row));p.exact_initial(reference,reference)
    changed=copy.deepcopy(reference);changed['repeat']['rows'][0]['normalized_predictions']['ar'][0]+=1e-12
    with pytest.raises(RuntimeError,match='reproduction'):p.exact_initial(changed,reference)


def test_budget_and_fixed_protocol():
    assert p.PROTOCOL['learning_rates']==dict(STAT=1e-5) and p.PROTOCOL['statistic_coefficient']==5/16
    assert p.PROTOCOL['updates_per_arm']==16
    p.limits(dict(MemFree=30,MemAvailable=50),1,True)
    with pytest.raises(RuntimeError):p.limits(dict(MemFree=40,MemAvailable=80),1201)
    with pytest.raises(RuntimeError):p.limits(dict(MemFree=19,MemAvailable=80),1)


def test_pinned_real_control():
    prior,control=p.read_control(REPO/'artifacts/fcp023_input_block_20261005/result.json')
    assert control['arm']=='HIGH' and control['learning_rate']==1e-5
    p.exact_initial(control['initial'],control['initial'])


def test_historical_sources_and_precision_bound(tmp_path):
    path=tmp_path/'source';path.write_text('source')
    prior=dict(source_sha256=dict(config=p.sha(path)),precision=dict(tf32=True))
    args=SimpleNamespace(config=path)
    p.verify_control_execution(prior,args,dict(tf32=True))
    with pytest.raises(ValueError,match='precision'):p.verify_control_execution(prior,args,dict(tf32=False))
    path.write_text('changed')
    with pytest.raises(ValueError,match='dependency'):p.verify_control_execution(prior,args)
