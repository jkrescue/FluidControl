import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch
import pytest

S=Path(__file__).resolve().parents[1]
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
m=load(S/'scripts/p064_temporal_increment_objective.py','new_objective')
old=load(Path('/workspace/fluid_control/scripts/p026_history_objective.py'),'old_objective')

def balanced(p,t):
    mse=(p-t).square().mean(tuple(range(p.ndim-1)))
    return {'balanced':.5*mse.mean()+.5*mse[3], 'channel_mse':mse}
obj=SimpleNamespace(balanced_force_objective=balanced)

def fixture(batch=2):
    torch.manual_seed(72)
    kw=dict(flow_states=torch.randn(batch,100,3,1,1,dtype=torch.float64),
        h1_states=torch.randn(batch,100,3,1,1,dtype=torch.float64),
        mask=torch.ones(batch,1,1,1,dtype=torch.float64),
        omega=torch.randn(batch,101,1,dtype=torch.float64),
        target_force=torch.randn(batch,100,4,dtype=torch.float64),
        preceding_states=torch.empty(batch,0,3,1,1,dtype=torch.float64),
        preceding_actions=torch.empty(batch,0,1,dtype=torch.float64))
    return kw

def predict(model,inputs,masks):
    return None,torch.tanh(inputs[:,:,0,0] @ model)

def test_all_99_edges_full_sequence_gradient_oracle(monkeypatch):
    monkeypatch.setattr(m,'TEMPORAL_INCREMENT_WEIGHT',1.)
    kw=fixture();p=torch.randn(6,4,dtype=torch.float64,requires_grad=True);q=p.detach().clone().requires_grad_()
    calls=[]
    def recorded(model,x,mask):calls.append(x.shape[0]);return predict(model,x,mask)
    result=m.chunk_force_objective(p,**kw,predict_fn=recorded,original_objective=obj,backward=True)
    values=[]
    for states in (kw['h1_states'],kw['flow_states']):
        x,mask=m.chunk_inputs(states,kw['preceding_states'],kw['mask'],kw['omega'],kw['preceding_actions'],0,100)
        values.append(predict(q,x,mask)[1].reshape(2,100,4))
    base=.5*balanced(values[0],kw['target_force'])['balanced']+.5*balanced(values[1],kw['target_force'])['balanced']
    auxiliary=m.temporal_increment_objective(values[0],kw['target_force'],obj)
    (base+auxiliary).backward()
    assert torch.allclose(p.grad,q.grad,atol=1e-12,rtol=1e-12)
    # Existing detached diagnostic accumulators are float32, unlike this float64 gradient oracle.
    assert result['training_objective']==pytest.approx(float((base+auxiliary).detach()),rel=1e-6)
    assert result['temporal_increment_loss']==pytest.approx(float(auxiliary.detach()),rel=1e-6)
    assert calls.count(40)==10 and calls.count(4)==9

def test_lambda_zero_original_loss_gradient_and_forward_exact(monkeypatch):
    monkeypatch.setattr(m,'TEMPORAL_INCREMENT_WEIGHT',0.)
    kw=fixture();p=torch.randn(6,4,dtype=torch.float64,requires_grad=True);q=p.detach().clone().requires_grad_()
    a=m.chunk_force_objective(p,**kw,predict_fn=predict,original_objective=obj,backward=True)
    b=old.chunk_force_objective(q,**kw,predict_fn=predict,original_objective=obj,backward=True)
    assert torch.equal(p.grad,q.grad)
    for key in ('total','h1_balanced','ar_balanced','h1_channel_mse','ar_channel_mse'):assert a[key]==b[key]
    for domain in ('h1','ar'):assert torch.equal(a['normalized_predictions'][domain],b['normalized_predictions'][domain])

def test_evaluation_original_components_unchanged():
    kw=fixture();p=torch.randn(6,4,dtype=torch.float64,requires_grad=True)
    a=m.chunk_force_objective(p,**kw,predict_fn=predict,original_objective=obj,backward=False)
    b=old.chunk_force_objective(p,**kw,predict_fn=predict,original_objective=obj,backward=False)
    assert a['total']==b['total'] and a['temporal_increment_loss'] is None and p.grad is None

def test_increment_nullspace_and_both_endpoint_gradients():
    p=torch.tensor([[[1.,2,3,4],[2.,4,6,8]]],requires_grad=True)
    t=torch.zeros_like(p)
    loss=m.temporal_increment_objective(p,t,obj);loss.backward()
    assert torch.equal(p.grad[:,0],-p.grad[:,1])
    assert m.temporal_increment_objective(p+3,t,obj)==loss

def test_target_gradient_rejected():
    p=torch.zeros(1,2,4,requires_grad=True)
    with pytest.raises(ValueError,match='target'):
        m.temporal_increment_objective(p,p,obj)
