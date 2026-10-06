import importlib.util
from pathlib import Path
import torch
import json
import hashlib
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('h1', ROOT/'scripts/p064_h1_only_objective.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class Objective:
    @staticmethod
    def balanced_force_objective(pred, target):
        mse = ((pred-target)**2).mean((0,1))
        return dict(balanced=.5*mse.mean()+.5*mse[3], channel_mse=mse)

def run(weight, backward=True, ar_value=3.):
    h1 = torch.ones(1,100,3,1,1)
    ar = torch.full_like(h1, ar_value)
    calls = []
    def predict(model, inputs, masks):
        calls.append(inputs.detach().clone())
        return None, inputs[:,0,0,0,None]*weight[None,:]
    result = m.chunk_force_objective(None,ar,h1,torch.ones(1,1,1,1),
        torch.zeros(1,101,1),torch.zeros(1,100,4),torch.empty(1,0,3,1,1),
        torch.empty(1,0,1),predict,Objective,backward=backward)
    return result,calls

def test_actual_chunk_backward_matches_pure_h1_and_keeps_mixed_batch():
    w = torch.tensor([1.,2.,3.,4.],requires_grad=True)
    result,calls=run(w)
    expected=w.detach().clone().requires_grad_()
    Objective.balanced_force_objective(expected[None,None,:],torch.zeros(1,1,4))['balanced'].backward()
    torch.testing.assert_close(w.grad,expected.grad)
    b_component=w.detach().clone().requires_grad_()
    (.5*Objective.balanced_force_objective(b_component[None,None,:],torch.zeros(1,1,4))['balanced']).backward()
    torch.testing.assert_close(w.grad,2*b_component.grad)
    assert len(calls)==10 and all(x.shape==(20,6,1,1) for x in calls)
    assert all(torch.equal(x[:10,0],torch.ones(10,1,1)) and torch.equal(x[10:,0],torch.full((10,1,1),3.)) for x in calls)
    assert abs(result['total']-(.5*result['h1_balanced']+.5*result['ar_balanced']))<1e-5

def test_ar_values_change_diagnostics_not_training_gradient():
    a=torch.ones(4,requires_grad=True); b=a.detach().clone().requires_grad_()
    ra,_=run(a,ar_value=3.); rb,_=run(b,ar_value=7.)
    torch.testing.assert_close(a.grad,b.grad)
    assert rb['ar_balanced']>ra['ar_balanced']

def test_evaluation_does_not_backward_and_retains_original_components():
    w=torch.ones(4,requires_grad=True)
    r,_=run(w,False)
    assert w.grad is None
    assert abs(r['total']-5.)<1e-5 and abs(r['h1_balanced']-1.)<1e-5
    assert abs(r['ar_balanced']-9.)<1e-5

def test_eight_window_average_and_one_adam_step():
    w=torch.ones(4,requires_grad=True)
    for _ in range(8): run(w)
    w.grad.div_(8)
    expected=w.detach().clone().requires_grad_(); run(expected)
    torch.testing.assert_close(w.grad,expected.grad)
    opt=torch.optim.AdamW([w],lr=1.5625e-7)
    torch.nn.utils.clip_grad_norm_([w],1.)
    opt.step()
    assert opt.state[w]['step']==1

def test_new_consumer_protocol_and_old_profile_rejection(tmp_path):
    spec=importlib.util.spec_from_file_location('f_consumer',(ROOT/'dual_fno.py' if (ROOT/'dual_fno.py').is_file() else ROOT/'src/fluid_control/dual_fno_h1_only.py'))
    loader=importlib.util.module_from_spec(spec);sys.modules[spec.name]=loader
    spec.loader.exec_module(loader)
    repo=Path('/workspace/fluid_control')
    parent=repo/'artifacts/fcp064_controlled_aero_arm_b_20261006'
    payload=json.loads((parent/'dual_model_manifest.json').read_text())
    p=json.loads((parent/'training_protocol.json').read_text())
    p.update(training_experiment='FC-P064-H1-ONLY',
        objective='H1_only_half_equal_four_half_rearCl_normalized_MSE',
        diagnostic_objective='equal_H1_AR_half_equal_four_half_rearCl_normalized_MSE',
        mixed_forward_preserved=True,candidate_profile='FC_P064_H1_ONLY_K1_FRESH',
        h1_gradient_multiplier_vs_b_component=2.0,ar_gradient_multiplier_vs_b_component=0.0,
        history_objective_sha256='1cb9ea57f873a167352a962037a09b777cbf3fab0a4d83078c1fba67dab7f716')
    contract=loader._experiment_contract('FC_P064_H1_ONLY_K1_FRESH_FORCE_FNO')
    old=loader._experiment_contract(loader.P064_SYSTEM_KIND['B'])
    def write():
        target=tmp_path/'training_protocol.json';target.write_text(json.dumps(p))
        payload.update(training_semantics=p,training_protocol_file=target.name,
                       training_protocol_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    write();loader._validate_p026_protocol(tmp_path,payload,contract)
    with pytest.raises(ValueError):loader._validate_p026_protocol(tmp_path,payload,old)
    p['mixed_forward_preserved']=False;write()
    with pytest.raises(ValueError):loader._validate_p026_protocol(tmp_path,payload,contract)
