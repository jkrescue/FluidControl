import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import train_fcp015_window_accumulation as p


def fixture_models():
    torch.manual_seed(123)
    a = torch.nn.Linear(3, 4)
    return a, copy.deepcopy(a), [(torch.randn(5,3)*i,torch.randn(5,4)) for i in range(1,9)]


def window_loss(model, window):
    x,y = window
    error = (model(x)-y).square().mean(dim=0)
    return .5*error.mean()+.5*error[3]


def run_for(model):
    def run(window):
        loss = window_loss(model,window)
        loss.backward()
        return dict(h1_balanced=loss.item(),ar_balanced=loss.item(),total=loss.item())
    return run


def test_raw_average_matches_scaled_loss_before_clip_and_adam_step():
    a,b,windows = fixture_models()
    oa = torch.optim.AdamW(a.parameters(),lr=1e-5,weight_decay=1e-4)
    ob = torch.optim.AdamW(b.parameters(),lr=1e-5,weight_decay=1e-4)
    captured = []
    def audit(model):
        captured.extend(q.grad.clone() for q in model.parameters())
        return {'finite':True}
    result = p.accumulation_step(a,oa,windows,run_for(a),audit)
    ob.zero_grad(set_to_none=True)
    for window in windows: (window_loss(b,window)/8).backward()
    for got,param in zip(captured,b.parameters()):
        torch.testing.assert_close(got,param.grad,rtol=2e-6,atol=2e-6)
    norm = torch.nn.utils.clip_grad_norm_(b.parameters(),1.0)
    assert norm > 1
    assert result['preclip_mean_gradient_norm'] == pytest.approx(norm.item(),rel=2e-6)
    ob.step()
    for x,y in zip(a.parameters(),b.parameters()): torch.testing.assert_close(x,y,rtol=1e-7,atol=1e-7)
    assert {int(s['step']) for s in oa.state.values()} == {1}
    assert result['windows'] == 8 and result['optimizer_steps'] == 1


def test_not_per_window_clipping():
    a,b,windows = fixture_models()
    total = [torch.zeros_like(q) for q in b.parameters()]
    for window in windows:
        b.zero_grad(set_to_none=True)
        window_loss(b,window).backward()
        torch.nn.utils.clip_grad_norm_(b.parameters(),1.0)
        for target,q in zip(total,b.parameters()): target.add_(q.grad/8)
    captured=[]
    def audit(model):
        captured.extend(q.grad.clone() for q in model.parameters())
        return {}
    p.accumulation_step(a,torch.optim.AdamW(a.parameters(),lr=1e-5),windows,run_for(a),audit)
    assert any(not torch.allclose(x,y) for x,y in zip(captured,total))


@pytest.mark.parametrize('count',[0,7,9])
def test_incomplete_or_oversized_group_cannot_update(count):
    a,_,windows=fixture_models()
    before=copy.deepcopy(a.state_dict())
    opt=torch.optim.AdamW(a.parameters(),lr=1e-5)
    with pytest.raises(ValueError):
        p.accumulation_step(a,opt,(windows*2)[:count],run_for(a),lambda _: {})
    assert not opt.state
    assert all(torch.equal(v,a.state_dict()[k]) for k,v in before.items())


def test_nonfinite_no_step():
    a,_,windows=fixture_models()
    windows[0]=(windows[0][0]*float('nan'),windows[0][1])
    opt=torch.optim.AdamW(a.parameters(),lr=1e-5)
    with pytest.raises(FloatingPointError): p.accumulation_step(a,opt,windows,run_for(a),lambda _: {})
    assert not opt.state


def test_full_fixed_budget_and_panel_boundaries():
    assert p.WINDOW_COUNT == 1368 and p.UPDATE_COUNT == 171 and p.GROUP == 8
    assert p.WINDOW_COUNT == p.UPDATE_COUNT*p.GROUP
    assert p.PANEL_COUNTS == (0,456,912,1368)
    assert tuple(x//8 for x in p.PANEL_COUNTS) == (0,57,114,171)
    model=torch.nn.Linear(1,1)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-4)
    seen=[]
    def run(i):
        seen.append(i)
        model(torch.ones(1,1)).square().mean().backward()
        return dict(h1_balanced=0.,ar_balanced=0.,total=0.)
    for start in range(0,1368,8): p.accumulation_step(model,opt,range(start,start+8),run,lambda _: {})
    assert seen == list(range(1368))
    assert {int(s['step']) for s in opt.state.values()} == {171}


def test_manifest_is_explicit_p015_same_dual_structure(tmp_path):
    o=SimpleNamespace(CONFIG_SHA='c',NORMALIZATION_SHA='n',PARENT_MODEL_SHA='m',PARENT_STATE_SHA='s',
                      PARENT_KIND='parent',FORCE_CHANNELS=('a','b','c','d'),sha256=lambda _: 'testhash')
    manifest=p.manifest_payload(o,tmp_path,tmp_path/'FNO.0.1.mdlus',tmp_path/'checkpoint.0.1.pt',{})
    assert manifest['status'] == 'FC_P015_DUAL_FNO_MANIFEST_VERIFIED'
    assert manifest['kind'] == 'FC_P015_WINDOW_ACCUMULATION_FORCE_FNO'
    assert manifest['aerodynamic']['metadata_kind'] == p.AERO_KIND
    for key,value in p.experiment_fields().items():
        assert manifest[key] == value and manifest['training_semantics'][key] == value
    assert manifest['training_semantics']['batch_size'] == 1
    assert manifest['training_semantics']['chunk_size'] == 10
    assert manifest['flow']['checkpoint_epoch'] == 0
    assert manifest['aerodynamic']['checkpoint_epoch'] == 1


def test_pinned_dependencies():
    root=Path(__file__).resolve().parents[1]
    diagnostic,objective=p.load_dependencies(root/'artifacts/fcp013_training_source_1634c05_immutable',
                                            root/'scripts/diagnose_fcp014_train_objective.py')
    assert objective.TOTAL_STEPS == 1368
    assert diagnostic.sha256(root/'scripts/diagnose_fcp014_train_objective.py') == p.P014_SHA


def test_reject_changed_dependency(tmp_path):
    path=tmp_path/'changed.py'
    path.write_text('raise AssertionError("must not run")')
    with pytest.raises(ValueError,match='differs'): p.load_dependencies(tmp_path,path)


def test_diagnostic_preserves_gradients_modes_rng_and_tensor_state():
    import random
    import numpy as np
    root=Path(__file__).resolve().parents[1]
    diagnostic,objective=p.load_dependencies(root/'artifacts/fcp013_training_source_1634c05_immutable',
                                            root/'scripts/diagnose_fcp014_train_objective.py')
    flow=torch.nn.Linear(1,4).eval()
    for q in flow.parameters(): q.requires_grad_(False)
    aero=torch.nn.Linear(1,4).train()
    for q in aero.parameters(): q.grad=torch.ones_like(q)
    grads=[q.grad.clone() for q in aero.parameters()]
    sample={'state':torch.zeros(3,1,1),'mask':torch.ones(1,1,1),'omega':torch.zeros(101),
            'target_state':torch.zeros(100,3,1,1),'target_force':torch.zeros(100,4)}
    items=[dict(global_index=i,family=f,identity=dict(case=c,start=s,split='train',rollout_steps=100),
                sample=sample) for i,f,c,s in diagnostic.WINDOWS]
    def predict(model,inputs,mask):
        # Deliberate RNG use verifies diagnostics restore all three states.
        random.random(); np.random.random(); torch.rand(1)
        return inputs[:,:3]*0,model(inputs[:,0,0,0,None])
    d=SimpleNamespace(validate_windows=diagnostic.validate_windows,check_memory=lambda: {},
                      capture_force_panel=diagnostic.capture_force_panel)
    rng=random.getstate(); nrng=np.random.get_state(); trng=torch.get_rng_state().clone()
    result=p.diagnostic_panel(d,objective,flow,aero,items,torch.ones(4),predict,torch.device('cpu'))
    assert len(result['rows']) == 6
    assert result['tensor_sha256_before'] == result['tensor_sha256_after']
    assert aero.training and not flow.training
    assert all(torch.equal(g,q.grad) for g,q in zip(grads,aero.parameters()))
    assert random.getstate() == rng and torch.equal(trng,torch.get_rng_state())
    now=np.random.get_state()
    assert now[0] == nrng[0] and now[2:] == nrng[2:]
    np.testing.assert_array_equal(now[1],nrng[1])
