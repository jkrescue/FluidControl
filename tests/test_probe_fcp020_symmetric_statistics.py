import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import probe_fcp020_symmetric_statistics as p


@pytest.fixture
def pinned():
    root=Path('/workspace/fluid_control')
    hp=root/'scripts/probe_fcp019_gradient_alignment.py'
    op=root/'artifacts/fcp013_training_source_1634c05_immutable/scripts/train_fcp013_independent_force_fno.py'
    if not hp.exists() or not op.exists():pytest.skip('pinned Main artifacts required')
    assert p.sha(hp)==p.P019_SHA
    assert p.sha(op)=='f3cf4b9a745cc0cbee39db9385cbfc398e4834e25bc487d8fb2d6eca07b483d7'
    def load(name,path):
        spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
    return load('test_p020_helper',hp),load('test_p020_objective',op)


@pytest.mark.parametrize('statistics',[False,True])
@pytest.mark.parametrize('dtype',[torch.float64,torch.float32])
def test_streamed_combined_gradient_equals_actual_objective_fullgraph(pinned,statistics,dtype):
    helper,objective=pinned;torch.manual_seed(42)
    model=torch.nn.Linear(6,4,dtype=dtype)
    before={n:t.detach().clone() for n,t in model.named_parameters()}
    states,h1=torch.randn(1,100,3,1,1,dtype=dtype),torch.randn(1,100,3,1,1,dtype=dtype)
    batch=dict(mask=torch.ones(1,1,1,1,dtype=dtype),omega=torch.randn(1,101,1,dtype=dtype),target_force=torch.randn(1,100,4,dtype=dtype))
    predict=lambda model,x,mask:(None,model(x[:,:,0,0]))
    observed=p.window_backward(helper,objective,model,states,h1,batch,predict,statistics=statistics)
    gradients={n:t.grad.clone() for n,t in model.named_parameters()};model.zero_grad(set_to_none=True)
    outputs=[]
    for x in (h1,states):
        inp=objective.make_inputs(x[0],batch['mask'].expand(100,-1,-1,-1),batch['omega'][0,:100],batch['omega'][0,1:])
        outputs.append(predict(model,inp,None)[1])
    original=sum(.5*objective.balanced_force_objective(force[None],batch['target_force'])['balanced'] for force in outputs)
    total=original
    if statistics:
        target=batch['target_force'][0,38:,3]
        for force in outputs:
            tail=force[38:,3]
            total=total+(5/16)*((tail-target).mean().square()+(tail.std(correction=0)-target.std(correction=0)).square())
    total.backward()
    for name,param in model.named_parameters():
        torch.testing.assert_close(gradients[name],param.grad,rtol=2e-5 if dtype==torch.float32 else 1e-11,atol=2e-7 if dtype==torch.float32 else 1e-12)
        torch.testing.assert_close(before[name],param,rtol=0,atol=0)
    assert observed['training_total']==pytest.approx(float(total.detach()),rel=2e-6)
    assert len(observed['normalized_statistic_errors'])==(4 if statistics else 0)


@pytest.mark.parametrize('targets',[[-10.,10.,-10.,10.,1.,1.],[4.]*6])
def test_six_raw_average_then_one_clip_matches_manual_adam(targets):
    model=torch.nn.Linear(1,1,bias=False,dtype=torch.float64);model.weight.data.fill_(0)
    ref=torch.nn.Parameter(model.weight.detach().clone())
    opt=torch.optim.AdamW(model.parameters(),lr=p.PROTOCOL['learning_rate'],weight_decay=1e-4)
    control=torch.optim.AdamW([ref],lr=p.PROTOCOL['learning_rate'],weight_decay=1e-4)
    def run(target):
        loss=(model.weight-target).square().sum();loss.backward();return {'training_total':float(loss.detach())}
    record=p.update(model,opt,targets,run)
    sum((ref-target).square().sum() for target in targets).div(6).backward()
    norm=torch.nn.utils.clip_grad_norm_([ref],1.);control.step()
    assert record['preclip_mean_gradient_norm']==pytest.approx(float(norm))
    torch.testing.assert_close(model.weight,ref,rtol=1e-14,atol=1e-22)
    assert float(next(iter(opt.state.values()))['step'])==1


def test_update_rejects_nonfinite_gradient():
    model=torch.nn.Linear(1,1);opt=torch.optim.AdamW(model.parameters())
    def bad(_):
        for param in model.parameters():param.grad=torch.full_like(param,float('nan'))
        return {'training_total':0.}
    with pytest.raises(FloatingPointError):p.update(model,opt,[None]*6,bad)
    assert not opt.state


def test_zero_rms_statistic_stops_without_update(pinned):
    helper,objective=pinned
    model=torch.nn.Linear(6,4,dtype=torch.float64)
    with torch.no_grad():model.weight.zero_();model.bias.zero_()
    states=torch.zeros(1,100,3,1,1,dtype=torch.float64)
    batch=dict(mask=torch.ones(1,1,1,1,dtype=torch.float64),omega=torch.zeros(1,101,1,dtype=torch.float64),target_force=torch.ones(1,100,4,dtype=torch.float64))
    with pytest.raises(ValueError,match='undefined'):
        p.window_backward(helper,objective,model,states,states,batch,lambda m,x,k:(None,m(x[:,:,0,0])),statistics=True)
    assert all(param.grad is None for param in model.parameters())


def test_same_initial_restore_and_fresh_optimizer():
    model=torch.nn.Linear(1,1);initial={n:t.detach().clone() for n,t in model.state_dict().items()}
    terminal=[]
    for _ in range(2):
        model.load_state_dict(initial);model.zero_grad(set_to_none=True)
        opt=torch.optim.AdamW(model.parameters(),lr=p.PROTOCOL['learning_rate'],weight_decay=1e-4)
        assert not opt.state
        loss=model(torch.ones(1,1)).square().mean();loss.backward();opt.step()
        terminal.append({n:t.detach().clone() for n,t in model.state_dict().items()})
    for n in terminal[0]:torch.testing.assert_close(terminal[0][n],terminal[1][n],rtol=0,atol=0)
    model.load_state_dict(initial)
    for n,t in model.state_dict().items():torch.testing.assert_close(t,initial[n],rtol=0,atol=0)


def test_fixed_criteria_reject_h1_tradeoff_and_equal_control():
    def summary(value):return {d:dict(six_window_original_objective=value,five_nonzero=dict(bias_mse=value,rms_error_mse=value,centered_residual_mse=value)) for d in ('h1','ar')}
    initial,control,changed=summary(3.),summary(2.),summary(1.)
    assert p.compare(initial,control,changed)['strict_local_conditions']
    assert not p.compare(initial,changed,changed)['strict_local_conditions']
    changed['h1']['five_nonzero']['rms_error_mse']=4.
    assert not p.compare(initial,control,changed)['strict_local_conditions']
    assert p.compare(initial,control,changed)['scientific_admission'] is False


def test_statistical_metric_is_not_centered_waveform_error(pinned):
    helper,_=pinned
    t=torch.zeros(100,4,dtype=torch.float64);t[:,3]=torch.linspace(-1,1,100)
    pred=t.clone();pred[:,3]=-t[:,3]
    result=p.metrics(helper,{'h1':pred,'ar':pred},t,torch.ones(4),{})
    assert result['domains']['h1']['rms_error_mse']==0
    assert result['domains']['h1']['centered_residual_mse']>0


def test_protocol_is_fixed_and_no_checkpoint_save():
    import ast
    assert p.PROTOCOL['updates_per_arm']==16 and p.PROTOCOL['windows_per_update']==6
    assert p.PROTOCOL['statistic_coefficient']==5/16 and p.PROTOCOL['endpoint_diagnostics']==[0,16]
    tree=ast.parse(Path(p.__file__).read_text())
    calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
    assert not any(name.endswith(('save_checkpoint','torch.save')) for name in calls)


def endpoint(value,repeat_value=None):
    def base(x):
        return dict(rows=[dict(global_index=160,panel={'value':x})],aggregate={d:dict(six_window_original_objective=x,
            five_nonzero=dict(bias_mse=x,rms_error_mse=x,centered_residual_mse=x)) for d in ('h1','ar')})
    return {**base(value),'repeat':base(value if repeat_value is None else repeat_value)}


def test_repeated_panel_records_exact_deltas():
    values=iter([endpoint(2.),endpoint(2.01)])
    result=p.repeated_panel(lambda:next(values))
    assert result['repeat_aggregate_signed_delta']['h1']['six_window_original_objective']==pytest.approx(.01)
    assert result['repeat_per_window_signed_delta'][0]['panel']['value']==pytest.approx(.01)
    assert 'not a rigorous' in result['repeat_uncertainty']


def test_improvement_below_repeat_spread_is_inconclusive():
    result=p.assess_repeat_resolution(endpoint(3.),endpoint(1.00001,1.01),endpoint(3.),endpoint(1.,1.01))
    assert result['initials_agree_at_observed_repeat_resolution']
    assert not result['numerically_resolved']
    assert not result['domains']['ar']['comparisons']['rms_error_mse_vs_control']['numerically_resolved_at_observed_repeats']


def test_clear_improvement_exceeds_repeat_spread():
    result=p.assess_repeat_resolution(endpoint(3.),endpoint(2.,2.000001),endpoint(3.),endpoint(1.,1.000001))
    assert result['numerically_resolved']


def test_initials_must_agree_per_window_not_only_macro():
    a,b=endpoint(3.),endpoint(3.)
    b['rows'][0]['panel']['value']=3.1;b['repeat']['rows'][0]['panel']['value']=3.1
    result=p.assess_repeat_resolution(a,endpoint(2.),b,endpoint(1.))
    assert not result['initials_agree_at_observed_repeat_resolution']
    assert not result['numerically_resolved']


def test_exact_unchanged_centered_error_preserves_nonincrease_rule():
    a,t,b=endpoint(3.),endpoint(2.),endpoint(1.)
    for panel in (b,b['repeat']):
        for domain in ('h1','ar'):panel['aggregate'][domain]['five_nonzero']['centered_residual_mse']=3.
    result=p.assess_repeat_resolution(a,t,a,b)
    assert result['numerically_resolved']
