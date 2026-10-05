from pathlib import Path
from types import SimpleNamespace
import copy
import pytest
import torch
import probe_fcp023_input_block as p

ROOT=Path('/workspace/fluid_control')
@pytest.fixture
def block():return p.load(ROOT/'scripts/p023_force_block.py',p.BLOCK_SHA,'test_p023_adapter')

@pytest.fixture
def metrics():return p.load(ROOT/'scripts/probe_fcp020_symmetric_statistics.py',p.P020_SHA,'test_p023_metrics')

def adapter(block):
    base=torch.nn.Conv2d(12,24,1,dtype=torch.float64)
    with torch.no_grad():base.weight[:,6:10].zero_()
    return block.FrozenForceBlock(base,lift_name='weight')

def optimizer(model,rate):return torch.optim.AdamW([model.block],lr=rate,weight_decay=p.PROTOCOL['weight_decay'],betas=tuple(p.PROTOCOL['betas']),eps=p.PROTOCOL['eps'])

@pytest.mark.parametrize('rate',[1.5625e-7,1e-5])
def test_mean_six_before_clip_and_only96_updates(block,rate):
    a=adapter(block);o=optimizer(a,rate);base=a.diagnostic_identity();seen=[]
    reference=torch.nn.Parameter(a.block.detach().clone());r=torch.optim.AdamW([reference],lr=rate,weight_decay=1e-4,betas=(.9,.999),eps=1e-8)
    def run(i):
        loss=a.block.sum()*i;loss.backward();return {'training_total':float(loss.detach())}
    result=p.accumulation_step(a,o,list(range(1,7)),run,lambda:seen.append(len(o.state)) or {'states':len(o.state)})
    (sum(reference.sum()*i for i in range(1,7))/6).backward();norm=torch.nn.utils.clip_grad_norm_([reference],1.);r.step()
    torch.testing.assert_close(a.block,reference,rtol=0,atol=0)
    assert result['preclip_mean_gradient_norm']==pytest.approx(float(norm)) and seen==[0,1]
    assert result['actual_adam_moment_bytes']==2*96*8
    p.check_optimizer(a,o,1,rate,block);assert a.diagnostic_identity()['base_tensor_sha256']==base['base_tensor_sha256']
    updates=block.block_update_metrics(torch.zeros_like(a.block),a.block);assert len(updates['values'])==96 and updates['update_l2']>0

def test_low_high_fresh_restore_no_old_moments(block):
    a=adapter(block);initial=a.diagnostic_identity()
    for rate in p.PROTOCOL['learning_rates'].values():
        a.restore_zero();assert a.diagnostic_identity()==initial
        o=optimizer(a,rate);block.assert_optimizer_scope(a,o)
        a.block.sum().backward();o.step();p.check_optimizer(a,o,1,rate,block)
        assert set(o.state)=={a.block}
        assert all(not x.requires_grad and x.grad is None for x in a.base.parameters())

@pytest.mark.parametrize('bad',['lr','step','moment','old_param'])
def test_optimizer_fail_closed(block,bad):
    a=adapter(block);o=optimizer(a,1e-5);a.block.sum().backward();o.step()
    if bad=='lr':o.param_groups[0]['lr']=1e-4
    if bad=='step':o.state[a.block]['step'].fill_(2)
    if bad=='moment':o.state[a.block]['exp_avg'].fill_(float('nan'))
    if bad=='old_param':o.param_groups[0]['params'].append(a.base.weight)
    with pytest.raises((ValueError,FloatingPointError)):p.check_optimizer(a,o,1,1e-5,block)

def test_nan_raw_gradient_stops_before_optimizer(block):
    a=adapter(block);o=optimizer(a,1e-5)
    def run(_):
        (a.block.sum()*float('nan')).backward();return {'training_total':float('nan')}
    with pytest.raises(FloatingPointError):p.accumulation_step(a,o,[1]*6,run,lambda:{})
    assert not o.state

def panel(value,prediction=0.):
    stats={k:value for k in ('bias_mse','rms_error_mse','centered_residual_mse')}
    x=dict(aggregate={d:dict(six_window_original_objective=value,five_nonzero=dict(stats)) for d in ('h1','ar')},rows=[dict(global_index=i,panel={'score':value},normalized_predictions={d:[[prediction]*4]*100 for d in ('h1','ar')}) for i in (160,816,923,975,1077,1233)])
    x['repeat']=copy.deepcopy(x);return x

def test_high_vs_low_rules_and_noise(metrics):
    initial,low,high=panel(3),panel(2),panel(1)
    assert p.comparison(metrics,initial,low,high)['strict_local_conditions']
    assert p.repeat_resolution(metrics,initial,low,initial,high)['numerically_resolved']
    high['aggregate']['ar']['six_window_original_objective']=2
    assert not p.comparison(metrics,initial,low,high)['strict_local_conditions']
    assert not p.repeat_resolution(metrics,initial,low,initial,high)['numerically_resolved']

def test_ablation_sensitivity_scale_and_repeat(metrics):
    conditioned,ablated=panel(1,2),panel(2,1)
    before=copy.deepcopy(conditioned)
    r=p.ablation_difference(conditioned,ablated,torch.tensor([1.,2.,3.,4.]),metrics)
    for domain in ('h1','ar'):
        assert r['rows'][0]['domains'][domain]['physical_output_difference_l2_per_force']==[10,20,30,40]
        assert r['rows'][0]['domains'][domain]['physical_output_difference_max_abs_per_force']==[1,2,3,4]
        assert r['rows'][0]['domains'][domain]['observed_combined_repeat_max_abs_per_force']==[0]*4
    assert conditioned==before
    assert r['conditioned_minus_ablated_aggregate']['h1']['six_window_original_objective']==-1

def test_optimizer_identity_detects_moment_mutation(block):
    a=adapter(block);o=optimizer(a,1e-5);a.block.sum().backward();o.step()
    objective=SimpleNamespace(tensor_sha256=lambda v:block.tensor_digest([('value',v)]))
    before=p.optimizer_identity(o,objective);o.state[a.block]['exp_avg'].add_(1)
    assert before!=p.optimizer_identity(o,objective)

@pytest.mark.parametrize('free,available,elapsed,startup',[(29.9,70,0,True),(31,49,0,True),(19.9,100,0,False),(31,100,1800.01,False)])
def test_memory_deadline(free,available,elapsed,startup):
    with pytest.raises(RuntimeError):p.limits({'MemFree':free,'MemAvailable':available},elapsed,startup)

def test_counts_scope_and_no_save():
    import ast
    assert p.PROTOCOL['learning_rates']=={'LOW':1.5625e-7,'HIGH':1e-5}
    assert 2*p.PROTOCOL['updates_per_arm']*p.PROTOCOL['windows_per_update']==192
    assert p.PROTOCOL['trainable_scalars']==96
    text=Path(p.__file__).read_text();tree=ast.parse(text)
    assert not any(isinstance(x,ast.Attribute) and x.attr in ('save','save_checkpoint') for x in ast.walk(tree))
    assert text.index('expanded.to(dist.device)')<text.index('block_module.FrozenForceBlock(expanded)')
    assert 'model.to(' not in text and 'model.load_state_dict(' not in text
