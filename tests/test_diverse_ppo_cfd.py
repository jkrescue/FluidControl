import ast
import copy
import importlib.util
from pathlib import Path
import pytest

HERE = Path(__file__).parent
if not (HERE/'run_exploratory_diverse_ppo_real_cfd.py').exists():
    HERE = HERE.parent/'scripts'
spec = importlib.util.spec_from_file_location('diverse_cfd', HERE/'run_exploratory_diverse_ppo_real_cfd.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture():
    return dict(status='EXPLORATORY_DIVERSE_H5_PPO_TRAINING_COMPLETE_NOT_ADMISSION',timesteps=4096,
        ppo_n_updates=32,optimizer_steps=[{'optimizer_step':i} for i in range(1,65)],fno_tensors_unchanged=True,
        artifacts={'ppo_final.zip':'a'*64,'vecnormalize.pkl':'b'*64},scientific_admission=False,
        reset_counts_by_phase={p:[35,34,34,34,34,34] for p in ('00','02','04','06')},
        protocol=dict(observation_dimension=69,action_limit=.75,action_delta_limit=.1,control_dt=.1,
                      norm_obs=False,norm_reward=False,reset_count=24,episode_steps=5,
                      reset_selection='deterministic_phase_cycle_no_reward_selection'))


def test_completed_diverse_training():
    m.validate_training(fixture(),'a'*64,'b'*64)


@pytest.mark.parametrize('key,value', [('status','EXPLORATORY_H5_PPO_TRAINING_COMPLETE_NOT_ADMISSION'),
    ('timesteps',4095),('fno_tensors_unchanged',False),('reset_counts_by_phase',{})])
def test_incomplete_or_old_training_rejected(key,value):
    r=fixture(); r[key]=value
    with pytest.raises(ValueError): m.validate_training(r,'a'*64,'b'*64)


def binding():
    return {'training_execution':{'unit':m.TRAIN_UNIT,'invocation_id':'c'*32},
        'inputs':{k:{'path':m.PAYLOAD+'/'+f,'sha256':'a'*64} for k,f in
                  [('training_result','result.json'),('policy','ppo_final.zip'),('vecnormalize','vecnormalize.pkl')]}}


def test_actual_bindings_only():
    assert m.training_bindings(binding()) == ('a'*64,)*3
    for key in ('invocation_id','unit'):
        b=binding();b['training_execution'][key]='PENDING'
        with pytest.raises(ValueError):m.training_bindings(b)
    for key in ('path','sha256'):
        b=binding();b['inputs']['policy'][key]=None
        with pytest.raises(ValueError):m.training_bindings(b)


def test_original_numerical_helpers_and_loop_unchanged():
    old=Path('/workspace/fluid_control/scripts/run_exploratory_ppo_real_cfd.py')
    if not old.exists():pytest.skip('canonical source unavailable')
    before=ast.parse(old.read_text());after=ast.parse((HERE/'run_exploratory_diverse_ppo_real_cfd.py').read_text())
    for name in ('observation','validate_vec','predict','fixed_window','summarize'):
        a=next(n for n in before.body if isinstance(n,ast.FunctionDef) and n.name==name)
        b=next(n for n in after.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(a)==ast.dump(b)
    def loop(tree):
        return next(n for n in ast.walk(tree) if isinstance(n,ast.With) and
                    any(isinstance(i.context_expr,ast.Call) and isinstance(i.context_expr.func,ast.Attribute)
                        and i.context_expr.func.attr=='PairSolvers' for i in n.items))
    assert ast.dump(loop(before))==ast.dump(loop(after))
