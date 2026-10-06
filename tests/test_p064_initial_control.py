import ast
import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).parent
SCRIPTS=ROOT if (ROOT/'run_p064_initial_projected_ppo_long_cfd.py').exists() else ROOT.parent/'scripts'
FILE=SCRIPTS/'run_p064_initial_projected_ppo_long_cfd.py'
BASE=ROOT/'base.py' if (ROOT/'base.py').exists() else SCRIPTS/'run_p064_candidate_projected_32768_ppo_long_cfd.py'
RECONSTRUCTION=ROOT/'reconstruct_initial_policy_r3.py' if (ROOT/'reconstruct_initial_policy_r3.py').exists() else SCRIPTS/'reconstruct_p064_initial_policy_cpu.py'
s=importlib.util.spec_from_file_location('initial',FILE);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def function(source,name):
    return next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)

@pytest.mark.parametrize('name',['observation','predict','reflect_physical69','projected_request','fixed_window','summarize','values_summary'])
def test_numerical_helpers_unchanged(name):
    assert ast.dump(function(FILE.read_text(),name))==ast.dump(function(BASE.read_text(),name))

def test_entire_eight_hundred_cycle_loop_unchanged():
    def loop(p):
        tree=function(p.read_text(),'execute')
        return next(n for n in ast.walk(tree) if isinstance(n,ast.For) and ast.unparse(n.iter)=='range(1, 801)')
    assert ast.dump(loop(FILE))==ast.dump(loop(BASE))

def test_initial_weights_are_explicit_not_trained_policy_load():
    source=FILE.read_text()
    assert "PPO.load(inputs['initial_policy'], device='cpu')" in source
    assert "PPO.load(inputs['policy']" not in source
    assert 'not model.policy.optimizer.state' in source
    assert 'pickle.dumps(vec.__dict__) == vec_state_before' in source
    assert source.index('initial_record = validate_initial') < source.index('model = PPO.load') < source.index('output.mkdir()')

def test_reconstruction_no_training_and_exact_hash():
    source=RECONSTRUCTION.read_text()
    tree=ast.parse(source)
    forbidden={'learn','backward','train'}
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in forbidden for n in ast.walk(tree))
    assert 'assert digest(model.policy)==INITIAL' in source
    assert "no environment step authorized" in source and "no environment reset authorized" in source
    assert "pickle.dumps(vec.__dict__)" in source
    assert 'observation_space=vec.observation_space' in source
    assert 'reloaded.observation_space == vec.observation_space' in source

def test_reconstruction_proof_rejects_nonzero_updates(tmp_path):
    r={'status':'P064_EXACT_INITIAL_POLICY_RECONSTRUCTED_CPU_NOT_CFD','initial_tensor_sha256':m.INITIAL_TENSOR_SHA,
       'seed':20261006,'optimizer_steps':1,'timesteps':0,'ppo_n_updates':0,'environment_steps':0,'fno_loaded':False,'gpu_used':False}
    p=tmp_path/'r.json';p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='no-training'):
        m.validate_initial({'initial_reconstruction':p},{'policy_tensor_sha256_before':m.INITIAL_TENSOR_SHA},'a','b')
