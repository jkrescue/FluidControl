import ast
import importlib.util
from pathlib import Path
import pytest

HERE=Path(__file__).parent
if not (HERE/'train_exploratory_diverse_h5_32768_ppo.py').exists():HERE=HERE.parent/'scripts'
REPO=Path('/workspace/fluid_control')
def load(path):
    spec=importlib.util.spec_from_file_location('budget_test_'+path.stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
m=load(HERE/'train_exploratory_diverse_h5_32768_ppo.py')
old=load(REPO/'scripts/train_exploratory_diverse_h5_ppo.py')

def test_only_protocol_budget_changes():
    p=dict(m.PROTOCOL);assert p.pop('timesteps')==32768
    q=dict(old.PROTOCOL);assert q.pop('timesteps')==4096
    assert p==q
    assert m.fixed_panel()==old.fixed_panel()

def test_exact_trainer_delta():
    expected=(REPO/'scripts/train_exploratory_diverse_h5_ppo.py').read_text()
    for a,b in [
      ('Separate diverse-real-reset PPO run; original H5 algorithm/guards unchanged.','Separate fresh-init 32768 budget; diverse H5 algorithm/data/guards unchanged.'),
      ('EXPLORATORY_DIVERSE_H5_PPO','EXPLORATORY_DIVERSE_H5_32768_PPO'),
      ('EXPLORATORY_DIVERSE_H5_SUPERVISED','EXPLORATORY_DIVERSE_H5_32768_SUPERVISED'),
      ('4096','32768'),('policy._n_updates == 32 and len(optimizer_steps) == 64','policy._n_updates == 256 and len(optimizer_steps) == 512')]:
        expected=expected.replace(a,b)
    assert expected==(HERE/'train_exploratory_diverse_h5_32768_ppo.py').read_text()

def test_supervisor_lifecycle_unchanged():
    expected=(REPO/'scripts/supervise_exploratory_diverse_h5_ppo.py').read_text()
    for a,b in [('EXPLORATORY_DIVERSE_H5_PPO','EXPLORATORY_DIVERSE_H5_32768_PPO'),
                ('EXPLORATORY_DIVERSE_H5_SUPERVISED','EXPLORATORY_DIVERSE_H5_32768_SUPERVISED'),('4096','32768')]:
        expected=expected.replace(a,b)
    assert expected==(HERE/'supervise_exploratory_diverse_h5_32768_ppo.py').read_text()

def test_expected_accounting():
    assert 32768//(4*128)==64
    assert 64*4==256 and 64*4*(512//256)==512
    steps=32768//4; assert divmod(steps,5)==(1638,2)
    counts=[0]*6
    for i in range(1+steps//5):counts[i%6]+=1
    assert counts==[274,273,273,273,273,273]

def test_fresh_policy_no_load_or_resume():
    tree=ast.parse((HERE/'train_exploratory_diverse_h5_32768_ppo.py').read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
    assert sum(isinstance(n.func,ast.Name) and n.func.id=='PPO' for n in calls)==1
    assert not any(isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name)
                   and n.func.value.id=='PPO' and n.func.attr=='load' for n in calls)
    assert m.PROTOCOL['seed']==20261006 and m.PROTOCOL['deadline_seconds']==1800

def test_old_approval_rejected():
    with pytest.raises(ValueError):m.validate_spec({'status':old.STATUS,'execution_authorized':True})

def test_final_serialized_approval_when_present():
    import json
    path=REPO/'docs/EXPLORATORY_DIVERSE_H5_32768_PPO_R2_APPROVAL_20261006.json'
    if not path.exists():pytest.skip('actual approved R2 not available')
    s=json.loads(path.read_text());m.validate_spec(s)
    assert all(type(s['protocol'][k]) is type(v) for k,v in m.PROTOCOL.items())
    s['protocol']['ent_coef']=0
    with pytest.raises(ValueError):m.validate_spec(s)
