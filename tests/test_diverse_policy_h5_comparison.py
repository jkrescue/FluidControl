import importlib.util
from pathlib import Path
import pytest
HERE=Path(__file__).parent
if not (HERE/'evaluate_diverse_policy_h5_comparison.py').exists():HERE=HERE.parent/'scripts'
sp=importlib.util.spec_from_file_location('eval_test',HERE/'evaluate_diverse_policy_h5_comparison.py')
m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)

class Env:
    def __init__(self):self.slot=-1
    def reset(self):self.slot+=1;self.stepno=0;return [self.slot],{}
    def step(self,a):
        self.stepno+=1
        return [self.slot],float(a),False,self.stepno==5,dict(reset_panel_index=self.slot,reset_case=str(self.slot),
            reset_frame=0 if self.slot==0 else 62,reward_drag=-float(a),predicted_force=[1,2,3,4],applied_omega=a)
class Policy:
    def predict(self,obs,deterministic):assert deterministic is True;return obs[0]+1,None

def test_fixed_panel_same_starts_equal_aggregation():
    guards=[];rows=m.evaluate_panel([(p,Env()) for p in ('00','02','04','06')],Policy(),lambda:guards.append(1))
    assert len(guards)==120 and len(rows)==24
    assert [r['return'] for r in rows]==[5.,10.,15.,20.,25.,30.]*4
    assert m.aggregate(rows)=={'mean_case_return':17.5,'mean_case_component_sums':{'reward_drag':-17.5}}
    assert len(rows[0]['steps'])==5

def test_early_termination_rejected():
    class Bad(Env):
        def step(self,a):
            o,r,t,tr,i=super().step(a);return o,r,True,tr,i
    with pytest.raises(ValueError):m.evaluate_panel([('00',Bad())],Policy(),lambda:None)

def test_incomplete_panel_rejected():
    with pytest.raises(ValueError):m.aggregate([])

def test_no_optimizer_or_training_calls():
    import ast
    tree=ast.parse((HERE/'evaluate_diverse_policy_h5_comparison.py').read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    assert not any(n.func.attr in ('learn','backward','save') for n in calls)
    source=(HERE/'evaluate_diverse_policy_h5_comparison.py').read_text()
    assert "device='cuda:0'" in source and "deterministic=True" in source
    assert source.index("torch.set_float32_matmul_precision('highest')")>source.index('network,identity=load_dual_fno')

def test_supervisor_only_scope_and_deadline_delta():
    p=Path('/workspace/fluid_control/scripts/supervise_exploratory_diverse_h5_32768_ppo.py')
    old=p.read_text()
    for a,b in [('EXPLORATORY_DIVERSE_H5_32768_PPO_EXECUTION_APPROVED','DIVERSE_POLICY_H5_COMPARISON_EXECUTION_APPROVED'),
        ('EXPLORATORY_DIVERSE_H5_32768_SUPERVISED','DIVERSE_POLICY_H5_EVAL_SUPERVISED'),
        ('1800','240'),('EXPLORATORY_DIVERSE_H5_32768_PPO_TRAINING_COMPLETE_NOT_ADMISSION','DIVERSE_POLICY_H5_COMPARISON_COMPLETE_NOT_ADMISSION'),
        ('result["timesteps"] == 32768','result["optimizer_steps"] == 0')]:old=old.replace(a,b)
    assert old==(HERE/'supervise_diverse_policy_h5_comparison.py').read_text()
