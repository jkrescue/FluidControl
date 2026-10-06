import ast
import importlib.util
from pathlib import Path
import numpy as np
import pytest
HERE=Path(__file__).parent
if not (HERE/'run_exploratory_diverse_32768_ppo_long_cfd.py').exists():HERE=HERE.parent/'scripts'
spec=importlib.util.spec_from_file_location('long_cfd',HERE/'run_exploratory_diverse_32768_ppo_long_cfd.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_exact_windows_and_boundary():
    t=np.round(148+.005*np.arange(1,16001),8);a=np.column_stack([t,t*0+1,np.sin(t)])
    assert len(m.fixed_window(a,148,228))==16000
    assert len(m.fixed_window(a,168,228))==12000
    assert len(m.fixed_window(a,168,228,True))==12001
    assert len(m.fixed_window(a,148,160.4))==2480
    assert len(m.fixed_window(a,148,154.2))==1240
    assert len(m.fixed_window(a,154.2,160.4))==1240
    assert m.fixed_window(a,168,228,True)[0,0]==168
    with pytest.raises(ValueError):m.fixed_window(a[:-1],168,228)
    duplicate=a.copy();duplicate[-2]=duplicate[-1]
    with pytest.raises(ValueError):m.fixed_window(duplicate,168,228)
    with pytest.raises(ValueError):m.fixed_window(a[a[:,0]!=168],168,228,True)

def test_all_predeclared_statistics(tmp_path):
    t=np.round(148+.005*np.arange(1,16001),8)
    cases={r:tmp_path/r for r in ('ppo','zero')}
    for role,case in cases.items():
        for body in ('forceFront','forceRear'):
            p=case/'postProcessing'/body/'148';p.mkdir(parents=True)
            np.savetxt(p/'coefficient.dat',np.column_stack([t,np.ones(len(t)),t*0,t*0,np.sin(t)]))
    def metric(front,rear):
        return dict(total_cd_mean=float(np.mean(front[:,1]+rear[:,1])),rear_cl_mean=float(np.mean(rear[:,2])),
                    rear_cl_fluctuation_rms=float(np.std(rear[:,2])),samples=len(rear))
    windows,hashes=m.summarize(cases,metric)
    assert [x['branches']['ppo']['samples'] for x in windows.values()]==[2480,1240,1240,12000,12001,16000]
    assert len(hashes)==4
    assert all(x['paired_drag_reduction']==0 and x['paired_rear_cl_fluctuation_rms_ratio']==1 for x in windows.values())
    assert windows['historical_inclusive_final_60']['left_endpoint_included'] is True
    assert all(x['original_mean_bias_reference']==.1 and x['mean_bias_sensitivity_reference']==.2 for x in windows.values())

def test_solver_loop_only_budget_delta():
    old=Path('/workspace/fluid_control/scripts/run_exploratory_diverse_ppo_real_cfd.py')
    if not old.exists():pytest.skip('reviewed short source unavailable')
    before=ast.parse(old.read_text());after=ast.parse((HERE/'run_exploratory_diverse_32768_ppo_long_cfd.py').read_text())
    def ownedloop(tree):
        return next(n for n in ast.walk(tree) if isinstance(n,ast.With) and any(
            isinstance(i.context_expr,ast.Call) and isinstance(i.context_expr.func,ast.Attribute)
            and i.context_expr.func.attr=='PairSolvers' for i in n.items))
    a=ownedloop(before);b=ownedloop(after)
    for n in ast.walk(a):
        if isinstance(n,ast.Constant) and n.value==125:n.value=801
    assert ast.dump(a)==ast.dump(b)
    for name in ('observation','predict','validate_vec','training_bindings'):
        f=lambda tree:next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(f(before))==ast.dump(f(after))

def test_long_time_only_budget():
    source=(HERE/'run_exploratory_diverse_32768_ppo_long_cfd.py').read_text()
    assert "spec['deadline_seconds'] == 3600" in source
    assert 'time.monotonic()-started < 3600' in source
    assert '8*2**30' in source and "memory.swap.max" in source
