import ast
import importlib.util
from pathlib import Path
import numpy as np
import pytest

HERE = Path(__file__).parent
SCRIPTS = HERE if (HERE/'run_p064_candidate_projected_32768_ppo_b01_long_cfd.py').exists() else HERE.parent/'scripts'
BASE = HERE/'base.py' if (HERE/'base.py').exists() else SCRIPTS/'run_p064_candidate_projected_32768_ppo_long_cfd.py'
OLD = HERE/'old_b01.py' if (HERE/'old_b01.py').exists() else SCRIPTS/'run_exploratory_projected_32768_ppo_b01_long_cfd.py'
spec = importlib.util.spec_from_file_location('b01', SCRIPTS/'run_p064_candidate_projected_32768_ppo_b01_long_cfd.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def functions(path):
    return {n.name: ast.dump(n) for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)}

def test_unchanged_numerical_functions():
    old, new = functions(BASE), functions(Path(m.__file__))
    for name in old.keys() - {'execute', 'summarize', 'validate_spec'}:
        assert old[name] == new[name], name

def test_historical_phase_helpers():
    old, new = functions(OLD), functions(Path(m.__file__))
    for name in ('declared_windows', 'build_pair_at'):
        assert old[name] == new[name]

def test_six_windows():
    data = np.zeros((16000, 3))
    data[:, 0] = 130 + np.arange(1,16001)*.005
    assert [len(m.fixed_window(data,a,b,c)) for _,a,b,c in m.declared_windows()] == [2480,1240,1240,12000,12001,16000]

@pytest.mark.parametrize('change', ['missing','duplicate'])
def test_grid_rejection(change):
    data = np.zeros((16000,3)); data[:,0] = 130+np.arange(1,16001)*.005
    if change == 'missing': data = data[:-1]
    else: data[-1,0] = data[-2,0]
    with pytest.raises(ValueError): m.fixed_window(data,130,210)

def test_actual_loop_only_time_shift():
    def loop(path):
        tree=ast.parse(path.read_text())
        return next(n for n in ast.walk(tree) if isinstance(n, ast.For) and isinstance(n.target,ast.Name) and n.target.id=='step')
    old = loop(BASE); new = loop(Path(m.__file__))
    class Clock(ast.NodeTransformer):
        def visit_Name(self,n):
            return ast.copy_location(ast.Constant(148),n) if n.id=='START' else n
    assert ast.dump(old) == ast.dump(Clock().visit(new))
