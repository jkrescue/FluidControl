import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('dashboard', Path(__file__).resolve().parents[1]/'scripts/serve_live_research_dashboard.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

BASE = dict(InvocationID='c41e61fcaa5f49e3be9e0092c1e19bc3',
            ExecStart='/artifacts/fcp015_formal_supervisor_bf3668d852f8_immutable/supervise_fcp015_formal.py',
            MainPID='3962454',ActiveState='active',SubState='running')

@pytest.mark.parametrize('change,verified,running', [
    ({},True,True),
    ({'MainPID':'0','SubState':'exited'},True,False),
    ({'ActiveState':'failed','SubState':'failed','MainPID':'0'},True,False),
    ({'InvocationID':'old'},False,False),
    ({'ExecStart':'unrelated'},False,False),
    ({'MainPID':'unknown'},False,False),
    ({'MainPID':'-1'},True,False),
])
def test_formal_status(change,verified,running):
    fields={**BASE,**change}
    result=module._parse_fcp015_posteval('\n'.join(k+'='+v for k,v in fields.items()))
    assert result['verified'] is verified
    assert result.get('running',False) is running
    assert result.get('admission',False) is False

def test_missing_binding():
    assert module._parse_fcp015_posteval('') == {'verified':False}

def test_missing_receipts_fail_closed(tmp_path):
    assert module._fcp015_posteval(tmp_path) == {'verified':False}
