import ast
import json
from pathlib import Path
from types import SimpleNamespace

SOURCE=Path(__file__).resolve().parents[1]/'scripts/serve_live_research_dashboard.py'
DIGEST='d6303082b0f97087961a9171505a2b4289623eb5f634f2502ac4f7d65639143b'

def fixture(tmp_path,n=12,terminal=False):
    approval=tmp_path/'docs/P064_ABSOLUTE64_SYMMETRY_CANONICAL_B01_CFD_APPROVAL_20261007.json';approval.parent.mkdir();approval.write_text('{}')
    path=tmp_path/'artifacts/p064_absolute64_symmetry_canonical_b01_cfd_20261007/progress.json';path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'completed_cycles':n,'rows':[{'step':i,'end_time':130+.1*i} for i in range(1,n+1)]}))
    state=f'InvocationID=6fa0baeb90034371b3b49f3d8d51192a\nMainPID={0 if terminal else 123}\nActiveState=active\nSubState={"exited" if terminal else "running"}\nResult=success\nExecMainStatus=0\nExecStart={approval} {DIGEST}\n'
    ns=dict(json=json,hashlib=SimpleNamespace(sha256=lambda b:SimpleNamespace(hexdigest=lambda:DIGEST)),subprocess=SimpleNamespace(check_output=lambda *a,**k:state,SubprocessError=RuntimeError))
    node=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_absolute64_cfd')
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),ns)
    return ns['_absolute64_cfd'],approval

def test_live(tmp_path):
    f,_=fixture(tmp_path);r=f(tmp_path);assert r['running'] and r['cycles']==12

def test_terminal(tmp_path):
    f,_=fixture(tmp_path,800,True);r=f(tmp_path);assert not r['running'] and '待独审' in r['status']

def test_missing(tmp_path):
    f,p=fixture(tmp_path);p.unlink();assert not f(tmp_path)['verified']

def test_budget(tmp_path):
    f,_=fixture(tmp_path,801);assert not f(tmp_path)['verified']

def test_append_not_replace():
    s=SOURCE.read_text();line=next(x for x in s.splitlines() if '并行 Absolute64 b01真实CFD' in x)
    assert 'textContent+=' in line and 'lead-now' not in line
