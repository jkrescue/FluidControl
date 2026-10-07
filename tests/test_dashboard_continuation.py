import ast
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import pytest

SOURCE=Path(__file__).resolve().parents[1]/'scripts/serve_live_research_dashboard.py'

def fixture(tmp_path,live=False):
    digest='6e08555e336e271fe0baf6790feaf3d92c7c6e0771fc9aa93d7b0fd608f3f812'
    approval=tmp_path/'docs/P064_B_E109_CONTINUATION_328_408_APPROVAL_20261007.json';approval.parent.mkdir();approval.write_text('{}')
    state=f'InvocationID=73a2568857074c70b494639269a602bf\nMainPID={123 if live else 0}\nActiveState=active\nSubState={"running" if live else "exited"}\nResult=success\nExecMainStatus=0\nExecStart={approval} {digest}\n'
    ns=dict(json=json,hashlib=SimpleNamespace(sha256=lambda b:SimpleNamespace(hexdigest=lambda:digest)),subprocess=SimpleNamespace(check_output=lambda *a,**k:state,SubprocessError=subprocess.SubprocessError))
    node=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_continuation_cfd')
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),ns)
    return ns['_continuation_cfd']

@pytest.mark.parametrize('live',[True,False])
def test_actual_state_not_fabricated_from_counts(tmp_path,live):
    r=fixture(tmp_path,live)(tmp_path)
    assert r['running']==live and r['process_completed']!=live
    assert r['cycles']==0 and r['last_time'] is None and not r['gpu_training']
    assert not r.get('terminal_verified',False)

def test_timeout_does_not_claim_terminal(tmp_path):
    f=fixture(tmp_path)
    def timeout(*a,**k):raise subprocess.TimeoutExpired('systemctl',3)
    f.__globals__['subprocess'].check_output=timeout
    assert f(tmp_path)==dict(verified=False,observation_state='unavailable')

def test_bad_identity_does_not_claim_running(tmp_path):
    f=fixture(tmp_path,True);f.__globals__['hashlib'].sha256=lambda b:SimpleNamespace(hexdigest=lambda:'bad')
    assert not f(tmp_path)['verified']

def test_current_stage_does_not_hide_b_delivery():
    s=SOURCE.read_text();assert '基本闭环已验证' in s
    assert '328→408' in s and '不是GPU训练' in s

def test_real_progress_clock(tmp_path):
    f=fixture(tmp_path,True)
    p=tmp_path/'artifacts/p064_b_continuation_328_408_cfd_20261007/progress.json';p.parent.mkdir(parents=True)
    p.write_text(json.dumps(dict(completed_cycles=2,rows=[dict(step=1,end_time=328.1),dict(step=2,end_time=328.2)])))
    r=f(tmp_path);assert r['cycles']==2 and r['last_time']==328.2
    p.write_text(json.dumps(dict(completed_cycles=800,rows=[])))
    assert not f(tmp_path)['verified']
