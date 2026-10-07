import ast
import math
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace
import pytest

SOURCE=Path(__file__).resolve().parents[1]/'scripts/serve_live_research_dashboard.py'

def fixture(tmp_path,live=True):
    digest='f2bfcd8f6d1526af266946448ca452d3c271047466d77a9e439d843081338536'
    approval=tmp_path/'docs/P064_B04_LONG_EXCITATION_RAW_CFD_APPROVAL_20261007.json';approval.parent.mkdir();approval.write_text('{}')
    state=f'InvocationID=e2fa5d4e5fa4465a90a2dfbbdd402fc8\nMainPID={123 if live else 0}\nActiveState=active\nSubState={"running" if live else "exited"}\nResult=success\nExecMainStatus=0\nExecStart={approval} {digest}\n'
    ns=dict(re=re,math=math,hashlib=SimpleNamespace(sha256=lambda b:SimpleNamespace(hexdigest=lambda:digest)),subprocess=SimpleNamespace(check_output=lambda *a,**k:state,SubprocessError=subprocess.SubprocessError))
    node=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_b04_raw_cfd')
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),ns)
    output=tmp_path/'artifacts/p064_b04_long_excitation_raw_20261007';output.mkdir(parents=True)
    return ns['_b04_raw_cfd'],output

def test_actual_log_clock_and_only_complete_file_pairs(tmp_path):
    f,out=fixture(tmp_path);(out/'log.pimpleFoam').write_text('Time = 123.770\nTime = 123.775\n')
    for t in ('120','120.1','120.2','120.005','119','constant'):
        p=out/'case'/t;p.mkdir(parents=True);(p/'U').write_text('u')
        if t!='120.2':(p/'p').write_text('p')
    r=f(tmp_path);assert r['running'] and r['live_time']==123.775 and r['frames']==2
    assert not r['gpu_training'] and not r['closed_loop_control']

@pytest.mark.parametrize('live',[True,False])
def test_initialization_or_unreviewed_terminal(tmp_path,live):
    f,_=fixture(tmp_path,live);r=f(tmp_path)
    assert r['running']==live and r['process_completed']!=live
    assert r['live_time'] is None and r['frames']==0 and not r.get('terminal_verified',False)

def test_timeout_unknown(tmp_path):
    f,_=fixture(tmp_path)
    def timeout(*a,**k):raise subprocess.TimeoutExpired('systemctl',3)
    f.__globals__['subprocess'].check_output=timeout
    assert f(tmp_path)==dict(verified=False,observation_state='unavailable')

def test_bad_identity_or_clock_unknown(tmp_path):
    f,out=fixture(tmp_path);(out/'log.pimpleFoam').write_text('Time = 201\n')
    assert not f(tmp_path)['verified']
    f.__globals__['hashlib'].sha256=lambda b:SimpleNamespace(hexdigest=lambda:'bad')
    assert not f(tmp_path)['verified']

def test_current_priority_and_curve_preservation():
    s=SOURCE.read_text();assert s.index('if(d.continuation_cfd?.terminal_verified)')<s.index('if(d.b04_raw_cfd)')
    assert 'renderBSelectedCurves(d)' in s and '不是反馈闭环控制，不是GPU训练' in s
