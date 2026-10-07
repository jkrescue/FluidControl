import ast
import json
import subprocess
import pytest
from pathlib import Path
from types import SimpleNamespace

SOURCE=Path(__file__).resolve().parents[1]/'scripts/serve_live_research_dashboard.py'
DIGEST='6e5ca18a42a1ad4410260fb9df4f14b457907ff2b5bceadfcfb0f3d621bb81e6'

def fixture(tmp_path,terminal=False,branches=15):
    approval=tmp_path/'docs/P064_Y_REFLECTION_PAIRED_TRAINING_APPROVAL_20261007.json';approval.parent.mkdir();approval.write_text('{}')
    state=f'InvocationID=766ad5ca993d483bb6a42d0f2fc09bd8\nMainPID={0 if terminal else 123}\nActiveState=active\nSubState={"exited" if terminal else "running"}\nResult=success\nExecMainStatus=0\nExecStart={approval} {DIGEST}\n'
    log='\n'.join(json.dumps(x) for x in [dict(event='training_window_complete',consumed=7,completed_transformed_branches=14),dict(event='training_reflection_branch_complete',completed_transformed_branches=branches)])
    ns=dict(Path=Path,json=json,hashlib=SimpleNamespace(sha256=lambda b:SimpleNamespace(hexdigest=lambda:DIGEST)),subprocess=SimpleNamespace(check_output=lambda argv,**k:state if argv[0]=='systemctl' else log,SubprocessError=RuntimeError))
    node=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_reflection_training')
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),ns)
    return ns['_reflection_training'],approval

def test_partial_pair_actual_counts(tmp_path):
    f,_=fixture(tmp_path);r=f(tmp_path)
    assert r['running'] and r['windows']==7 and r['branches']==15 and r['updates']==0

def test_terminal_never_running(tmp_path):
    f,_=fixture(tmp_path,True);r=f(tmp_path);assert not r['running'] and '待终态核验' in r['status']

def test_missing(tmp_path):
    f,p=fixture(tmp_path);p.unlink();assert not f(tmp_path)['verified']

def test_bad_count(tmp_path):
    f,_=fixture(tmp_path,branches=513);assert not f(tmp_path)['verified']

@pytest.mark.parametrize('command',['systemctl','journalctl'])
def test_timeout_is_unknown_not_terminal(tmp_path,command):
    f,_=fixture(tmp_path)
    old=f.__globals__['subprocess'].check_output
    def timeout(argv,**kwargs):
        if argv[0]==command:raise subprocess.TimeoutExpired(argv,3)
        return old(argv,**kwargs)
    f.__globals__['subprocess']=SimpleNamespace(check_output=timeout,SubprocessError=subprocess.SubprocessError)
    r=f(tmp_path);assert r=={'verified':False,'observation_state':'unavailable'}
    assert 'running' not in r and 'terminal_verified' not in r

def test_identity_mismatch_is_not_terminal(tmp_path):
    f,_=fixture(tmp_path)
    f.__globals__['hashlib'].sha256=lambda b:SimpleNamespace(hexdigest=lambda:'wrong')
    assert f(tmp_path)=={'verified':False,'observation_state':'identity_mismatch'}

def test_reflection_has_current_priority_without_hiding_curves():
    s=SOURCE.read_text();active=s.split('function renderActiveExperiment(d){')[1].split('function renderHistoricalClosedLoopEvidence')[0]
    assert active.index('renderAbsolute64ClosedLoop(d)')<active.index('if(d.reflection_training?.verified)')
    assert active.index('本轮训练与两项CFD任务均已结束')<active.index('if(d.reflection_training?.verified)')
    assert 'else if(d.reflection_training)' in active and '不能据此判断训练已停止或完成' in active

def test_bound_terminal_survives_unit_gc(tmp_path):
    f,_=fixture(tmp_path)
    pins=[('docs/P064_Y_REFLECTION_PAIRED_TERMINAL_REVIEW_20261007.md','8b129933430694791609083c0609503cbcd0ad9636c30ef907704022e70295c9'),('artifacts/p064_y_reflection_terminal_audit_20261007/receipt.json','10ab43ae2c8f7d73833e1b119da4f7f0dca5c74fa9b1962df118b5b706a350f5'),('artifacts/p064_y_reflection_paired_20261007/result.json','fa57ada9b47414c20c5ee245e8b9e9663e6f568c5f06e4ac9d17dd16b9b53d60')]
    receipt=dict(status='Y_REFLECTION_TERMINAL_ENGINEERING_ACCEPT_NOT_ADMISSION',terminal_evidence={'invocation':'766ad5ca993d483bb6a42d0f2fc09bd8'},approval_sha256=DIGEST,result_sha256=pins[2][1],original_windows=256,branches=512,updates=32,min_available_gib=105.)
    values=[b'review',json.dumps(receipt).encode(),b'result'];digests={b'{}':DIGEST}
    for (name,expected),value in zip(pins,values):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(value);digests[value]=expected
    f.__globals__['hashlib'].sha256=lambda b:SimpleNamespace(hexdigest=lambda:digests.get(b,'bad'))
    def unavailable(*args,**kwargs):raise AssertionError('terminal must not query GC unit')
    f.__globals__['subprocess'].check_output=unavailable
    r=f(tmp_path);assert r['terminal_verified'] and not r['running'] and not r['promoted']
    assert r['development_evaluation']=='not_run_unknown' and r['updates']==32
    (tmp_path/pins[2][0]).write_bytes(b'changed');assert f(tmp_path)['observation_state']=='identity_mismatch'
