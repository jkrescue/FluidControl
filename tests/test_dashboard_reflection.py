import ast
import json
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

def test_reflection_has_current_priority_without_hiding_curves():
    s=SOURCE.read_text();active=s.split('function renderActiveExperiment(d){')[1].split('function renderHistoricalClosedLoopEvidence')[0]
    assert active.index('renderAbsolute64ClosedLoop(d)')<active.index('if(d.reflection_training?.verified)')
    assert active.index('本轮训练与两项CFD任务均已结束')<active.index('if(d.reflection_training?.verified)')
