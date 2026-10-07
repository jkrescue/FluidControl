import ast
import hashlib
import json
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE=Path(__file__).resolve().parents[1]/'scripts/serve_live_research_dashboard.py'

def fixture(tmp_path, rows=(), terminal=False):
    node=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_pressure_aux_training')
    literals={n.targets[0].id:ast.literal_eval(n.value) for n in node.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and isinstance(n.value,ast.Constant)}
    digest=literals['expected_approval'];inv=literals['inv']
    assert digest and inv, 'test only final actual binding'
    approval=tmp_path/'docs'/literals['approval_name'];approval.parent.mkdir();approval.write_text('{}')
    state=f'InvocationID={inv}\nMainPID={0 if terminal else 123}\nActiveState=active\nSubState={"exited" if terminal else "running"}\nResult=success\nExecMainStatus=0\nExecStart={approval} {digest}\n'
    ns=dict(Path=Path,json=json,math=math,hashlib=SimpleNamespace(sha256=lambda b:SimpleNamespace(hexdigest=lambda:digest)),subprocess=SimpleNamespace(check_output=lambda argv,**k:state if argv[0]=='systemctl' else '\n'.join(map(json.dumps,rows)),SubprocessError=subprocess.SubprocessError))
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),ns)
    return ns['_pressure_aux_training'],approval

def test_initializing_unknown_losses(tmp_path):
    f,_=fixture(tmp_path);r=f(tmp_path)
    assert r['running'] and r['windows']==r['updates']==0 and r['latest_window_losses'] is None

def test_four_loss_semantics_and_actual_updates(tmp_path):
    f,_=fixture(tmp_path,[dict(event='training_window_complete',consumed=8,original_total_loss=.2,pressure_h1_loss=.3,pressure_aux_weighted_loss=.03,training_objective=.23),dict(event='accumulation_update_complete',update=1)])
    r=f(tmp_path);assert r['windows']==8 and r['updates']==1
    assert r['latest_window_losses']==dict(window=8,original_total_loss=.2,pressure_h1_loss=.3,pressure_aux_weighted_loss=.03,training_objective=.23)

def test_terminal_is_not_scientific_pass(tmp_path):
    f,_=fixture(tmp_path,terminal=True);r=f(tmp_path)
    assert not r['running'] and '待终态核验' in r['status'] and not r.get('terminal_verified',False)

@pytest.mark.parametrize('command',['systemctl','journalctl'])
def test_observation_timeout_unknown(tmp_path,command):
    f,_=fixture(tmp_path);old=f.__globals__['subprocess'].check_output
    def read(argv,**kwargs):
        if argv[0]==command:raise subprocess.TimeoutExpired(argv,3)
        return old(argv,**kwargs)
    f.__globals__['subprocess'].check_output=read
    assert f(tmp_path)==dict(verified=False,observation_state='unavailable')

@pytest.mark.parametrize('row',[dict(event='training_window_complete',consumed=257),dict(event='accumulation_update_complete',update=33),dict(event='training_window_complete',consumed=1,training_objective=float('nan'))])
def test_bad_metrics_fail_closed(tmp_path,row):
    f,_=fixture(tmp_path,[row]);assert not f(tmp_path)['verified']

def test_bad_approval_not_running(tmp_path):
    f,_=fixture(tmp_path);f.__globals__['hashlib'].sha256=lambda b:SimpleNamespace(hexdigest=lambda:'wrong')
    assert f(tmp_path)==dict(verified=False,observation_state='identity_mismatch')

def test_current_line_preserves_delivery_and_history():
    s=SOURCE.read_text();assert "data['pressure_aux_training']" in s
    assert '实际训练目标 ${fmt(l.training_objective)}' in s and '原主任务loss ${fmt(l.original_total_loss)}' in s
    assert 'B默认与已交付闭环保持' in s and '不能据此判断结束' in s
