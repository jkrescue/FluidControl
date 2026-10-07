import ast
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

SOURCE=Path(__file__).resolve().parents[1]/'scripts/serve_live_research_dashboard.py'
DIGEST='f380a076a92e1ffa98a497c1fe7933cfb6cc880b4b5e9c3e1708d4cd93a66cf6'

def fixture(tmp_path, live=True, count=4):
    runner=tmp_path/'runner.py';runner.write_text('fixture')
    approval=tmp_path/'docs/P064_B04_COVERAGE_I_TRAINING_APPROVAL_20261007.json';approval.parent.mkdir()
    unit='fluid-control-p064-b04-coverage-i-training-20261007.service'
    approval.write_text(json.dumps(dict(execution_authorized=True,source={'runner':{'path':'runner.py','sha256':DIGEST}},planned_unit=unit,argv=[str(runner),'--execute'])))
    state=f'InvocationID=7c1d1d634af94321b638ba6da2febe45\nMainPID={42 if live else 0}\nActiveState=active\nSubState={"running" if live else "exited"}\nResult=success\nExecMainStatus=0\nExecStart={runner} --execute\nMemoryCurrent=1024\nMemoryMax=12884901888\nMemorySwapMax=0'
    log=json.dumps(dict(event='training_window_complete',consumed=count,source='controlled_b04'))+'\n'+json.dumps(dict(event='accumulation_update_complete',update=1))
    ns=dict(json=json,hashlib=SimpleNamespace(sha256=lambda b:SimpleNamespace(hexdigest=lambda:DIGEST)),subprocess=SimpleNamespace(check_output=lambda a,**k:state if a[0]=='systemctl' else log,SubprocessError=subprocess.SubprocessError))
    node=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_b04_i_training')
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),ns)
    return ns['_b04_i_training']

def test_actual_counts_without_invented_loss(tmp_path):
    r=fixture(tmp_path)(tmp_path);assert r['running'] and r['windows']==4 and r['updates']==1 and r['loss'] is None and r['flow_frozen']
    assert r['memory']['MemorySwapMax']==0

def test_terminal_not_admission(tmp_path):
    r=fixture(tmp_path,False)(tmp_path);assert not r['running'] and r['process_completed'] and not r['terminal_verified']

def test_budget_rejected(tmp_path):
    assert not fixture(tmp_path,count=257)(tmp_path)['verified']

def test_timeout_unknown(tmp_path):
    f=fixture(tmp_path)
    def fail(*a,**k):raise subprocess.TimeoutExpired('systemctl',3)
    f.__globals__['subprocess'].check_output=fail
    assert f(tmp_path)==dict(verified=False,observation_state='unavailable')

def test_wrong_invocation_rejected(tmp_path):
    f=fixture(tmp_path);old=f.__globals__['subprocess'].check_output
    f.__globals__['subprocess'].check_output=lambda *a,**k:old(*a,**k).replace('7c1d1d634af94321b638ba6da2febe45','wrong')
    assert not f(tmp_path)['verified']

def test_ui_preserves_delivery():
    s=SOURCE.read_text();assert "data['b04_i_training']" in s and 'loss未在当前日志报告，不估算' in s
    assert '基本闭环已验证：E114新增800次真实CFD反馈' in s
