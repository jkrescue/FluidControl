import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/serve_live_research_dashboard.py'

def helper(tmp_path, terminal=False, windows=512, updates=64):
    tree=ast.parse(SOURCE.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_absolute64_training')
    approval=tmp_path/'docs/P064_ABSOLUTE64_ARM_B_TRAINING_APPROVAL_20261007.json'
    approval.parent.mkdir(); runner=tmp_path/'runner.py';runner.write_text('# fixture')
    argv=['python','-u',str(runner),'--output',str(tmp_path/'artifacts/p064_absolute64_arm_b_20261007'),'--execute']
    approval.write_text(json.dumps({'execution_authorized':True,'argv':argv,'planned_unit':'fluid-control-p064-absolute64-arm-b-20261007.service','planned_output':argv[4],'runner_sha256':hashlib.sha256(runner.read_bytes()).hexdigest()}))
    def command(command_args,**kwargs):
        if command_args[0]=='systemctl':
            return f'InvocationID=actual\nMainPID={0 if terminal else 123}\nActiveState=active\nSubState={"exited" if terminal else "running"}\nResult=success\nExecMainStatus=0\nExecStart={" ".join(argv)}\n'
        return '\n'.join(map(json.dumps,[{'event':'training_window_complete','consumed':windows},{'event':'accumulation_update_complete','update':updates,'updates_total':64}]))
    ns=dict(Path=Path,json=json,hashlib=hashlib,subprocess=SimpleNamespace(check_output=command,SubprocessError=RuntimeError))
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),ns)
    return ns['_absolute64_training'],approval

def test_live_budget_and_no_fabricated_loss(tmp_path):
    f,_=helper(tmp_path,windows=264,updates=33);r=f(tmp_path)
    assert r['running'] and r['updates']==33 and r['windows']==264 and r['loss'] is None

def test_terminal_never_running(tmp_path):
    f,_=helper(tmp_path,terminal=True);r=f(tmp_path)
    assert not r['running'] and r['updates']==64 and '待独审' in r['status']

def test_missing_approval_is_not_started(tmp_path):
    f,p=helper(tmp_path);p.unlink();r=f(tmp_path)
    assert not r['running'] and r['status']=='未启动'

def test_over_budget_fails_closed(tmp_path):
    f,_=helper(tmp_path,updates=65);r=f(tmp_path)
    assert not r['running'] and r['status']=='状态不可核验'

def test_changed_runner_fails_closed(tmp_path):
    f,_=helper(tmp_path);(tmp_path/'runner.py').write_text('# changed')
    assert f(tmp_path)['status']=='状态不可核验'

def test_pending_approval_never_running(tmp_path):
    f,p=helper(tmp_path);s=json.loads(p.read_text());s['execution_authorized']=False;p.write_text(json.dumps(s))
    assert f(tmp_path)['status']=='未启动'

def test_short_argv_does_not_break_api(tmp_path):
    f,p=helper(tmp_path);s=json.loads(p.read_text());s['argv']=[];p.write_text(json.dumps(s))
    assert f(tmp_path)['status']=='状态不可核验'

def test_current_summary_keeps_g_curves():
    s=SOURCE.read_text();active=s.split('function renderActiveExperiment(d){')[1].split('function renderHistoricalClosedLoopEvidence')[0]
    assert active.index('renderCurrentGClosedLoop')<active.index('if(d.absolute64_training)')
    assert '更新${a.updates}/64，窗口${a.windows}/512' in active
