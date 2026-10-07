import ast
import json
from pathlib import Path
from types import SimpleNamespace

SOURCE=Path(__file__).resolve().parents[1]/'scripts/serve_live_research_dashboard.py'
DIGEST='5cf4862882caa99eb600c76737927d45fdbc6fe7d9099a66c134fbd5bc73fa4b'

def fixture(tmp_path,terminal=False,steps=15360):
    approval=tmp_path/'docs/P064_ABSOLUTE64_SYMMETRY_CANONICAL_PPO_APPROVAL_20261007.json';approval.parent.mkdir();approval.write_text('{}')
    progress=tmp_path/'artifacts/p064_absolute64_symmetry_canonical_32768_ppo_20261007/payload/progress.json';progress.parent.mkdir(parents=True)
    progress.write_text(json.dumps({'time/total_timesteps':steps,'train/n_updates':116})+'\n{"partial":')
    state=f'InvocationID=5e1c7db703874125bb1556b411d3169c\nMainPID={0 if terminal else 123}\nActiveState=active\nSubState={"exited" if terminal else "running"}\nResult=success\nExecMainStatus=0\nExecStart={approval} {DIGEST}\n'
    ns=dict(json=json,hashlib=SimpleNamespace(sha256=lambda b:SimpleNamespace(hexdigest=lambda:DIGEST)),subprocess=SimpleNamespace(check_output=lambda *a,**k:state,SubprocessError=RuntimeError))
    node=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_absolute64_ppo')
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),ns)
    return ns['_absolute64_ppo'],approval

def test_live_jsonl_partial_tail(tmp_path):
    f,_=fixture(tmp_path);r=f(tmp_path)
    assert r['running'] and r['timesteps']==15360 and r['ppo_epochs']==116

def test_terminal_not_running(tmp_path):
    f,_=fixture(tmp_path,True);r=f(tmp_path)
    assert not r['running'] and '待独审' in r['status']

def test_missing_no_claim(tmp_path):
    f,p=fixture(tmp_path);p.unlink();assert not f(tmp_path)['verified']

def test_budget_failclosed(tmp_path):
    f,_=fixture(tmp_path,steps=32769);assert not f(tmp_path)['verified']

def test_appends_after_cfd_not_replaces():
    s=SOURCE.read_text();line=next(x for x in s.splitlines() if '并行 E110 PPO' in x)
    assert "$('lead-monitor').textContent+=" in line and 'lead-now' not in line
    assert s.index('if(d.future_time_cfd?.verified)')<s.index('并行 E110 PPO')
