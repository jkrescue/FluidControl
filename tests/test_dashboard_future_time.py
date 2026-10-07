import ast
import json
from pathlib import Path
from types import SimpleNamespace

SOURCE=Path(__file__).resolve().parents[1]/'scripts/serve_live_research_dashboard.py'
DIGEST='dbcedcce5d680c98e9e611ca092fcd4953879ffef9e8ef9fbf613cd11c93da25'

def fixture(tmp_path,baseline,paired,terminal=False):
    approval=tmp_path/'docs/P064_B_FUTURE_TIME_CFD_APPROVAL_20261007.json';approval.parent.mkdir();approval.write_text('{}')
    out=tmp_path/'artifacts/p064_b_future_time_248_328_cfd_20261007';out.mkdir(parents=True)
    for name,n,start in [('baseline_progress.json',baseline,228),('progress.json',paired,248)]:
        if n is not None:(out/name).write_text(json.dumps({'completed_cycles':n,'rows':[dict(step=i,end_time=start+.1*i) for i in range(1,n+1)]}))
    state=f'InvocationID=4b3eb2a2fcab4585aafd5740f8d0cea3\nMainPID={0 if terminal else 123}\nActiveState=active\nSubState={"exited" if terminal else "running"}\nResult=success\nExecMainStatus=0\nExecStart={approval} {DIGEST}\n'
    ns=dict(Path=Path,json=json,hashlib=SimpleNamespace(sha256=lambda b:SimpleNamespace(hexdigest=lambda:DIGEST)),subprocess=SimpleNamespace(check_output=lambda *a,**k:state,SubprocessError=RuntimeError))
    node=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_future_time_cfd')
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),ns)
    return ns['_future_time_cfd'],approval

def test_baseline_live(tmp_path):
    f,_=fixture(tmp_path,100,None);r=f(tmp_path)
    assert r['running'] and r['baseline_cycles']==100 and r['paired_cycles']==0 and r['phase']=='baseline'

def test_paired_live(tmp_path):
    f,_=fixture(tmp_path,200,70);r=f(tmp_path)
    assert r['running'] and r['paired_cycles']==70 and r['phase']=='paired'

def test_baseline_complete_waiting_pair(tmp_path):
    f,_=fixture(tmp_path,200,None);assert f(tmp_path)['phase']=='handoff'

def test_terminal_not_running(tmp_path):
    f,_=fixture(tmp_path,200,800,True);r=f(tmp_path)
    assert not r['running'] and '待独审' in r['status']

def test_invalid_phase_or_budget(tmp_path):
    f,_=fixture(tmp_path,199,5);assert not f(tmp_path)['verified']

def test_missing_approval_no_claim(tmp_path):
    f,p=fixture(tmp_path,200,800);p.unlink();assert not f(tmp_path)['verified']

def test_verified_terminal_requires_both_proofs(tmp_path):
    f,_=fixture(tmp_path,200,800,True)
    proof=tmp_path/'docs/P064_B_FUTURE_TIME_CFD_TERMINAL_REVIEW_20261007.md';proof.write_bytes(b'proof')
    result=tmp_path/'artifacts/p064_b_future_time_248_328_cfd_20261007/result.json';result.write_bytes(b'result')
    digests={b'{}':DIGEST,b'proof':'da193912d5c6a773d382d4b874c304660e3f732f247f679bf0f6636ea6d6fc75',b'result':'d53cb2ea32f66af6c4067d0c7eb90e7aecc8b815634588b6fe448d291acc5b98'}
    f.__globals__['hashlib']=SimpleNamespace(sha256=lambda b:SimpleNamespace(hexdigest=lambda:digests.get(b,'bad')))
    assert f(tmp_path)['terminal_verified']
    result.write_bytes(b'changed');assert not f(tmp_path)['terminal_verified']

def test_curves_and_priority_preserved():
    s=SOURCE.read_text();active=s.split('function renderActiveExperiment(d){')[1].split('function renderHistoricalClosedLoopEvidence')[0]
    assert active.index('renderCurrentGClosedLoop')<active.index('if(d.absolute64_training)')<active.index('if(d.future_time_cfd?.verified)')
