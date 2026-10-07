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

def test_terminal_survives_gc_and_hash_mismatch_fails(tmp_path):
    f=fixture(tmp_path)
    node=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_continuation_cfd')
    pins=next(ast.literal_eval(n.value) for n in ast.walk(node) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='pins')
    digest='6e08555e336e271fe0baf6790feaf3d92c7c6e0771fc9aa93d7b0fd608f3f812'
    receipt=dict(status='CONTINUATION_RAW_REVIEW_NOT_INDEPENDENT_CONDITION',invocation='73a2568857074c70b494639269a602bf',approval_sha256=digest,result_sha256=pins[2][1],new_cycles=800)
    result=dict(approval_sha256=digest,cycles=800,rows=[dict(step=i+1,end_time=328+(i+1)*.1) for i in range(800)],windows={'prospective_continuation_windows':{'tail_full_80':{'metric':1},'joined_full_160':{'metric':2}}})
    blobs=[b'review',json.dumps(receipt).encode(),json.dumps(result).encode()];digests={b'{}':digest}
    for (name,h),b in zip(pins,blobs):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);digests[b]=h
    f.__globals__['hashlib'].sha256=lambda b:SimpleNamespace(hexdigest=lambda:digests.get(b,'bad'))
    def no_unit(*a,**k):raise AssertionError('GC unit not queried')
    f.__globals__['subprocess'].check_output=no_unit
    r=f(tmp_path);assert r['terminal_verified'] and not r['running'] and len(r['rows'])==800
    assert r['primary']!=r['joined']
    (tmp_path/pins[2][0]).write_bytes(b'changed');assert not f(tmp_path)['verified']

def test_js_selects_actual_800_continuation_rows():
    tree=ast.parse(SOURCE.read_text());html=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PAGE' for t in n.targets))
    mapper='function currentGSeries(rows){'+html.split('function currentGSeries(rows){',1)[1].split('function renderCurrentGClosedLoop',1)[0]
    selector='function selectedBRows(d, selection){'+html.split('function selectedBRows(d, selection){',1)[1].split('function renderBSelectedCurves',1)[0]
    root=SOURCE.parents[1]
    result_path=root/'artifacts/p064_b_continuation_328_408_cfd_20261007/result.json'
    if not result_path.exists():pytest.skip('actual Spark artifact only')
    program=mapper+selector+'''const fs=require('fs');const r=JSON.parse(fs.readFileSync(process.argv[1]));const original=[{time:130.1}];const d={continuation_cfd:{terminal_verified:true,rows:r.rows},canonical_b01_reproduction:{rows:original}};const x=selectedBRows(d,'continuation');if(x.length!==800||x[0].time!==328.1||x[799].time!==408)throw Error('clock');x.forEach((a,i)=>{const b=r.rows[i];if(a.omega!==b.applied_omega||a.requested_omega!==b.requested_omega||a.ppo_total_cd!==b.output_observation[64]+b.output_observation[66]||a.ppo_rear_cl!==b.output_observation[67]||a.zero_total_cd!==b.zero_observation[64]+b.zero_observation[66]||a.zero_rear_cl!==b.zero_observation[67])throw Error('mapping');});if(selectedBRows(d,'original')!==original)throw Error('old selection');d.continuation_cfd.terminal_verified=false;if(selectedBRows(d,'continuation').length)throw Error('unreviewed');'''
    subprocess.run(['node','-e',program,str(result_path)],check=True,capture_output=True,text=True)
