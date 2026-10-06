import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

path=Path(__file__).with_name('serve_live_research_dashboard.py')
if not path.exists():path=Path(__file__).parents[1]/'scripts/serve_live_research_dashboard.py'
spec=importlib.util.spec_from_file_location('bridge_dashboard',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

@pytest.fixture
def example(tmp_path,monkeypatch):
    registration={'unit':'fluid-control-two-segment-frame-cpu-r4-20261006.service','invocation':'99e5c019c08240668241f7ac036320f3'}
    state={'InvocationID':registration['invocation'],'MainPID':'0','ActiveState':'active','SubState':'exited','Result':'success','ExecMainCode':'1','ExecMainStatus':'0'}
    base=tmp_path/'artifacts/online_two_segment_cpu_20261006_r4';base.mkdir(parents=True)
    docs=tmp_path/'docs';docs.mkdir()
    result={'status':'TWO_SEGMENT_ENGINEERING_ONLY_NOT_CONTROL_ADMISSION','model_loaded':False,'policy_loaded':False,'scientific_admission':False,'source_unchanged':True,'rows':[{'step':i,'start':s,'end':e,'applied_now':0,'applied_next':0,'health':{'solver_ended_cleanly':True,'steps':20}} for i,s,e in [(1,148.0,148.1),(2,148.1,148.2)]]}
    fixtures=[(docs/'TWO_SEGMENT_CURRENT_FRAME_R4_APPROVAL_20261006.json',{},'9e268e48f2201eeefbbcb67fefcd3f3184d82877339a429b3f847a2134a9ddd6'),(base/'result.json',result,'be3c57e00003d7092b116058604a47d2ea2b2c1f033551adb39188f1c91f7584'),(base/'cleanup.json',{'cid':'synthetic','errors':[]},'176c4419be678560ac81af6c1b98b88649ac95f7b739f8a441cd1cb24459bdee'),(base/'container_terminal.json',{'Id':'synthetic','State':{'Running':False,'OOMKilled':False,'Status':'exited','ExitCode':137}},'4f9948cf37959af9d9a50c32d2a924cf139561116d55c384deca87f84ea24229')]
    hashes={}
    for file,value,digest in fixtures:
        file.write_text(json.dumps(value));hashes[file.read_bytes()]=digest
    monkeypatch.setattr(module.hashlib,'sha256',lambda raw:SimpleNamespace(hexdigest=lambda:hashes.get(raw,'invalid')))
    return tmp_path,registration,state,base

def test_completed_engineering_not_control(example):
    root,registration,state,_=example
    result=module._two_segment_bridge(root,registration,state)
    assert result['bridge_completed'] and result['completed_segments']==2
    assert result['machine_compute'] is False and result['scientific_admission'] is False
    assert result['container_exit_code']==137

@pytest.mark.parametrize('key,value',[('InvocationID','wrong'),('MainPID','42'),('ExecMainStatus','1'),('SubState','running')])
def test_wrong_or_live_unit_rejected(example,key,value):
    root,registration,state,_=example;state[key]=value
    assert module._two_segment_bridge(root,registration,state)=={'verified':False}

def test_changed_cleanup_rejected(example):
    root,registration,state,base=example
    (base/'cleanup.json').write_text('{"errors":["cleanup failed"]}')
    assert module._two_segment_bridge(root,registration,state)=={'verified':False}
