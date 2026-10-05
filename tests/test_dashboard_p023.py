import importlib.util
import json
import sys
from pathlib import Path

path=Path(__file__).parents[1]/'scripts/serve_live_research_dashboard.py'
spec=importlib.util.spec_from_file_location('dashboard_p023',path)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

def state():
    return dict(InvocationID='39aec740a9914226bb1f74c2d29e7917',MainPID='42',ActiveState='active',SubState='running',Result='success',ExecMainCode='0',ExecMainStatus='0')

def log(arm,steps):return '\n'.join(json.dumps(dict(event='arm_update_complete',arm=arm,update=i)) for i in steps)

def test_live_and_not_accepted():
    x=m._parse_fcp023_live(state(),log('LOW',[1,2]),True)
    assert x['running'] and x['updates']=={'LOW':2,'HIGH':0} and not x['admission']

def test_wrong_identity():
    s=state();s['InvocationID']='old'
    assert not m._parse_fcp023_live(s,'',True)['verified']
    assert not m._parse_fcp023_live(state(),'',False)['verified']

def test_bad_progress():
    for text in [log('LOW',[2]),log('LOW',[1,1]),log('HIGH',[1]),log('LOW',[17]),log('A_zero',[1])]:
        assert not m._parse_fcp023_live(state(),text,True)['verified']

def test_terminal_is_not_admission():
    s=state();s.update(MainPID='0',SubState='exited',ExecMainCode='1')
    x=m._parse_fcp023_live(s,log('LOW',range(1,17))+'\n'+log('HIGH',range(1,17)),False)
    assert x['exited_success'] and not x['running'] and not x['admission']

def test_failed_not_running():
    s=state();s.update(MainPID='0',ActiveState='failed',SubState='failed',Result='exit-code',ExecMainCode='1',ExecMainStatus='1')
    x=m._parse_fcp023_live(s,log('LOW',[1]),False)
    assert x['verified'] and not x['exited_success'] and not x['running']

def test_terminal_review_missing_or_wrong_identity(tmp_path):
    assert not m._fcp023_terminal_review(tmp_path)['verified']
    p=tmp_path/'artifacts/fcp023_input_block_20261005/result.json'
    p.parent.mkdir(parents=True)
    p.write_text('{"comparison":{"local_support":true}}')
    assert not m._fcp023_terminal_review(tmp_path)['verified']
