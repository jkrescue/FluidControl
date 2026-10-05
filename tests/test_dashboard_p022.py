import importlib.util
import json
import sys
from pathlib import Path

path=Path(__file__).parents[1]/'scripts/serve_live_research_dashboard.py'
spec=importlib.util.spec_from_file_location('dashboard_p022',path)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

def state(**kw):
    return dict(InvocationID='f1f3f7b31e70440693da2661a12cfc04',MainPID='42',ActiveState='active',SubState='running',Result='success',ExecMainCode='0',ExecMainStatus='0',**kw)

def log(arm,steps):return '\n'.join(json.dumps(dict(event='arm_update_complete',arm=arm,update=i)) for i in steps)

def test_live_and_not_accepted():
    x=m._parse_fcp022_live(state(),log('A_zero',[1,2]),True)
    assert x['running'] and x['updates']=={'A_zero':2,'B_causal':0} and not x['admission']

def test_wrong_invocation_and_process():
    s=state();s['InvocationID']='old'
    assert not m._parse_fcp022_live(s,'',True)['verified']
    assert not m._parse_fcp022_live(state(),'',False)['verified']

def test_bad_counts():
    for text in [log('A_zero',[2]),log('A_zero',[1,1]),log('B_causal',[1]),log('A_zero',[17])]:
        assert not m._parse_fcp022_live(state(),text,True)['verified']

def test_terminal_not_running_or_admission():
    s=state();s.update(MainPID='0',SubState='exited',ExecMainCode='1')
    x=m._parse_fcp022_live(s,log('A_zero',range(1,17))+'\n'+log('B_causal',range(1,17)),False)
    assert x['exited_success'] and not x['running'] and not x['admission']

def test_failure_is_not_success():
    s=state();s.update(MainPID='0',ActiveState='failed',SubState='failed',Result='exit-code',ExecMainCode='1',ExecMainStatus='1')
    x=m._parse_fcp022_live(s,log('A_zero',[1]),False)
    assert x['verified'] and not x['exited_success'] and not x['running']
