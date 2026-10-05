import importlib.util,json,sys
from pathlib import Path

path=Path(__file__).parents[1]/'scripts/serve_live_research_dashboard.py'
spec=importlib.util.spec_from_file_location('registered_dashboard',path)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
REG=dict(invocation='current',planned_updates={'STAT':16},label='test',description='test')
def state():return dict(InvocationID='current',MainPID='42',ActiveState='active',SubState='running',Result='success',ExecMainCode='0',ExecMainStatus='0')
def log(steps,arm='STAT'):return '\n'.join(json.dumps(dict(event='arm_update_complete',arm=arm,update=s)) for s in steps)
def test_running():
    r=m._parse_registered_progress(state(),log([1,2]),True,REG)
    assert r['running'] and r['updates']=={'STAT':2} and not r['admission']
def test_identity():
    s=state();s['InvocationID']='old'
    assert not m._parse_registered_progress(s,'',True,REG)['verified']
    assert not m._parse_registered_progress(state(),'',False,REG)['verified']
def test_bad_progress():
    for text in (log([2]),log([1,1]),log([17]),log([True]),log([1],arm='old')):
        assert not m._parse_registered_progress(state(),text,True,REG)['verified']
def test_terminal_no_admission():
    s=state();s.update(MainPID='0',SubState='exited',ExecMainCode='1')
    r=m._parse_registered_progress(s,log(range(1,17)),False,REG)
    assert r['exited_success'] and not r['running'] and not r['admission']
def test_bad_registry_plan():
    for plan in ({},{'STAT':True},{'STAT':0}):
        assert not m._parse_registered_progress(state(),'',True,{**REG,'planned_updates':plan})['verified']
def test_missing_registry(tmp_path):
    assert not m._registered_experiment_live(tmp_path)['verified']
