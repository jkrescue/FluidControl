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

def test_review_requires_terminal_complete_and_bound_false_result(tmp_path):
    import hashlib
    report=tmp_path/'report.md';result=tmp_path/'result.json'
    report.write_text('Independent review')
    result.write_text(json.dumps({'comparison':{'local_support':False}}))
    review=dict(report='report.md',result='result.json',summary='failed',next_action='CPU preparation')
    for key,path in [('report',report),('result',result)]:
        review[key+'_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    reg={'review':review}
    progress=dict(verified=True,exited_success=True,updates={'STAT':16},planned_updates={'STAT':16})
    assert m._registered_terminal_review(tmp_path,reg,progress)['verified']
    assert not m._registered_terminal_review(tmp_path,reg,{**progress,'exited_success':False})['verified']
    assert not m._registered_terminal_review(tmp_path,reg,{**progress,'updates':{'STAT':15}})['verified']
    report.write_text('changed')
    assert not m._registered_terminal_review(tmp_path,reg,progress)['verified']

def test_review_rejects_true_or_outside_result(tmp_path):
    import hashlib
    report=tmp_path/'report';report.write_text('review')
    result=tmp_path/'result';result.write_text(json.dumps({'comparison':{'local_support':True}}))
    review=dict(report='report',result='result',summary='bad',next_action='bad',
                report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
                result_sha256=hashlib.sha256(result.read_bytes()).hexdigest())
    progress=dict(verified=True,exited_success=True,updates={'STAT':16},planned_updates={'STAT':16})
    assert not m._registered_terminal_review(tmp_path,{'review':review},progress)['verified']
    review['result']='../outside'
    assert not m._registered_terminal_review(tmp_path,{'review':review},progress)['verified']

def test_resource_arm_progress_is_not_training():
    reg={**REG,'progress_kind':'resource_arms','planned_updates':{'K1':1,'K4':1}}
    s=state();s.update(MainPID='0',SubState='exited',ExecMainCode='1')
    logs='\n'.join(json.dumps({'event':'arm_complete','k':k}) for k in (1,4))
    value=m._parse_registered_progress(s,logs,False,reg)
    assert value['verified'] and value['updates']=={'K1':1,'K4':1}
    assert value['progress_unit']=='项无更新计算' and not value['admission']
    for invalid in (logs+'\n'+logs, json.dumps({'event':'arm_complete','k':True})):
        assert not m._parse_registered_progress(s,invalid,False,reg)['verified']

def test_resource_review_requires_no_update_and_no_candidate(tmp_path):
    import hashlib
    report=tmp_path/'report';report.write_text('review')
    result=tmp_path/'result'
    payload=dict(status='FC_P026_HISTORY_RESOURCE_COMPLETE_NOT_ADMISSION',optimizer_steps=0,
                 candidate_saved=False,scientific_admission=False,arms=[{'k':1},{'k':4}])
    review=dict(kind='history_resource',report='report',result='result',summary='engineering',
                next_action='integration',report_sha256=hashlib.sha256(report.read_bytes()).hexdigest())
    progress=dict(verified=True,exited_success=True,updates={'K1':1,'K4':1},planned_updates={'K1':1,'K4':1})
    for candidate in (False,True):
        result.write_text(json.dumps({**payload,'candidate_saved':candidate}))
        review['result_sha256']=hashlib.sha256(result.read_bytes()).hexdigest()
        actual=m._registered_terminal_review(tmp_path,{'review':review},progress)
        assert actual['verified'] is (not candidate)
        if not candidate: assert actual['engineering_pass'] and not actual['admission']
