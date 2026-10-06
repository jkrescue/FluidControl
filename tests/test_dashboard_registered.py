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

def test_p027_counts_cases_not_optimizer_steps():
    reg={**REG,'progress_kind':'diagnostic_origins','planned_updates':{'CFD':44}}
    rows=[dict(event='origin_complete',count=n,case=f'case{n}') for n in range(1,45)]
    s=state();s.update(MainPID='0',SubState='exited',ExecMainCode='1')
    actual=m._parse_registered_progress(s,'\n'.join(map(json.dumps,rows)),False,reg)
    assert actual['exited_success'] and actual['updates']=={'CFD':44}
    assert actual['progress_unit']=='个诊断工况（无参数更新）' and not actual['admission']
    for bad in (rows[1:],rows+[rows[-1]],rows[:-1]+[{**rows[-1],'case':'case1'}]):
        assert not m._parse_registered_progress(s,'\n'.join(map(json.dumps,bad)),False,reg)['verified']

def test_p027_bound_review_never_admits(tmp_path):
    import hashlib
    report=tmp_path/'review.md';report.write_text('independent review')
    result=tmp_path/'result.json'
    payload=dict(status='P027_OFFLINE_DIAGNOSTIC_COMPLETE_NOT_ADMISSION',
                 rows=[{'case':str(n)} for n in range(44)],flow_transitions=440,
                 aerodynamic_state_evaluations=1760,scientific_admission=False,
                 optimizer_created=False,model_saved=False,validation_accessed=False,
                 frozen_test_accessed=False)
    review=dict(kind='p027_diagnostic',report='review.md',result='result.json',
                report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
                summary='diagnostic',next_action='CPU implementation')
    progress=dict(verified=True,exited_success=True,updates={'CFD':44},planned_updates={'CFD':44})
    for changed in ({},{'model_saved':True},{'rows':payload['rows'][:-1]},{'scientific_admission':True}):
        result.write_text(json.dumps({**payload,**changed}))
        review['result_sha256']=hashlib.sha256(result.read_bytes()).hexdigest()
        value=m._registered_terminal_review(tmp_path,{'review':review},progress)
        assert value['verified'] is (not changed)
        if not changed: assert value['diagnostic_completed'] and not value['admission']
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

def test_history_training_counts_actual_windows_and_updates():
    reg={**REG,'progress_kind':'history_training','planned_updates':{'K4':171}}
    rows=[dict(event='training_window_complete',history_k=4,consumed=n) for n in range(1,9)]
    rows.append(dict(event='accumulation_update_complete',history_k=4,update=1))
    s=state();s.update(ActiveState='activating',SubState='start')
    value=m._parse_registered_progress(s,'\n'.join(map(json.dumps,rows)),True,reg)
    assert value['running'] and value['updates']=={'K4':1} and value['windows']=={'K4':8}
    assert not value['admission']
    for bad in (rows[1:],rows+rows[-1:],rows+[dict(event='training_window_complete',history_k=True,consumed=9)],rows+[dict(event='training_window_complete',history_k=1,consumed=9)]):
        assert not m._parse_registered_progress(s,'\n'.join(map(json.dumps,bad)),True,reg)['verified']

def test_history_training_cannot_report_update_without_eight_windows():
    reg={**REG,'progress_kind':'history_training','planned_updates':{'K1':171}}
    text=json.dumps(dict(event='accumulation_update_complete',history_k=1,update=1))
    assert not m._parse_registered_progress(state(),text,True,reg)['verified']

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
