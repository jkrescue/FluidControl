import json
from datetime import UTC,datetime
from pathlib import Path

import pytest
import watch_training_evaluation_state as w
import reconcile_training_evaluation_state as r

NOW=datetime(2026,10,5,13,0,tzinfo=UTC)


def live(repo,**changes):
    argv=f'/usr/bin/env FCP018_APPROVAL_SHA256={w.P018_APPROVAL_SHA256} /bin/bash {repo/w.P018_LAUNCHER} --execute'
    value=dict(load_state='loaded',invocation_id=w.P018_INVOCATION,exec_start='{ argv[]='+argv+' ; }',
        active_state='active',sub_state='running',main_pid=123,main_pid_alive=True,
        exec_main_status='0',exec_main_code='0',result='success')
    value.update(changes);return value


@pytest.fixture
def evidence(tmp_path,monkeypatch):
    def digest(path):
        name=str(path)
        if name.endswith('execution_approval.json') or name.endswith(str(w.P018_APPROVAL)):return w.P018_APPROVAL_SHA256
        if name.endswith(str(w.P018_OBSERVATION)):return w.P018_OBSERVATION_SHA256
        if 'protocol' in name.lower():return w.P018_PROTOCOL_SHA256
        return w.P018_LAUNCHER_SHA256
    monkeypatch.setattr(w,'file_sha256',digest)
    monkeypatch.setattr(w,'p018_progress',lambda *a:dict(completed_updates=12,consumed_windows=96,log_age_seconds=10))
    (tmp_path/w.P018_ROOT).mkdir(parents=True)
    return tmp_path


def test_running_precedes_partial_result(evidence):
    (evidence/w.P018_ROOT/'completion_receipt.json').write_text('{}')
    task=w.p018_authority(evidence,live(evidence),NOW)
    assert task['state']=='RUNNING' and task['running'] and not task['stage_complete']


def test_stale_is_not_terminal(evidence,monkeypatch):
    monkeypatch.setattr(w,'p018_progress',lambda *a:dict(log_age_seconds=301))
    task=w.p018_authority(evidence,live(evidence),NOW)
    assert task['state']=='RUNNING_PROGRESS_STALE' and task['running']
    plan=r.plan_recovery(dict(current_authority='p018_training',authority_tasks={'p018_training':task}),{}, {'bad':{}})
    assert plan['decision']=='LEAD_ACTION_QUEUED' and not plan['automatic_execution']


@pytest.mark.parametrize('changes',[dict(invocation_id='old'),dict(exec_start='wrong'),dict(main_pid_alive=False),
    dict(result='exit-code'),dict(exec_main_status='1'),dict(load_state='not-found')])
def test_identity_or_liveness_failure_no_retry(evidence,changes):
    task=w.p018_authority(evidence,live(evidence,**changes),NOW)
    assert task['state']=='NEEDS_AGENT_ANALYSIS'
    if 'result' not in changes and 'exec_main_status' not in changes:
        assert not task['running']
    assert task['approved_action_id'] is None


def test_missing_protocol_never_uses_old_authority(evidence,monkeypatch):
    monkeypatch.setattr(w,'file_sha256',lambda _: 'wrong')
    assert w.p018_present(evidence)
    assert w.p018_authority(evidence,live(evidence),NOW)['state']=='NEEDS_AGENT_ANALYSIS'


def test_retained_success_queues_audit_not_scientific_complete(evidence):
    task=w.p018_authority(evidence,live(evidence,sub_state='exited',main_pid=0,main_pid_alive=False,exec_main_code='1'),NOW)
    assert task['state']=='TERMINAL_AUDIT_PENDING' and not task['scientific_admission']
    assert 'finalizer' in task['next_action']
    plan=r.plan_recovery(dict(current_authority='p018_training',authority_tasks={'p018_training':task}),{}, {})
    assert plan['decision']=='LEAD_ACTION_QUEUED' and 'command' not in plan


def test_inactive_dead_not_success(evidence):
    assert w.p018_authority(evidence,live(evidence,active_state='inactive',sub_state='dead',main_pid=0),NOW)['state']=='NEEDS_AGENT_ANALYSIS'


def test_resource_marker_preserved(evidence):
    (evidence/w.P018_ROOT/'resource_or_deadline_violation').touch()
    task=w.p018_authority(evidence,live(evidence),NOW)
    assert task['state']=='NEEDS_AGENT_ANALYSIS' and any('watchdog' in x for x in task['identity_issues'])
    assert task['running']  # A failed guard does not make a still-live owned PID idle.


def test_new_authority_stale_running_not_idle_and_resource_alert(evidence,monkeypatch):
    legacy=dict(alerts=['TRAIN16_PENDING_WITH_NO_RUNNING_UNIT_FOR_300_SECONDS','RESOURCE_ALERT'],blocker_reasons=[],
        no_running_since_utc='old',no_running_duration_seconds=999)
    monkeypatch.setattr(w,'_build_legacy_sample',lambda *a:legacy)
    monkeypatch.setattr(w,'p018_progress',lambda *a:dict(log_age_seconds=301))
    sample=w.build_sample(evidence,{}, {w.P018_UNIT:live(evidence)},dict(mem_available_gib=25,mem_free_gib=19),NOW)
    assert sample['current_authority']=='p018_training' and sample['active_units']==[w.P018_UNIT]
    assert sample['no_running_duration_seconds']==0 and sample['no_running_since_utc'] is None
    assert 'RESOURCE_ALERT' in sample['alerts'] and any('MEM_FREE' in x for x in sample['alerts'])
    assert 'historical_legacy_sample' not in sample['historical_legacy_sample']


def test_progress_requires_actual_lr_protocol_and_update(tmp_path):
    path=tmp_path/'run.log'
    rows=[dict(event='accumulation_update',update=16,consumed_windows=128,actual_learning_rate=1e-5,training_protocol_sha256=w.P018_PROTOCOL_SHA256),
          dict(event='accumulation_update',update=12,consumed_windows=96,actual_learning_rate=1.5625e-7,training_protocol_sha256=w.P018_PROTOCOL_SHA256)]
    path.write_text('\n'.join(json.dumps(x) for x in rows))
    assert w.p018_progress(path,NOW)['completed_updates']==12


def test_running_resource_alert_is_actionable():
    task=dict(state='RUNNING',next_action='inspect resources')
    sample=dict(current_authority='p018_training',authority_tasks={'p018_training':task},alerts=['MEMORY'])
    decision=r.plan_recovery(sample,{}, {})
    assert decision['decision']=='LEAD_ACTION_QUEUED' and not decision['automatic_execution']
