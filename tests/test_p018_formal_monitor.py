import json
import shlex
from datetime import datetime, UTC, timedelta

import pytest
import watch_training_evaluation_state as w
import reconcile_training_evaluation_state as r


@pytest.fixture
def case(tmp_path, monkeypatch):
    now=datetime.now(UTC)
    hashes={w.P018_SUPERVISOR:w.P018_SUPERVISOR_SHA,w.P018_FORMAL_APPROVAL:w.P018_FORMAL_APPROVAL_SHA,
        w.P018_CHAIN/'receipt.json':w.P018_CHAIN_SHA,
        w.P018_CHAIN/'scripts/run_fcp008_posteval_spark.sh':w.P018_RUNNER_SHA,
        w.P018_APPROVAL:w.P018_APPROVAL_SHA256,w.P018_OBSERVATION:w.P018_OBSERVATION_SHA256}
    monkeypatch.setattr(w,'file_sha256',lambda p:hashes[p.relative_to(tmp_path)])
    monitor=tmp_path/w.P018_FORMAL_MONITOR;monitor.mkdir(parents=True)
    (monitor/'runner.log').write_text('evaluation is running\n')
    row=dict(timestamp_utc=now.isoformat(),mem_available_kib=30*1024**2,mem_free_kib=25*1024**2)
    (monitor/'memory.jsonl').write_text(json.dumps(row)+'\n')
    approval=tmp_path/w.P018_FORMAL_APPROVAL;approval.parent.mkdir(parents=True)
    approval.write_text(json.dumps(dict(candidate_model_sha256='a'*64,candidate_state_sha256='b'*64,dual_manifest_sha256='c'*64)))
    state=dict(load_state='loaded',invocation_id=w.P018_FORMAL_INVOCATION,
        exec_start='{ argv[]='+shlex.join(w.p018_formal_argv(tmp_path))+' ; }',
        active_state='activating',sub_state='start',main_pid=56992,main_pid_alive=True,
        result='success',exec_main_code='0',exec_main_status='0')
    return tmp_path,now,state,hashes


def test_oneshot_start_is_running_not_idle(case):
    repo,now,state,_=case
    value=w.p018_formal_authority(repo,state,now)
    assert value['state']=='RUNNING' and value['running']
    assert not value['stage_complete'] and not value['scientific_admission']
    plan=r.plan_recovery(dict(current_authority='p018_formal',authority_tasks={'p018_formal':value}),{},r.APPROVED_ACTIONS)
    assert plan['decision']=='NO_ACTION_AUTHORITY_RUNNING' and not plan['automatic_execution']


@pytest.mark.parametrize('key,value',[('invocation_id','wrong'),('exec_start','wrong'),('load_state','not-found')])
def test_wrong_identity_never_claims_running(case,key,value):
    repo,now,state,_=case;state[key]=value
    task=w.p018_formal_authority(repo,state,now)
    assert not task['running'] and task['state']=='NEEDS_AGENT_ANALYSIS'


def test_stale_memory_preserves_live_process(case):
    repo,now,state,_=case
    task=w.p018_formal_authority(repo,state,now+timedelta(seconds=40))
    assert task['running'] and task['state']=='RUNNING_REQUIRES_REVIEW'
    assert 'Formal resource samples stale' in task['identity_issues']


def test_early_result_never_overrides_running(case):
    repo,now,state,_=case
    path=repo/w.P018_ROOT/'posteval_fc_p018';path.mkdir()
    (path/'receipt.json').write_text('{"status":"FC_P018_POSTEVAL_COMPLETE"}')
    assert w.p018_formal_authority(repo,state,now)['state']=='RUNNING'


@pytest.mark.parametrize('value',[19*1024**2,float('nan'),None])
def test_bad_resource_observation_requires_review_not_restart(case,value):
    repo,now,state,_=case
    path=repo/w.P018_FORMAL_MONITOR/'memory.jsonl'
    row=json.loads(path.read_text());row['mem_free_kib']=value;path.write_text(json.dumps(row)+'\n')
    task=w.p018_formal_authority(repo,state,now)
    assert task['running'] and task['state']=='RUNNING_REQUIRES_REVIEW'
    assert not task['automatic_recovery_eligible']


def test_failed_unit_is_not_terminal_success(case):
    repo,now,state,_=case
    state.update(active_state='failed',sub_state='failed',main_pid=0,result='exit-code',exec_main_status='75')
    task=w.p018_formal_authority(repo,state,now)
    assert task['state']=='NEEDS_AGENT_ANALYSIS' and not task['running']
    assert task['identity_issues']


def test_terminal_requires_guard_and_is_only_pending_audit(case):
    repo,now,state,_=case
    state.update(active_state='active',sub_state='exited',main_pid=0,main_pid_alive=False,exec_main_code='1')
    assert w.p018_formal_authority(repo,state,now)['state']=='NEEDS_AGENT_ANALYSIS'
    guard=dict(status='FC_P018_FORMAL_RESOURCE_GUARD_COMPLETE_NOT_ADMISSION',
        supervisor_sha256=w.P018_SUPERVISOR_SHA,runner_sha256=w.P018_RUNNER_SHA,
        approval_sha256=w.P018_FORMAL_APPROVAL_SHA,chain_receipt_sha256=w.P018_CHAIN_SHA,runner_exit_code=0,
        training_approval_sha256=w.P018_APPROVAL_SHA256,observation_sha256=w.P018_OBSERVATION_SHA256,
        command=['bash',str(repo/w.P018_CHAIN/'scripts/run_fcp008_posteval_spark.sh'),'--execute'])
    (repo/w.P018_FORMAL_MONITOR/'receipt.json').write_text(json.dumps(guard))
    task=w.p018_formal_authority(repo,state,now+timedelta(hours=1))
    assert task['state']=='TERMINAL_AUDIT_PENDING' and not task['scientific_admission']
    assert not task['running'] and not task['stage_complete']


def test_step_proof_must_bind_approved_candidate(case):
    repo,now,state,_=case
    path=repo/w.P018_ROOT/'posteval_fc_p018/step_receipts';path.mkdir(parents=True)
    row=dict(status='FC_P018_POSTEVAL_STEP_COMPLETE',step='validation10',
        candidate_kind='fcp018_reduced_rate_dual_fno',formal_evaluation_approval_sha256=w.P018_FORMAL_APPROVAL_SHA,
        posteval_chain_receipt_sha256=w.P018_CHAIN_SHA,checkpoint_sha256='a'*64,
        checkpoint_state_sha256='b'*64,dual_manifest_sha256='c'*64)
    target=path/'validation10.json';target.write_text(json.dumps(row))
    assert w.p018_formal_authority(repo,state,now)['progress']['completed_steps']==['validation10']
    row['checkpoint_sha256']='wrong';target.write_text(json.dumps(row))
    task=w.p018_formal_authority(repo,state,now)
    assert task['progress']['completed_steps']==[] and task['running']
    assert task['state']=='RUNNING_REQUIRES_REVIEW'


def test_formal_authority_supersedes_training_without_historical_loop(case,monkeypatch):
    repo,now,state,_=case
    legacy=dict(alerts=[],blocker_reasons=[],resources={})
    monkeypatch.setattr(w,'_build_legacy_sample',lambda *a:legacy)
    sample=w.build_sample(repo,None,{w.P018_FORMAL_UNIT:state},dict(mem_available_gib=30,mem_free_gib=25),now)
    assert sample['current_authority']=='p018_formal'
    assert sample['active_units']==[w.P018_FORMAL_UNIT]
    assert sample['no_running_duration_seconds']==0


def test_slow_legacy_poll_uses_resource_read_time(case,monkeypatch):
    repo,start,state,_=case
    read_time=start+timedelta(seconds=8)
    path=repo/w.P018_FORMAL_MONITOR/'memory.jsonl'
    row=json.loads(path.read_text());row['timestamp_utc']=(start+timedelta(seconds=6)).isoformat()
    path.write_text(json.dumps(row)+'\n')
    def slow_legacy(*args):
        monkeypatch.setattr(w,'utc_now',lambda:read_time)
        return dict(timestamp_utc=start.isoformat(),alerts=[],blocker_reasons=[])
    monkeypatch.setattr(w,'_build_legacy_sample',slow_legacy)
    sample=w.build_sample(repo,None,{w.P018_FORMAL_UNIT:state},dict(mem_available_gib=30,mem_free_gib=25),start)
    task=sample['authority_tasks']['p018_formal']
    assert task['state']=='RUNNING'
    assert task['progress']['memory_age_seconds']==2
    assert sample['timestamp_utc']==read_time.isoformat()==task['observed_utc']


@pytest.mark.parametrize('offset',[-1,31])
def test_live_read_still_rejects_real_future_or_stale_clock(case,monkeypatch,offset):
    repo,now,state,_=case
    monkeypatch.setattr(w,'utc_now',lambda:now+timedelta(seconds=offset))
    task=w.p018_formal_authority(repo,state)
    assert task['running'] and task['state']=='RUNNING_REQUIRES_REVIEW'
    assert task['progress']['memory_age_seconds']==offset
