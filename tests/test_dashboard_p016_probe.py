import json
from pathlib import Path

import pytest
import serve_live_research_dashboard as d


def state(**changes):
    fields=dict(InvocationID=d.FCP016_INVOCATION,ExecStart=d.FCP016_LAUNCHER+' --execute FCP016_APPROVAL_SHA256='+d.FCP016_APPROVAL,
                ActiveState='active',SubState='running',MainPID='42',Result='success',ExecMainCode='0',ExecMainStatus='0')
    fields.update(changes)
    return '\n'.join(k+'='+v for k,v in fields.items())


def terminal():
    return dict(status='FC_P016_FIXED_PANEL_FIT_COMPLETE_NOT_ADMISSION',training_experiment='FC-P016',optimizer_steps=32,
        probe_sha256='18b210077a93bbd21a327ae6578a73fb371bf0d2f48d0541f6ca8bd87aa42bbb',candidate_saved=False,
        validation_accessed=False,frozen_test_accessed=False,ppo_executed=False,panels=[dict(update=i) for i in (0,8,16,32)])


def test_running_lower_bound_not_current_update():
    log=json.dumps(dict(event='panel_window',update=16,global_index=160))
    r=d._parse_fcp016_live(state(),log,3)
    assert r['running'] and r['minimum_completed_updates']==16 and not r['terminal']
    assert not r['admission'] and 'updates' not in r


def test_stale_log_does_not_fake_stopped_process():
    r=d._parse_fcp016_live(state(),' ',301)
    assert r['running'] and not r['progress_fresh'] and r['minimum_completed_updates'] is None


@pytest.mark.parametrize('change',[dict(InvocationID='old'),dict(ExecStart='wrong'),dict(MainPID='invalid')])
def test_wrong_identity(change):
    assert not d._parse_fcp016_live(state(**change),'',0)['verified']


def test_result_during_running_not_terminal():
    assert not d._parse_fcp016_live(state(),'',0,terminal())['terminal']


def test_terminal_not_admission():
    r=d._parse_fcp016_live(state(SubState='exited',MainPID='0',ExecMainCode='1'),'',0,terminal())
    assert r['terminal'] and not r['running'] and not r['admission'] and not r['independent_terminal_audit']


@pytest.mark.parametrize('change',[dict(Result='exit-code'),dict(ExecMainStatus='1'),dict(InvocationID='old')])
def test_failed_or_other_run_not_complete(change):
    args=dict(SubState='exited',MainPID='0',ExecMainCode='1');args.update(change)
    assert not d._parse_fcp016_live(state(**args),'',0,terminal()).get('terminal',False)


def test_log_never_proves_completion():
    log=json.dumps(dict(event='panel_window',update=32,global_index=160))
    r=d._parse_fcp016_live(state(SubState='exited',MainPID='0',ExecMainCode='1'),log,0)
    assert r['minimum_completed_updates']==32 and not r['terminal']


def test_invalid_progress_ignored():
    log='\n'.join(json.dumps(x) for x in [dict(event='training_window',update=31),dict(event='panel_window',update=17,global_index=160)])
    assert d._parse_fcp016_live(state(),log,0)['minimum_completed_updates'] is None


def test_missing_evidence_failclosed(tmp_path):
    assert not d._fcp016_probe(tmp_path)['verified']
    assert not d._fcp015_formal_result(tmp_path)['verified']


def test_actual_p015_receipt():
    result=d._fcp015_formal_result(Path(__file__).resolve().parents[1])
    assert result==dict(verified=True,admission=False,joint_pass=1,cd_pass=5,rms_pass=2,mean_pass=2)
