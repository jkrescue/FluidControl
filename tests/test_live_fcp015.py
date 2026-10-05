import json
import pytest
from test_live_fcp013_training import dashboard as MODULE


def live_state():
    return ('ActiveState=active\nSubState=running\nMainPID=123\n'
            'InvocationID=7842742926284d0c94b0383163d5dc0b\n'
            'ExecStart=/repo/artifacts/fcp015_window_accumulation_source_20261005_immutable/scripts/run_fcp015_window_accumulation_spark.sh --execute\n')


def test_actual_updates_are_distinct_from_windows():
    log = '\n'.join(json.dumps({'event':'accumulation_update','update':i,'consumed_windows':i*8}) for i in (1,2,3))
    result = MODULE._parse_fcp015_live(live_state(), log)
    assert result['running'] and result['updates'] == 3 and result['consumed_windows'] == 24
    assert result['admission'] is False


@pytest.mark.parametrize('old,new', [('SubState=running','SubState=exited'),
    ('7842742926284d0c94b0383163d5dc0b','other'),('MainPID=123','MainPID=0'),
    ('--execute','--dry-run')])
def test_wrong_or_terminal_invocation_not_running(old,new):
    assert not MODULE._parse_fcp015_live(live_state().replace(old,new), '')['running']


def test_partial_and_inconsistent_rows_do_not_inflate_progress():
    log = '{partial\n' + json.dumps({'event':'accumulation_update','update':171,'consumed_windows':171})
    assert MODULE._parse_fcp015_live(live_state(),log)['updates'] == 0


def test_missing_bound_evidence_not_verified(tmp_path):
    assert MODULE._fcp015_training(tmp_path) == {'verified': False}
