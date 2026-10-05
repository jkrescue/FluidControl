import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location('dashboard', Path(__file__).parent.parent / 'scripts/serve_live_research_dashboard.py')
dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard)


def state(**changes):
    return dict(InvocationID='4567f6d393414bba8baf2239d16960a7', MainPID='123',
                ActiveState='activating', SubState='start', Result='success',
                ExecMainCode='0', ExecMainStatus='0', **changes)


def event(step, arm='A_original'):
    return json.dumps(dict(event='arm_update_complete', arm=arm, update=step))


def test_running_progress_is_observed_updates_only():
    result = dashboard._parse_fcp020_live(state(), event(1)+'\n'+event(2), True)
    assert result['running'] and not result['exited_success']
    assert result['updates'] == dict(A_original=2, B_symmetric_tail=0)
    assert result['admission'] is False


def test_default_exit_zero_is_not_terminal_and_process_must_match():
    assert not dashboard._parse_fcp020_live(state(), '', True)['exited_success']
    assert dashboard._parse_fcp020_live(state(), '', False) == {'verified': False}
    wrong = state(); wrong['InvocationID'] = 'other'
    assert dashboard._parse_fcp020_live(wrong, '', True) == {'verified': False}


def test_duplicate_gap_and_invalid_progress_rejected():
    for log in (event(1)+'\n'+event(1), event(2), event(17), event(True), event(1, 'unknown')):
        assert dashboard._parse_fcp020_live(state(), log, True) == {'verified': False}


def test_retained_exit_never_means_scientific_acceptance():
    terminal = state(); terminal.update(MainPID='0', ActiveState='active', SubState='exited', ExecMainCode='1')
    result = dashboard._parse_fcp020_live(terminal, event(1), False)
    assert result['exited_success'] and not result['running'] and not result['admission']


def test_unverified_or_missing_terminal_result_not_promoted(tmp_path):
    assert dashboard._fcp020_result(tmp_path) == {'verified': False}
    artifact = tmp_path / 'artifacts/fcp020_symmetric_statistics_20261005/result.json'
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"comparison":{"local_support":true}}')
    assert dashboard._fcp020_result(tmp_path) == {'verified': False}
