import json
import pytest
import serve_live_research_dashboard as d


def state(**updates):
    fields=dict(InvocationID='1ca4654aab074278bb2efdfff8dbc1eb',
        ExecStart=d.FCP018_LAUNCHER+' --execute FCP018_APPROVAL_SHA256='+d.FCP018_APPROVAL,
        ActiveState='active',SubState='running',MainPID='42')
    fields.update(updates)
    return '\n'.join(k+'='+v for k,v in fields.items())


def log(**updates):
    row=dict(event='accumulation_update',update=7,consumed_windows=56,
        actual_learning_rate=1.5625e-7,training_protocol_sha256=d.FCP018_PROTOCOL)
    row.update(updates);return json.dumps(row)


def test_running_actual_counts_not_admission():
    p=d._parse_fcp018_live(state(),log(),2)
    assert p['verified'] and p['running'] and p['progress_fresh']
    assert (p['updates'],p['consumed_windows'])==(7,56)
    assert not p['admission']


@pytest.mark.parametrize('changes',[{'InvocationID':'old'},{'ExecStart':'wrong'},{'MainPID':'invalid'}])
def test_wrong_execution_rejected(changes):
    assert not d._parse_fcp018_live(state(**changes),log(),2)['verified']


@pytest.mark.parametrize('changes',[{'actual_learning_rate':1e-5},{'training_protocol_sha256':'old'},
    {'update':True},{'update':172},{'consumed_windows':55}])
def test_bad_progress_not_counted(changes):
    assert d._parse_fcp018_live(state(),log(**changes),2)['updates']==0


def test_stale_log_does_not_mean_stopped():
    p=d._parse_fcp018_live(state(),log(),301)
    assert p['running'] and not p['progress_fresh']


def test_exited_not_scientific_success():
    p=d._parse_fcp018_live(state(SubState='exited',MainPID='0'),log(update=171,consumed_windows=1368),2)
    assert p['verified'] and not p['running'] and not p['admission']
    assert not p['independent_terminal_audit']
