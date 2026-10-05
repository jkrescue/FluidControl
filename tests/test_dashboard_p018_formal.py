from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import sys
import serve_live_research_dashboard as d


def test_live_formal_preserves_authority_without_admission(monkeypatch):
    task = {'running': True, 'state': 'RUNNING', 'scientific_admission': False,
            'observed_utc': '2026-10-05T14:36:00+00:00'}
    def authority(root, state):
        assert root == Path('/project') and state == {'pid': 42}
        return task
    monkeypatch.setitem(sys.modules, 'watch_training_evaluation_state', SimpleNamespace(
        P018_FORMAL_UNIT='exact', unit_state=lambda unit: {'pid': 42},
        p018_formal_authority=authority))
    result = d._fcp018_formal_live(Path('/project'))
    assert result['observed'] and result['task'] is task
    assert result['sampled_at_utc'] == task['observed_utc']
    assert result['admission'] is False


def test_missing_monitor_does_not_invent_progress(monkeypatch):
    monkeypatch.setitem(sys.modules, 'watch_training_evaluation_state', SimpleNamespace())
    assert d._fcp018_formal_live(Path('/project')) == {'observed': False, 'admission': False}


def test_formal_card_precedes_completed_training():
    source = Path(d.__file__).read_text()
    assert source.index('const formal018=d.p018_formal_live;') < source.index('const reduced=d.reduced_rate_training;')
