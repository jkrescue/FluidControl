import importlib.util
from pathlib import Path
import numpy as np
import pytest

source = Path(__file__).with_name('recover_accelerated_long_h5_metrics.py')
if not source.exists():
    source = Path(__file__).resolve().parents[1] / 'scripts/recover_accelerated_long_h5_metrics.py'
spec = importlib.util.spec_from_file_location('recovery', source)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture():
    return np.column_stack((148 + .005 * np.arange(1, 2481), np.ones(2480), np.zeros(2480)))


def test_three_fixed_windows_open_left():
    a = fixture()
    full = m.window(a, 148, 160.4)
    first = m.window(a, 148, 154.2)
    last = m.window(a, 154.2, 160.4)
    assert [len(full), len(first), len(last)] == [2480, 1240, 1240]
    assert np.array_equal(np.concatenate((first, last)), full)
    assert first[-1, 0] == 154.2 and last[0, 0] > 154.2


def test_inclusive_legacy_bug_reproduced():
    a = fixture()
    assert len(a[(a[:, 0] >= 154.2 - 1e-8) & (a[:, 0] <= 160.4 + 1e-8)]) == 1241


@pytest.mark.parametrize('mode', ['missing', 'duplicate', 'offgrid', 'nan'])
def test_bad_window_rejected(mode):
    a = fixture()
    if mode == 'missing': a = a[1:]
    if mode == 'duplicate': a = np.concatenate((a, a[:1]))
    if mode == 'offgrid': a[10, 0] += .001
    if mode == 'nan': a[10, 0] = np.nan
    with pytest.raises(ValueError): m.window(a, 148, 160.4)


def test_default_no_execution(monkeypatch, capsys):
    import sys
    monkeypatch.setattr(sys, 'argv', ['recovery'])
    monkeypatch.setattr(m, 'recover', lambda: pytest.fail('unexpected execution'))
    m.main()
    assert 'PREPARATION_ONLY' in capsys.readouterr().out
